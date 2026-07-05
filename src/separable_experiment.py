"""
Score matching on WELL-SEPARATED data.

Previous failure root cause: data had NO structure (separability ratio 1.1x).
On a 1024-dim sphere, spread=0.15 produces points that are as far from their
own cluster center as from other centers.

This experiment: 5 tight clusters (spread=0.005, ratio=9x separable).
If score matching works here, the methodology is validated. Then we can
talk about scaling to more realistic data.
"""
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from phase2_score_matching_scaled import (
    ScoreNetworkV2, get_sigma_schedule, dsm_loss_v2, cosine_lr_schedule,
    evaluate_t1_direction, DIM, DEVICE
)

RESULTS_DIR = REPO / "results"


def make_separable_data(n_samples=2000, n_clusters=5, dim=DIM, cluster_spread=0.005):
    """Tight, well-separated clusters on the unit sphere."""
    torch.manual_seed(42)
    centers = torch.randn(n_clusters, dim)
    centers = F.normalize(centers, dim=-1)

    real = []
    labels = []
    for i in range(n_samples):
        c_idx = torch.randint(0, n_clusters, (1,)).item()
        c = centers[c_idx]
        point = c + torch.randn(dim) * cluster_spread
        point = F.normalize(point, dim=-1)
        real.append(point)
        labels.append(c_idx)
    real = torch.stack(real).to(DEVICE)

    noise = torch.randn(n_samples, dim)
    noise = F.normalize(noise, dim=-1).to(DEVICE)

    centers = centers.to(DEVICE)
    return real, noise, centers, labels


def train_and_evaluate(n_clusters, cluster_spread, n_epochs=2000):
    config_name = f"sep_c{n_clusters}_s{cluster_spread}"
    print(f"\n{'#'*60}")
    print(f"# {config_name}")
    print(f"{'#'*60}")

    real, noise, centers, labels = make_separable_data(
        n_samples=2000, n_clusters=n_clusters, cluster_spread=cluster_spread
    )

    # Verify separability
    dists = torch.cdist(real.cpu(), centers.cpu())
    assigned = dists.argmin(dim=1)
    dist_to_own = dists.gather(1, assigned.unsqueeze(1)).squeeze()
    intra = dist_to_own.mean().item()
    inter = torch.cdist(centers.cpu(), centers.cpu())
    inter = inter[inter > 0].mean().item()
    ratio = inter / intra
    print(f"Separability: intra={intra:.4f} inter={inter:.4f} ratio={ratio:.1f}x")

    # Train
    sigmas = get_sigma_schedule(10)
    score_net = ScoreNetworkV2(dim=DIM, hidden=1024, n_blocks=8).to(DEVICE)
    optimizer = torch.optim.AdamW(score_net.parameters(), lr=1e-3, weight_decay=1e-4)

    train_data = real[:1600]
    print(f"Training {n_epochs} epochs...")
    t0 = time.time()
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = dsm_loss_v2(score_net, train_data, sigmas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
        cosine_lr_schedule(optimizer, epoch, n_epochs, warmup=100, base_lr=1e-3)
        optimizer.step()

        if epoch % 400 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch}: loss={loss.item():.4f}")
            losses.append({"epoch": epoch, "loss": loss.item()})

    print(f"Training: {time.time()-t0:.0f}s")
    score_net.eval()

    # T1: direction
    t1 = evaluate_t1_direction(score_net, real, noise)

    # T2: gradient descent sampling
    print(f"\n--- T2: Gradient descent sampling (σ=0.5) ---")
    real_ref = real[:800]
    x = torch.randn(200, DIM, device=DEVICE)
    x = F.normalize(x, dim=-1)
    distances = []
    for step in range(1000):
        with torch.no_grad():
            sigma_batch = torch.full((200, 1), 0.5, device=DEVICE)
            score = score_net(x, sigma_batch)
        x = x + 0.01 * score
        x = F.normalize(x, dim=-1)
        if step % 100 == 0 or step == 999:
            with torch.no_grad():
                dist = torch.cdist(x, real_ref).min(dim=1)[0]
                d = {"step": step, "median": float(dist.median()),
                     "near_03": float((dist < 0.3).float().mean()),
                     "near_01": float((dist < 0.1).float().mean())}
                distances.append(d)
                print(f"  step {step:4d}: dist={d['median']:.4f}  "
                      f"near(<0.3)={d['near_03']:.2%}  near(<0.1)={d['near_01']:.2%}")

    medians = [d["median"] for d in distances]
    decrease = (medians[0] - medians[-1]) / medians[0] if medians[0] > 0 else 0

    # T3: sample quality
    print(f"\n--- T3: Final sample quality ---")
    samples = x.cpu()
    real_cpu = real_ref.cpu()
    with torch.no_grad():
        dist_all = torch.cdist(samples, real_cpu).min(dim=1)[0]
        near_03 = float((dist_all < 0.3).float().mean())
        near_01 = float((dist_all < 0.1).float().mean())
        sample_std = samples.std(dim=0).mean().item()
        # Which clusters do samples reach?
        dists_to_centers = torch.cdist(samples, centers.cpu())
        nearest_center = dists_to_centers.argmin(dim=1)
        n_clusters_reached = len(torch.unique(nearest_center))

    print(f"  Near data (<0.3): {near_03:.2%}")
    print(f"  Near data (<0.1): {near_01:.2%}")
    print(f"  Sample diversity (std): {sample_std:.4f}")
    print(f"  Clusters reached: {n_clusters_reached}/{n_clusters}")

    result = {
        "config": config_name,
        "n_clusters": n_clusters,
        "cluster_spread": cluster_spread,
        "separability_ratio": ratio,
        "losses": losses,
        "t1_direction_cos": t1["avg_cos_to_true"],
        "t2_decrease": decrease,
        "t2_distances": distances,
        "t3_near_03": near_03,
        "t3_near_01": near_01,
        "t3_clusters_reached": n_clusters_reached,
        "t3_sample_std": sample_std,
    }

    # Verdict
    if t1["avg_cos_to_true"] > 0.8 and decrease > 0.3 and near_03 > 0.3:
        verdict = "✅ FULL SUCCESS: score matching generates data"
    elif t1["avg_cos_to_true"] > 0.8 and decrease > 0.2:
        verdict = "🔶 PARTIAL: score guides correctly, samples near data"
    elif t1["avg_cos_to_true"] > 0.8:
        verdict = "🔶 Direction learned, sampling incomplete"
    else:
        verdict = "❌ FAIL"
    result["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")

    return result


if __name__ == "__main__":
    results = {"experiment": "separable_score_matching", "timestamp": datetime.now().isoformat()}

    # Test 1: 5 clusters, very tight
    r1 = train_and_evaluate(n_clusters=5, cluster_spread=0.005, n_epochs=2000)
    results["config_5c_0005"] = r1

    # Save
    out = RESULTS_DIR / "separable_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")

    # If that worked, try more clusters
    if "SUCCESS" in r1["verdict"]:
        print("\n\nExpanding to more clusters...")
        r2 = train_and_evaluate(n_clusters=10, cluster_spread=0.005, n_epochs=2000)
        results["config_10c_0005"] = r2
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
