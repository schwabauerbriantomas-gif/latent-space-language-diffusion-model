"""
PCA sweep: test multiple k values with optimized sampling.

Finding: k=4 gives 22% near data, k=388 gives 0%.
Question: what's the optimal k? And can we push k=4 higher with better sampling?
"""
import math
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from phase2_score_matching_scaled import DIM, DEVICE
from separable_experiment import make_separable_data
from pca_reduced import ScoreNetKD, dsm_loss

RESULTS_DIR = REPO / "results"


def run_single(n_clusters, cluster_spread, k, n_epochs=2000, n_steps_per_sigma=500):
    """Run one experiment with fixed k."""
    torch.manual_seed(42)
    real, _, centers, _ = make_separable_data(
        n_samples=2000, n_clusters=n_clusters, cluster_spread=cluster_spread
    )
    real_cpu = real.cpu()

    # PCA to exactly k dims
    mean = real_cpu.mean(dim=0)
    real_centered = real_cpu - mean
    U, S, Vt = torch.linalg.svd(real_centered, full_matrices=False)
    V = Vt.T
    basis = V[:, :k]
    real_low = real_centered @ basis

    # Train
    net = ScoreNetKD(k_dim=k, hidden=256, n_blocks=6).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    real_low_gpu = real_low.cuda()
    sigmas = torch.tensor([2.0, 1.0, 0.5, 0.2, 0.1, 0.05], device=DEVICE)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = dsm_loss(net, real_low_gpu[:1600], sigmas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
    net.eval()

    # Sample (more steps this time)
    x = torch.randn(300, k, device=DEVICE) * 3
    for si, sigma in enumerate(sigmas):
        alpha = 0.01 * sigma.item() ** 2
        for _ in range(n_steps_per_sigma):
            with torch.no_grad():
                s = net(x, torch.full((300, 1), sigma.item(), device=DEVICE))
            x = x + 0.5 * alpha * s + math.sqrt(alpha) * torch.randn_like(x)

    # Map back
    x_full = (x.cpu() @ basis.T) + mean
    x_sphere = F.normalize(x_full, dim=-1)

    dist = torch.cdist(x_sphere, real_cpu).min(dim=1)[0]
    near_03 = float((dist < 0.3).float().mean())
    near_01 = float((dist < 0.1).float().mean())
    near_005 = float((dist < 0.05).float().mean())

    dists_centers = torch.cdist(x_sphere, centers.cpu())
    modes = len(torch.unique(dists_centers.argmin(dim=1)))

    return {"k": k, "near_03": near_03, "near_01": near_01,
            "near_005": near_005, "modes": modes,
            "median_dist": float(dist.median())}


if __name__ == "__main__":
    print("=" * 70)
    print("PCA k-sweep: finding optimal intrinsic dimensionality")
    print("=" * 70)

    results = []
    for k in [2, 3, 4, 6, 8, 16, 32]:
        t0 = time.time()
        print(f"\n--- k={k} ---")
        r = run_single(n_clusters=5, cluster_spread=0.005, k=k,
                       n_epochs=2000, n_steps_per_sigma=500)
        r["time_s"] = time.time() - t0
        results.append(r)
        print(f"  near(<0.3)={r['near_03']:.2%}  near(<0.1)={r['near_01']:.2%}  "
              f"modes={r['modes']}/5  median={r['median_dist']:.4f}  "
              f"({r['time_s']:.0f}s)")

    print(f"\n{'='*70}")
    print(f"SUMMARY: k-sweep")
    print(f"{'='*70}")
    print(f"{'k':>4} {'near<0.3':>10} {'near<0.1':>10} {'modes':>8} {'median':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['k']:>4} {r['near_03']:>9.2%} {r['near_01']:>9.2%} "
              f"{r['modes']:>5}/5 {r['median_dist']:>9.4f}")

    best = max(results, key=lambda r: r["near_03"])
    print(f"\nBest: k={best['k']} with {best['near_03']:.2%} near data")

    out = RESULTS_DIR / "pca_ksweep_results.json"
    with open(out, "w") as f:
        json.dump({"results": results, "best_k": best["k"]}, f, indent=2)
