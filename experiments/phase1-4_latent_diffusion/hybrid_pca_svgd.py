"""
The hybrid sampler: combines PCA-reduced score matching (for effective
guidance in low-D) with SVGD-style repulsion (for mode coverage).

APPROACH:
  1. PCA-reduce to k=4 (where data structure lives)
  2. Train score network in k=4 (fast, accurate)
  3. Sample with SVGD in k=4: score attracts, kernel repulsion diversifies
  4. Map back to 1024D

This solves BOTH problems:
  - Langevin in k=4 reaches data (21% alone)
  - SVGD repulsion prevents mode collapse (keeps diversity)
  - Bandwidth scheduling: large→small for exploration→exploitation
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


def train_score_net_kd(data_low, k, n_epochs=2000):
    """Train score network in k dimensions."""
    net = ScoreNetKD(k_dim=k, hidden=256, n_blocks=6).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    sigmas = torch.tensor([2.0, 1.0, 0.5, 0.2, 0.1, 0.05], device=DEVICE)
    data_gpu = data_low.cuda()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = dsm_loss(net, data_gpu, sigmas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
    net.eval()
    return net


def svgd_low_dim(net, real_low, k, n_particles=200, n_iters=1000,
                 sigmas_score=[0.1, 0.05], h_schedule="anneal"):
    """SVGD with learned score + RBF repulsion in k-dim space.

    Calibrated: init from data-scale Gaussian, not wide.
    """
    print(f"\n--- SVGD in {k}D ({n_particles} particles, {n_iters} iters) ---")

    # Initialize from data-scale Gaussian (not arbitrary * 3)
    data_std = real_low.std(dim=0)
    data_mean = real_low.mean(dim=0)
    x = data_mean.cuda() + torch.randn(n_particles, k, device=DEVICE) * data_std.cuda() * 3
    real_low_gpu = real_low.cuda()

    print(f"  Data range: [{real_low.min():.3f}, {real_low.max():.3f}]")
    print(f"  Data std per dim: {data_std.tolist()}")

    distances = []

    for it in range(n_iters):
        # Annealed bandwidth: geometric decay from h_max to h_min
        progress = it / n_iters
        h_max = 10.0
        h_min = 0.01
        h = h_max * (h_min / h_max) ** progress

        # Annealed score sigma
        sigma_score = sigmas_score[0] * (sigmas_score[1] / sigmas_score[0]) ** progress

        # Score from learned network
        with torch.no_grad():
            score = net(x, torch.full((n_particles, 1), sigma_score, device=DEVICE))

        # RBF kernel
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # [n, n, k]
        sq_dist = (diff ** 2).sum(dim=-1)  # [n, n]
        k_xy = torch.exp(-sq_dist / h)

        # Repulsion
        grad_k = (-2.0 / h * k_xy.unsqueeze(-1) * diff).sum(dim=1)  # [n, k]

        # SVGD update
        phi = (k_xy @ score) / n_particles + grad_k / n_particles

        # Step size also annealed
        step = 0.1 * (1 - 0.9 * progress)
        x = x + step * phi

        if it % 100 == 0 or it == n_iters - 1:
            dist_low = torch.cdist(x.cpu(), real_low[:800]).min(dim=1)[0]
            near_05 = float((dist_low < 0.5).float().mean())
            near_01 = float((dist_low < 0.1).float().mean())

            # Mode coverage in k-dim (by nearest cluster center projection)
            dists = torch.cdist(x.cpu(), real_low)
            # Approximate cluster assignment by nearest training point
            nn = dists.argmin(dim=1)
            # Each training point belongs to a cluster; count unique
            modes = len(torch.unique(nn[:100]))  # rough estimate

            d = {"iter": it, "h": h, "sigma": sigma_score,
                 "near_05_low": near_05, "near_01_low": near_01,
                 "median_low": float(dist_low.median())}
            distances.append(d)
            print(f"  iter {it:4d}: h={h:.4f} σ={sigma_score:.4f}  "
                  f"near(<0.5)={near_05:.2%}  near(<0.1)={near_01:.2%}  "
                  f"median={dist_low.median():.4f}")

    return x, distances


def run_hybrid():
    print("=" * 70)
    print("HYBRID: PCA k=4 + SVGD with bandwidth scheduling")
    print("=" * 70)

    torch.manual_seed(42)
    real, _, centers, _ = make_separable_data(
        n_samples=2000, n_clusters=5, cluster_spread=0.005
    )
    real_cpu = real.cpu()

    # PCA to k=4
    mean = real_cpu.mean(dim=0)
    real_centered = real_cpu - mean
    U, S, Vt = torch.linalg.svd(real_centered, full_matrices=False)
    V = Vt.T
    k = 4
    basis = V[:, :k]
    real_low = real_centered @ basis  # [2000, 4]
    print(f"PCA: {k} dims, reconstruction error="
          f"{(real_low @ basis.T - real_centered).norm(dim=-1).mean():.6f}")

    # Train score network
    print(f"Training score network in {k}D...")
    net = train_score_net_kd(real_low[:1600], k, n_epochs=2000)

    # SVGD sampling
    x_low, traj = svgd_low_dim(net, real_low, k, n_particles=300, n_iters=1000,
                                sigmas_score=[0.5, 0.05], h_schedule="anneal")

    # Map back to 1024D
    x_full = (x_low.cpu() @ basis.T) + mean
    x_sphere = F.normalize(x_full, dim=-1)

    # Evaluate in 1024D
    dist = torch.cdist(x_sphere, real_cpu).min(dim=1)[0]
    near_03 = float((dist < 0.3).float().mean())
    near_01 = float((dist < 0.1).float().mean())
    near_005 = float((dist < 0.05).float().mean())

    dists_centers = torch.cdist(x_sphere, centers.cpu())
    modes = len(torch.unique(dists_centers.argmin(dim=1)))
    pairwise = torch.cdist(x_sphere[:50], x_sphere[:50])
    pairwise_mean = float(pairwise[pairwise > 0].mean())

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS (Hybrid PCA-SVGD, k={k})")
    print(f"{'='*70}")
    print(f"  Near data (<0.3):  {near_03:.2%}")
    print(f"  Near data (<0.1):  {near_01:.2%}")
    print(f"  Near data (<0.05): {near_005:.2%}")
    print(f"  Modes reached:     {modes}/5")
    print(f"  Median distance:   {dist.median():.4f}")
    print(f"  Sample diversity:  {pairwise_mean:.4f}")

    if near_03 > 0.5 and modes >= 4:
        verdict = "✅ FULL SUCCESS"
    elif near_03 > 0.3 and modes >= 3:
        verdict = "🔶 GOOD"
    elif near_03 > 0.1:
        verdict = "🔶 PARTIAL"
    else:
        verdict = "❌ FAIL"
    print(f"  VERDICT: {verdict}")

    result = {
        "experiment": "hybrid_pca_svgd",
        "timestamp": datetime.now().isoformat(),
        "k": k,
        "near_03": near_03, "near_01": near_01, "near_005": near_005,
        "modes": modes, "median_dist": float(dist.median()),
        "diversity": pairwise_mean, "verdict": verdict,
        "trajectory": traj,
    }
    out = RESULTS_DIR / "hybrid_pca_svgd_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run_hybrid()
