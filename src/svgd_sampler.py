"""
SVGD with exact KDE score — deterministic particle transport.

SVGD doesn't need Langevin noise. It deterministically transports
particles toward the target distribution using:
  φ(x_i) = (1/n) Σ_j [ k(x_j, x_i) ∇log p(x_i) + ∇_{x_i} k(x_j, x_i) ]

The kernel repulsion term prevents collapse.

Uses exact KDE score (no neural network). Fast: 200 particles, 300 iters.
"""
import math
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from phase2_score_matching_scaled import DIM, DEVICE
from separable_experiment import make_separable_data

RESULTS_DIR = REPO / "results"


def kde_score(x, data, sigma):
    """Exact score from Gaussian KDE. Batched for memory."""
    scores = []
    batch = 50
    for i in range(0, x.shape[0], batch):
        xb = x[i:i+batch]
        diff = xb.unsqueeze(1) - data.unsqueeze(0)  # [batch, n, d]
        sq_dist = (diff ** 2).sum(dim=-1)  # [batch, n]
        log_w = -sq_dist / (2 * sigma ** 2)
        log_w = log_w - log_w.max(dim=-1, keepdim=True).values
        w = log_w.exp()
        weighted_diff = (diff * w.unsqueeze(-1)).sum(dim=1)
        score = -weighted_diff / (w.sum(dim=-1, keepdim=True) * sigma ** 2)
        scores.append(score)
    return torch.cat(scores)


def svgd_step(x, data, sigma_score, h_bandwidth, step_size):
    """One SVGD iteration."""
    n = x.shape[0]

    # Score at each particle
    score = kde_score(x, data, sigma_score)  # [n, d]

    # RBF kernel + gradient
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # [n, n, d]
    sq_dist = (diff ** 2).sum(dim=-1)  # [n, n]
    k_xy = torch.exp(-sq_dist / h_bandwidth)

    # ∇_{x_i} k(x_j, x_i) = -2(x_i - x_j)/h * k(x_j, x_i)
    # Sum over j
    grad_k = (-2.0 / h_bandwidth * k_xy.unsqueeze(-1) * diff).sum(dim=1)  # [n, d]

    # SVGD update
    phi = (k_xy @ score) / n + grad_k / n
    return phi


def run_svgd(n_particles=200, n_iters=500, sigma_score=0.05,
             h_bandwidth="median", step_size=0.01):
    print("=" * 70)
    print(f"SVGD: {n_particles} particles, {n_iters} iters")
    print("=" * 70)

    torch.manual_seed(42)
    real, _, centers, _ = make_separable_data(
        n_samples=2000, n_clusters=5, cluster_spread=0.005
    )
    real_ref = real[:800].cpu()
    data = real[:1600].cpu()

    # Initialize particles on sphere
    x = F.normalize(torch.randn(n_particles, DIM), dim=-1)

    t0 = time.time()
    distances = []

    for it in range(n_iters):
        # Compute bandwidth (median heuristic)
        if h_bandwidth == "median":
            with torch.no_grad():
                sample = x[torch.randperm(min(100, n_particles))[:100]]
                sd = torch.cdist(sample, sample) ** 2
                sd = sd[sd > 0]
                h = sd.median().item() if len(sd) > 0 else 1.0
        else:
            h = h_bandwidth

        phi = svgd_step(x, data, sigma_score, h, step_size)
        x = x + step_size * phi
        x = F.normalize(x, dim=-1)  # project to sphere

        if it % 50 == 0 or it == n_iters - 1:
            dist = torch.cdist(x, real_ref).min(dim=1)[0]
            near_03 = float((dist < 0.3).float().mean())
            near_01 = float((dist < 0.1).float().mean())
            dists_c = torch.cdist(x, centers.cpu())
            modes = len(torch.unique(dists_c.argmin(dim=1)))
            distances.append({
                "iter": it, "near_03": near_03, "near_01": near_01,
                "median_dist": float(dist.median()), "modes": modes,
            })
            print(f"  iter {it:3d}: near(<0.3)={near_03:.2%}  "
                  f"near(<0.1)={near_01:.2%}  modes={modes}/5  "
                  f"median={dist.median():.4f}  h={h:.4f}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")

    # Final
    dist = torch.cdist(x, real_ref).min(dim=1)[0]
    final = {
        "near_03": float((dist < 0.3).float().mean()),
        "near_01": float((dist < 0.1).float().mean()),
        "near_005": float((dist < 0.05).float().mean()),
        "modes": len(torch.unique(torch.cdist(x, centers.cpu()).argmin(dim=1))),
        "median_dist": float(dist.median()),
        "distances": distances,
    }
    return final


if __name__ == "__main__":
    # Try multiple sigma and bandwidth configs
    configs = [
        {"sigma_score": 0.01, "h": "median", "step": 0.001},
        {"sigma_score": 0.05, "h": "median", "step": 0.001},
        {"sigma_score": 0.01, "h": "median", "step": 0.01},
        {"sigma_score": 0.01, "h": 0.1, "step": 0.01},
    ]

    all_results = {}
    for cfg in configs:
        name = f"sig{cfg['sigma_score']}_h{cfg['h']}_lr{cfg['step']}"
        print(f"\n{'#'*60}")
        print(f"# Config: {name}")
        print(f"{'#'*60}")
        r = run_svgd(n_particles=100, n_iters=300,
                     sigma_score=cfg["sigma_score"],
                     h_bandwidth=cfg["h"], step_size=cfg["step"])
        all_results[name] = r
        print(f"\n  Final: near(<0.3)={r['near_03']:.2%}  "
              f"near(<0.1)={r['near_01']:.2%}  modes={r['modes']}/5")

    print(f"\n{'='*70}")
    print("SVGD SUMMARY")
    print(f"{'='*70}")
    for name, r in all_results.items():
        print(f"  {name}: near<0.3={r['near_03']:.2%}  "
              f"near<0.1={r['near_01']:.2%}  modes={r['modes']}/5")

    out = RESULTS_DIR / "svgd_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
