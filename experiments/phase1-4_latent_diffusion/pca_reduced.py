"""
PCA-Reduced Score Matching: the mathematically principled fix.

FINDING: Data on S^1023 has intrinsic dimensionality k=4 (90% var).
Score matching in 1024D wastes 636+ dimensions fighting noise.

APPROACH:
  1. PCA-reduce data to k dimensions
  2. Score match in R^k (standard diffusion, no manifold constraint)
  3. Map samples back to 1024D via PCA inverse
  4. Project to sphere

This should work because:
  - In k=4D, distances are meaningful (no concentration of measure)
  - Standard Langevin works in R^k (proven in 2D control)
  - The data structure is fully captured in k dims
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

RESULTS_DIR = REPO / "results"


class ScoreNetKD(nn.Module):
    """Score network for arbitrary dimension k."""
    def __init__(self, k_dim, hidden=256, n_blocks=6):
        super().__init__()
        self.sigma_mlp = nn.Sequential(
            nn.Linear(64, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.input_proj = nn.Linear(k_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                          nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(n_blocks)
        ])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(),
                                  nn.Linear(hidden, k_dim))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def _sigma_emb(self, sigma):
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        half = 32
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=sigma.device) / half)
        args = sigma * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.sigma_mlp(emb)

    def forward(self, x, sigma):
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0).expand(x.shape[0], 1)
        elif sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        s_emb = self._sigma_emb(sigma)
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


def dsm_loss(net, x, sigmas):
    total = 0.0
    for sigma in sigmas:
        eps = torch.randn_like(x)
        x_noisy = x + sigma * eps
        target = -eps / sigma
        pred = net(x_noisy, sigma.expand(x.shape[0], 1))
        total = total + ((pred - target) ** 2).mean()
    return total / len(sigmas)


def run_pca_experiment(n_clusters=5, cluster_spread=0.005, variance_threshold=0.99):
    print("=" * 70)
    print(f"PCA-Reduced Score Matching (clusters={n_clusters}, spread={cluster_spread})")
    print("=" * 70)

    torch.manual_seed(42)
    real, _, centers, _ = make_separable_data(
        n_samples=2000, n_clusters=n_clusters, cluster_spread=cluster_spread
    )
    real_cpu = real.cpu()

    # --- PCA ---
    mean = real_cpu.mean(dim=0)
    real_centered = real_cpu - mean
    U, S, Vt = torch.linalg.svd(real_centered, full_matrices=False)
    V = Vt.T  # [d, k]

    cumvar = (S ** 2).cumsum(0) / (S ** 2).sum()
    k = (cumvar < variance_threshold).sum().item() + 1
    print(f"PCA: {k} dimensions capture {variance_threshold*100:.0f}% variance")
    print(f"  50% var: {(cumvar < 0.50).sum().item() + 1} dims")
    print(f"  90% var: {(cumvar < 0.90).sum().item() + 1} dims")

    basis = V[:, :k]  # [1024, k]
    real_low = real_centered @ basis  # [2000, k]
    print(f"  Low-dim data: {real_low.shape}")
    print(f"  Range: [{real_low.min():.3f}, {real_low.max():.3f}]")

    # Verify reconstruction quality
    reconstructed = (real_low @ basis.T) + mean
    recon_error = (reconstructed - real_cpu).norm(dim=-1).mean()
    print(f"  Reconstruction error: {recon_error:.6f}")

    # --- Train score network in k-dim ---
    net = ScoreNetKD(k_dim=k, hidden=256, n_blocks=6).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

    real_low_gpu = real_low.cuda()
    sigmas = torch.tensor([2.0, 1.0, 0.5, 0.2, 0.1, 0.05], device=DEVICE)

    print(f"\nTraining score network in {k}D...")
    t0 = time.time()
    for epoch in range(2000):
        optimizer.zero_grad()
        loss = dsm_loss(net, real_low_gpu[:1600], sigmas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        if epoch % 400 == 0 or epoch == 1999:
            print(f"  epoch {epoch}: loss={loss.item():.4f}")
    print(f"Training: {time.time()-t0:.0f}s")
    net.eval()

    # --- T1: score direction in k-dim ---
    print(f"\n--- T1: Score direction in {k}D ---")
    for sigma in [0.05, 0.1, 0.5, 1.0]:
        eps = torch.randn(500, k, device=DEVICE)
        x = real_low_gpu[:500] + sigma * eps
        target = -eps / sigma
        with torch.no_grad():
            pred = net(x, torch.full((500, 1), sigma, device=DEVICE))
            cos = F.cosine_similarity(pred, target, dim=-1).mean().item()
        print(f"  σ={sigma:.2f}: cos(score, true)={cos:+.4f}")

    # --- T2: Annealed Langevin sampling in k-dim ---
    print(f"\n--- T2: Annealed Langevin in {k}D ---")
    x = torch.randn(200, k, device=DEVICE) * 3
    distances_timeline = []

    for si, sigma in enumerate(sigmas):
        alpha = 0.01 * sigma.item() ** 2
        for step in range(300):
            with torch.no_grad():
                s = net(x, torch.full((200, 1), sigma.item(), device=DEVICE))
            x = x + 0.5 * alpha * s + math.sqrt(alpha) * torch.randn_like(x)

            global_step = si * 300 + step
            if global_step % 100 == 0:
                dist_low = torch.cdist(x.cpu(), real_low[:800]).min(dim=1)[0]
                distances_timeline.append({
                    "step": global_step,
                    "sigma": sigma.item(),
                    "median_dist_low": float(dist_low.median()),
                    "near_05_low": float((dist_low < 0.5).float().mean()),
                })

    # --- T3: Map back to 1024D and measure ---
    print(f"\n--- T3: Map back to 1024D ---")
    x_full = (x.cpu() @ basis.T) + mean
    x_sphere = F.normalize(x_full, dim=-1)

    dist_1024 = torch.cdist(x_sphere, real_cpu).min(dim=1)[0]
    near_03 = float((dist_1024 < 0.3).float().mean())
    near_01 = float((dist_1024 < 0.1).float().mean())
    near_005 = float((dist_1024 < 0.05).float().mean())

    # Cluster coverage
    dists_centers = torch.cdist(x_sphere, centers.cpu())
    assigned = dists_centers.argmin(dim=1)
    modes = len(torch.unique(assigned))

    # Diversity
    pairwise = torch.cdist(x_sphere[:50], x_sphere[:50])
    pairwise_mean = float(pairwise[pairwise > 0].mean())

    print(f"  Near data (<0.3): {near_03:.2%}")
    print(f"  Near data (<0.1): {near_01:.2%}")
    print(f"  Near data (<0.05): {near_005:.2%}")
    print(f"  Modes reached: {modes}/{n_clusters}")
    print(f"  Median distance: {dist_1024.median():.4f}")
    print(f"  Sample diversity: {pairwise_mean:.4f}")

    verdict = "✅ FULL SUCCESS" if near_03 > 0.5 and modes >= n_clusters - 1 else \
              "🔶 PARTIAL" if near_03 > 0.2 else "❌ FAIL"
    print(f"\n  VERDICT: {verdict}")

    result = {
        "experiment": "pca_reduced_score_matching",
        "timestamp": datetime.now().isoformat(),
        "n_clusters": n_clusters,
        "cluster_spread": cluster_spread,
        "intrinsic_dim_k": k,
        "variance_threshold": variance_threshold,
        "near_03": near_03,
        "near_01": near_01,
        "near_005": near_005,
        "modes_reached": modes,
        "median_dist": float(dist_1024.median()),
        "sample_diversity": pairwise_mean,
        "verdict": verdict,
        "distances_timeline": distances_timeline,
    }

    out = RESULTS_DIR / "pca_reduced_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out}")

    return result


if __name__ == "__main__":
    # Main experiment: 5 clusters, tight
    r1 = run_pca_experiment(n_clusters=5, cluster_spread=0.005, variance_threshold=0.99)
