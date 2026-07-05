"""
Root cause: is the data actually clusterable?

If intra-cluster distance ≈ inter-cluster distance, the data has NO
learnable structure. The score matching can't guide to clusters that
don't exist as separable regions.

This script measures cluster separability under different configurations
and tests sampling with WELL-SEPARATED clusters.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from phase2_score_matching_scaled import (
    ScoreNetworkV2, get_sigma_schedule, DIM, DEVICE
)


def make_separable_data(n_samples=2000, n_clusters=5, dim=DIM, cluster_spread=0.01):
    """Data with FEW, WELL-SEPARATED clusters (tight spread).

    With 5 clusters and spread=0.01, clusters should be clearly separable
    even after sphere normalization.
    """
    torch.manual_seed(42)
    centers = torch.randn(n_clusters, dim)
    centers = F.normalize(centers, dim=-1)

    real = []
    for i in range(n_samples):
        c = centers[torch.randint(0, n_clusters, (1,)).item()]
        point = c + torch.randn(dim) * cluster_spread
        point = F.normalize(point, dim=-1)
        real.append(point)
    real = torch.stack(real).to(DEVICE)

    noise = torch.randn(n_samples, dim)
    noise = F.normalize(noise, dim=-1).to(DEVICE)

    centers = centers.to(DEVICE)
    return real, noise, centers


def measure_separability(real, centers, n_clusters):
    """Measure how separable the clusters are."""
    # Assign each point to nearest center
    dists_to_centers = torch.cdist(real, centers)  # [N, K]
    assigned = dists_to_centers.argmin(dim=1)

    # Distance from each point to its assigned center
    dist_to_own = dists_to_centers.gather(1, assigned.unsqueeze(1)).squeeze()
    intra = dist_to_own.mean().item()

    # Distance between centers
    inter = torch.cdist(centers, centers)
    inter = inter[inter > 0].mean().item()

    # Silhouette-like ratio: should be >> 1 for separable clusters
    ratio = inter / intra if intra > 0 else float('inf')

    print(f"  Clusters: {n_clusters}")
    print(f"  Intra-cluster dist (point to own center): {intra:.4f}")
    print(f"  Inter-cluster dist (center to center):    {inter:.4f}")
    print(f"  Separability ratio (inter/intra):         {ratio:.2f}x")
    print(f"  {'✅ SEPARABLE' if ratio > 5 else '❌ NOT SEPARABLE'} (need >5x)")
    return ratio, intra, inter


def quick_train_and_sample(real, n_clusters, n_epochs=1500):
    """Quick training on separable data, then sample."""
    from phase2_score_matching_scaled import train_score_network, evaluate_t1_direction

    print(f"\n{'='*60}")
    print(f"Train + Sample on {n_clusters} separable clusters")
    print(f"{'='*60}")

    score_net, info = train_score_network(real, n_epochs=n_epochs, base_lr=1e-3)

    # T1-direction
    t1 = evaluate_t1_direction(score_net, real, real)

    # Sample with gradient descent
    print(f"\n--- Gradient descent sampling ---")
    x = torch.randn(200, DIM, device=DEVICE)
    x = F.normalize(x, dim=-1)
    real_ref = real[:800]

    for step in range(500):
        with torch.no_grad():
            sigma_batch = torch.full((200, 1), 0.5, device=DEVICE)
            score = score_net(x, sigma_batch)
        x = x + 0.01 * score
        x = F.normalize(x, dim=-1)
        if step % 100 == 0 or step == 499:
            with torch.no_grad():
                dist = torch.cdist(x, real_ref).min(dim=1)[0]
                print(f"  step {step:3d}: dist={dist.median():.4f}  "
                      f"near(<0.3)={float((dist<0.3).float().mean()):.2%}")

    # Also try starting from NEAR data to see if score holds us there
    print(f"\n--- Score at data points (should be restoring) ---")
    for sigma in [0.05, 0.1, 0.2, 0.5]:
        with torch.no_grad():
            sigma_batch = torch.full((200, 1), sigma, device=DEVICE)
            score = score_net(real[:200], sigma_batch)
            print(f"  σ={sigma:.2f}: ||score||={score.norm(dim=-1).mean():.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("SEPARABILITY ANALYSIS")
    print("=" * 60)

    # Test 1: original config (20 clusters, spread=0.15)
    print("\n--- Original config: 20 clusters, spread=0.15 ---")
    from phase2_score_matching_scaled import make_unit_norm_data
    real, _, centers = make_unit_norm_data(n_samples=2000, n_clusters=20, cluster_spread=0.15)
    measure_separability(real, centers, 20)

    # Test 2: fewer clusters, tighter spread
    configs = [
        (5, 0.01, "5 clusters, spread=0.01"),
        (5, 0.005, "5 clusters, spread=0.005"),
        (3, 0.01, "3 clusters, spread=0.01"),
        (10, 0.02, "10 clusters, spread=0.02"),
    ]

    for n_clusters, spread, desc in configs:
        print(f"\n--- {desc} ---")
        real, _, centers = make_separable_data(
            n_samples=2000, n_clusters=n_clusters, cluster_spread=spread
        )
        ratio, _, _ = measure_separability(real, centers, n_clusters)

    # Test 3: train and sample on the MOST separable config
    print("\n" + "=" * 60)
    print("FULL EXPERIMENT: 5 clusters, spread=0.005")
    print("=" * 60)
    real, noise, centers = make_separable_data(
        n_samples=2000, n_clusters=5, cluster_spread=0.005
    )
    measure_separability(real, centers, 5)
    quick_train_and_sample(real, n_clusters=5, n_epochs=2000)
