"""
Diagnose why T2/T3 sampling fails despite cos(score, true) = 0.970.

The score direction is excellent. The problem must be in the sampler.
This script tests multiple step sizes and approaches to isolate the issue.
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
    ScoreNetworkV2, make_unit_norm_data, get_sigma_schedule, DIM, DEVICE
)

def load_trained_model():
    net = ScoreNetworkV2(dim=DIM, hidden=1024, n_blocks=8).to(DEVICE)
    net.load_state_dict(torch.load(REPO / "checkpoints" / "score_net_v2.pt"))
    net.eval()
    return net


def diagnose_score_magnitude(net, real):
    """What is ||score|| at different distances from data?"""
    print("=" * 60)
    print("Diagnosis 1: Score magnitude vs distance to data")
    print("=" * 60)

    real_ref = real[:800]
    sigmas = [0.05, 0.2, 0.5, 1.0, 1.5]

    for sigma in sigmas:
        # Points at various distances: real data + controlled noise
        for dist_scale in [0.1, 0.5, 1.0, 2.0]:
            eps = torch.randn(200, DIM, device=DEVICE)
            x = real_ref[:200] + dist_scale * eps
            with torch.no_grad():
                sigma_batch = torch.full((200, 1), float(sigma), device=DEVICE)
                score = net(x, sigma_batch)
                actual_dist = torch.cdist(x, real_ref).min(dim=1)[0]
            print(f"  σ={sigma:.2f} noise_scale={dist_scale:.1f}: "
                  f"actual_dist={actual_dist.mean():.3f}  "
                  f"||score||={score.norm(dim=-1).mean():.3f}")

    print()

    # Points uniformly on sphere (where sampling STARTS)
    print("  --- Starting points for sampling (random on sphere) ---")
    x_start = torch.randn(200, DIM, device=DEVICE)
    x_start = F.normalize(x_start, dim=-1)
    with torch.no_grad():
        dist = torch.cdist(x_start, real_ref).min(dim=1)[0]
        print(f"  Distance to nearest data: {dist.mean():.3f} (median {dist.median():.3f})")

        for sigma in [0.05, 0.2, 0.5, 1.0, 1.5]:
            sigma_batch = torch.full((200, 1), float(sigma), device=DEVICE)
            score = net(x_start, sigma_batch)
            print(f"    σ={sigma:.2f}: ||score||={score.norm(dim=-1).mean():.3f}")


def test_langevin_freely(net, real, step_lr=0.01, n_steps=500):
    """Langevin WITHOUT sphere projection — let points move freely."""
    print(f"\n{'='*60}")
    print(f"Test: Free Langevin (no sphere projection), lr={step_lr}")
    print(f"{'='*60}")

    real_ref = real[:800]
    sigmas = get_sigma_schedule(10)

    x = torch.randn(200, DIM, device=DEVICE)  # NOT normalized — free space
    steps_per_sigma = n_steps // len(sigmas)

    for si, sigma in enumerate(sigmas):
        for step in range(steps_per_sigma):
            with torch.no_grad():
                sigma_batch = torch.full((200, 1), sigma.item(), device=DEVICE)
                score = net(x, sigma_batch)
            x = x + 0.5 * step_lr * score + math.sqrt(step_lr) * torch.randn_like(x)

            global_step = si * steps_per_sigma + step
            if global_step % 50 == 0 or global_step == n_steps - 1:
                with torch.no_grad():
                    dist = torch.cdist(x, real_ref).min(dim=1)[0]
                    print(f"  step {global_step:3d} (σ={sigma:.3f}): "
                          f"dist={dist.median():.4f}  "
                          f"||x||={x.norm(dim=-1).mean():.3f}  "
                          f"near(<0.5)={float((dist<0.5).float().mean()):.2%}")

    return x


def test_langevin_large_steps(net, real, step_lr=0.1, n_steps=500):
    """Langevin with MUCH larger step sizes."""
    print(f"\n{'='*60}")
    print(f"Test: Large step Langevin, lr={step_lr}")
    print(f"{'='*60}")

    real_ref = real[:800]
    sigmas = get_sigma_schedule(10)

    x = torch.randn(200, DIM, device=DEVICE)
    steps_per_sigma = n_steps // len(sigmas)

    for si, sigma in enumerate(sigmas):
        for step in range(steps_per_sigma):
            with torch.no_grad():
                sigma_batch = torch.full((200, 1), sigma.item(), device=DEVICE)
                score = net(x, sigma_batch)
            # Standard Langevin: step proportional to sigma^2
            alpha = step_lr * sigma.item() ** 2
            x = x + 0.5 * alpha * score + math.sqrt(alpha) * torch.randn_like(x)

            global_step = si * steps_per_sigma + step
            if global_step % 50 == 0 or global_step == n_steps - 1:
                with torch.no_grad():
                    dist = torch.cdist(x, real_ref).min(dim=1)[0]
                    print(f"  step {global_step:3d} (σ={sigma:.3f}): "
                          f"dist={dist.median():.4f}  "
                          f"||x||={x.norm(dim=-1).mean():.3f}  "
                          f"near(<0.5)={float((dist<0.5).float().mean()):.2%}")

    return x


def test_score_at_data(net, real):
    """If we START at real data, does the score keep us there?"""
    print(f"\n{'='*60}")
    print("Test: Score at real data points (should be ~0 or restoring)")
    print(f"{'='*60}")

    for sigma in [0.05, 0.1, 0.2, 0.5]:
        with torch.no_grad():
            sigma_batch = torch.full((200, 1), sigma, device=DEVICE)
            score = net(real[:200], sigma_batch)
            # Compare to theoretical: for Gaussian, score = -(x-mu)/sigma^2
            # At the data point itself (no noise added), score should be small
            print(f"  σ={sigma:.2f}: ||score|| at data = {score.norm(dim=-1).mean():.3f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    net = load_trained_model()
    real, noise, centers = make_unit_norm_data(n_samples=2000, n_clusters=20)

    print(f"Model loaded. cos(score,true) was measured at 0.970.")
    print()

    diagnose_score_magnitude(net, real)
    test_score_at_data(net, real)
    test_langevin_freely(net, real, step_lr=0.01, n_steps=500)
    test_langevin_large_steps(net, real, step_lr=0.1, n_steps=500)
