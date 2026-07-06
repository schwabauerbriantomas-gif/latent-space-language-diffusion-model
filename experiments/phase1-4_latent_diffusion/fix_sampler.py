"""
Fix the sampler. The score direction is near-perfect (cos=0.970).
The issue is step size calibration. Test multiple approaches.
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


def sample_gradient_descent(net, real, sigma=0.5, lr=0.01, n_steps=500):
    """Pure gradient descent on the score (no noise). 

    This tests: if we follow the score deterministically, do we reach data?
    """
    print(f"\n{'='*60}")
    print(f"Gradient descent (no noise): σ={sigma}, lr={lr}, {n_steps} steps")
    print(f"{'='*60}")

    real_ref = real[:800]
    x = torch.randn(200, DIM, device=DEVICE)
    x = F.normalize(x, dim=-1)  # start on sphere

    for step in range(n_steps):
        with torch.no_grad():
            sigma_batch = torch.full((200, 1), sigma, device=DEVICE)
            score = net(x, sigma_batch)
        # Gradient descent: follow the score (points toward high density)
        x = x + lr * score
        x = F.normalize(x, dim=-1)  # stay on sphere

        if step % 50 == 0 or step == n_steps - 1:
            with torch.no_grad():
                dist = torch.cdist(x, real_ref).min(dim=1)[0]
                print(f"  step {step:3d}: dist={dist.median():.4f}  "
                      f"near(<0.5)={float((dist<0.5).float().mean()):.2%}")

    return x


def sample_annealed_proper(net, real, n_steps_per_sigma=200, lr_base=0.01):
    """Properly calibrated annealed Langevin.

    Uses NCSN-style step sizes: α_i = lr_base * σ_i²
    But with ENOUGH steps per level.
    """
    print(f"\n{'='*60}")
    print(f"Annealed Langevin (NCSN-style): {n_steps_per_sigma} steps/σ, lr={lr_base}")
    print(f"{'='*60}")

    real_ref = real[:800]
    sigmas = get_sigma_schedule(10)

    x = torch.randn(200, DIM, device=DEVICE)
    x = F.normalize(x, dim=-1)

    for si, sigma in enumerate(sigmas):
        alpha = lr_base * sigma.item() ** 2
        for step in range(n_steps_per_sigma):
            with torch.no_grad():
                sigma_batch = torch.full((200, 1), sigma.item(), device=DEVICE)
                score = net(x, sigma_batch)
            # Langevin: x += α/2 * score + sqrt(α) * noise
            x = x + 0.5 * alpha * score + math.sqrt(alpha) * torch.randn_like(x)
            x = F.normalize(x, dim=-1)

            global_step = si * n_steps_per_sigma + step
            if global_step % 200 == 0 or global_step == n_steps_per_sigma * len(sigmas) - 1:
                with torch.no_grad():
                    dist = torch.cdist(x, real_ref).min(dim=1)[0]
                    print(f"  step {global_step:4d} (σ={sigma:.3f}): "
                          f"dist={dist.median():.4f}  "
                          f"near(<0.5)={float((dist<0.5).float().mean()):.2%}")

    return x


def sample_single_sigma_descent(net, real, sigma=0.5, lr=0.001, n_steps=2000):
    """Single-sigma gradient descent with small lr, many steps."""
    print(f"\n{'='*60}")
    print(f"Single-σ descent: σ={sigma}, lr={lr}, {n_steps} steps")
    print(f"{'='*60}")

    real_ref = real[:800]
    x = torch.randn(200, DIM, device=DEVICE)
    x = F.normalize(x, dim=-1)

    for step in range(n_steps):
        with torch.no_grad():
            sigma_batch = torch.full((200, 1), sigma, device=DEVICE)
            score = net(x, sigma_batch)
        x = x + lr * score
        x = F.normalize(x, dim=-1)

        if step % 200 == 0 or step == n_steps - 1:
            with torch.no_grad():
                dist = torch.cdist(x, real_ref).min(dim=1)[0]
                print(f"  step {step:4d}: dist={dist.median():.4f}  "
                      f"near(<0.5)={float((dist<0.5).float().mean()):.2%}")

    return x


if __name__ == "__main__":
    torch.manual_seed(42)
    net = load_trained_model()
    real, noise, centers = make_unit_norm_data(n_samples=2000, n_clusters=20)

    # Test 1: gradient descent at various sigmas and lrs
    for sigma in [0.05, 0.2, 0.5, 1.0]:
        for lr in [0.001, 0.01]:
            sample_gradient_descent(net, real, sigma=sigma, lr=lr, n_steps=300)

    # Test 2: proper annealed Langevin with more steps
    sample_annealed_proper(net, real, n_steps_per_sigma=200, lr_base=0.01)

    # Test 3: long single-sigma descent
    sample_single_sigma_descent(net, real, sigma=0.5, lr=0.001, n_steps=2000)
