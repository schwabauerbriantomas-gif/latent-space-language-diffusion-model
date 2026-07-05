"""
The score has correct direction near data (cos=0.97) but the magnitude is
constant (~57) regardless of position. This means:
  - Near data: score pulls correctly
  - Far from data: score still points vaguely but doesn't get stronger

The fix: train with LARGER sigmas so the score learns to guide from far away.
Also: normalize the score during sampling and use adaptive step sizes.

This experiment uses a much wider sigma range and longer training.
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from phase2_score_matching_scaled import (
    ScoreNetworkV2, cosine_lr_schedule, DIM, DEVICE
)
from separable_experiment import make_separable_data

RESULTS_DIR = REPO / "results"


def get_wide_sigma_schedule(n_levels=10, sigma_max=5.0, sigma_min=0.01):
    """Wide sigma schedule: from 5.0 down to 0.01.

    sigma_max=5.0 means the score learns to denoise from VERY far away
    (noise 5x the cluster spread). This should teach it to guide from
    random points on the sphere.
    """
    sigmas = torch.tensor(
        [sigma_max * (sigma_min / sigma_max) ** (i / (n_levels - 1))
         for i in range(n_levels)]
    )
    return sigmas.to(DEVICE)


def dsm_loss_weighted(score_net, x, sigmas):
    """DSM with Song & Ermon (2019) optimal weighting: λ(σ) = σ².

    Each sigma level contributes equally to the loss. Without this, the
    loss is dominated by small sigmas (target = -eps/σ has magnitude 1/σ).
    """
    batch_size = x.shape[0]
    total_loss = 0.0

    for sigma in sigmas:
        eps = torch.randn_like(x)
        x_noisy = x + sigma * eps
        target = -eps / sigma

        sigma_batch = sigma.expand(batch_size, 1)
        pred = score_net(x_noisy, sigma_batch)

        # Weight by σ² to normalize contribution across sigmas
        loss_per_point = ((pred - target) ** 2).mean(dim=-1)
        weight = sigma ** 2  # Song & Ermon optimal weight
        total_loss = total_loss + weight * loss_per_point.mean()

    return total_loss / len(sigmas)


def train_with_wide_sigmas(real, n_epochs=3000, sigma_max=5.0):
    sigmas = get_wide_sigma_schedule(n_levels=10, sigma_max=sigma_max)
    print(f"Wide sigma schedule: {[round(s.item(),3) for s in sigmas]}")
    print(f"Sigma range covers {sigma_max/sigmas[-1].item():.0f}x of the scale")

    net = ScoreNetworkV2(dim=DIM, hidden=1024, n_blocks=8).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-4)

    train_data = real[:1600]
    print(f"\nTraining {n_epochs} epochs with σ²-weighted loss (lr=1e-2)...")
    t0 = time.time()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = dsm_loss_weighted(net, train_data, sigmas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        cosine_lr_schedule(optimizer, epoch, n_epochs, warmup=200, base_lr=1e-2)
        optimizer.step()

        if epoch % 300 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch}: loss={loss.item():.6f}")

    print(f"Training: {time.time()-t0:.0f}s")
    net.eval()
    return net, sigmas


def sample_with_normalized_score(net, real, sigmas, n_steps_per_sigma=200):
    """Sampling with score normalization + adaptive step size.

    Key insight: the score DIRECTION is correct (cos=0.97) but magnitude
    is constant. So we normalize the score and use a FIXED step size in
    the direction of the score. This is like following a compass regardless
    of how strong the magnetic field is.
    """
    print(f"\n--- Sampling with normalized score (compass mode) ---")

    real_ref = real[:800]
    x = torch.randn(200, DIM, device=DEVICE)
    x = F.normalize(x, dim=-1)

    step_size = 0.02  # fixed angular step on the sphere

    for si, sigma in enumerate(sigmas):
        for step in range(n_steps_per_sigma):
            with torch.no_grad():
                sigma_batch = torch.full((200, 1), sigma.item(), device=DEVICE)
                score = net(x, sigma_batch)
                # Normalize score to unit length (use direction only)
                score_normalized = F.normalize(score, dim=-1)

            # Fixed step in score direction + decreasing noise
            noise_scale = sigma.item() * 0.3  # less noise than sigma
            x = x + step_size * score_normalized + noise_scale * torch.randn_like(x) * 0.01
            x = F.normalize(x, dim=-1)

            global_step = si * n_steps_per_sigma + step
            if global_step % 200 == 0 or global_step == n_steps_per_sigma * len(sigmas) - 1:
                with torch.no_grad():
                    dist = torch.cdist(x, real_ref).min(dim=1)[0]
                    print(f"  step {global_step:4d} (σ={sigma:.3f}): "
                          f"dist={dist.median():.4f}  "
                          f"near(<0.3)={float((dist<0.3).float().mean()):.2%}  "
                          f"near(<0.1)={float((dist<0.1).float().mean()):.2%}")

    return x


def sample_annealed_sphere(net, real, sigmas, n_steps_per_sigma=300):
    """Annealed Langevin on sphere with proper sigma-scaled steps.

    At each sigma level, use step α = lr_base * σ²
    """
    print(f"\n--- Annealed Langevin on sphere ---")

    real_ref = real[:800]
    x = torch.randn(200, DIM, device=DEVICE)
    x = F.normalize(x, dim=-1)

    lr_base = 0.005

    for si, sigma in enumerate(sigmas):
        alpha = lr_base * sigma.item() ** 2
        for step in range(n_steps_per_sigma):
            with torch.no_grad():
                sigma_batch = torch.full((200, 1), sigma.item(), device=DEVICE)
                score = net(x, sigma_batch)
            x = x + 0.5 * alpha * score + math.sqrt(alpha) * torch.randn_like(x)
            x = F.normalize(x, dim=-1)

            global_step = si * n_steps_per_sigma + step
            if global_step % 300 == 0 or global_step == n_steps_per_sigma * len(sigmas) - 1:
                with torch.no_grad():
                    dist = torch.cdist(x, real_ref).min(dim=1)[0]
                    print(f"  step {global_step:4d} (σ={sigma:.3f}, α={alpha:.6f}): "
                          f"dist={dist.median():.4f}  "
                          f"near(<0.3)={float((dist<0.3).float().mean()):.2%}  "
                          f"near(<0.1)={float((dist<0.1).float().mean()):.2%}")

    return x


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("Phase 2b: Wide-sigma score matching + improved sampling")
    print("=" * 60)

    real, noise, centers, _ = make_separable_data(
        n_samples=2000, n_clusters=5, cluster_spread=0.005
    )

    # Train with wide sigmas
    net, sigmas = train_with_wide_sigmas(real, n_epochs=3000, sigma_max=5.0)

    # Evaluate direction at various distances
    print(f"\n--- Score direction at various distances ---")
    real_ref = real[:800]
    for sigma in [0.05, 0.2, 0.5, 1.0, 2.0, 5.0]:
        eps = torch.randn(200, DIM, device=DEVICE)
        x = real[:200] + sigma * eps
        x = F.normalize(x, dim=-1)
        with torch.no_grad():
            sigma_batch = torch.full((200, 1), sigma, device=DEVICE)
            score = net(x, sigma_batch)
            true = -eps / sigma
            cos = F.cosine_similarity(score, true, dim=-1).mean().item()
            dist = torch.cdist(x, real_ref).min(dim=1)[0].mean().item()
        print(f"  σ={sigma:.2f}: dist={dist:.3f}  cos={cos:+.3f}  ||score||={score.norm(dim=-1).mean():.1f}")

    # Try both samplers
    s1 = sample_with_normalized_score(net, real, sigmas)
    s2 = sample_annealed_sphere(net, real, sigmas)

    # Final quality
    print(f"\n--- Final sample quality ---")
    for name, samples in [("normalized", s1), ("annealed", s2)]:
        with torch.no_grad():
            dist = torch.cdist(samples.cpu(), real_ref.cpu()).min(dim=1)[0]
            print(f"  {name}: near(<0.3)={float((dist<0.3).float().mean()):.2%}  "
                  f"near(<0.1)={float((dist<0.1).float().mean()):.2%}  "
                  f"median_dist={dist.median():.4f}")
