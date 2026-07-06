"""
Critical test: does the score RESTORE us to data when we start near it?

cos(score, true) = 0.96 is measured by perturbing real data with noise.
But sampling STARTS from random. The question is: does the score field
between random and data actually point toward data?

We test: gradient flow from different starting distances.
"""
import sys
import math
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from phase2_score_matching_scaled import ScoreNetworkV2, DIM, DEVICE, get_sigma_schedule
from separable_experiment import make_separable_data


def load_model(path):
    # Can't easily reload from the separable experiment (no save).
    # Retrain quickly, or load phase2 model and adapt.
    pass


def score_field_analysis(net, real, centers, n_clusters=5):
    """Analyze the score field in detail.

    Sample points along a LINE from random to a cluster center,
    measure score direction and magnitude at each point.
    """
    print(f"\n{'='*60}")
    print("Score field analysis: line from random → cluster center")
    print(f"{'='*60}")

    real_ref = real[:800]

    for cluster_idx in range(min(3, n_clusters)):
        center = centers[cluster_idx]
        # Start far away (random direction)
        start = torch.randn(1, DIM, device=DEVICE)
        start = F.normalize(start, dim=-1)

        print(f"\n  Cluster {cluster_idx}: center at cos(start, center)="
              f"{F.cosine_similarity(start, center.unsqueeze(0)).item():.4f}")

        # Walk from start to center in 20 steps
        for t in range(21):
            alpha = t / 20.0
            # Spherical interpolation (slerp)
            cos_angle = F.cosine_similarity(start, center.unsqueeze(0)).clamp(-0.9999, 0.9999)
            omega = torch.acos(cos_angle).item()
            if omega < 1e-6:
                point = start
            else:
                sin_omega = math.sin(omega)
                a = math.sin((1 - alpha) * omega) / sin_omega
                b = math.sin(alpha * omega) / sin_omega
                point = a * start + b * center.unsqueeze(0)
                point = F.normalize(point, dim=-1)

            # Measure score at this point
            with torch.no_grad():
                sigma_batch = torch.full((point.shape[0], 1), 0.5, device=DEVICE)
                score = net(point, sigma_batch)

                # Direction to center
                dir_to_center = center.unsqueeze(0) - point
                dir_to_center = F.normalize(dir_to_center, dim=-1)
                score_normalized = F.normalize(score, dim=-1)

                cos_to_center = F.cosine_similarity(score_normalized, dir_to_center).item()
                score_norm = score.norm(dim=-1).item()

                # Distance to nearest real data
                dist = torch.cdist(point, real_ref).min(dim=1)[0].item()

            if t % 4 == 0 or t == 20:
                print(f"    t={alpha:.2f}: dist_to_data={dist:.4f}  "
                      f"||score||={score_norm:.1f}  "
                      f"cos(score→center)={cos_to_center:+.3f}")


def gradient_flow_from_nearby(net, real, centers):
    """Start from points that are CLOSE to data (small perturbation)."""
    print(f"\n{'='*60}")
    print("Gradient flow: start NEAR data (σ=0.1 perturbation)")
    print(f"{'='*60}")

    real_ref = real[:800]

    # Perturb real data slightly
    eps = torch.randn(200, DIM, device=DEVICE) * 0.1
    x = real[:200] + eps
    x = F.normalize(x, dim=-1)

    print(f"  Starting distance to nearest data: "
          f"{torch.cdist(x, real_ref).min(dim=1)[0].median():.4f}")

    for step in range(300):
        with torch.no_grad():
            sigma_batch = torch.full((200, 1), 0.1, device=DEVICE)
            score = net(x, sigma_batch)
        x = x + 0.001 * score
        x = F.normalize(x, dim=-1)
        if step % 50 == 0 or step == 299:
            with torch.no_grad():
                dist = torch.cdist(x, real_ref).min(dim=1)[0]
                print(f"  step {step:3d}: dist={dist.median():.4f}  "
                      f"near(<0.1)={float((dist<0.1).float().mean()):.2%}  "
                      f"near(<0.05)={float((dist<0.05).float().mean()):.2%}")


def gradient_flow_multi_lr(net, real, centers):
    """Try many different step sizes from random starts."""
    print(f"\n{'='*60}")
    print("Gradient flow: sweep step sizes from random start")
    print(f"{'='*60}")

    real_ref = real[:800]

    for lr in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]:
        x = torch.randn(100, DIM, device=DEVICE)
        x = F.normalize(x, dim=-1)
        for step in range(300):
            with torch.no_grad():
                sigma_batch = torch.full((100, 1), 0.5, device=DEVICE)
                score = net(x, sigma_batch)
            x = x + lr * score
            x = F.normalize(x, dim=-1)
        with torch.no_grad():
            dist = torch.cdist(x, real_ref).min(dim=1)[0]
            print(f"  lr={lr:.4f}: final dist={dist.median():.4f}  "
                  f"near(<0.3)={float((dist<0.3).float().mean()):.2%}")


if __name__ == "__main__":
    torch.manual_seed(42)

    # Retrain quickly on separable data
    from phase2_score_matching_scaled import dsm_loss_v2, cosine_lr_schedule

    print("Training score network on 5 separable clusters (1000 epochs)...")
    real, noise, centers, _ = make_separable_data(
        n_samples=2000, n_clusters=5, cluster_spread=0.005
    )
    net = ScoreNetworkV2(dim=DIM, hidden=1024, n_blocks=8).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sigmas = get_sigma_schedule(10)

    for epoch in range(1000):
        optimizer.zero_grad()
        loss = dsm_loss_v2(net, real[:1600], sigmas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        cosine_lr_schedule(optimizer, epoch, 1000, warmup=100, base_lr=1e-3)
        optimizer.step()
        if epoch % 200 == 0:
            print(f"  epoch {epoch}: loss={loss.item():.4f}")
    net.eval()
    print("Training done.\n")

    # Run all analyses
    score_field_analysis(net, real, centers, n_clusters=5)
    gradient_flow_from_nearby(net, real, centers)
    gradient_flow_multi_lr(net, real, centers)
