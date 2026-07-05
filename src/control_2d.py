"""
2D validation: does the full score matching → sampling pipeline work?

If it works in 2D, the 1024D failure is the curse of dimensionality.
If it fails in 2D, there's a bug in our pipeline.

This is the scientific control: lowest dimensionality, clearest signal.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


class ScoreNet2D(nn.Module):
    """Simple score network for 2D data with sigma conditioning."""
    def __init__(self, hidden=128, n_blocks=4):
        super().__init__()
        self.sigma_mlp = nn.Sequential(
            nn.Linear(64, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.input_proj = nn.Linear(2, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                          nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(n_blocks)
        ])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, 2))
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
            h = h + block(h)  # residual (ignore sigma emb for simplicity in 2D)
        return self.out(h)


def make_2d_data(n_samples=2000, n_clusters=5):
    """2D clusters — clearly separable."""
    torch.manual_seed(42)
    centers = torch.tensor([
        [2.0, 2.0], [-2.0, 2.0], [0.0, -2.5], [3.0, -1.0], [-3.0, -0.5]
    ])
    real = []
    labels = []
    for _ in range(n_samples):
        idx = torch.randint(0, n_clusters, (1,)).item()
        point = centers[idx] + torch.randn(2) * 0.3
        real.append(point)
        labels.append(idx)
    return torch.stack(real), centers, labels


def dsm_loss_2d(net, x, sigmas):
    total = 0.0
    for sigma in sigmas:
        eps = torch.randn_like(x)
        x_noisy = x + sigma * eps
        target = -eps / sigma
        pred = net(x_noisy, sigma.expand(x.shape[0], 1))
        total = total + ((pred - target) ** 2).mean()
    return total / len(sigmas)


def run_2d_experiment():
    print("=" * 60)
    print("2D CONTROL Experiment: Score Matching Pipeline")
    print("=" * 60)

    real, centers, labels = make_2d_data(n_samples=2000)
    print(f"Data: {real.shape}, {len(centers)} clusters")
    print(f"Cluster centers: {centers.tolist()}")

    # Train
    sigmas = torch.tensor([2.0, 1.0, 0.5, 0.2, 0.1, 0.05], device='cuda')
    net = ScoreNet2D().cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

    real = real.cuda()
    train_data = real[:1600]

    print(f"\nTraining 2000 epochs...")
    for epoch in range(2000):
        optimizer.zero_grad()
        loss = dsm_loss_2d(net, train_data, sigmas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        if epoch % 400 == 0 or epoch == 1999:
            print(f"  epoch {epoch}: loss={loss.item():.6f}")
    net.eval()

    # T1: score direction
    print(f"\n--- T1: Score direction ---")
    for sigma in [0.05, 0.1, 0.5, 1.0]:
        eps = torch.randn(500, 2, device='cuda')
        x = train_data[:500] + sigma * eps
        target = -eps / sigma
        with torch.no_grad():
            pred = net(x, torch.full((500, 1), sigma, device='cuda'))
            cos = F.cosine_similarity(pred, target, dim=-1).mean().item()
        print(f"  σ={sigma:.2f}: cos(score, true)={cos:+.4f}")

    # T2: Langevin sampling
    print(f"\n--- T2: Annealed Langevin sampling ---")
    x = torch.randn(500, 2, device='cuda') * 3  # wide start
    real_ref = real[:800]

    trajectory = [x.cpu().clone()]
    for si, sigma in enumerate(sigmas):
        alpha = 0.01 * sigma.item() ** 2
        for step in range(300):
            with torch.no_grad():
                score = net(x, torch.full((500, 1), sigma.item(), device='cuda'))
            x = x + 0.5 * alpha * score + math.sqrt(alpha) * torch.randn_like(x)
            if step % 100 == 0:
                trajectory.append(x.cpu().clone())

    # Measure
    with torch.no_grad():
        dist = torch.cdist(x.cpu(), real_ref.cpu()).min(dim=1)[0]
        near_05 = (dist < 0.5).float().mean().item()
        near_03 = (dist < 0.3).float().mean().item()
        median_dist = dist.median().item()

    print(f"  Final dist to data: median={median_dist:.4f}")
    print(f"  Near (<0.5): {near_05:.2%}")
    print(f"  Near (<0.3): {near_03:.2%}")

    # T3: mode coverage
    dists_to_centers = torch.cdist(x.cpu(), centers)
    assigned = dists_to_centers.argmin(dim=1)
    modes_reached = len(torch.unique(assigned))
    print(f"  Modes reached: {modes_reached}/{len(centers)}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: data + samples
    ax = axes[0]
    ax.scatter(real.cpu()[:, 0], real.cpu()[:, 1], c='blue', alpha=0.3, s=5, label='Real data')
    ax.scatter(x.cpu()[:, 0], x.cpu()[:, 1], c='red', alpha=0.5, s=10, label='Samples')
    ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, label='Centers')
    ax.set_title('Data (blue) vs Samples (red)')
    ax.legend()
    ax.set_aspect('equal')

    # Plot 2: sampling trajectory
    ax = axes[1]
    for i in range(0, len(trajectory), 5):
        alpha_traj = 0.3 + 0.7 * i / len(trajectory)
        ax.scatter(trajectory[i][:, 0], trajectory[i][:, 1], c='green',
                   alpha=0.1, s=3)
    ax.scatter(real.cpu()[:, 0], real.cpu()[:, 1], c='blue', alpha=0.2, s=3)
    ax.set_title('Sampling trajectory')
    ax.set_aspect('equal')

    # Plot 3: score field
    ax = axes[2]
    grid_x = np.linspace(-5, 5, 20)
    grid_y = np.linspace(-5, 5, 20)
    GX, GY = np.meshgrid(grid_x, grid_y)
    grid = torch.tensor(np.stack([GX.ravel(), GY.ravel()], axis=1), dtype=torch.float32).cuda()
    with torch.no_grad():
        scores = net(grid, torch.full((grid.shape[0], 1), 0.5, device='cuda'))
    ax.quiver(GX.ravel(), GY.ravel(),
              scores.cpu()[:, 0].numpy(), scores.cpu()[:, 1].numpy(),
              alpha=0.5)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
    ax.set_title('Score field (σ=0.5)')
    ax.set_aspect('equal')

    plt.tight_layout()
    plot_path = REPO / "results" / "2d_control_experiment.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")

    # Verdict
    if near_05 > 0.5 and modes_reached >= 4:
        verdict = "✅ FULL SUCCESS: Score matching works in 2D"
    elif near_05 > 0.3:
        verdict = "🔶 PARTIAL: Most samples near data"
    else:
        verdict = "❌ FAIL: Pipeline broken even in 2D"

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"  near(<0.5): {near_05:.2%}")
    print(f"  modes: {modes_reached}/{len(centers)}")
    print(f"{'='*60}")

    return verdict, near_05, modes_reached


if __name__ == "__main__":
    run_2d_experiment()
