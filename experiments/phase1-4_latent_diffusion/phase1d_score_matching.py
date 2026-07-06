"""
Score Matching energy model — the control experiment.

If FF failed because of the objective (goodness → spikes), then score matching
(trains ∇log p directly) should SUCCEED in the same latent space.

Denoising Score Matching (Vincent 2011):
  x_noisy = x + σ·ε       (ε ~ N(0,I))
  target  = -ε/σ          (the score of Gaussian noise)
  loss    = ||s_θ(x_noisy) - target||²

The score network outputs a vector in latent space. No energy, no goodness.
The gradient IS the learned quantity.

Usage:
    python src/phase1d_score_matching.py
"""
import json
import sys
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from phase1_test import make_latent_data, make_sequence_data, DEVICE, DIM, _save

RESULTS_DIR = REPO / "results"


class ScoreNetwork(nn.Module):
    """Score network s_θ(x) ≈ ∇_x log p(x).

    MLP with skip connections. Outputs a vector in latent space (the score).
    No energy involved — the gradient is the direct target of training.
    """

    def __init__(self, dim: int = 1024, hidden: int = 512, n_layers: int = 4):
        super().__init__()
        self.dim = dim
        layers = []
        prev = dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev, hidden),
                nn.SiLU(),  # smooth activation (no ReLU kinks — better for score)
                nn.Linear(hidden, dim),
            ])
            if prev == dim:
                # residual on first block
                pass
            prev = dim
        # Stack with residual: input → block → +input → block → +...
        self.blocks = nn.ModuleList([
            self._make_block(dim, hidden) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def _make_block(self, dim, hidden):
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns score s(x) ∈ R^dim."""
        h = x
        for block in self.blocks:
            h = h + block(h)  # residual
        return self.norm(h) if False else h  # final raw score


def dsm_loss(score_net, x, sigmas):
    """Multi-scale denoising score matching.

    For each σ in sigmas, add noise at that scale and predict the score.
    Multi-scale ensures the score is learned at all noise levels (needed
    for annealed Langevin sampling).
    """
    total_loss = torch.tensor(0.0, device=x.device)
    for sigma in sigmas:
        eps = torch.randn_like(x)
        x_noisy = x + sigma * eps
        target = -eps / sigma  # score of N(x, σ²I)
        pred = score_net(x_noisy)
        total_loss = total_loss + ((pred - target) ** 2).sum(dim=-1).mean()
    return total_loss / len(sigmas)


def langevin_sample(score_net, n_samples, dim, n_steps=100, lr=0.01,
                    sigmas=None, device="cuda"):
    """Annealed Langevin sampling using the learned score.

    Start at high noise, anneal σ down. At each scale, run Langevin steps
    guided by the score. This is the standard score-based sampling (Song &
    Ermon 2019).
    """
    if sigmas is None:
        sigmas = [2.0, 1.0, 0.5, 0.1]

    x = torch.randn(n_samples, dim, device=device) * sigmas[0]
    steps_per_sigma = n_steps // len(sigmas)

    for sigma in sigmas:
        step_lr = lr * sigma  # scale lr with noise level
        for _ in range(steps_per_sigma):
            with torch.no_grad():
                score = score_net(x)
            x = x + 0.5 * step_lr * score + math.sqrt(step_lr) * torch.randn_like(x)
    return x


def run_score_matching_test():
    """Train score matching, measure T1/T2/T3. Compare to FF/CD."""
    print("=" * 60)
    print("Phase 1d: Denoising Score Matching (control experiment)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    torch.manual_seed(42)
    real, noise, centers = make_latent_data(n_samples=1000)

    results = {
        "test": "phase1d_score_matching",
        "timestamp": datetime.now().isoformat(),
        "method": "denoising_score_matching",
    }

    # --- Train score network ---
    score_net = ScoreNetwork(dim=DIM, hidden=512, n_layers=4).to(DEVICE)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=1e-3)

    # Multi-scale sigmas (annealed)
    sigmas = [2.0, 1.0, 0.5, 0.2]
    n_epochs = 500

    print(f"\nTraining score network ({n_epochs} epochs, sigmas={sigmas})...")
    for ep in range(n_epochs):
        optimizer.zero_grad()
        loss = dsm_loss(score_net, real[:800], sigmas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
        optimizer.step()
        if ep % 100 == 0 or ep == n_epochs - 1:
            print(f"  epoch {ep:3d}: DSM loss = {loss.item():.4f}")

    score_net.eval()

    # --- T1: discrimination via score magnitude (lower score = in-distribution) ---
    # Heuristic: ||s(x)|| should be small at data (the score is ~0 at density peaks)
    # and large at noise. Measure as a discriminator.
    print("\n--- T1: Discrimination (score magnitude) ---")
    with torch.no_grad():
        s_real = score_net(real[800:]).norm(dim=-1).cpu().numpy().ravel()
        s_noise = score_net(noise[800:]).norm(dim=-1).cpu().numpy().ravel()
    labels = np.concatenate([np.zeros(len(s_real)), np.ones(len(s_noise))])
    # Lower score norm = more in-distribution → invert for AUROC
    scores_disc = -np.concatenate([s_real, s_noise])
    auroc = roc_auc_score(labels, scores_disc)
    print(f"  ||s|| real: {s_real.mean():.4f}  ||s|| noise: {s_noise.mean():.4f}")
    print(f"  AUROC: {auroc:.4f}  {'✅ PASS' if auroc > 0.85 else '❌ FAIL'}")
    results["t1_auroc"] = float(auroc)

    # --- T2: score gradient direction (THE CRITICAL TEST) ---
    print("\n--- T2: Score direction (does s(x) point toward data?) ---")
    n_test = 200
    x = torch.randn(n_test, DIM, device=DEVICE) * 2.0
    real_ref = real[:500]
    lr = 0.01
    distances = []
    for step in range(50):
        with torch.no_grad():
            score = score_net(x)
        # Langevin: x += lr*s + noise (follow the score toward high density)
        x = x + 0.5 * lr * score + np.sqrt(2 * lr) * torch.randn_like(x)
        if step % 10 == 0 or step == 49:
            with torch.no_grad():
                dist = torch.cdist(x, real_ref).min(dim=1)[0]
                d = {"step": step, "median": float(dist.median()),
                     "mean": float(dist.mean())}
                distances.append(d)
                print(f"  step {step:3d}: median={d['median']:.4f} mean={d['mean']:.4f}")
    medians = [d["median"] for d in distances]
    decrease = (medians[0] - medians[-1]) / medians[0]
    results["t2_decrease"] = decrease
    results["t2_distances"] = distances
    results["t2_verdict"] = "PASS" if decrease > 0.15 else "FAIL"
    print(f"  Distance decrease: {decrease*100:.1f}%  {'✅ PASS' if decrease > 0.15 else '❌ FAIL'} (>15%)")

    # --- T3: sampling near data (annealed Langevin) ---
    print("\n--- T3: Annealed Langevin sampling ---")
    samples = langevin_sample(score_net, n_samples=200, dim=DIM, n_steps=200,
                               lr=0.01, sigmas=sigmas, device=DEVICE)
    with torch.no_grad():
        s_cpu = samples.cpu()
        dist_to_real = torch.cdist(s_cpu, real[:500].cpu()).min(dim=1)[0]
        near_ratio = (dist_to_real < 1.0).float().mean().item()
        near_ratio_2 = (dist_to_real < 2.0).float().mean().item()
        sample_std = s_cpu.std(dim=0).mean().item()
        pairwise = torch.cdist(s_cpu[:50], s_cpu[:50])
        pairwise_mean = pairwise[pairwise > 0].mean().item()
    results["t3_near_ratio"] = near_ratio
    results["t3_near_ratio_2"] = near_ratio_2
    results["t3_sample_std"] = sample_std
    results["t3_verdict"] = "PASS" if near_ratio > 0.3 else "FAIL"
    print(f"  Near real (<1.0): {near_ratio:.3f}  {('✅ PASS' if near_ratio > 0.3 else '❌ FAIL')}")
    print(f"  Near real (<2.0): {near_ratio_2:.3f}")
    print(f"  Sample std: {sample_std:.4f}  (diversity)")
    print(f"  Pairwise dist: {pairwise_mean:.4f}")

    # --- Verdict ---
    t2_pass = decrease > 0.15
    t3_pass = near_ratio > 0.3
    if t2_pass and t3_pass:
        verdict = "SCORE MATCHING WORKS — latent space is viable, FF was the problem"
    elif t2_pass:
        verdict = "Score guides correctly but samples miss data (sampler issue)"
    else:
        verdict = "Score matching ALSO fails — latent space itself is problematic"
    results["final_verdict"] = verdict
    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*60}")

    _save(results, "phase1d_score_matching")
    return score_net


if __name__ == "__main__":
    run_score_matching_test()
