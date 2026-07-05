"""
Scaled-up Denoising Score Matching — the real attempt.

Previous run (phase1d): cos=0.38 (correct direction but WEAK). T2/T3 failed.
Diagnosis: underfitting in 1024D — small network, no sigma conditioning,
loss summed (not averaged), few epochs.

This version fixes all of that:
  1. ScoreNetwork v2: 8 residual blocks, 1024 hidden, FiLM sigma conditioning
  2. Data: unit-norm on hypersphere (matches bge-m3 exactly)
  3. Loss: MEAN over batch and dimensions (correct normalization)
  4. Training: 3000 epochs, cosine LR schedule, gradient clipping
  5. Sampling: annealed Langevin with 10 sigma levels, calibrated step sizes

Usage:
    python src/phase2_score_matching_scaled.py
"""
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 1024
RESULTS_DIR = REPO / "results"


# ---------------------------------------------------------------------------
# Data: clusters on the unit hypersphere (matches bge-m3 normalized embeddings)
# ---------------------------------------------------------------------------
def make_unit_norm_data(n_samples=2000, n_clusters=20, dim=DIM, cluster_spread=0.15):
    """Generate data on the unit hypersphere, organized in clusters.

    bge-m3 embeddings are L2-normalized → all points live on S^1023.
    Clusters represent semantic groups (like SplatsDB splats).

    cluster_spread: angular spread of each cluster (lower = tighter).
    """
    torch.manual_seed(42)
    # Cluster centers on the unit sphere
    centers = torch.randn(n_clusters, dim)
    centers = F.normalize(centers, dim=-1)

    real = []
    cluster_labels = []
    for i in range(n_samples):
        c = centers[torch.randint(0, n_clusters, (1,)).item()]  # [dim]
        # Add noise, then re-normalize to stay on sphere
        point = c + torch.randn(dim) * cluster_spread
        point = F.normalize(point, dim=-1)
        real.append(point)
        cluster_labels.append(c)
    real = torch.stack(real)  # [n_samples, dim]

    # Noise: uniform on sphere (random direction)
    noise = torch.randn(n_samples, dim)
    noise = F.normalize(noise, dim=-1)

    return real.to(DEVICE), noise.to(DEVICE), centers.to(DEVICE)


# ---------------------------------------------------------------------------
# ScoreNetwork v2: deep, wide, sigma-conditioned
# ---------------------------------------------------------------------------
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: condition features on sigma.

    gamma, beta = MLP(sigma_embedding)
    output = (1 + gamma) * features + beta

    sigma_emb is already an [batch, hidden] embedding from ScoreNetworkV2.
    FiLM maps it to per-feature gain/bias.
    Zero-init so block starts as identity.
    """

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.gb_net = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),  # gamma and beta
        )
        nn.init.zeros_(self.gb_net[-1].weight)
        nn.init.zeros_(self.gb_net[-1].bias)

    def forward(self, x, sigma_emb):
        gb = self.gb_net(sigma_emb)
        gamma, beta = gb.chunk(2, dim=-1)
        return (1.0 + gamma) * x + beta


class ResidualBlock(nn.Module):
    """Pre-norm residual block with FiLM conditioning."""

    def __init__(self, dim, hidden, film: FiLMLayer):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.film = film

    def forward(self, x, sigma_emb):
        h = self.norm1(x)
        h = self.film(h, sigma_emb)  # condition on sigma
        h = self.act(self.linear1(h))
        h = self.linear2(h)
        h = self.norm2(h + x)  # residual
        return h


class ScoreNetworkV2(nn.Module):
    """Score network with sigma conditioning via FiLM.

    Architecture:
      input (1024) → [ResBlock × 8] → output (1024)

    Each ResBlock is conditioned on the noise level sigma via FiLM.
    This is the standard approach from NCSN (Song & Ermon 2019).
    """

    def __init__(self, dim=DIM, hidden=1024, n_blocks=8):
        super().__init__()
        self.dim = dim

        # Input projection
        self.input_proj = nn.Linear(dim, hidden)

        # Sigma embedding: sinusoidal positional encoding for continuous sigma
        self.sigma_emb_dim = hidden
        self.sigma_mlp = nn.Sequential(
            nn.Linear(self.sigma_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Residual blocks with shared FiLM
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            film = FiLMLayer(hidden, hidden)  # cond_dim = sigma_emb dim = hidden
            block = ResidualBlock(hidden, hidden, film)
            self.blocks.append(block)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        # Initialize output to small values (start with weak score)
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def _sigma_embedding(self, sigma):
        """Sinusoidal embedding for continuous sigma values."""
        # sigma: [batch, 1] or [batch]
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        half = self.sigma_emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=sigma.device) / half
        )
        args = sigma * freqs.unsqueeze(0)  # [batch, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [batch, emb_dim]
        return self.sigma_mlp(emb)

    def forward(self, x, sigma):
        """Predict score s(x, sigma) ≈ ∇_x log p_sigma(x).

        Args:
            x: [batch, dim] — noisy data point
            sigma: [batch, 1] or scalar — noise level

        Returns:
            score: [batch, dim] — estimated score
        """
        if sigma.dim() == 0:
            sigma = sigma.expand(x.shape[0], 1)
        elif sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1) if sigma.shape[0] == x.shape[0] else sigma.unsqueeze(-1)

        sigma_emb = self._sigma_embedding(sigma)

        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, sigma_emb)
        score = self.output_proj(h)
        return score


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def get_sigma_schedule(n_levels=10, sigma_max=1.5, sigma_min=0.05):
    """Geometric schedule of noise levels."""
    sigmas = torch.tensor(
        [sigma_max * (sigma_min / sigma_max) ** (i / (n_levels - 1))
         for i in range(n_levels)]
    )
    return sigmas.to(DEVICE)


def dsm_loss_v2(score_net, x, sigmas):
    """Multi-scale denoising score matching with CORRECT normalization.

    For each sigma:
      x_noisy = x + sigma * eps
      target = -eps / sigma
      loss = mean over batch AND dimensions of ||s(x_noisy, sigma) - target||²

    This is per-dimension MSE, not sum. Critical for 1024D.
    """
    batch_size = x.shape[0]
    total_loss = 0.0

    for sigma in sigmas:
        # Sample one sigma per data point (stochastic — better gradient estimate)
        sigma_per_point = sigma.expand(batch_size, 1)

        eps = torch.randn_like(x)
        x_noisy = x + sigma * eps
        target = -eps / sigma

        pred = score_net(x_noisy, sigma_per_point)

        # MEAN over dimensions (not sum!)
        loss_per_point = ((pred - target) ** 2).mean(dim=-1)  # [batch]
        total_loss = total_loss + loss_per_point.mean()

    return total_loss / len(sigmas)


def cosine_lr_schedule(optimizer, step, total_steps, warmup=100, base_lr=1e-3):
    """Cosine annealing with linear warmup."""
    if step < warmup:
        lr = base_lr * step / warmup
    else:
        progress = (step - warmup) / (total_steps - warmup)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_score_network(real_data, n_epochs=3000, base_lr=1e-3):
    """Train ScoreNetworkV2 with cosine schedule."""
    sigmas = get_sigma_schedule(n_levels=10)
    print(f"Sigma schedule: {[f'{s:.3f}' for s in sigmas.cpu()]}")

    score_net = ScoreNetworkV2(dim=DIM, hidden=1024, n_blocks=8).to(DEVICE)
    n_params = sum(p.numel() for p in score_net.parameters())
    print(f"ScoreNetworkV2: {n_params:,} parameters ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(score_net.parameters(), lr=base_lr, weight_decay=1e-4)

    train_data = real_data[:1600]  # 80% train

    print(f"\nTraining {n_epochs} epochs...")
    losses = []
    start_time = time.time()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        loss = dsm_loss_v2(score_net, train_data, sigmas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)

        lr = cosine_lr_schedule(optimizer, epoch, n_epochs, warmup=100, base_lr=base_lr)
        optimizer.step()

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            elapsed = time.time() - start_time
            print(f"  epoch {epoch:4d}: loss={loss.item():.6f}  lr={lr:.6f}  ({elapsed:.0f}s)")
            losses.append({"epoch": epoch, "loss": loss.item(), "lr": lr})

    total_time = time.time() - start_time
    print(f"Training complete: {total_time:.0f}s ({total_time/60:.1f} min)")

    score_net.eval()
    return score_net, {"losses": losses, "n_params": n_params, "train_time_s": total_time}


# ---------------------------------------------------------------------------
# Evaluation: T1 (discrimination), T2 (gradient direction), T3 (sampling)
# ---------------------------------------------------------------------------
def evaluate_t1_direction(score_net, real, noise):
    """T1-direction: does the score point toward data?

    The ultimate test: cosine similarity between learned score and the
    actual direction toward nearest data point.

    cos > 0: correct direction (score guides toward data)
    cos > 0.5: strong guidance (sufficient for Langevin)
    cos > 0.8: excellent (near-optimal)
    """
    print("\n" + "=" * 60)
    print("T1-direction: Score direction quality (THE KEY METRIC)")
    print("=" * 60)

    real_ref = real[:800]
    sigmas_test = [0.05, 0.1, 0.2, 0.5, 1.0]

    results = {}
    for sigma in sigmas_test:
        # Add noise to real data, then check if score points back
        eps = torch.randn(500, DIM, device=DEVICE)
        x_noisy = real_ref[:500] + sigma * eps

        # True direction: from noisy point back to clean data
        true_direction = -eps / sigma  # = -eps/sigma (normalized in cosine)

        with torch.no_grad():
            sigma_batch = torch.full((500, 1), sigma, device=DEVICE)
            pred_score = score_net(x_noisy, sigma_batch)

        # Cosine similarity (over dim axis)
        cos_sim = F.cosine_similarity(pred_score, true_direction, dim=-1)
        cos_mean = cos_sim.mean().item()
        cos_std = cos_sim.std().item()

        # Also check: does score point toward ANY real data (nearest neighbor)?
        with torch.no_grad():
            dists = torch.cdist(x_noisy, real_ref)  # [500, 800]
            nn_idx = dists.argmin(dim=1)
            nn_direction = real_ref[nn_idx] - x_noisy
            nn_direction = F.normalize(nn_direction, dim=-1)
            cos_to_nn = F.cosine_similarity(pred_score, nn_direction, dim=-1)

        print(f"  σ={sigma:.2f}: cos(score, true)={cos_mean:+.3f}±{cos_std:.3f}  "
              f"cos(score, nearest_data)={cos_to_nn.mean():+.3f}")

        results[f"sigma_{sigma}"] = {
            "cos_to_true": cos_mean,
            "cos_to_true_std": cos_std,
            "cos_to_nearest_data": cos_to_nn.mean().item(),
        }

    # Overall verdict: average cos across sigmas
    avg_cos = np.mean([results[f"sigma_{s}"]["cos_to_true"] for s in sigmas_test])
    avg_cos_nn = np.mean([results[f"sigma_{s}"]["cos_to_nearest_data"] for s in sigmas_test])
    results["avg_cos_to_true"] = avg_cos
    results["avg_cos_to_nearest"] = avg_cos_nn

    verdict = "EXCELLENT" if avg_cos > 0.8 else \
              "STRONG" if avg_cos > 0.5 else \
              "WEAK" if avg_cos > 0.2 else "ADVERSARIAL"
    results["verdict"] = verdict
    print(f"\n  Average cos(score, true direction): {avg_cos:+.3f} → {verdict}")
    print(f"  Average cos(score, nearest data):   {avg_cos_nn:+.3f}")

    return results


def evaluate_t2_sampling(score_net, real, n_steps=500):
    """T2: Does Langevin sampling move noise toward data?"""
    print("\n" + "=" * 60)
    print("T2: Langevin sampling trajectory")
    print("=" * 60)

    real_ref = real[:800]
    sigmas = get_sigma_schedule(n_levels=10)

    # Start from random point on sphere (like random noise)
    x = torch.randn(200, DIM, device=DEVICE)
    x = F.normalize(x, dim=-1)

    distances = []
    steps_per_sigma = n_steps // len(sigmas)
    step_lr_base = 0.001

    for si, sigma in enumerate(sigmas):
        for step in range(steps_per_sigma):
            with torch.no_grad():
                sigma_batch = torch.full((200, 1), sigma.item(), device=DEVICE)
                score = score_net(x, sigma_batch)

            # Langevin dynamics on the sphere (project back after each step)
            step_lr = step_lr_base * (sigma.item() ** 2)
            x = x + 0.5 * step_lr * score
            x = x + math.sqrt(step_lr) * torch.randn_like(x) * 0.1  # small noise
            x = F.normalize(x, dim=-1)  # project back to sphere

            global_step = si * steps_per_sigma + step
            if global_step % 50 == 0 or global_step == n_steps - 1:
                with torch.no_grad():
                    dist = torch.cdist(x, real_ref).min(dim=1)[0]
                    d = {
                        "step": global_step,
                        "sigma": float(sigma),
                        "median_dist": float(dist.median()),
                        "mean_dist": float(dist.mean()),
                        "pct_near": float((dist < 0.5).float().mean()),
                    }
                    distances.append(d)
                    print(f"  step {global_step:3d} (σ={sigma:.3f}): "
                          f"median={d['median_dist']:.4f}  "
                          f"near(<0.5)={d['pct_near']:.2%}")

    medians = [d["median_dist"] for d in distances]
    decrease = (medians[0] - medians[-1]) / medians[0] if medians[0] > 0 else 0

    results = {
        "distances": distances,
        "decrease_pct": decrease,
        "final_median": medians[-1] if medians else None,
        "verdict": "PASS" if decrease > 0.15 else "FAIL",
    }
    print(f"\n  Distance decrease: {decrease*100:.1f}% → {results['verdict']}")
    return results


def evaluate_t3_generation(score_net, real, n_samples=200):
    """T3: Generate samples and measure quality."""
    print("\n" + "=" * 60)
    print("T3: Sample generation quality")
    print("=" * 60)

    real_ref = real[:800]
    sigmas = get_sigma_schedule(n_levels=10)

    # Annealed Langevin
    x = torch.randn(n_samples, DIM, device=DEVICE)
    x = F.normalize(x, dim=-1)

    n_steps_total = 1000
    steps_per_sigma = n_steps_total // len(sigmas)
    step_lr_base = 0.001

    for sigma in sigmas:
        for _ in range(steps_per_sigma):
            with torch.no_grad():
                sigma_batch = torch.full((n_samples, 1), sigma.item(), device=DEVICE)
                score = score_net(x, sigma_batch)
            step_lr = step_lr_base * (sigma.item() ** 2)
            x = x + 0.5 * step_lr * score
            x = x + math.sqrt(step_lr) * torch.randn_like(x) * 0.1
            x = F.normalize(x, dim=-1)

    samples = x.cpu()
    real_cpu = real_ref.cpu()

    with torch.no_grad():
        # Distance to nearest real data
        dist_to_real = torch.cdist(samples, real_cpu).min(dim=1)[0]
        near_05 = (dist_to_real < 0.5).float().mean().item()
        near_10 = (dist_to_real < 1.0).float().mean().item()

        # Sample diversity
        sample_std = samples.std(dim=0).mean().item()
        pairwise = torch.cdist(samples[:50], samples[:50])
        pairwise_mean = pairwise[pairwise > 0].mean().item()

        # Mode coverage: how many clusters do samples reach?
        from collections import Counter
        # Assign each sample to nearest cluster center
        # (approximate by nearest real point's implied cluster)
        nn_idx = torch.cdist(samples, real_cpu).argmin(dim=1)
        unique_nn = len(torch.unique(nn_idx))

    results = {
        "near_data_05": near_05,
        "near_data_10": near_10,
        "sample_std": sample_std,
        "pairwise_dist": pairwise_mean,
        "unique_modes_reached": unique_nn,
        "total_modes": 800,
        "verdict": "PASS" if near_05 > 0.3 else "FAIL",
    }

    print(f"  Near data (<0.5 dist): {near_05:.2%}  {('✅' if near_05 > 0.3 else '❌')}")
    print(f"  Near data (<1.0 dist): {near_10:.2%}")
    print(f"  Sample std (diversity): {sample_std:.4f}")
    print(f"  Pairwise dist (spread): {pairwise_mean:.4f}")
    print(f"  Modes reached: {unique_nn}/{800}")
    print(f"  Verdict: {results['verdict']}")

    return results


def evaluate_t1_discrimination(score_net, real, noise):
    """T1-classic: can score magnitude discriminate real from noise?"""
    print("\n" + "=" * 60)
    print("T1-classic: Discrimination via score magnitude")
    print("=" * 60)

    real_test = real[800:]   # [1200, 1024]
    noise_test = noise[800:]  # [1200, 1024]

    with torch.no_grad():
        sigma_batch_real = torch.full((real_test.shape[0], 1), 0.2, device=DEVICE)
        sigma_batch_noise = torch.full((noise_test.shape[0], 1), 0.2, device=DEVICE)
        s_real = score_net(real_test, sigma_batch_real).norm(dim=-1).cpu().numpy()
        s_noise = score_net(noise_test, sigma_batch_noise).norm(dim=-1).cpu().numpy()

    labels = np.concatenate([np.zeros(len(s_real)), np.ones(len(s_noise))])
    scores = -np.concatenate([s_real, s_noise])
    auroc = roc_auc_score(labels, scores)

    print(f"  ||s|| real: {s_real.mean():.2f}  ||s|| noise: {s_noise.mean():.2f}")
    print(f"  AUROC: {auroc:.4f}  {'✅ PASS' if auroc > 0.85 else '❌ FAIL'}")
    return {"auroc": float(auroc), "s_real_mean": float(s_real.mean()),
            "s_noise_mean": float(s_noise.mean())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_scaled_experiment():
    print("=" * 60)
    print("Phase 2: SCALED Score Matching")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Dim: {DIM}")

    torch.manual_seed(42)

    # Data
    real, noise, centers = make_unit_norm_data(n_samples=2000, n_clusters=20)
    print(f"\nData: {real.shape[0]} real (unit-norm), {noise.shape[0]} noise")
    print(f"  Real ||x||: {real.norm(dim=-1).mean():.4f} (should be 1.0)")
    print(f"  Noise ||x||: {noise.norm(dim=-1).mean():.4f} (should be 1.0)")

    # Data statistics for calibration
    with torch.no_grad():
        inter_cluster = torch.cdist(centers, centers)
        inter_cluster = inter_cluster[inter_cluster > 0]
        intra_cluster_dists = torch.cdist(real[:100], real[:100])
    print(f"  Inter-cluster distance: {inter_cluster.mean():.4f} ± {inter_cluster.std():.4f}")
    print(f"  Intra-cluster distance: {intra_cluster_dists[intra_cluster_dists > 0].mean():.4f}")

    # Train
    score_net, train_info = train_score_network(real, n_epochs=3000, base_lr=1e-3)

    # Save model IMMEDIATELY (before evaluation — eval may crash, model shouldn't be lost)
    torch.save(score_net.state_dict(), REPO / "checkpoints" / "score_net_v2.pt")
    print(f"\nModel saved to checkpoints/score_net_v2.pt")

    # Evaluate
    results = {
        "experiment": "phase2_scaled_score_matching",
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE,
        "architecture": {
            "hidden": 1024,
            "n_blocks": 8,
            "n_params": train_info["n_params"],
            "conditioning": "FiLM (sinusoidal sigma embedding)",
        },
        "training": {
            "epochs": 3000,
            "base_lr": 1e-3,
            "schedule": "cosine with warmup",
            "final_loss": train_info["losses"][-1]["loss"],
            "train_time_s": train_info["train_time_s"],
        },
        "loss_history": train_info["losses"],
    }

    results["t1_discrimination"] = evaluate_t1_discrimination(score_net, real, noise)
    results["t1_direction"] = evaluate_t1_direction(score_net, real, noise)
    results["t2_sampling"] = evaluate_t2_sampling(score_net, real)
    results["t3_generation"] = evaluate_t3_generation(score_net, real)

    # Overall verdict
    t1d_cos = results["t1_direction"]["avg_cos_to_true"]
    t2_pass = results["t2_sampling"]["verdict"] == "PASS"
    t3_pass = results["t3_generation"]["verdict"] == "PASS"

    if t1d_cos > 0.5 and t2_pass and t3_pass:
        verdict = "SCORE MATCHING WORKS AT SCALE — generation viable"
    elif t1d_cos > 0.5 and t2_pass:
        verdict = "Score guides correctly, sampling partially works"
    elif t1d_cos > 0.5:
        verdict = "Score direction learned but sampling needs tuning"
    elif t1d_cos > 0.2:
        verdict = "Weak score direction — more scaling needed"
    else:
        verdict = "Score matching fails even at scale"

    results["final_verdict"] = verdict
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {verdict}")
    print(f"  cos(score, true): {t1d_cos:+.3f}")
    print(f"  T2 sampling: {results['t2_sampling']['verdict']}")
    print(f"  T3 generation: {results['t3_generation']['verdict']}")
    print(f"{'='*60}")

    # Save
    out_path = RESULTS_DIR / "phase2_scaled_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    return results


if __name__ == "__main__":
    run_scaled_experiment()
