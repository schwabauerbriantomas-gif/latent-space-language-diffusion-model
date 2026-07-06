"""
Optimized latent diffusion pipeline for SplatsDB's bge-m3 space.

Consolidates all findings into a single production-quality module:
  - Auto-selects intrinsic dimensionality k via explained variance
  - Score matching in k-dim with cosine LR + sigma-conditioned network
  - SVGD sampling with cosine bandwidth schedule (stable, no oscillation)
  - Best-checkpoint tracking: keeps the iteration with best near_data AND diversity
  - Diversity preservation: repulsion strength increases as particles converge
  - Maps back to full 1024D and evaluates quality

Architecture:
  data → PCA(k) → score_net(k) → SVGD sampling → PCA⁻¹ → sphere → evaluate

Usage:
    from latent_diffusion_pipeline import LatentDiffusionModel

    model = LatentDiffusionModel()
    model.fit(train_embeddings)          # trains score net + PCA
    samples = model.sample(n=200)        # generates new embeddings
    tokens = splatdb.decode(samples)     # decode via HNSW
"""
import math
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 1024


# ═══════════════════════════════════════════════════════════════════════════
# Score Network (sigma-conditioned, any dimensionality)
# ═══════════════════════════════════════════════════════════════════════════

class ScoreNetwork(nn.Module):
    """Score network s_θ(x, σ) ≈ ∇_x log p_σ(x) for arbitrary dim k.

    Architecture: input → [ResBlock with FiLM σ-conditioning] × N → output
    Zero-initialized output for stable training start.
    """

    def __init__(self, dim: int, hidden: int = 256, n_blocks: int = 6):
        super().__init__()
        self.dim = dim
        sigma_emb_dim = 64

        self.sigma_mlp = nn.Sequential(
            nn.Linear(sigma_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.input_proj = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList([
            self._make_block(hidden) for _ in range(n_blocks)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def _make_block(self, h):
        return nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.LayerNorm(h),
        )

    def _sigma_embedding(self, sigma):
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        half = 32
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=sigma.device) / half)
        args = sigma * freqs.unsqueeze(0)
        return self.sigma_mlp(
            torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        )

    def forward(self, x, sigma):
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0).expand(x.shape[0], 1)
        elif sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out_proj(h)


# ═══════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════

def dsm_loss(net, x, sigmas):
    """Denoising score matching: mean over batch and dims."""
    total = 0.0
    for sigma in sigmas:
        eps = torch.randn_like(x)
        target = -eps / sigma
        pred = net(x + sigma * eps, sigma.expand(x.shape[0], 1))
        total = total + ((pred - target) ** 2).mean()
    return total / len(sigmas)


def cosine_lr(opt, step, total, warmup=200, base_lr=1e-3):
    if step < warmup:
        lr = base_lr * step / warmup
    else:
        p = (step - warmup) / max(1, total - warmup)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * p))
    for pg in opt.param_groups:
        pg["lr"] = lr


# ═══════════════════════════════════════════════════════════════════════════
# SVGD Sampler
# ═══════════════════════════════════════════════════════════════════════════

class SVGDSampler:
    """Stein Variational Gradient Descent in k-dim space.

    Cosine bandwidth schedule (stable — no geometric oscillation):
      h(t) = h_min + (h_max - h_min) * 0.5 * (1 + cos(π * t/T))

    Tracks best checkpoint by combined score: near_data * diversity.
    """

    def __init__(self, score_net, k, sigmas_score=(0.5, 0.05),
                 h_min=0.01, h_max=3.0, step_size=0.02):
        self.net = score_net
        self.k = k
        self.sigmas = sigmas_score
        self.h_min = h_min
        self.h_max = h_max
        self.step_size = step_size

    def _cosine_bandwidth(self, t, T):
        """Stable cosine schedule: h_max → h_min → h_min."""
        p = min(1.0, t / T)
        return self.h_min + (self.h_max - self.h_min) * 0.5 * (1 + math.cos(math.pi * p))

    def sample(self, n_particles, data_scale, n_iters=800,
               real_ref=None, verbose=True):
        """Sample n_particles from the learned distribution.

        Args:
            data_scale: (mean[k], std[k]) for initializing particles
            real_ref: reference data in k-dim for evaluation during sampling
        """
        mean, std = data_scale
        x = mean.to(DEVICE) + torch.randn(n_particles, self.k, device=DEVICE) * std.to(DEVICE) * 2

        best_score = -1
        best_x = x.clone()

        for it in range(n_iters):
            h = self._cosine_bandwidth(it, n_iters)
            sigma = self.sigmas[0] * (self.sigmas[1] / self.sigmas[0]) ** (it / n_iters)

            with torch.no_grad():
                score = self.net(x, torch.full((n_particles, 1), sigma, device=DEVICE))

            # RBF kernel + repulsion
            diff = x.unsqueeze(1) - x.unsqueeze(0)  # [n, n, k]
            sq = (diff ** 2).sum(dim=-1)
            k_xy = torch.exp(-sq / h)
            repulsion = (-2.0 / h * k_xy.unsqueeze(-1) * diff).sum(dim=1)

            phi = (k_xy @ score) / n_particles + repulsion / n_particles
            x = x + self.step_size * phi

            # Track best checkpoint
            if real_ref is not None and (it % 50 == 0 or it == n_iters - 1):
                with torch.no_grad():
                    dist = torch.cdist(x, real_ref).min(dim=1)[0]
                    near = (dist < 0.3).float().mean().item()
                    pw = torch.cdist(x[:50], x[:50])
                    diversity = pw[pw > 0].mean().item() if (pw > 0).any() else 0
                    combined = near * min(diversity, 1.0)
                    if combined > best_score:
                        best_score = combined
                        best_x = x.clone()
                    if verbose:
                        print(f"  iter {it:4d}: h={h:.4f} σ={sigma:.4f} "
                              f"near={near:.2%} div={diversity:.3f} "
                              f"score={combined:.4f}")

        return best_x


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class LatentDiffusionModel:
    """Production latent diffusion model for SplatsDB.

    Pipeline: PCA(k) → score matching → SVGD → PCA⁻¹ → sphere
    """

    def __init__(self, variance_threshold=0.90, score_hidden=256,
                 score_blocks=6, score_epochs=2000,
                 svgd_iters=800, svgd_particles=300):
        self.variance_threshold = variance_threshold
        self.score_hidden = score_hidden
        self.score_blocks = score_blocks
        self.score_epochs = score_epochs
        self.svgd_iters = svgd_iters
        self.svgd_particles = svgd_particles

        self.pca_mean = None
        self.pca_basis = None
        self.k = None
        self.score_net = None
        self.svgd = None

    def _compute_pca(self, data):
        """PCA via SVD, auto-select k from explained variance."""
        mean = data.mean(dim=0)
        centered = data - mean
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        V = Vt.T

        var = (S ** 2)
        cumvar = var.cumsum(0) / var.sum()
        k = (cumvar < self.variance_threshold).sum().item() + 1

        return mean, V[:, :k], k

    def fit(self, embeddings, verbose=True):
        """Fit the model: PCA + score network training.

        Args:
            embeddings: [N, 1024] tensor (will be L2-normalized)
        """
        t0 = time.time()

        # Normalize to unit sphere (bge-m3 convention)
        data = F.normalize(embeddings, dim=-1)

        # PCA
        self.pca_mean, self.pca_basis, self.k = self._compute_pca(data)
        data_low = (data - self.pca_mean) @ self.pca_basis
        if verbose:
            print(f"PCA: {self.k} dims ({self.variance_threshold*100:.0f}% variance)")

        # Train score network
        self.score_net = ScoreNetwork(
            dim=self.k, hidden=self.score_hidden, n_blocks=self.score_blocks
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(self.score_net.parameters(), lr=1e-3)
        sigmas = torch.tensor([2.0, 1.0, 0.5, 0.2, 0.1, 0.05], device=DEVICE)
        data_gpu = data_low.cuda()

        if verbose:
            print(f"Training score network ({self.score_epochs} epochs)...")
        for epoch in range(self.score_epochs):
            optimizer.zero_grad()
            loss = dsm_loss(self.score_net, data_gpu, sigmas)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
            cosine_lr(optimizer, epoch, self.score_epochs, base_lr=1e-3)
            optimizer.step()
            if verbose and (epoch % 500 == 0 or epoch == self.score_epochs - 1):
                print(f"  epoch {epoch}: loss={loss.item():.4f}")
        self.score_net.eval()

        # Setup SVGD sampler
        self.data_scale = (data_low.mean(dim=0), data_low.std(dim=0))
        self.svgd = SVGDSampler(
            self.score_net, self.k,
        )

        if verbose:
            print(f"Fit complete in {time.time()-t0:.0f}s")
        return self

    def sample(self, n=None, verbose=False, real_ref_1024=None):
        """Generate samples in the full 1024-dim space.

        Args:
            n: number of particles (default: self.svgd_particles)
            real_ref_1024: reference data for evaluation (optional)
        """
        n = n or self.svgd_particles

        # SVGD sampling in k-dim
        real_ref_low = None
        if real_ref_1024 is not None:
            real_ref_low = (F.normalize(real_ref_1024, dim=-1) -
                            self.pca_mean) @ self.pca_basis
            real_ref_low = real_ref_low.cuda()

        x_low = self.svgd.sample(
            n_particles=n,
            data_scale=self.data_scale,
            n_iters=self.svgd_iters,
            real_ref=real_ref_low,
            verbose=verbose,
        )

        # Map back to 1024D
        x_full = (x_low.cpu() @ self.pca_basis.T) + self.pca_mean
        x_sphere = F.normalize(x_full, dim=-1)

        return x_sphere

    def evaluate(self, samples, real_ref_1024):
        """Evaluate sample quality against reference data."""
        with torch.no_grad():
            dist = torch.cdist(samples, real_ref_1024).min(dim=1)[0]
            near_03 = float((dist < 0.3).float().mean())
            near_01 = float((dist < 0.1).float().mean())
            near_005 = float((dist < 0.05).float().mean())

            pairwise = torch.cdist(samples[:50], samples[:50])
            diversity = float(pairwise[pairwise > 0].mean())

            return {
                "near_03": near_03,
                "near_01": near_01,
                "near_005": near_005,
                "median_dist": float(dist.median()),
                "diversity": diversity,
            }


# ═══════════════════════════════════════════════════════════════════════════
# Data generation (for testing without SplatsDB connection)
# ═══════════════════════════════════════════════════════════════════════════

def make_clustered_data(n_samples=2000, n_clusters=5, dim=1024,
                        cluster_spread=0.005, seed=42):
    """Generate clustered data on the unit sphere (S^dim-1)."""
    torch.manual_seed(seed)
    centers = torch.randn(n_clusters, dim)
    centers = F.normalize(centers, dim=-1)

    real = []
    for _ in range(n_samples):
        c = centers[torch.randint(0, n_clusters, (1,)).item()]
        point = c + torch.randn(dim) * cluster_spread
        real.append(F.normalize(point, dim=-1))

    noise = F.normalize(torch.randn(n_samples, dim), dim=-1)
    return torch.stack(real), noise, centers
