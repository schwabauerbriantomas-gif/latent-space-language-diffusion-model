"""
FF-SplatDiffusion: Forward-Forward energy head over SplatsDB latent space.

Training: each layer learns goodness(x) = sum of squared activations.
  Positive pass: real token sequences → goodness should be LOW (low energy)
  Negative pass: corrupted sequences → goodness should be HIGH

Inference: E(x) = sum_L goodness_L(x), score = autodiff ∇E.

Usage:
    # Smoke test
    python src/ff_energy.py
"""
import json
import os
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# FF Layer — all layers operate in the SAME latent dim (1024).
# Each layer: LayerNorm → Linear(dim, hidden) → ReLU → Linear(hidden, dim)
# This way, greedy layer-wise training passes [batch, dim] to each layer.
# ---------------------------------------------------------------------------
class FFLayer(nn.Module):
    """A single Forward-Forward layer operating in the latent space.

    Architecture: x → LayerNorm → Linear(dim, hidden) → ReLU → Linear(hidden, dim)
    Goodness is computed on the hidden activations.
    Output (for the next layer) is the projected-back vector in latent space.

    FF training: local loss only, no backprop to earlier layers.
    Inference: autodiff through the frozen stack gives ∇E.
    """

    def __init__(self, dim: int = 1024, hidden: int = 512,
                 threshold_pos: float = 0.1, threshold_neg: float = 0.3,
                 lr: float = 0.5):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        self.linear_down = nn.Linear(dim, hidden)
        self.linear_up = nn.Linear(hidden, dim)
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.lr = lr

    def hidden_activations(self, x: torch.Tensor) -> torch.Tensor:
        """x → LayerNorm → Linear → ReLU. Returns hidden activations."""
        x = self.ln(x)
        h = torch.relu(self.linear_down(x))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full pass: returns projected-back vector in latent dim."""
        h = self.hidden_activations(x)
        return self.linear_up(h)

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        """Goodness = mean of squared hidden activations."""
        h = self.hidden_activations(x)
        return (h ** 2).mean(dim=-1)  # [batch]

    def ff_loss(self, x: torch.Tensor, positive: bool) -> torch.Tensor:
        g = self.goodness(x)
        if positive:
            loss = torch.log1p(torch.exp(g - self.threshold_pos)).mean()
        else:
            loss = torch.log1p(torch.exp(self.threshold_neg - g)).mean()
        return loss

    def local_update(self, x_pos: torch.Tensor, x_neg: torch.Tensor):
        """One FF training step: local gradient only, no backprop to earlier layers."""
        self.zero_grad()
        # Detach inputs so gradient does NOT flow to earlier layers or the input
        x_pos_d = x_pos.detach()
        x_neg_d = x_neg.detach()
        loss = self.ff_loss(x_pos_d, positive=True) + self.ff_loss(x_neg_d, positive=False)
        loss.backward()
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param -= self.lr * param.grad
            self.zero_grad()


# ---------------------------------------------------------------------------
# FF Energy Model — stack of FF layers defining E(x)
# ---------------------------------------------------------------------------
class FFEnergy(nn.Module):
    """Stack of FF layers. Energy E(x) = sum_L goodness_L(x).

    During training: layers trained greedily, one at a time.
    During inference: E(x) is differentiable → ∇E usable for diffusion.
    """

    def __init__(self, dim: int = 1024, hidden: int = 512, n_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            FFLayer(dim, hidden) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy E(x) = sum_L goodness_L(x).

        x: [batch, seq_len, dim] or [batch, dim]
        Returns: [batch] energy values.
        """
        if x.dim() == 3:
            x = x.mean(dim=1)  # pool sequence → single vector
        energy = torch.zeros(x.shape[0], device=x.device)
        h = x
        for layer in self.layers:
            energy = energy + layer.goodness(h)
            h = layer.forward(h).detach()  # detach: no backprop between layers at inference-energy
        # NOTE: for score(), we need gradients. We recompute without detach there.
        return energy

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Score function ∇_x E(x) via autodiff. Used in diffusion sampling.

        Here we ALLOW gradient flow through all layers (they're frozen at
        inference, so this is just forward+backward, no training).
        """
        x = x.detach().requires_grad_(True)
        if x.dim() == 1:
            x_in = x.unsqueeze(0)
        else:
            x_in = x

        h = x_in
        energy = torch.zeros(x_in.shape[0], device=x_in.device)
        for layer in self.layers:
            energy = energy + layer.goodness(h)
            h = layer.forward(h)  # NO detach — gradient flows for score computation

        if x.dim() == 1:
            energy = energy[0]
        grad = torch.autograd.grad(energy.sum(), x_in)[0]
        return grad.detach()

    def train_ff(self, x_pos: torch.Tensor, x_neg: torch.Tensor, epochs: int = 100,
                 verbose: bool = True):
        """Greedy layer-wise FF training."""
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f"[FF] Training layer {i+1}/{len(self.layers)}...")
            for ep in range(epochs):
                inp_pos = self._layer_input(x_pos, up_to=i)
                inp_neg = self._layer_input(x_neg, up_to=i)
                layer.local_update(inp_pos, inp_neg)
                if verbose and (ep % 20 == 0 or ep == epochs - 1):
                    with torch.no_grad():
                        gp = layer.goodness(inp_pos).mean().item()
                        gn = layer.goodness(inp_neg).mean().item()
                        print(f"  epoch {ep:3d}: pos_goodness={gp:.3f}  neg_goodness={gn:.3f}")

    def train_cd(self, x_pos: torch.Tensor, epochs: int = 200, k_steps: int = 15,
                 lr_langevin: float = 0.05, cd_lr: float = 0.05,
                 grad_penalty: float = 1.0,
                 verbose: bool = True):
        """Contrastive Divergence training (Hinton 2002) + gradient penalty.

        The gradient penalty (R1, Mescheder et al. 2018) penalizes ||∇E(x)||^2
        at data points. This prevents energy collapse (sharp spikes on data)
        and forces smooth valleys — making the energy useful for generation.

        Per epoch:
          1. x_neg = x_pos + noise, then k Langevin steps
          2. Loss = E(x_pos) - E(x_neg) + λ||∇E(x_pos)||^2
          3. Backprop + update
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=cd_lr)
        for ep in range(epochs):
            # 1. Generate negatives via Langevin from current energy
            #    NOTE: score computation needs grad context (no torch.no_grad),
            #    but we detach the result so it doesn't backprop through sampling
            with torch.enable_grad():
                x_neg = (x_pos.clone() + torch.randn_like(x_pos) * 0.5).detach()
                if x_neg.dim() == 3:
                    x_neg = x_neg.mean(dim=1)
                for _ in range(k_steps):
                    x_neg = x_neg.detach().requires_grad_(True)
                    energy_neg = self._energy_raw(x_neg)
                    score = torch.autograd.grad(energy_neg.sum(), x_neg)[0]
                    x_neg = (x_neg - 0.5 * lr_langevin * score +
                             (2 * lr_langevin) ** 0.5 * torch.randn_like(x_neg)).detach()

            # 2. Contrastive loss with L2 reg + R1 gradient penalty
            optimizer.zero_grad()
            xp = x_pos.mean(dim=1) if x_pos.dim() == 3 else x_pos
            xn = x_neg
            # E(x_pos) needs grad for penalty
            xp_grad = xp.detach().requires_grad_(True)
            e_pos = self._energy_raw(xp_grad)
            e_neg = self._energy_raw(xn)
            # L2 reg on energy magnitude
            l2_reg = 0.001 * (e_pos.pow(2).mean() + e_neg.pow(2).mean())
            # R1 gradient penalty: ||∇E(x_pos)||^2 — forces smooth valleys
            if grad_penalty > 0:
                grads = torch.autograd.grad(e_pos.sum(), xp_grad, create_graph=True)[0]
                gp = grads.pow(2).sum(dim=-1).mean()
            else:
                gp = torch.tensor(0.0, device=xp.device)
            loss = e_pos.mean() - e_neg.mean() + l2_reg + grad_penalty * gp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            if verbose and (ep % 20 == 0 or ep == epochs - 1):
                print(f"  [CD] ep {ep:3d}: E(pos)={e_pos.mean():.4f} E(neg)={e_neg.mean():.4f} "
                      f"margin={e_neg.mean()-e_pos.mean():.4f} |∇E|={gp.sqrt().item():.4f}")

    def _energy_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Raw energy (differentiable, for CD training)."""
        energy = torch.zeros(x.shape[0], device=x.device)
        h = x
        for layer in self.layers:
            energy = energy + layer.goodness(h)
            h = layer.forward(h)
        return energy

    def _layer_input(self, x: torch.Tensor, up_to: int) -> torch.Tensor:
        """Get input to layer `up_to` by passing through layers 0..up_to-1 (detached)."""
        if x.dim() == 3:
            x = x.mean(dim=1)
        h = x
        for j in range(up_to):
            h = self.layers[j].forward(h).detach()
        return h

    def save(self, path: str):
        torch.save({
            "dim": self.dim,
            "n_layers": len(self.layers),
            "hidden": self.layers[0].linear_down.out_features,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str):
        ckpt = torch.load(path, map_location="cpu")
        model = cls(dim=ckpt["dim"], hidden=ckpt["hidden"], n_layers=ckpt["n_layers"])
        model.load_state_dict(ckpt["state_dict"])
        return model


# ---------------------------------------------------------------------------
# Continuous diffusion sampler (Langevin guided by FF score)
# ---------------------------------------------------------------------------
class LatentDiffusionSampler:
    """Score-based sampler in SplatsDB latent space, guided by FF energy.

    Langevin dynamics: x_{t-1} = x_t - (lr/2) * ∇E(x_t) + sqrt(lr) * ε
    """

    def __init__(self, energy_model: FFEnergy, dim: int = 1024,
                 n_steps: int = 100, lr: float = 0.01):
        self.E = energy_model
        self.dim = dim
        self.n_steps = n_steps
        self.lr = lr

    def sample(self, n_samples: int, device: str = "cuda",
               return_trajectory: bool = False):
        x = torch.randn(n_samples, self.dim, device=device)
        traj = [x.detach().cpu()] if return_trajectory else None

        for t in range(self.n_steps):
            score = self.E.score(x)
            step_lr = self.lr * (1.0 - t / self.n_steps)
            noise_scale = math.sqrt(2 * step_lr)
            x = x - 0.5 * step_lr * score + noise_scale * torch.randn_like(x)
            if return_trajectory and t % 10 == 0:
                traj.append(x.detach().cpu())

        if return_trajectory:
            return x, traj
        return x


if __name__ == "__main__":
    print("=== FF Energy smoke test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(42)
    real = torch.randn(200, 1024, device=device) * 0.3 + 0.5
    noise = torch.randn(200, 1024, device=device) * 2.0

    model = FFEnergy(dim=1024, hidden=256, n_layers=2).to(device)
    print("Training FF energy (50 epochs)...")
    model.train_ff(real[:160], noise[:160], epochs=50, verbose=False)

    model.eval()
    with torch.no_grad():
        e_real = model(real[160:])
        e_noise = model(noise[160:])

    print(f"\nEnergy on held-out real:  mean={e_real.mean():.3f} std={e_real.std():.3f}")
    print(f"Energy on held-out noise: mean={e_noise.mean():.3f} std={e_noise.std():.3f}")

    from sklearn.metrics import roc_auc_score
    labels = torch.cat([torch.zeros(40), torch.ones(40)]).numpy()
    scores = torch.cat([e_real, e_noise]).cpu().numpy()
    auroc = roc_auc_score(labels, scores)
    print(f"AUROC (real=0 vs noise=1): {auroc:.3f}")
    print(f"  (>0.85 = FF learns meaningful energy, <0.70 = fails)")
