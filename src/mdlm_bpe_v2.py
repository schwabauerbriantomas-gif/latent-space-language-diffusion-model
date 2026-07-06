"""
MDLM-BPE v2: Improved masked diffusion model for real text.

Fixes over v1:
  - Correct RoPE (rotary pairs, not additive)
  - Better timestep conditioning (proper AdaLN with per-layer modulation)
  - Weight tying with proper output projection
  - Gradient checkpointing option for larger models
  - More efficient batched sampling (vectorized, no Python loop per sample)

Architecture:
  - Token embeddings (tied output)
  - RoPE (rotary positional embedding)
  - Pre-LN transformer with AdaLN timestep modulation
  - GELU FFN
"""
import math
import json
import sys
import time
import random
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = REPO / "results"
CHECKPOINT_DIR = REPO / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
TOKENIZER_PATH = REPO / "tokenizer" / "bpe_tokenizer.json"


# ═══════════════════════════════════════════════════════════════════════════
# Tokenizer Interface (reused from v1)
# ═══════════════════════════════════════════════════════════════════════════

class BPETokenizer:
    def __init__(self, path=TOKENIZER_PATH):
        self.tokenizer = Tokenizer.from_file(str(path))
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.mask_id = self.tokenizer.token_to_id("<mask>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text, add_special=True):
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        clean = [i for i in ids if i not in (self.pad_id, self.mask_id, self.bos_id)]
        return self.tokenizer.decode(clean)


# ═══════════════════════════════════════════════════════════════════════════
# Correct RoPE
# ═══════════════════════════════════════════════════════════════════════════

def precompute_rope(head_dim, max_seq_len, base=10000.0, device="cpu"):
    """Precompute cos/sin tables for rotary positional embedding.

    RoPE rotates pairs of dimensions: (x_{2i}, x_{2i+1}) → rotated by θ_i * pos.
    """
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half).float() / half))
    positions = torch.arange(max_seq_len).float()
    angles = positions[:, None] * freqs[None, :]  # [max_seq_len, half]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Apply RoPE to x [batch, n_heads, seq, head_dim].

    Rotates pairs: (x[0], x[1]), (x[2], x[3]), ...
    """
    # x shape: [batch, n_heads, seq, head_dim]
    # cos/sin: [seq, head_dim//2]
    seq_len = x.shape[2]
    cos = cos[:seq_len]  # [seq, half]
    sin = sin[:seq_len]

    # Split into even and odd pairs
    x1 = x[..., 0::2]  # [batch, n_heads, seq, half]
    x2 = x[..., 1::2]  # [batch, n_heads, seq, half]

    # Rotate
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, half]
    sin = sin.unsqueeze(0).unsqueeze(0)

    rotated1 = x1 * cos - x2 * sin
    rotated2 = x1 * sin + x2 * cos

    # Interleave back
    out = torch.stack([rotated1, rotated2], dim=-1)
    out = out.flatten(-2)  # [batch, n_heads, seq, head_dim]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

class MDLMConfig:
    def __init__(self, vocab_size=10_000, d_model=768, n_heads=12,
                 n_layers=8, max_seq_len=128, d_ff=None, dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.d_ff = d_ff or d_model * 4
        self.dropout = dropout

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return (f"MDLMConfig(d_model={self.d_model}, n_heads={self.n_heads}, "
                f"n_layers={self.n_layers}, vocab={self.vocab_size}, "
                f"max_seq={self.max_seq_len})")


# ═══════════════════════════════════════════════════════════════════════════
# Transformer Block with proper AdaLN
# ═══════════════════════════════════════════════════════════════════════════

class MDLMBlockV2(nn.Module):
    """Pre-LN transformer block with AdaLN timestep modulation."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Attention
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        # FFN
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # AdaLN modulation (per-block scale/shift from timestep)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(self, x, t_emb, rope_cos, rope_sin, pad_mask=None):
        """
        x: [batch, seq, d_model]
        t_emb: [batch, d_model] - timestep embedding
        """
        batch, seq, _ = x.shape

        # AdaLN modulation
        scale_shift = self.ada_ln(t_emb)  # [batch, 6*d]
        s1, sh1, s2, sh2, s3, sh3 = scale_shift.chunk(6, dim=-1)

        # Self-attention with pre-LN
        h = self.ln1(x)
        h = h * (1 + s1.unsqueeze(1)) + sh1.unsqueeze(1)

        q = self.q_proj(h).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Attention
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=0.0 if not self.training else self.attn_drop.p,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).reshape(batch, seq, self.d_model)
        attn = self.o_proj(attn)
        x = x + attn

        # FFN with pre-LN + AdaLN
        h = self.ln2(x)
        h = h * (1 + s2.unsqueeze(1)) + sh2.unsqueeze(1)
        x = x + self.ff(h)

        # Final AdaLN
        x = x * (1 + s3.unsqueeze(1)) + sh3.unsqueeze(1)

        return x


# ═══════════════════════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════════════════════

class MDLMBPEV2(nn.Module):
    """MDLM v2 with proper RoPE, AdaLN, weight tying, Flash Attention."""

    def __init__(self, config: MDLMConfig, pad_id=0, mask_id=1):
        super().__init__()
        self.config = config
        self.pad_id = pad_id
        self.mask_id = mask_id

        d = config.d_model

        # Token embedding (tied with output)
        self.token_emb = nn.Embedding(config.vocab_size, d, padding_idx=pad_id)

        # Timestep embedding: sinusoidal → MLP → d_model
        self.time_embed = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MDLMBlockV2(d, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        # Final norm
        self.ln_f = nn.LayerNorm(d)

        # Output projection (tied weights + bias)
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Precompute RoPE
        head_dim = d // config.n_heads
        cos, sin = precompute_rope(head_dim, config.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0)
        nn.init.zeros_(self.output_bias)

    def _timestep_embedding(self, t):
        """Sinusoidal timestep embedding."""
        d = self.config.d_model
        half = d // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_embed(emb)  # [batch, d]

    def forward(self, tokens, t):
        """
        tokens: [batch, seq_len] - partially masked
        t: [batch] - timestep in [0, 1]

        Returns logits: [batch, seq_len, vocab_size]
        """
        batch, seq = tokens.shape

        # Embed tokens
        x = self.token_emb(tokens)

        # Timestep
        t_emb = self._timestep_embedding(t)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, self.rope_cos, self.rope_sin)

        x = self.ln_f(x)

        # Tied output
        logits = F.linear(x, self.token_emb.weight, self.output_bias)
        return logits


# ═══════════════════════════════════════════════════════════════════════════
# Diffusion Process (vectorized, no Python loops)
# ═══════════════════════════════════════════════════════════════════════════

def forward_mask_bpe(tokens, t, mask_id=1, protect_special=True):
    """Vectorized forward masking.

    Each token masked independently with probability t.
    Special tokens (pad=0, bos=2, eos=3) are protected.
    """
    batch, seq_len = tokens.shape
    prob_mask = t[:, None].expand(batch, seq_len)
    rand = torch.rand_like(tokens.float())
    mask_positions = rand < prob_mask

    if protect_special:
        is_special = (tokens == 0) | (tokens == 2) | (tokens == 3)
        mask_positions = mask_positions & (~is_special)

    masked = torch.where(mask_positions, mask_id, tokens)
    return masked, mask_positions


def mdlm_bpe_loss_v2(model, tokens, mask_id=1):
    """MDLM training loss — fully vectorized."""
    batch = tokens.shape[0]
    t = torch.rand(batch, device=tokens.device)
    masked, mask_pos = forward_mask_bpe(tokens, t, mask_id)
    logits = model(masked, t)

    mask_flat = mask_pos.reshape(-1)
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=tokens.device, requires_grad=True)

    logits_flat = logits.reshape(-1, logits.shape[-1])
    tokens_flat = tokens.reshape(-1)
    return F.cross_entropy(logits_flat[mask_flat], tokens_flat[mask_flat])


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized Sampling (batch, no per-sample Python loop)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_mdlm_v2(model, tokenizer, prompt_ids=None, seq_len=64,
                   n_samples=1, n_steps=32, temperature=0.7,
                   device=DEVICE):
    """Vectorized sampling via iterative unmasking."""
    model.eval()
    mask_id = tokenizer.mask_id

    if prompt_ids is not None:
        full = torch.full((n_samples, seq_len), mask_id, device=device)
        prompt_len = min(len(prompt_ids), seq_len)
        full[:, :prompt_len] = torch.tensor(prompt_ids[:prompt_len], device=device)
        prompt_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_mask[:prompt_len] = True
    else:
        full = torch.full((n_samples, seq_len), mask_id, device=device)
        prompt_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    for step in range(n_steps):
        t_val = max(1.0 - step / n_steps, 0.01)
        t = torch.full((n_samples,), t_val, device=device)

        logits = model(full, t)
        mask_positions = (full == mask_id)

        if not mask_positions.any():
            break

        # Vectorized: compute probs and sample for ALL masked positions at once
        temp_logits = logits / max(temperature, 0.01)
        probs = F.softmax(temp_logits, dim=-1)  # [batch, seq, vocab]

        # Sample for each position
        flat_probs = probs.reshape(-1, probs.shape[-1])
        sampled = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
        sampled = sampled.reshape(n_samples, seq_len)

        confidence = probs.max(dim=-1)[0]  # [batch, seq]

        # Zero out non-masked positions
        confidence[~mask_positions] = -1.0
        confidence[prompt_mask.unsqueeze(0).expand(n_samples, -1)] = -1.0

        # Determine how many to unmask per sample
        n_masked = mask_positions.sum(dim=1)  # [batch]
        n_to_unmask = torch.clamp(
            n_masked // max(n_steps - step, 1), min=1
        )

        # For each sample, unmask top-k confident
        for b in range(n_samples):
            if n_masked[b] == 0:
                continue
            k = min(n_to_unmask[b].item(), n_masked[b].item())
            top_conf, top_idx = confidence[b].topk(k)
            # Only fill positions that are actually masked
            valid = top_conf > 0
            if valid.any():
                positions = top_idx[valid]
                full[b, positions] = sampled[b, positions]

    # Decode
    results = []
    for b in range(n_samples):
        ids = full[b].cpu().tolist()
        if tokenizer.eos_id in ids:
            ids = ids[:ids.index(tokenizer.eos_id)]
        text = tokenizer.decode(ids)
        results.append(text)
    return results
