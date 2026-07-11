"""
AR Control Model: Autoregressive transformer trained from scratch with
identical hyperparameters to MDLM-BPE v3, for controlled comparison.

Architecture differences from MDLM v3:
  1. Causal (masked) attention instead of bidirectional
  2. No timestep embedding, no AdaLN (standard pre-LN)
  3. Next-token prediction loss (shifted CE) instead of masked CE
  4. 15 layers (vs 10) to match ~201M params after removing AdaLN/time_embed
  5. KV cache for fast autoregressive generation

Parameter budget (identical to MDLM v3 ~201M):
  - Token embedding: 10K vocab × 1024 = 10.2M
  - 15 × Transformer blocks (no AdaLN): 15 × 12.6M = 188.9M
  - Final LN + output bias: 0.01M
  - Total: ~199.1M (within 1% of MDLM's 201.3M)
"""
import math
import sys
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = REPO / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
TOKENIZER_PATH = REPO / "tokenizer" / "bpe_tokenizer.json"


# ═══════════════════════════════════════════════════════════════════════════
# Reuse tokenizer from MDLM (identical BPE vocab)
# ═══════════════════════════════════════════════════════════════════════════

class BPETokenizer:
    """Same tokenizer as MDLM v3 — identical vocab, identical special tokens."""
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
# RoPE (identical implementation to MDLM v3)
# ═══════════════════════════════════════════════════════════════════════════

def precompute_rope(head_dim, max_seq_len, base=10000.0):
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half).float() / half))
    positions = torch.arange(max_seq_len).float()
    angles = positions[:, None] * freqs[None, :]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Apply RoPE, slicing cos/sin to match current sequence length."""
    seq_len = x.shape[2]
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated1 = x1 * cos - x2 * sin
    rotated2 = x1 * sin + x2 * cos
    out = torch.stack([rotated1, rotated2], dim=-1)
    return out.flatten(-2)


def apply_rope_offset(x, cos, sin, offset):
    """Apply RoPE with a position offset (for KV cache incremental decoding).

    When generating token at position `offset`, the RoPE angles must use
    cos[offset] / sin[offset], not cos[0] / sin[0].
    """
    seq_len = x.shape[2]
    cos = cos[offset:offset + seq_len]
    sin = sin[offset:offset + seq_len]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated1 = x1 * cos - x2 * sin
    rotated2 = x1 * sin + x2 * cos
    out = torch.stack([rotated1, rotated2], dim=-1)
    return out.flatten(-2)


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

class ARConfig:
    """Configuration for AR control model.

    Defaults match MDLM v3's d_model/n_heads but with 15 layers (vs 10)
    to compensate for the absence of AdaLN + time_embed parameters.
    """
    def __init__(self, vocab_size=10_000, d_model=1024, n_heads=16,
                 n_layers=15, max_seq_len=256, d_ff=None, dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.d_ff = d_ff or d_model * 4
        self.dropout = dropout

    def to_dict(self):
        return self.__dict__


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

class ARBlock(nn.Module):
    """Standard pre-LN transformer block with causal attention + RoPE.

    Key differences from MDLMBlockV3:
      - No AdaLN (standard LayerNorm instead of adaptive)
      - Causal attention mask (is_causal=True)
      - No timestep conditioning
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, rope_cos, rope_sin, kv_cache=None, use_cache=False,
                rope_offset=0):
        """Forward pass.

        Args:
            x: [batch, seq, d_model]
            rope_cos, rope_sin: RoPE buffers
            kv_cache: optional tuple (past_k, past_v) for incremental decoding
            use_cache: whether to return updated kv_cache
            rope_offset: position offset for RoPE when using KV cache
                         (set to past sequence length so new tokens get correct positions)
        """
        batch, seq, _ = x.shape

        # Pre-LN attention
        h = self.ln1(x)
        q = self.q_proj(h).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE with correct position offset for incremental decoding
        if kv_cache is not None:
            past_k, past_v = kv_cache
            past_len = past_k.shape[2]

            # Slice RoPE for the new tokens' positions (past_len .. past_len + seq)
            q = apply_rope_offset(q, rope_cos, rope_sin, past_len)
            k = apply_rope_offset(k, rope_cos, rope_sin, past_len)

            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
            new_cache = (k, v) if use_cache else None

        # Causal mask: only needed when seq > 1 (prefill).
        # During incremental decoding (seq=1), no mask needed.
        is_causal = (seq > 1)

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=is_causal,
        )
        attn = attn.transpose(1, 2).reshape(batch, seq, self.d_model)
        x = x + self.o_proj(attn)

        # FFN
        h = self.ln2(x)
        x = x + self.ff(h)

        if use_cache:
            return x, new_cache
        return x


class ARControlModel(nn.Module):
    """Autoregressive control model for MDLM comparison.

    ~199M params (15 layers, d_model=1024, 16 heads, 10K vocab).
    Matches MDLM v3's parameter count within 1%.

    Uses next-token prediction (shifted cross-entropy) with causal attention.
    """

    def __init__(self, config: ARConfig, pad_id=0):
        super().__init__()
        self.config = config
        self.pad_id = pad_id

        d = config.d_model
        self.token_emb = nn.Embedding(config.vocab_size, d, padding_idx=pad_id)

        self.blocks = nn.ModuleList([
            ARBlock(d, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(d)
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))

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

    def forward(self, tokens, return_hidden=False, use_cache=False, kv_caches=None):
        """Forward pass.

        Args:
            tokens: [batch, seq] token IDs
            return_hidden: if True, also return hidden states
            use_cache: whether to return KV caches
            kv_caches: list of per-layer cache tuples for incremental decoding

        Returns:
            logits [batch, seq, vocab] or (logits, hidden) or (logits, caches)
        """
        batch, seq = tokens.shape
        x = self.token_emb(tokens)

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            if use_cache:
                x, cache = block(x, self.rope_cos, self.rope_sin,
                                 kv_cache=cache, use_cache=True)
                new_caches.append(cache)
            else:
                x = block(x, self.rope_cos, self.rope_sin)

        x = self.ln_f(x)
        logits = F.linear(x, self.token_emb.weight, self.output_bias)

        if use_cache:
            return logits, new_caches
        if return_hidden:
            return logits, x
        return logits


# ═══════════════════════════════════════════════════════════════════════════
# Loss: next-token prediction (shifted CE)
# ═══════════════════════════════════════════════════════════════════════════

def ar_loss(model, tokens, pad_id=0):
    """Standard autoregressive next-token loss.

    Predict token[t+1] from token[t] using causal attention.
    Ignore padding tokens in loss.
    """
    # Input: all tokens except last; Target: all tokens except first
    inp = tokens[:, :-1]
    target = tokens[:, 1:]

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(inp)

    # Flatten and compute CE, ignoring pad
    batch, seq, vocab = logits.shape
    logits_flat = logits.reshape(-1, vocab)
    target_flat = target.reshape(-1)

    return F.cross_entropy(logits_flat, target_flat, ignore_index=pad_id)


# ═══════════════════════════════════════════════════════════════════════════
# Sampling (greedy + temperature)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_ar(model, tokenizer, prompt_ids=None, max_new_tokens=64,
              temperature=0.7, top_p=0.95, device=DEVICE):
    """Autoregressive sampling with KV cache.

    Generates one token at a time (standard AR decoding).
    """
    model.eval()
    eos_id = tokenizer.eos_id
    pad_id = tokenizer.pad_id

    if prompt_ids is None:
        tokens = [tokenizer.bos_id]
    else:
        tokens = [tokenizer.bos_id] + list(prompt_ids)

    # Prefill: process prompt in one pass
    inp = torch.tensor([tokens], device=device)
    logits, caches = model(inp, use_cache=True)
    next_logits = logits[0, -1] / max(temperature, 0.01)

    # Top-p filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_mask = cum_probs > top_p
        sorted_mask[1:] = sorted_mask[:-1].clone()
        sorted_mask[0] = False
        sorted_logits[sorted_mask] = float('-inf')
        next_logits = next_logits.scatter(0, sorted_indices, sorted_logits)

    probs = torch.softmax(next_logits, dim=-1)
    next_token = torch.multinomial(probs, 1).item()

    generated = [next_token]

    # Decode one token at a time using KV cache
    for _ in range(max_new_tokens - 1):
        if next_token == eos_id:
            break

        inp = torch.tensor([[next_token]], device=device)
        logits, caches = model(inp, use_cache=True, kv_caches=caches)
        next_logits = logits[0, -1] / max(temperature, 0.01)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_mask = cum_probs > top_p
            sorted_mask[1:] = sorted_mask[:-1].clone()
            sorted_mask[0] = False
            sorted_logits[sorted_mask] = float('-inf')
            next_logits = next_logits.scatter(0, sorted_indices, sorted_logits)

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)

    return tokenizer.decode(generated)


@torch.no_grad()
def generate_response_ar(model, tokenizer, prompt, max_new_tokens=64,
                          temperature=0.6, top_p=0.95, device=DEVICE):
    """Chatbot response via AR generation."""
    model.eval()

    user_tok = tokenizer.tokenizer.token_to_id("<|user|>")
    asst_tok = tokenizer.tokenizer.token_to_id("<|assistant|>")

    ctx_ids = tokenizer.tokenizer.encode(prompt).ids
    prefix = [tokenizer.bos_id, user_tok] + ctx_ids + [asst_tok]

    result = sample_ar(model, tokenizer, prompt_ids=ctx_ids,
                       max_new_tokens=max_new_tokens,
                       temperature=temperature, top_p=top_p, device=device)
    return result


@torch.no_grad()
def measure_perplexity(model, tokens, batch_size=32, device=DEVICE):
    """Compute perplexity on a held-out set.

    Uses next-token prediction (standard AR perplexity).
    Returns (avg_loss, perplexity).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    pad_id = model.pad_id

    n_samples = tokens.shape[0]
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = tokens[i:i+batch_size].to(device)
            inp = batch[:, :-1]
            target = batch[:, 1:]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(inp)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                ignore_index=pad_id,
                reduction='sum',
            )
            n_valid = (target != pad_id).sum().item()
            total_loss += loss.item()
            total_tokens += n_valid

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 15))
    return avg_loss, ppl
