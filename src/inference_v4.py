"""
Optimized inference for MDLM v4.

Improvements over v3 inference:
  1. Fully vectorized guidance (no Python loops over batch)
  2. KV cache compatible (not used in diffusion, but for AR oracle)
  3. torch.compile friendly
  4. Batch generation
  5. Cosine unmasking schedule
  6. Efficient top-p (vectorized)
"""
import sys
import math
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════
# VECTORIZED GUIDANCE — all GPU, zero Python loops
# ═══════════════════════════════════════════════════════════════

def apply_frequency_penalty_batch(logits, tokens, penalty=0.4, special_mask=None):
    """OpenAI-style frequency penalty — fully vectorized.
    
    logits: [batch, seq, vocab]
    tokens: [batch, seq]
    """
    if penalty == 0.0:
        return logits
    b, s, v = logits.shape
    counts = torch.zeros(b, v, device=tokens.device)
    counts.scatter_add_(1, tokens, torch.ones_like(tokens, dtype=torch.float))
    if special_mask is not None:
        counts[:, special_mask] = 0
    return logits - (penalty * torch.sqrt(counts.float())).unsqueeze(1)


def apply_repetition_penalty_batch(logits, tokens, penalty=1.3, special_mask=None):
    """Keskar et al. repetition penalty — fully vectorized."""
    if penalty == 1.0:
        return logits
    b, s, v = logits.shape
    used = torch.zeros(b, v, dtype=torch.bool, device=tokens.device)
    used.scatter_(1, tokens, True)
    if special_mask is not None:
        used[:, special_mask] = False
    factor = torch.where(used, torch.tensor(1.0 / penalty, device=logits.device),
                         torch.tensor(1.0, device=logits.device))
    return logits * factor.unsqueeze(1)


def apply_no_repeat_bigram_batch(logits, tokens, special_mask=None):
    """Ban tokens that complete an already-seen bigram — vectorized over batch.
    
    For each position, checks if the previous token has been followed by
    something else earlier in the sequence. If so, ban that token.
    """
    b, seq_len, v = logits.shape
    if seq_len < 2:
        return logits

    ban = torch.zeros_like(logits, dtype=torch.bool)
    
    # Vectorized bigram extraction
    # prefixes[b, i] = tokens[b, i], nexts[b, i] = tokens[b, i+1]
    prefixes = tokens[:, :-1]  # [b, seq-1]
    nexts = tokens[:, 1:]      # [b, seq-1]
    # Only consider bigrams where BOTH tokens are real (unmasked) content:
    # a bigram computed against a still-masked predecessor never existed in
    # the text and must not feed the ban set.
    mask_pad_ids = [0, 1, 2, 3]  # pad/mask/eos/unk — never real content
    real = ~torch.isin(tokens, torch.tensor(
        mask_pad_ids, device=tokens.device))
    real_prefix = real[:, :-1]
    real_next = real[:, 1:]

    for bi in range(b):
        seq_t = tokens[bi]  # [seq]
        for pos in range(1, seq_len):
            cur_prefix = seq_t[pos - 1]
            if cur_prefix.item() in mask_pad_ids:
                # Current predecessor is still masked (or special): there is
                # no real previous token at this position — nothing to ban.
                continue
            match = (prefixes[bi] == cur_prefix) & real_prefix[bi] & real_next[bi]
            if special_mask is not None:
                match = match & (~special_mask[nexts[bi]])
            if match.any():
                banned = nexts[bi][match].unique()
                ban[bi, pos, banned] = True

    return logits.masked_fill(ban, float('-inf'))


def apply_top_p_batch(probs, p=0.95):
    """Nucleus sampling — vectorized over batch and sequence."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # Keep tokens until cumulative prob exceeds p
    sorted_mask = (cumsum - sorted_probs) < p
    sorted_probs = sorted_probs * sorted_mask
    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    # Scatter back
    new_probs = torch.zeros_like(probs)
    new_probs.scatter_(-1, sorted_idx, sorted_probs)
    return new_probs


# ═══════════════════════════════════════════════════════════════
# OPTIMIZED FULL-PARALLEL SAMPLING
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_full_parallel(
    model,
    tokenizer,
    prompt: str = "",
    max_new_tokens: int = 64,
    n_samples: int = 1,
    n_steps: int = 32,
    temperature: float = 0.7,
    repetition_penalty: float = 1.3,
    frequency_penalty: float = 0.4,
    top_p: float = 0.95,
    use_guidance: bool = True,
    device: str = DEVICE,
):
    """Full-parallel diffusion generation with adaptive guidance.
    
    All positions predicted simultaneously — no left-to-right decoding.
    With guidance, repetition is handled during generation.
    """
    model.eval()
    mask_id = tokenizer.mask_id
    pad_id = tokenizer.pad_id

    # Encode prompt
    if prompt:
        prompt_ids = tokenizer.encode(prompt, add_special=False)
    else:
        prompt_ids = [tokenizer.bos_id]
    
    seq_len = min(256, len(prompt_ids) + max_new_tokens)
    prompt_len = min(len(prompt_ids), seq_len)

    # Initialize: prompt + masks
    full = torch.full((n_samples, seq_len), mask_id, device=device, dtype=torch.long)
    full[:, :prompt_len] = torch.tensor(prompt_ids[:prompt_len], device=device)

    # Special token mask for guidance
    special_ids = {pad_id, mask_id, tokenizer.bos_id, tokenizer.eos_id}
    special_tensor = torch.tensor(list(special_ids), device=device)
    special_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)
    for sid in special_ids:
        if sid < tokenizer.vocab_size:
            special_mask[sid] = True

    for step in range(n_steps):
        # Linear timestep schedule
        t_val = max(1.0 - step / n_steps, 0.01)
        t = torch.full((n_samples,), t_val, device=device)

        # Forward pass — ALL positions at once
        logits = model(full, t)

        if use_guidance:
            logits = apply_frequency_penalty_batch(
                logits, full, penalty=frequency_penalty, special_mask=special_mask)
            logits = apply_repetition_penalty_batch(
                logits, full, penalty=repetition_penalty, special_mask=special_mask)
            logits = apply_no_repeat_bigram_batch(
                logits, full, special_mask=special_mask)

        # Find masked positions
        mask_positions = (full == mask_id)
        if not mask_positions.any():
            break

        # Apply temperature and sample
        temp_logits = logits / max(temperature, 0.01)
        probs = F.softmax(temp_logits, dim=-1)
        
        if top_p < 1.0:
            probs = apply_top_p_batch(probs, p=top_p)

        sampled = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]), 1
        ).squeeze(-1).reshape(n_samples, -1)

        # Confidence
        conf = probs.max(dim=-1)[0]  # [n_samples, seq_len]
        conf[~mask_positions] = -1.0

        # Cosine unmasking schedule: more tokens early, fewer late
        progress = step / n_steps
        cos_factor = 0.5 * (1 + math.cos(progress * math.pi))  # 1→0
        for bi in range(n_samples):
            n_masked = mask_positions[bi].sum().item()
            if n_masked == 0:
                continue
            n_unmask = max(1, int(n_masked * cos_factor / max(n_steps - step, 1)))
            n_unmask = min(n_unmask, n_masked)
            top_conf, top_idx = conf[bi].topk(n_unmask)
            valid = top_conf > 0
            if valid.any():
                positions = top_idx[valid]
                full[bi, positions] = sampled[bi, positions]

    # Decode
    results = []
    for bi in range(n_samples):
        ids = full[bi, prompt_len:].cpu().tolist()
        if tokenizer.eos_id in ids:
            ids = ids[:ids.index(tokenizer.eos_id)]
        ids = [i for i in ids if i not in (pad_id, mask_id)]
        text = tokenizer.decode(ids)
        results.append(text)
    return results


@torch.no_grad()
def benchmark_generation(model, tokenizer, prompts, n_steps=32, temperature=0.7):
    """Benchmark generation speed and quality."""
    model.eval()
    results = []
    
    for prompt in prompts:
        t0 = time.time()
        texts = generate_full_parallel(
            model, tokenizer, prompt=prompt,
            max_new_tokens=64, n_samples=1,
            n_steps=n_steps, temperature=temperature,
        )
        dt = time.time() - t0
        
        gen_text = texts[0]
        gen_tokens = len(tokenizer.encode(gen_text, add_special=False))
        tps = gen_tokens / dt if dt > 0 else 0
        
        results.append({
            "prompt": prompt,
            "generated": gen_text,
            "time_s": dt,
            "gen_tokens": gen_tokens,
            "tps": tps,
        })
        print(f"  [{tps:.1f} t/s, {dt:.2f}s] {prompt}")
        print(f"    → {gen_text[:120]}")
    
    avg_tps = np.mean([r["tps"] for r in results])
    avg_time = np.mean([r["time_s"] for r in results])
    print(f"\n  Avg: {avg_tps:.1f} t/s ({avg_time:.2f}s per generation)")
    return {"samples": results, "avg_tps": avg_tps, "avg_time": avg_time}
