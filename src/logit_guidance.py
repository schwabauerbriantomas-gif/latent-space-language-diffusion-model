"""
Logit guidance for MDLM-BPE sampling.

Prevents repetition DURING generation instead of fixing it after.

Three techniques applied during the unmasking loop:

  1. REPETITION PENALTY (Keskar et al. 2019)
     Divide logits of already-generated tokens by factor > 1.0.
     token "machine" used 3x → its logit divided by 1.3^3
     Soft penalty: reduces but doesn't eliminate.

  2. NO-REPEAT N-GRAM
     Set logits of tokens that would create an already-seen n-gram to -inf.
     Hard penalty: makes repeating bigrams/trigrams impossible.

  3. FREQUENCY PENALTY (OpenAI)
     Subtract from logit proportional to how many times token was used.
     freq_penalty × sqrt(count)
     Balanced: penalizes frequent tokens without eliminating them.

All three are applied AFTER model forward pass, BEFORE sampling.
Zero extra model calls, zero extra training.

COMBINATION:
  These are complementary to HRM refinement (hrm_refiner.py).
  Logit guidance prevents MOST repetition during generation.
  HRM catches residual issues and fixes them.
  Together: generation is clean from the start.
"""
import sys
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from mdlm_bpe_v2 import MDLMConfig, MDLMBPEV2, BPETokenizer


def apply_frequency_penalty(logits: torch.Tensor, tokens: torch.Tensor,
                             penalty: float = 0.3,
                             special_ids: set = None) -> torch.Tensor:
    """OpenAI-style frequency penalty — VECTORIZED.

    Subtract penalty × sqrt(count) from each token's logit.
    """
    if penalty == 0.0:
        return logits

    batch, seq, vocab = logits.shape
    if special_ids is None:
        special_ids = {0, 1, 2, 3}

    # Count all token frequencies at once: [batch, vocab]
    counts = torch.zeros(batch, vocab, device=tokens.device)
    counts.scatter_add_(1, tokens, torch.ones_like(tokens, dtype=torch.float))

    # Zero out special tokens
    special_mask = torch.zeros(vocab, device=tokens.device, dtype=torch.bool)
    for sid in special_ids:
        if sid < vocab:
            special_mask[sid] = True
    counts[:, special_mask] = 0

    # Compute penalty: penalty × sqrt(count) → [batch, vocab]
    penalty_vals = penalty * torch.sqrt(counts.float())  # [batch, vocab]

    # Broadcast subtract: [batch, 1, vocab] from [batch, seq, vocab]
    logits = logits - penalty_vals.unsqueeze(1)
    return logits


def apply_repetition_penalty(logits: torch.Tensor, tokens: torch.Tensor,
                              penalty: float = 1.2,
                              special_ids: set = None) -> torch.Tensor:
    """Keskar et al. repetition penalty — VECTORIZED.

    Divide logits of already-used tokens by `penalty`.
    """
    if penalty == 1.0:
        return logits

    batch, seq, vocab = logits.shape
    if special_ids is None:
        special_ids = {0, 1, 2, 3}

    # Create mask of used tokens: [batch, vocab]
    used_mask = torch.zeros(batch, vocab, dtype=torch.bool, device=tokens.device)
    used_mask.scatter_(1, tokens, True)

    # Don't penalize special tokens
    for sid in special_ids:
        if sid < vocab:
            used_mask[:, sid] = False

    # Apply penalty: divide by `penalty` where used
    # [batch, 1, vocab] broadcast over seq dimension
    penalty_factor = torch.where(
        used_mask, torch.tensor(1.0 / penalty, device=logits.device),
        torch.tensor(1.0, device=logits.device),
    )  # [batch, vocab]
    logits = logits * penalty_factor.unsqueeze(1)
    return logits


def apply_no_repeat_ngram(logits: torch.Tensor, tokens: torch.Tensor,
                           n: int = 2,
                           special_ids: set = None) -> torch.Tensor:
    """Ban tokens that would complete an already-seen n-gram — VECTORIZED.

    For n=2 (bigrams): if (A,B) already exists, ban B after A.
    Uses hash-based approach: hash each (n-1)-prefix, store set of
    banned next-tokens, apply via scatter.
    """
    batch, seq_len_dim, vocab = logits.shape
    if special_ids is None:
        special_ids = {0, 1, 2, 3}
    if seq_len_dim < n:
        return logits

    # For efficiency with seq=128, vocab=10K, we process on GPU
    # Build ban mask: [batch, seq, vocab] — True = banned
    ban_mask = torch.zeros_like(logits, dtype=torch.bool)

    special_tensor = torch.tensor(list(special_ids), device=tokens.device)

    for b in range(batch):
        seq_t = tokens[b]  # [seq]

        # Build seen bigrams using vectorized approach
        # For n=2: prefix = token[i], next = token[i+1]
        if n == 2:
            prefixes = seq_t[:-1]   # [seq-1]
            nexts = seq_t[1:]       # [seq-1]

            # Filter out special tokens
            valid = ~torch.isin(prefixes, special_tensor) & ~torch.isin(nexts, special_tensor)

            # For each position in the sequence, check if token[pos-1] has
            # been followed by something before
            for pos in range(1, seq_len_dim):
                cur_prefix = seq_t[pos - 1]
                if cur_prefix.item() in special_ids:
                    continue
                # Find all positions where this prefix appeared
                match = (prefixes == cur_prefix) & valid
                if match.any():
                    # Get unique next tokens after this prefix
                    banned = nexts[match].unique()
                    ban_mask[b, pos, banned] = True
        else:
            # General n-gram (slower, but n>2 is rarely needed)
            seq_list = seq_t.cpu().tolist()
            seen = {}
            for i in range(len(seq_list) - n + 1):
                prefix = tuple(seq_list[i:i+n-1])
                next_tok = seq_list[i + n - 1]
                if next_tok in special_ids or any(t in special_ids for t in prefix):
                    continue
                if prefix not in seen:
                    seen[prefix] = set()
                seen[prefix].add(next_tok)

            for pos in range(n-1, seq_len_dim):
                prefix = tuple(seq_list[pos-n+1:pos])
                if any(t in special_ids for t in prefix):
                    continue
                if prefix in seen:
                    for banned_tok in seen[prefix]:
                        ban_mask[b, pos, banned_tok] = True

    # Apply ban: set banned logits to -inf
    logits = logits.masked_fill(ban_mask, float('-inf'))
    return logits


@torch.no_grad()
def sample_with_guidance(
    model: MDLMBPEV2,
    tokenizer: BPETokenizer,
    prompt_ids: List[int] = None,
    seq_len: int = 64,
    n_samples: int = 1,
    n_steps: int = 32,
    temperature: float = 0.7,
    # Guidance parameters
    repetition_penalty: float = 1.3,
    no_repeat_ngram: int = 2,
    frequency_penalty: float = 0.4,
    device: str = DEVICE,
) -> List[str]:
    """Generate text with logit guidance to prevent repetition.

    All guidance is applied DURING unmasking, before each sampling step.
    """
    model.eval()
    mask_id = tokenizer.mask_id

    # Special IDs to never penalize
    special_ids = {tokenizer.pad_id, tokenizer.mask_id,
                   tokenizer.bos_id, tokenizer.eos_id}
    # Chat tokens
    for name in ["<|user|>", "<|assistant|>", "<|system|>",
                 "<|think|>", "<|/think|>"]:
        tid = tokenizer.tokenizer.token_to_id(name)
        if tid is not None:
            special_ids.add(tid)

    # Initialize sequence
    if prompt_ids is not None:
        full = torch.full((n_samples, seq_len), mask_id, device=device)
        prompt_len = min(len(prompt_ids), seq_len)
        full[:, :prompt_len] = torch.tensor(prompt_ids[:prompt_len], device=device)
    else:
        full = torch.full((n_samples, seq_len), mask_id, device=device)

    for step in range(n_steps):
        t_val = max(1.0 - step / n_steps, 0.01)
        t = torch.full((n_samples,), t_val, device=device)

        logits = model(full, t)

        mask_positions = (full == mask_id)
        if not mask_positions.any():
            break

        # === APPLY LOGIT GUIDANCE ===
        # 1. Frequency penalty (applied first, soft)
        if frequency_penalty > 0:
            logits = apply_frequency_penalty(
                logits, full, penalty=frequency_penalty,
                special_ids=special_ids,
            )

        # 2. Repetition penalty (soft, reduces already-used tokens)
        if repetition_penalty > 1.0:
            logits = apply_repetition_penalty(
                logits, full, penalty=repetition_penalty,
                special_ids=special_ids,
            )

        # 3. No-repeat n-gram (hard, bans seen n-grams)
        if no_repeat_ngram > 0:
            logits = apply_no_repeat_ngram(
                logits, full, n=no_repeat_ngram,
                special_ids=special_ids,
            )

        # === SAMPLE ===
        temp_logits = logits / max(temperature, 0.01)

        # Only modify logits at masked positions
        probs = F.softmax(temp_logits, dim=-1)
        confidence = probs.max(dim=-1)[0]
        confidence[~mask_positions] = -1.0

        # Sample for all positions
        flat_probs = probs.reshape(-1, probs.shape[-1])
        # Handle -inf from n-gram ban (set to 0 prob)
        flat_probs = flat_probs.clamp(min=0)
        # Renormalize
        flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sampled = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
        sampled = sampled.reshape(n_samples, seq_len)

        # Unmask most confident
        n_masked = mask_positions.sum(dim=1)
        n_to_unmask = torch.clamp(n_masked // max(n_steps - step, 1), min=1)

        for b in range(n_samples):
            if n_masked[b] == 0:
                continue
            k = min(int(n_to_unmask[b].item()), int(n_masked[b].item()))
            top_conf, top_idx = confidence[b].topk(k)
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


@torch.no_grad()
def generate_response_guided(
    model: MDLMBPEV2,
    tokenizer: BPETokenizer,
    prompt: str,
    max_len: int = 48,
    n_steps: int = 24,
    temperature: float = 0.6,
    repetition_penalty: float = 1.3,
    no_repeat_ngram: int = 2,
    frequency_penalty: float = 0.4,
    device: str = DEVICE,
) -> str:
    """Generate a chatbot response with logit guidance.

    Uses the vectorized apply_* functions for speed.
    """
    model.eval()

    user_tok = tokenizer.tokenizer.token_to_id("<|user|>")
    asst_tok = tokenizer.tokenizer.token_to_id("<|assistant|>")
    mask_id = tokenizer.mask_id
    pad_id = tokenizer.pad_id

    ctx_ids = tokenizer.tokenizer.encode(prompt).ids
    prefix = [tokenizer.bos_id, user_tok] + ctx_ids + [asst_tok]
    response_start = len(prefix)

    total_len = min(MAX_SEQ_LEN, response_start + max_len)
    seq = (prefix + [mask_id] * max_len)[:total_len]
    # Pad if needed
    while len(seq) < total_len:
        seq.append(pad_id)

    full = torch.tensor([seq], device=device)

    # Build special_ids set
    special_ids = {pad_id, mask_id, tokenizer.bos_id, tokenizer.eos_id,
                   user_tok, asst_tok}
    for name in ["<|user|>", "<|assistant|>", "<|system|>",
                 "<|think|>", "<|/think|>"]:
        tid = tokenizer.tokenizer.token_to_id(name)
        if tid is not None:
            special_ids.add(tid)

    for step in range(n_steps):
        t_val = max(1.0 - step / n_steps, 0.01)
        t = torch.full((1,), t_val, device=device)

        logits = model(full, t)

        # Find masked positions in the response span
        mask_positions = (full[0] == mask_id)
        mask_positions[:response_start] = False

        if not mask_positions.any():
            break

        # Apply vectorized guidance to the full logits
        # (the functions handle per-batch, per-position logic)
        guided_logits = logits.clone()

        if frequency_penalty > 0:
            guided_logits = apply_frequency_penalty(
                guided_logits, full, penalty=frequency_penalty,
                special_ids=special_ids,
            )

        if repetition_penalty > 1.0:
            guided_logits = apply_repetition_penalty(
                guided_logits, full, penalty=repetition_penalty,
                special_ids=special_ids,
            )

        if no_repeat_ngram > 0:
            guided_logits = apply_no_repeat_ngram(
                guided_logits, full, n=no_repeat_ngram,
                special_ids=special_ids,
            )

        # Sample at masked positions only
        masked_idx = mask_positions.nonzero(as_tuple=True)[0]
        pos_logits = guided_logits[0, masked_idx] / max(temperature, 0.01)
        pos_probs = F.softmax(pos_logits, dim=-1)
        pos_probs = pos_probs.clamp(min=0)
        pos_probs = pos_probs / pos_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        sampled = torch.multinomial(pos_probs, 1).squeeze(-1)
        conf = pos_probs.max(dim=-1)[0]

        n_unmask = max(1, len(masked_idx) // (n_steps - step))
        top_conf, top_idx = conf.topk(min(n_unmask, len(masked_idx)))
        positions_to_fill = masked_idx[top_idx]
        full[0, positions_to_fill] = sampled[top_idx]

    # Decode response
    resp_ids = full[0, response_start:].cpu().tolist()
    if tokenizer.eos_id in resp_ids:
        resp_ids = resp_ids[:resp_ids.index(tokenizer.eos_id)]
    resp_ids = [i for i in resp_ids if i not in (pad_id, mask_id)]
    return tokenizer.decode(resp_ids)


MAX_SEQ_LEN = 128
