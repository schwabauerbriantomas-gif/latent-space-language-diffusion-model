"""
Evaluate MDLM-BPE v3 WITH logit guidance integrated into semi-AR sampling.

Compares:
  1. Baseline (no guidance)
  2. With guidance (repetition_penalty + no_repeat_ngram + frequency_penalty)

Measures: text quality, repetition scores, TPS throughput.
"""
import json
import sys
import time
import math
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = REPO / "checkpoints"

from mdlm_bpe_v3 import (
    MDLMConfig, MDLMBPEV3, BPETokenizer,
)
from hrm_refiner import RepetitionReviewer


# ═══════════════════════════════════════════════════════════════════
# LOGIT GUIDANCE FUNCTIONS (adapted for v3 — operate on [B,S,V] logits)
# ═══════════════════════════════════════════════════════════════════

def apply_frequency_penalty(logits, tokens, penalty=0.3, special_ids=None):
    """OpenAI-style frequency penalty — VECTORIZED."""
    if penalty == 0.0:
        return logits
    batch, seq, vocab = logits.shape
    if special_ids is None:
        special_ids = {0, 1, 2, 3}
    counts = torch.zeros(batch, vocab, device=tokens.device)
    counts.scatter_add_(1, tokens, torch.ones_like(tokens, dtype=torch.float))
    special_mask = torch.zeros(vocab, device=tokens.device, dtype=torch.bool)
    for sid in special_ids:
        if sid < vocab:
            special_mask[sid] = True
    counts[:, special_mask] = 0
    penalty_vals = penalty * torch.sqrt(counts.float())
    return logits - penalty_vals.unsqueeze(1)


def apply_repetition_penalty(logits, tokens, penalty=1.3, special_ids=None):
    """Keskar et al. repetition penalty — VECTORIZED."""
    if penalty == 1.0:
        return logits
    batch, seq, vocab = logits.shape
    if special_ids is None:
        special_ids = {0, 1, 2, 3}
    used_mask = torch.zeros(batch, vocab, dtype=torch.bool, device=tokens.device)
    used_mask.scatter_(1, tokens, True)
    for sid in special_ids:
        if sid < vocab:
            used_mask[:, sid] = False
    penalty_factor = torch.where(
        used_mask, torch.tensor(1.0 / penalty, device=logits.device),
        torch.tensor(1.0, device=logits.device),
    )
    return logits * penalty_factor.unsqueeze(1)


def apply_no_repeat_ngram(logits, tokens, n=2, special_ids=None):
    """Ban tokens that would complete an already-seen n-gram."""
    batch, seq_len_dim, vocab = logits.shape
    if special_ids is None:
        special_ids = {0, 1, 2, 3}
    if seq_len_dim < n:
        return logits

    ban_mask = torch.zeros_like(logits, dtype=torch.bool)
    special_tensor = torch.tensor(list(special_ids), device=tokens.device)

    for b in range(batch):
        seq_t = tokens[b]
        if n == 2:
            prefixes = seq_t[:-1]
            nexts = seq_t[1:]
            valid = ~torch.isin(prefixes, special_tensor) & ~torch.isin(nexts, special_tensor)
            for pos in range(1, seq_len_dim):
                cur_prefix = seq_t[pos - 1]
                if cur_prefix.item() in special_ids:
                    continue
                match = (prefixes == cur_prefix) & valid
                if match.any():
                    banned = nexts[match].unique()
                    ban_mask[b, pos, banned] = True
        else:
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

    return logits.masked_fill(ban_mask, float('-inf'))


# ═══════════════════════════════════════════════════════════════════
# GUIDED SEMI-AR SAMPLING
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_semi_ar_guided(
    model, tokenizer, prompt_ids=None, seq_len=128,
    n_samples=1, block_size=4, temperature=0.7,
    repetition_penalty=1.3, no_repeat_ngram=2, frequency_penalty=0.4,
    device=DEVICE,
):
    """Semi-AR sampling with logit guidance integrated."""
    model.eval()
    mask_id = tokenizer.mask_id
    pad_id = tokenizer.pad_id

    special_ids = {pad_id, mask_id, tokenizer.bos_id, tokenizer.eos_id}
    for name in ["<|user|>", "<|assistant|>", "<|system|>",
                 "<|think|>", "<|/think|>"]:
        tid = tokenizer.tokenizer.token_to_id(name)
        if tid is not None:
            special_ids.add(tid)

    if prompt_ids is not None:
        full = torch.full((n_samples, seq_len), mask_id, device=device)
        prompt_len = min(len(prompt_ids), seq_len)
        full[:, :prompt_len] = torch.tensor(prompt_ids[:prompt_len], device=device)
    else:
        full = torch.full((n_samples, seq_len), mask_id, device=device)
        prompt_len = 0

    n_steps_per_block = max(2, block_size)

    for block_start in range(prompt_len, seq_len, block_size):
        block_end = min(block_start + block_size, seq_len)

        for step in range(n_steps_per_block):
            t_val = max(0.5 - step / (n_steps_per_block * 2), 0.01)
            t = torch.full((n_samples,), t_val, device=device)

            logits = model(full, t)

            # === APPLY GUIDANCE ===
            if frequency_penalty > 0:
                logits = apply_frequency_penalty(
                    logits, full, penalty=frequency_penalty,
                    special_ids=special_ids,
                )
            if repetition_penalty > 1.0:
                logits = apply_repetition_penalty(
                    logits, full, penalty=repetition_penalty,
                    special_ids=special_ids,
                )
            if no_repeat_ngram > 0:
                logits = apply_no_repeat_ngram(
                    logits, full, n=no_repeat_ngram,
                    special_ids=special_ids,
                )

            mask_in_block = (full[:, block_start:block_end] == mask_id)
            if not mask_in_block.any():
                break

            pos_logits = logits[:, block_start:block_end] / max(temperature, 0.01)
            probs = F.softmax(pos_logits, dim=-1)
            probs = probs.clamp(min=0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            sampled = torch.multinomial(
                probs.reshape(-1, probs.shape[-1]), 1
            ).squeeze(-1).reshape(n_samples, -1)

            confidence = probs.max(dim=-1)[0]
            confidence[~mask_in_block] = -1

            n_masked = mask_in_block.sum(dim=1)
            n_to_unmask = torch.clamp(n_masked // max(n_steps_per_block - step, 1), min=1)

            for b in range(n_samples):
                if n_masked[b] == 0:
                    continue
                k = min(int(n_to_unmask[b].item()), int(n_masked[b].item()))
                top_conf, top_idx = confidence[b].topk(k)
                valid = top_conf > 0
                if valid.any():
                    positions = top_idx[valid]
                    full[b, block_start + positions] = sampled[b, positions]

    results = []
    for b in range(n_samples):
        ids = full[b].cpu().tolist()
        if tokenizer.eos_id in ids:
            ids = ids[:ids.index(tokenizer.eos_id)]
        text = tokenizer.decode(ids)
        results.append(text)
    return results


@torch.no_grad()
def generate_response_guided_v3(
    model, tokenizer, prompt, max_len=64,
    block_size=4, temperature=0.6,
    repetition_penalty=1.3, no_repeat_ngram=2, frequency_penalty=0.4,
    device=DEVICE,
):
    """Chatbot response with semi-AR + logit guidance."""
    model.eval()

    user_tok = tokenizer.tokenizer.token_to_id("<|user|>")
    asst_tok = tokenizer.tokenizer.token_to_id("<|assistant|>")
    mask_id = tokenizer.mask_id
    pad_id = tokenizer.pad_id

    special_ids = {pad_id, mask_id, tokenizer.bos_id, tokenizer.eos_id,
                   user_tok, asst_tok}
    for name in ["<|user|>", "<|assistant|>", "<|system|>",
                 "<|think|>", "<|/think|>"]:
        tid = tokenizer.tokenizer.token_to_id(name)
        if tid is not None:
            special_ids.add(tid)

    ctx_ids = tokenizer.tokenizer.encode(prompt).ids
    prefix = [tokenizer.bos_id, user_tok] + ctx_ids + [asst_tok]
    response_start = len(prefix)

    seq_len = min(128, response_start + max_len)
    seq = (prefix + [mask_id] * max_len)[:seq_len]
    while len(seq) < seq_len:
        seq.append(pad_id)

    full = torch.tensor([seq], device=device)
    n_steps = max(2, block_size)

    for block_start in range(response_start, seq_len, block_size):
        block_end = min(block_start + block_size, seq_len)

        for step in range(n_steps):
            t_val = max(0.5 - step / (n_steps * 2), 0.01)
            t = torch.full((1,), t_val, device=device)

            logits = model(full, t)

            # === APPLY GUIDANCE ===
            if frequency_penalty > 0:
                logits = apply_frequency_penalty(
                    logits, full, penalty=frequency_penalty,
                    special_ids=special_ids,
                )
            if repetition_penalty > 1.0:
                logits = apply_repetition_penalty(
                    logits, full, penalty=repetition_penalty,
                    special_ids=special_ids,
                )
            if no_repeat_ngram > 0:
                logits = apply_no_repeat_ngram(
                    logits, full, n=no_repeat_ngram,
                    special_ids=special_ids,
                )

            mask_in_block = (full[0, block_start:block_end] == mask_id)
            if not mask_in_block.any():
                break

            idxs = mask_in_block.nonzero(as_tuple=True)[0]
            pos_logits = logits[0, block_start:block_end][idxs] / max(temperature, 0.01)
            probs = F.softmax(pos_logits, dim=-1)
            probs = probs.clamp(min=0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            sampled = torch.multinomial(probs, 1).squeeze(-1)
            conf = probs.max(dim=-1)[0]

            n_unmask = max(1, len(idxs) // (n_steps - step))
            top_conf, top_idx = conf.topk(min(n_unmask, len(idxs)))
            positions_in_block = idxs[top_idx]
            full[0, block_start + positions_in_block] = sampled[top_idx]

    resp_ids = full[0, response_start:].cpu().tolist()
    if tokenizer.eos_id in resp_ids:
        resp_ids = resp_ids[:resp_ids.index(tokenizer.eos_id)]
    resp_ids = [i for i in resp_ids if i not in (pad_id, mask_id)]
    return tokenizer.decode(resp_ids)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def load_model():
    ckpt_path = CHECKPOINT_DIR / "mdlm_bpe_v3_best.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "mdlm_bpe_v3_final.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = MDLMConfig(**ckpt["config"])
    model = MDLMBPEV3(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokenizer = BPETokenizer()
    print(f"Loaded: {ckpt_path.name} (step {ckpt.get('step', '?')}, "
          f"PPL={ckpt.get('ppl', '?'):.1f})")
    return model, tokenizer, config


def main():
    print("=" * 70)
    print("MDLM-BPE v3 — GUIDED EVALUATION")
    print("=" * 70)

    model, tokenizer, config = load_model()
    reviewer = RepetitionReviewer()

    gpu = torch.cuda.get_device_name(0)
    params = sum(p.numel() for p in model.parameters())
    print(f"  GPU: {gpu}")
    print(f"  Params: {params:,} ({params/1e6:.1f}M)")

    prompts = [
        "The future of artificial intelligence",
        "To build a reliable system, you need",
        "Climate change is one of the biggest challenges",
        "The key to success in any project is",
    ]

    # ── BASELINE (no guidance) ──
    print("\n" + "─" * 70)
    print("BASELINE (no guidance)")
    print("─" * 70)
    baseline_scores = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special=False)
        result = sample_semi_ar_guided(
            model, tokenizer, prompt_ids=ids, seq_len=64,
            n_samples=1, block_size=4, temperature=0.7,
            repetition_penalty=1.0, no_repeat_ngram=0, frequency_penalty=0.0,
        )
        text = result[0].strip()[:150]
        ids_out = tokenizer.encode(result[0], add_special=False)
        if ids_out:
            t = torch.tensor([ids_out[:64]], device=DEVICE)
            if t.shape[1] < 64:
                t = torch.cat([t, torch.full((1, 64-t.shape[1]), tokenizer.pad_id, device=DEVICE)], 1)
            score = reviewer.score_sequence(t[0])
        else:
            score = 1.0
        baseline_scores.append(score)
        print(f"  [{score:.2f}] {prompt}")
        print(f"         → {text}")
        print()

    # ── GUIDED ──
    print("\n" + "─" * 70)
    print("GUIDED (rep_penalty=1.3, no_repeat_bigram, freq_penalty=0.4)")
    print("─" * 70)
    guided_scores = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special=False)
        result = sample_semi_ar_guided(
            model, tokenizer, prompt_ids=ids, seq_len=64,
            n_samples=1, block_size=4, temperature=0.7,
            repetition_penalty=1.3, no_repeat_ngram=2, frequency_penalty=0.4,
        )
        text = result[0].strip()[:150]
        ids_out = tokenizer.encode(result[0], add_special=False)
        if ids_out:
            t = torch.tensor([ids_out[:64]], device=DEVICE)
            if t.shape[1] < 64:
                t = torch.cat([t, torch.full((1, 64-t.shape[1]), tokenizer.pad_id, device=DEVICE)], 1)
            score = reviewer.score_sequence(t[0])
        else:
            score = 1.0
        guided_scores.append(score)
        print(f"  [{score:.2f}] {prompt}")
        print(f"         → {text}")
        print()

    # ── CHATBOT (guided) ──
    print("\n" + "─" * 70)
    print("CHATBOT RESPONSES (guided)")
    print("─" * 70)
    chat_prompts = [
        "What is machine learning?",
        "How do I write a Python function?",
        "Explain neural networks simply",
        "What is Python?",
        "Tell me about the solar system",
    ]
    chat_scores = []
    for prompt in chat_prompts:
        resp = generate_response_guided_v3(
            model, tokenizer, prompt, max_len=64, block_size=4,
            temperature=0.6, repetition_penalty=1.3,
            no_repeat_ngram=2, frequency_penalty=0.4,
        )
        text = resp.strip()[:200]
        ids_out = tokenizer.encode(resp, add_special=False)
        if ids_out:
            t = torch.tensor([ids_out[:64]], device=DEVICE)
            if t.shape[1] < 64:
                t = torch.cat([t, torch.full((1, 64-t.shape[1]), tokenizer.pad_id, device=DEVICE)], 1)
            score = reviewer.score_sequence(t[0])
        else:
            score = 1.0
        chat_scores.append(score)
        print(f"  Q: {prompt}")
        print(f"  A [{score:.2f}]: {text}")
        print()

    # ── THROUGHPUT ──
    print("\n" + "─" * 70)
    print("THROUGHPUT (forward pass)")
    print("─" * 70)
    for bs in [1, 10, 50]:
        tokens = torch.full((bs, 128), tokenizer.mask_id, device=DEVICE)
        t = torch.full((bs,), 0.5, device=DEVICE)
        for _ in range(3):
            with torch.no_grad():
                _ = model(tokens, t)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = model(tokens, t)
        torch.cuda.synchronize()
        tps = bs * 128 * 10 / (time.time() - start)
        print(f"  Batch {bs:3d}: {tps:>10,.0f} TPS")

    # ── GUIDED GENERATION LATENCY ──
    print("\n── Guided generation latency (seq_len=64, block=4) ──")
    ids = tokenizer.encode(prompts[0], add_special=False)
    for _ in range(2):
        _ = sample_semi_ar_guided(
            model, tokenizer, prompt_ids=ids, seq_len=64,
            n_samples=1, block_size=4, temperature=0.7,
        )
    start = time.time()
    for _ in range(5):
        _ = sample_semi_ar_guided(
            model, tokenizer, prompt_ids=ids, seq_len=64,
            n_samples=1, block_size=4, temperature=0.7,
        )
    elapsed = time.time() - start
    gen_tps = 5 * 64 / elapsed
    print(f"  Generation: {gen_tps:>10,.1f} tok/s (incl. guidance overhead)")
    print(f"  Latency:    {elapsed/5*1000:.0f} ms per 64-token sequence")

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline avg repetition score: {np.mean(baseline_scores):.2f}")
    print(f"  Guided avg repetition score:   {np.mean(guided_scores):.2f}")
    print(f"  Chatbot avg repetition score:   {np.mean(chat_scores):.2f}")
    print(f"  Improvement (baseline→guided):  {np.mean(guided_scores) - np.mean(baseline_scores):+.2f}")
    print(f"  Model: {params/1e6:.1f}M params, PPL=102.6")


if __name__ == "__main__":
    main()
