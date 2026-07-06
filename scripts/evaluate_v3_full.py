"""
MDLM-BPE v3 — Full Pipeline: Semi-AR + Adaptive Guidance + Multi-Layer HRM

Pipeline:
  1. Semi-AR generation (left-to-right blocks)
  2. Adaptive logit guidance (cooling temp + adaptive penalties + top-p)
  3. Repetition HRM (token-level: exact repeat, window, bigram)
  4. Semantic Coherence HRM (embedding-level: drift detection + correction)

This is the most complete version of the text generation system.
"""
import sys
import math
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = REPO / "checkpoints"

from mdlm_bpe_v3 import MDLMConfig, MDLMBPEV3, BPETokenizer
from hrm_refiner import RepetitionReviewer
from semantic_hrm import SemanticCoherenceHRM, SemanticState


# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE LOGIT GUIDANCE
# ═══════════════════════════════════════════════════════════════════

def apply_frequency_penalty(logits, tokens, penalty=0.3, special_ids=None):
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
    return logits - (penalty * torch.sqrt(counts.float())).unsqueeze(1)


def apply_repetition_penalty(logits, tokens, penalty=1.3, special_ids=None):
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
    factor = torch.where(
        used_mask, torch.tensor(1.0 / penalty, device=logits.device),
        torch.tensor(1.0, device=logits.device),
    )
    return logits * factor.unsqueeze(1)


def apply_no_repeat_ngram(logits, tokens, n=2, special_ids=None):
    batch, seq_len_dim, vocab = logits.shape
    if seq_len_dim < n:
        return logits
    ban_mask = torch.zeros_like(logits, dtype=torch.bool)
    special_tensor = torch.tensor(list(special_ids), device=tokens.device) if special_ids else torch.tensor([], device=tokens.device, dtype=tokens.dtype)
    for b in range(batch):
        seq_t = tokens[b]
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
    return logits.masked_fill(ban_mask, float('-inf'))


def apply_top_p(probs, top_p=0.95):
    """Nucleus sampling: keep only tokens in top-p cumulative probability."""
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # Remove tokens above threshold
    sorted_mask = cumsum - sorted_probs > top_p
    sorted_probs[sorted_mask] = 0
    # Scatter back and renormalize
    new_probs = torch.zeros_like(probs)
    new_probs.scatter_(1, sorted_idx, sorted_probs)
    return new_probs / new_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)


# ═══════════════════════════════════════════════════════════════════
# FULL PIPELINE GENERATION
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_full_pipeline(
    model, tokenizer, prompt_ids,
    seq_len=64, block_size=4,
    # Adaptive guidance params
    temp_schedule=(1.0, 0.5),
    top_p=0.95,
    rep_schedule=(1.2, 1.5),
    freq_schedule=(0.3, 0.6),
    # HRM params
    use_repetition_hrm=True,
    use_semantic_hrm=True,
    # Semantic HRM params
    drift_threshold=0.20,
    semantic_guidance_strength=0.3,
    device=DEVICE,
):
    """Full pipeline: Semi-AR + Adaptive Guidance + Repetition HRM + Semantic HRM.

    Returns:
        tokens: [1, seq_len] generated token IDs
        stats: dict with all metrics
    """
    model.eval()
    mask_id = tokenizer.mask_id
    pad_id = tokenizer.pad_id

    special_ids = {pad_id, mask_id, tokenizer.bos_id, tokenizer.eos_id}
    for name in ["<|user|>", "<|assistant|>", "<|system|>",
                 "<|think|>", "<|/think|>"]:
        tid = tokenizer.tokenizer.token_to_id(name)
        if tid is not None:
            special_ids.add(tid)

    prompt_len = min(len(prompt_ids), seq_len)
    full = torch.full((1, seq_len), mask_id, device=device)
    full[:, :prompt_len] = torch.tensor(prompt_ids[:prompt_len], device=device)

    n_steps = max(2, block_size)
    n_blocks = math.ceil((seq_len - prompt_len) / block_size)

    # Initialize HRMs
    rep_reviewer = RepetitionReviewer(
        pad_id=pad_id, mask_id=mask_id,
        bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
    ) if use_repetition_hrm else None

    semantic_hrm = SemanticCoherenceHRM(
        model, tokenizer,
        drift_threshold=drift_threshold,
        guidance_strength=semantic_guidance_strength,
    ) if use_semantic_hrm else None

    # Initialize semantic state from prompt
    semantic_state = None
    if semantic_hrm:
        semantic_state = semantic_hrm.init_state(full, prompt_len)

    block_idx = 0
    drift_stats = []

    for block_start in range(prompt_len, seq_len, block_size):
        block_end = min(block_start + block_size, seq_len)
        frac = block_idx / max(n_blocks - 1, 1)

        # Adaptive parameters
        temp = temp_schedule[0] + frac * (temp_schedule[1] - temp_schedule[0])
        rep_p = rep_schedule[0] + frac * (rep_schedule[1] - rep_schedule[0])
        freq_p = freq_schedule[0] + frac * (freq_schedule[1] - freq_schedule[0])

        # ── DIFFUSION WITHIN BLOCK ──
        for step in range(n_steps):
            t_val = max(0.5 - step / (n_steps * 2), 0.01)
            t = torch.full((1,), t_val, device=device)

            logits = model(full, t)

            # Adaptive logit guidance
            logits = apply_frequency_penalty(logits, full, penalty=freq_p, special_ids=special_ids)
            logits = apply_repetition_penalty(logits, full, penalty=rep_p, special_ids=special_ids)
            logits = apply_no_repeat_ngram(logits, full, n=2, special_ids=special_ids)

            mask_in_block = (full[0, block_start:block_end] == mask_id)
            if not mask_in_block.any():
                break

            idxs = mask_in_block.nonzero(as_tuple=True)[0]
            pos_logits = logits[0, block_start:block_end][idxs] / max(temp, 0.01)
            probs = F.softmax(pos_logits, dim=-1)

            # Top-p filtering
            probs = apply_top_p(probs, top_p)

            probs = probs.clamp(min=0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            conf = probs.max(dim=-1)[0]

            n_unmask = max(1, len(idxs) // (n_steps - step))
            tc, ti = conf.topk(min(n_unmask, len(idxs)))
            full[0, block_start + idxs[ti]] = sampled[ti]

        # ── POST-BLOCK HRM CHECKS ──
        # Semantic drift check (updates state too)
        if semantic_hrm and semantic_state:
            drift_mask, avg_sim = semantic_hrm.detect_drift(
                full, block_start, block_end, semantic_state
            )
            n_drift = drift_mask.sum().item()
            if n_drift > 0:
                full, drift_stat = semantic_hrm.refine_block(
                    full, block_start, block_end, semantic_state,
                    n_steps=4, temperature=max(0.3, temp - 0.2),
                )
                drift_stats.append({
                    "block": f"{block_start}-{block_end}",
                    "drifted": n_drift,
                    "pre_sim": drift_stat["pre_sim"],
                    "post_sim": drift_stat["post_sim"],
                })

        block_idx += 1

    # ── FINAL REPETITION HRM PASS ──
    rep_stats = None
    if rep_reviewer:
        rep_tokens, rep_stats_dict = _rep_hrm_pass(
            model, tokenizer, full, rep_reviewer,
            special_ids, temperature=0.5,
        )
        full = rep_tokens
        rep_stats = rep_stats_dict

    stats = {
        "n_blocks": n_blocks,
        "drift_corrections": drift_stats,
        "rep_stats": rep_stats,
    }
    return full, stats


@torch.no_grad()
def _rep_hrm_pass(model, tokenizer, tokens, reviewer, special_ids,
                  temperature=0.5, max_rounds=3):
    """Final repetition HRM pass over the full sequence."""
    mask_id = tokenizer.mask_id
    batch, seq_len = tokens.shape

    initial_score = reviewer.score_sequence(tokens[0])
    total_masked = 0
    rounds = 0

    for r in range(max_rounds):
        bad_mask = reviewer.detect_bad_positions(tokens)
        n_bad = bad_mask.sum().item()
        if n_bad == 0:
            rounds = r
            break
        rounds = r + 1
        tokens = torch.where(bad_mask, mask_id, tokens)

        # Regenerate
        for step in range(6):
            t_val = max(1.0 - step / 6, 0.01)
            t = torch.full((batch,), t_val, device=tokens.device)
            logits = model(tokens, t)

            logits = apply_frequency_penalty(logits, tokens, penalty=0.4, special_ids=special_ids)
            logits = apply_repetition_penalty(logits, tokens, penalty=1.3, special_ids=special_ids)

            current_mask = (tokens == mask_id)
            if not current_mask.any():
                break
            temp_logits = logits / max(temperature, 0.01)
            probs = F.softmax(temp_logits, dim=-1)
            confidence = probs.max(dim=-1)[0]
            confidence[~current_mask] = -1.0

            n_masked = current_mask.sum(dim=1)
            n_to_unmask = torch.clamp(n_masked // max(6 - step, 1), min=1)
            for b in range(batch):
                if n_masked[b] == 0:
                    continue
                k = min(int(n_to_unmask[b].item()), int(n_masked[b].item()))
                if k <= 0:
                    continue
                top_conf, top_idx = confidence[b].topk(k)
                valid = top_conf > 0
                if valid.any():
                    positions = top_idx[valid]
                    pos_logits = logits[b, positions] / max(temperature, 0.01)
                    pos_probs = F.softmax(pos_logits, dim=-1)
                    sampled = torch.multinomial(pos_probs, 1).squeeze(-1)
                    tokens[b, positions] = sampled
        total_masked += n_bad

    final_score = reviewer.score_sequence(tokens[0])
    return tokens, {
        "rounds": rounds,
        "masked": total_masked,
        "initial_score": initial_score,
        "final_score": final_score,
    }


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def load_model():
    ckpt_path = CHECKPOINT_DIR / "mdlm_bpe_v3_best.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = MDLMConfig(**ckpt["config"])
    model = MDLMBPEV3(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokenizer = BPETokenizer()
    print(f"Loaded: {ckpt_path.name} (step {ckpt.get('step', '?')}, "
          f"PPL={ckpt.get('ppl', '?'):.1f})")
    return model, tokenizer, config


def score_text(text, tokenizer, reviewer):
    ids = tokenizer.encode(text, add_special=False)
    if not ids:
        return 1.0
    t = torch.tensor([ids[:64]], device=DEVICE)
    if t.shape[1] < 64:
        t = torch.cat([t, torch.full((1, 64-t.shape[1]), tokenizer.pad_id, device=DEVICE)], 1)
    return reviewer.score_sequence(t[0])


def main():
    print("=" * 70)
    print("MDLM-BPE v3 — FULL PIPELINE EVALUATION")
    print("Semi-AR + Adaptive Guidance + Repetition HRM + Semantic HRM")
    print("=" * 70)

    model, tokenizer, config = load_model()
    reviewer = RepetitionReviewer(
        pad_id=tokenizer.pad_id, mask_id=tokenizer.mask_id,
        bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
    )

    prompts = [
        "The future of artificial intelligence",
        "To build a reliable system, you need",
        "Climate change is one of the biggest challenges",
        "The key to success in any project is",
        "Education is important because",
        "Programming is a skill that requires",
        "Reading books helps you to",
        "The internet has changed the way we",
    ]

    # ═══ Compare 4 configurations ═══
    configs = [
        ("A) Baseline (semi-AR only)", {
            "use_repetition_hrm": False, "use_semantic_hrm": False,
        }),
        ("B) + Adaptive Guidance", {
            "use_repetition_hrm": False, "use_semantic_hrm": False,
            # guidance is always on via params below
        }),
        ("C) + Guidance + Rep HRM", {
            "use_repetition_hrm": True, "use_semantic_hrm": False,
        }),
        ("D) FULL: + Semantic HRM", {
            "use_repetition_hrm": True, "use_semantic_hrm": True,
        }),
    ]

    all_results = {}

    for config_name, extra_params in configs:
        scores = []
        times = []
        drift_counts = []
        print(f"\n{'─' * 70}")
        print(f"{config_name}")
        print(f"{'─' * 70}")

        for prompt in prompts:
            ids = tokenizer.encode(prompt, add_special=False)

            # For baseline (A), disable guidance
            if "Baseline" in config_name:
                params = {
                    "temp_schedule": (0.7, 0.7),
                    "top_p": 1.0,
                    "rep_schedule": (1.0, 1.0),
                    "freq_schedule": (0.0, 0.0),
                    **extra_params,
                }
            else:
                params = {
                    "temp_schedule": (1.0, 0.5),
                    "top_p": 0.95,
                    "rep_schedule": (1.2, 1.5),
                    "freq_schedule": (0.3, 0.6),
                    **extra_params,
                }

            t0 = time.time()
            tokens, stats = generate_full_pipeline(
                model, tokenizer, ids, seq_len=64, block_size=4,
                **params,
            )
            elapsed = time.time() - t0
            text = tokenizer.decode(tokens[0].cpu().tolist())
            s = score_text(text, tokenizer, reviewer)
            scores.append(s)
            times.append(elapsed)

            n_drift = len(stats.get("drift_corrections", []))
            drift_counts.append(n_drift)

            print(f"  [{s:.2f}] ({elapsed:.1f}s) {prompt}")
            print(f"         → {text.strip()[:140]}")
            if n_drift > 0:
                print(f"         (semantic drift corrections: {n_drift})")

        avg_s = float(np.mean(scores))
        avg_t = float(np.mean(times))
        avg_d = float(np.mean(drift_counts))
        all_results[config_name] = {
            "score": avg_s, "time": avg_t, "tps": 64/avg_t, "drift": avg_d
        }
        print(f"\n  >>> Avg score: {avg_s:.3f}, Avg time: {avg_t:.1f}s, "
              f"TPS: {64/avg_t:.1f}, Drift corrections: {avg_d:.1f}")

    # ═══ Summary ═══
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Config':<35} {'Score':>7} {'Time':>7} {'TPS':>7} {'Drift':>7}")
    print(f"  {'-'*63}")
    for name, r in all_results.items():
        print(f"  {name:<35} {r['score']:>7.3f} {r['time']:>6.1f}s {r['tps']:>6.1f} {r['drift']:>6.1f}")


if __name__ == "__main__":
    main()
