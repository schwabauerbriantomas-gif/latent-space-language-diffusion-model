"""
MDLM-BPE v3 — Full pipeline evaluation with HRM refinement.

Pipeline:
  1. Semi-AR generation (left-to-right blocks)
  2. Logit guidance during generation (prevents most repetition)
  3. HRM refinement loop (detect remaining repetition → mask → regenerate)

Compares 3 modes:
  A) Baseline (semi-AR only)
  B) Semi-AR + Logit Guidance
  C) Semi-AR + Logit Guidance + HRM Refinement
"""
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
# LOGIT GUIDANCE
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
    penalty_vals = penalty * torch.sqrt(counts.float())
    return logits - penalty_vals.unsqueeze(1)


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
    penalty_factor = torch.where(
        used_mask, torch.tensor(1.0 / penalty, device=logits.device),
        torch.tensor(1.0, device=logits.device),
    )
    return logits * penalty_factor.unsqueeze(1)


def apply_no_repeat_ngram(logits, tokens, n=2, special_ids=None):
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
    return logits.masked_fill(ban_mask, float('-inf'))


# ═══════════════════════════════════════════════════════════════════
# HRM REFINER (adapted for v3)
# ═══════════════════════════════════════════════════════════════════

class HRMRefinerV3:
    """HRM refinement loop for MDLM-BPE v3.

    detect bad positions → mask → regenerate via semi-AR infilling.
    """

    def __init__(self, model, tokenizer, max_rounds=3, temperature=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.reviewer = RepetitionReviewer(
            pad_id=tokenizer.pad_id,
            mask_id=tokenizer.mask_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
        )
        self.max_rounds = max_rounds
        self.temperature = temperature

    @torch.no_grad()
    def refine(self, tokens, n_infill_steps=8):
        """Refine a batch of token sequences [batch, seq_len]."""
        self.model.eval()
        batch, seq_len = tokens.shape
        mask_id = self.tokenizer.mask_id

        initial_scores = [self.reviewer.score_sequence(tokens[b])
                          for b in range(batch)]

        total_masked = 0
        rounds_used = 0
        scores_per_round = [np.mean(initial_scores)]

        for round_num in range(self.max_rounds):
            bad_mask = self.reviewer.detect_bad_positions(tokens)
            n_bad = bad_mask.sum().item()

            if n_bad == 0:
                rounds_used = round_num
                break

            rounds_used = round_num + 1

            # Mask bad positions
            tokens = torch.where(bad_mask, mask_id, tokens)

            # Regenerate masked positions via infilling
            tokens = self._infill(tokens, bad_mask, n_infill_steps, batch, seq_len)

            total_masked += n_bad
            round_scores = [self.reviewer.score_sequence(tokens[b])
                           for b in range(batch)]
            scores_per_round.append(np.mean(round_scores))

        final_scores = [self.reviewer.score_sequence(tokens[b])
                       for b in range(batch)]

        stats = {
            "rounds": rounds_used,
            "total_masked": total_masked,
            "initial_score": np.mean(initial_scores),
            "final_score": np.mean(final_scores),
            "improvement": np.mean(final_scores) - np.mean(initial_scores),
            "scores_per_round": scores_per_round,
        }
        return tokens, stats

    @torch.no_grad()
    def _infill(self, tokens, mask_positions, n_steps, batch, seq_len):
        """Regenerate only masked positions using the v3 model."""
        mask_id = self.tokenizer.mask_id

        for step in range(n_steps):
            t_val = max(1.0 - step / n_steps, 0.01)
            t = torch.full((batch,), t_val, device=tokens.device)

            logits = self.model(tokens, t)

            current_mask = (tokens == mask_id)
            if not current_mask.any():
                break

            # Apply logit guidance during infilling too
            special_ids = {self.tokenizer.pad_id, mask_id,
                          self.tokenizer.bos_id, self.tokenizer.eos_id}
            logits = apply_frequency_penalty(logits, tokens, penalty=0.3,
                                            special_ids=special_ids)
            logits = apply_repetition_penalty(logits, tokens, penalty=1.3,
                                             special_ids=special_ids)

            temp_logits = logits / max(self.temperature, 0.01)
            probs = F.softmax(temp_logits, dim=-1)
            confidence = probs.max(dim=-1)[0]
            confidence[~current_mask] = -1.0

            n_masked = current_mask.sum(dim=1)
            n_to_unmask = torch.clamp(
                n_masked // max(n_steps - step, 1), min=1
            )

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
                    pos_logits = logits[b, positions] / max(self.temperature, 0.01)
                    pos_probs = F.softmax(pos_logits, dim=-1)
                    sampled = torch.multinomial(pos_probs, 1).squeeze(-1)
                    tokens[b, positions] = sampled

        return tokens


# ═══════════════════════════════════════════════════════════════════
# SEMI-AR GENERATION WITH GUIDANCE
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_semi_ar_guided(
    model, tokenizer, prompt_ids=None, seq_len=64,
    n_samples=1, block_size=4, temperature=0.7,
    use_guidance=True,
    repetition_penalty=1.3, no_repeat_ngram=2, frequency_penalty=0.4,
    device=DEVICE,
):
    """Semi-AR generation with optional logit guidance."""
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

            if use_guidance:
                if frequency_penalty > 0:
                    logits = apply_frequency_penalty(logits, full, penalty=frequency_penalty, special_ids=special_ids)
                if repetition_penalty > 1.0:
                    logits = apply_repetition_penalty(logits, full, penalty=repetition_penalty, special_ids=special_ids)
                if no_repeat_ngram > 0:
                    logits = apply_no_repeat_ngram(logits, full, n=no_repeat_ngram, special_ids=special_ids)

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

    return full


# ═════════════════════════════════════════════════════════ →
# HELPER
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


def score_text(text, tokenizer, reviewer, max_len=64):
    ids = tokenizer.encode(text, add_special=False)
    if not ids:
        return 1.0
    t = torch.tensor([ids[:max_len]], device=DEVICE)
    if t.shape[1] < max_len:
        t = torch.cat([t, torch.full((1, max_len-t.shape[1]), tokenizer.pad_id, device=DEVICE)], 1)
    return reviewer.score_sequence(t[0])


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MDLM-BPE v3 — FULL PIPELINE (Semi-AR + Guidance + HRM)")
    print("=" * 70)

    model, tokenizer, config = load_model()
    reviewer = RepetitionReviewer(
        pad_id=tokenizer.pad_id, mask_id=tokenizer.mask_id,
        bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
    )
    refiner = HRMRefinerV3(model, tokenizer, max_rounds=3, temperature=0.5)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params:,} ({params/1e6:.1f}M)")

    prompts = [
        "The future of artificial intelligence",
        "To build a reliable system, you need",
        "Climate change is one of the biggest challenges",
        "The key to success in any project is",
        "Education is important because",
    ]

    all_scores = {"baseline": [], "guided": [], "guided_hrm": []}
    all_times = {"baseline": [], "guided": [], "guided_hrm": []}

    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special=False)
        print(f"\n{'─' * 70}")
        print(f"PROMPT: {prompt}")
        print(f"{'─' * 70}")

        # === A) BASELINE ===
        t0 = time.time()
        tokens = generate_semi_ar_guided(
            model, tokenizer, prompt_ids=ids, seq_len=64,
            n_samples=1, block_size=4, temperature=0.7,
            use_guidance=False,
        )
        text_baseline = tokenizer.decode(tokens[0].cpu().tolist())
        t_baseline = time.time() - t0
        score_baseline = score_text(text_baseline, tokenizer, reviewer)
        all_scores["baseline"].append(score_baseline)
        all_times["baseline"].append(t_baseline)
        print(f"  A) BASELINE [{score_baseline:.2f}] ({t_baseline:.1f}s)")
        print(f"     → {text_baseline.strip()[:160]}")

        # === B) SEMI-AR + GUIDANCE ===
        t0 = time.time()
        tokens = generate_semi_ar_guided(
            model, tokenizer, prompt_ids=ids, seq_len=64,
            n_samples=1, block_size=4, temperature=0.7,
            use_guidance=True,
            repetition_penalty=1.3, no_repeat_ngram=2, frequency_penalty=0.4,
        )
        text_guided = tokenizer.decode(tokens[0].cpu().tolist())
        t_guided = time.time() - t0
        score_guided = score_text(text_guided, tokenizer, reviewer)
        all_scores["guided"].append(score_guided)
        all_times["guided"].append(t_guided)
        print(f"  B) GUIDED  [{score_guided:.2f}] ({t_guided:.1f}s)")
        print(f"     → {text_guided.strip()[:160]}")

        # === C) SEMI-AR + GUIDANCE + HRM ===
        t0 = time.time()
        # Start from guided tokens
        tokens = generate_semi_ar_guided(
            model, tokenizer, prompt_ids=ids, seq_len=64,
            n_samples=1, block_size=4, temperature=0.7,
            use_guidance=True,
            repetition_penalty=1.3, no_repeat_ngram=2, frequency_penalty=0.4,
        )
        # Apply HRM refinement
        refined_tokens, hrml_stats = refiner.refine(tokens, n_infill_steps=8)
        text_hrm = tokenizer.decode(refined_tokens[0].cpu().tolist())
        t_hrm = time.time() - t0
        score_hrm = score_text(text_hrm, tokenizer, reviewer)
        all_scores["guided_hrm"].append(score_hrm)
        all_times["guided_hrm"].append(t_hrm)
        print(f"  C) +HRM    [{score_hrm:.2f}] ({t_hrm:.1f}s) rounds={hrml_stats['rounds']} masked={hrml_stats['total_masked']}")
        print(f"     → {text_hrm.strip()[:160]}")

    # === SUMMARY ===
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Mode':<25} {'Avg Score':>10} {'Avg Time':>10} {'TPS':>10}")
    print(f"  {'-'*55}")
    for mode in ["baseline", "guided", "guided_hrm"]:
        avg_score = np.mean(all_scores[mode])
        avg_time = np.mean(all_times[mode])
        tps = 64 / avg_time
        print(f"  {mode:<25} {avg_score:>10.2f} {avg_time:>9.1f}s {tps:>9.1f}t/s")

    print(f"\n  Improvement baseline → guided:     {np.mean(all_scores['guided']) - np.mean(all_scores['baseline']):+.2f}")
    print(f"  Improvement guided → guided+hrm:   {np.mean(all_scores['guided_hrm']) - np.mean(all_scores['guided']):+.2f}")
    print(f"  Improvement baseline → guided+hrm: {np.mean(all_scores['guided_hrm']) - np.mean(all_scores['baseline']):+.2f}")


if __name__ == "__main__":
    main()
