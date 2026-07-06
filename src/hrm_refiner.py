"""
HRM Refinement for MDLM-BPE v2.

Instead of the old HRM (which used the 648-word vocab_cfg), this module
works with the real-text BPE model.

PIPELINE:
  1. MDLM-BPE generates text (or SFT model generates a response)
  2. Reviewer detects bad positions:
     - Exact repetition: token[i] == token[i-1]
     - Window repetition: same token appears >N times in window of K
     - Bigram repetition: same bigram appears multiple times
  3. Bad positions get MASKed
  4. MDLM-BPE regenerates ONLY masked positions (infilling)
  5. Repeat until clean or max rounds

WHY THIS WORKS:
  MDLM's parallel generation causes repetition because all positions
  are predicted independently — position 5 doesn't know position 4
  chose the same token. The Reviewer + re-mask loop fixes exactly
  this failure mode by adding a sequential correction pass.
"""
import sys
import time
import math
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RepetitionReviewer:
    """Detects repetition and poor local coherence in token sequences.

    This is a GEOMETRIC reviewer (0 params, 0 training) — like the
    DensityConfidenceScore that replaced the neural ConfidenceNet.
    It uses deterministic rules to find bad positions:

    1. EXACT REPEAT: token[i] == token[i-1] → mask token[i]
    2. WINDOW REPEAT: token appears >max_repeat times in window → mask extras
    3. BIGRAM REPEAT: same bigram seen before → mask second occurrence

    Returns a position-level mask: True = "this position is bad, regenerate it".
    """

    def __init__(self, window_size=8, max_repeat=2, pad_id=0, mask_id=1,
                 bos_id=2, eos_id=3):
        self.window = window_size
        self.max_repeat = max_repeat
        self.special_ids = {pad_id, mask_id, bos_id, eos_id, 5, 6, 7}  # chat tokens too

    def detect_bad_positions(self, tokens: torch.Tensor) -> torch.Tensor:
        """Detect positions that should be regenerated.

        Args:
            tokens: [batch, seq_len] token IDs

        Returns:
            bad_mask: [batch, seq_len] bool — True = regenerate this position
        """
        batch, seq_len = tokens.shape
        bad = torch.zeros_like(tokens, dtype=torch.bool)

        for b in range(batch):
            seq = tokens[b].cpu().tolist()

            # Rule 1: Exact consecutive repeat
            for i in range(1, seq_len):
                if seq[i] in self.special_ids:
                    continue
                if seq[i] == seq[i - 1]:
                    bad[b, i] = True

            # Rule 2: Window repetition (>max_repeat in window_size window)
            for i in range(seq_len):
                if seq[i] in self.special_ids:
                    continue
                if bad[b, i]:
                    continue  # Already marked
                start = max(0, i - self.window)
                window = seq[start:i + 1]
                count = window.count(seq[i])
                if count > self.max_repeat:
                    bad[b, i] = True

            # Rule 3: Bigram repetition (same 2-token sequence seen before)
            seen_bigrams = {}
            for i in range(1, seq_len):
                if seq[i] in self.special_ids or seq[i-1] in self.special_ids:
                    continue
                bigram = (seq[i-1], seq[i])
                if bigram in seen_bigrams:
                    # Mark the second occurrence
                    bad[b, i] = True
                else:
                    seen_bigrams[bigram] = i

        return bad

    def score_sequence(self, tokens: torch.Tensor) -> float:
        """Score a single sequence [0, 1]. 1 = no repetition, 0 = heavy repeat."""
        bad = self.detect_bad_positions(tokens.unsqueeze(0))
        special_tensor = torch.tensor(list(self.special_ids), device=tokens.device)
        non_special = ~torch.isin(tokens.unsqueeze(0), special_tensor)
        total = non_special.sum().item()
        if total == 0:
            return 1.0
        bad_count = bad.sum().item()
        return 1.0 - (bad_count / total)


class HRMRefiner:
    """HRM refinement loop: generate → detect → mask → regenerate.

    Uses the MDLM model for generation/infilling and the
    RepetitionReviewer for quality detection.
    """

    def __init__(self, model, tokenizer, max_rounds=5, temperature=0.6):
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

    def refine(self, tokens: torch.Tensor, n_unmask_steps=16) -> Tuple[torch.Tensor, dict]:
        """Refine a batch of token sequences.

        Args:
            tokens: [batch, seq_len] generated token IDs
            n_unmask_steps: diffusion steps for regeneration

        Returns:
            refined_tokens, stats dict
        """
        self.model.eval()
        batch, seq_len = tokens.shape
        mask_id = self.tokenizer.mask_id

        initial_scores = [self.reviewer.score_sequence(tokens[b])
                          for b in range(batch)]
        total_masked = 0
        rounds_used = 0

        with torch.no_grad():
            for round_num in range(self.max_rounds):
                # Detect bad positions
                bad_mask = self.reviewer.detect_bad_positions(tokens)
                n_bad = bad_mask.sum().item()

                if n_bad == 0:
                    rounds_used = round_num
                    break

                rounds_used = round_num + 1

                # Mask bad positions
                masked_tokens = torch.where(bad_mask, mask_id, tokens)

                # Regenerate masked positions via infilling
                tokens = self._infill(masked_tokens, bad_mask,
                                      n_unmask_steps, batch, seq_len)

                total_masked += n_bad

        final_scores = [self.reviewer.score_sequence(tokens[b])
                        for b in range(batch)]

        stats = {
            "rounds": rounds_used,
            "total_masked": total_masked,
            "initial_score": np.mean(initial_scores),
            "final_score": np.mean(final_scores),
            "improvement": np.mean(final_scores) - np.mean(initial_scores),
        }
        return tokens, stats

    def _infill(self, tokens: torch.Tensor, mask_positions: torch.Tensor,
                n_steps: int, batch: int, seq_len: int) -> torch.Tensor:
        """Regenerate only masked positions using the MDLM."""
        mask_id = self.tokenizer.mask_id

        for step in range(n_steps):
            t_val = max(1.0 - step / n_steps, 0.01)
            t = torch.full((batch,), t_val, device=tokens.device)

            logits = self.model(tokens, t)

            # Only look at masked positions
            current_mask = (tokens == mask_id)
            if not current_mask.any():
                break

            # Vectorized sampling
            temp_logits = logits / max(self.temperature, 0.01)
            probs = F.softmax(temp_logits, dim=-1)
            confidence = probs.max(dim=-1)[0]
            confidence[~current_mask] = -1.0

            # Unmask proportionally
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


def generate_with_hrm(model, tokenizer,
                      prompt: str = None, seq_len: int = 64,
                      n_steps: int = 32, temperature: float = 0.7,
                      max_rounds: int = 5, device: str = DEVICE) -> Tuple[str, dict]:
    """Generate text and refine it with the HRM loop.

    Args:
        prompt: optional prompt text (for SFT model)
        max_rounds: max HRM refinement rounds

    Returns:
        refined_text, stats
    """
    model.eval()
    refiner = HRMRefiner(model, tokenizer, max_rounds=max_rounds,
                         temperature=temperature)

    # Step 1: Initial generation
    if prompt:
        from finetune_sft_v2 import generate_response
        text = generate_response(model, tokenizer, prompt,
                                 max_len=seq_len, n_steps=n_steps,
                                 temperature=temperature)
        # Re-encode to get token IDs
        ids = tokenizer.encode(text, add_special=False)
        tokens = torch.tensor([ids[:seq_len]], device=device)
        if tokens.shape[1] < seq_len:
            pad = torch.full((1, seq_len - tokens.shape[1]),
                             tokenizer.pad_id, device=device)
            tokens = torch.cat([tokens, pad], dim=1)
    else:
        # Free generation
        from mdlm_bpe_v2 import sample_mdlm_v2
        samples = sample_mdlm_v2(model, tokenizer, seq_len=seq_len,
                                n_samples=1, n_steps=n_steps,
                                temperature=temperature)
        ids = tokenizer.encode(samples[0], add_special=False)
        tokens = torch.tensor([ids[:seq_len]], device=device)
        if tokens.shape[1] < seq_len:
            pad = torch.full((1, seq_len - tokens.shape[1]),
                             tokenizer.pad_id, device=device)
            tokens = torch.cat([tokens, pad], dim=1)

    initial_text = tokenizer.decode(tokens[0].cpu().tolist())

    # Step 2: HRM refinement
    refined, stats = refiner.refine(tokens)
    final_text = tokenizer.decode(refined[0].cpu().tolist())

    stats["initial_text"] = initial_text.strip()[:200]
    stats["final_text"] = final_text.strip()[:200]

    return final_text, stats
