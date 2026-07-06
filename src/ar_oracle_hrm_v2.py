"""
AR-Oracle HRM v2 — Direct token replacement (no MDLM regeneration).

DESIGN CHANGE FROM v1:
  v1: detect bad → mask → MDLM regenerates with oracle bias
      PROBLEM: MDLM (201M) too weak to generate good replacements
      even with guidance. Net effect ≈ 0 or negative.

  v2: detect bad → oracle DIRECTLY provides replacement tokens
      The oracle's argmax at each flagged position replaces the
      MDLM's output. No regeneration round-trip.

  This is "speculative decoding in reverse":
  - Fast: MDLM generates everything in parallel (60K TPS)
  - Verify: Oracle scores each token (1 forward pass)
  - Replace: Oracle's predictions overwrite bad positions

  The oracle is the quality floor. The MDLM provides speed.
  Together: fast generation + quality guaranteed by the oracle.

PIPELINE:
  1. MDLM generates full text (semi-AR, ~3s for 64 tokens)
  2. Oracle scores all tokens (1 forward pass, ~50ms)
  3. Positions below threshold → oracle's argmax replaces them
  4. Repeat: re-score, re-replace (converges in 2-3 rounds)
"""
import sys
import time
import math
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from ar_oracle_hrm import (
    AROracleHRM as _Base,
    load_oracle,
    build_token_alignment,
    generated_region,
)


class AROracleHRMv2(_Base):
    """AR-Oracle HRM v2: Direct replacement instead of guided regeneration.

    When the oracle detects a bad position, it replaces the token directly
    with its own prediction — no MDLM regeneration round-trip.

    This is fundamentally more reliable because:
    - The oracle (596M) has better language understanding
    - No round-trip through the weak MDLM
    - Deterministic: oracle's best guess always wins
    - Converges fast (2-3 rounds max)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def refine(
        self,
        tokens: torch.Tensor,
        prompt_len: int = 0,
        temperature: float = 0.0,  # 0 = greedy oracle, >0 = sampling
        max_rounds: int = 3,
    ) -> Tuple[torch.Tensor, dict]:
        """Refine by direct oracle replacement.

        Args:
            tokens: [1, seq_len] MDLM tokens
            prompt_len: prompt tokens preserved
            temperature: oracle sampling temp (0=greedy)
            max_rounds: max detect→replace iterations

        Returns:
            refined_tokens, stats
        """
        mask_id = self.mdlm_tok.mask_id
        pad_id = self.mdlm_tok.pad_id
        seq_len = tokens.shape[1]

        # Score before
        scores_before, _ = self.score_sequence(tokens, prompt_len)

        current_tokens = tokens.clone()
        round_stats = []
        total_replaced = 0

        for round_num in range(max_rounds):
            # ── DETECT: single oracle pass ──
            scores, oracle_log_probs = self.score_sequence_with_logits(
                current_tokens, prompt_len
            )

            incoherent_mask, _ = self._threshold_incoherent(
                scores, current_tokens, prompt_len
            )
            n_bad = incoherent_mask.sum().item()

            if n_bad == 0:
                round_stats.append({
                    "round": round_num, "replaced": 0
                })
                break

            if oracle_log_probs is None:
                break

            # ── REPLACE: oracle's prediction overwrites bad positions ──
            # For each bad MDLM position, find the oracle position and
            # get the oracle's predicted token (greedy or sampled)
            replacements = self._get_oracle_replacements(
                current_tokens, incoherent_mask, oracle_log_probs,
                prompt_len, temperature,
            )

            # Apply replacements
            n_replaced = 0
            for pos, new_text in replacements.items():
                # Encode the oracle's suggestion in MDLM vocab
                try:
                    new_ids = self.mdlm_tok.encode(new_text, add_special=False)
                    if new_ids:
                        # Replace this position with the first token
                        # (if multi-token, we can only fit one)
                        current_tokens[0, pos] = new_ids[0]
                        n_replaced += 1
                except Exception:
                    pass

            total_replaced += n_replaced

            # Re-score to check improvement
            scores_after, _ = self.score_sequence(current_tokens, prompt_len)
            gen_mask = generated_region(current_tokens, prompt_len, mask_id)
            lp_before = scores[gen_mask].mean().item()
            lp_after = scores_after[gen_mask].mean().item()

            round_stats.append({
                "round": round_num,
                "detected": n_bad,
                "replaced": n_replaced,
                "lp_before": lp_before,
                "lp_after": lp_after,
                "improved": lp_after - lp_before,
            })

            # Early stopping if no improvement
            if lp_after <= lp_before + 0.01:
                break

        # Final scores
        scores_final, _ = self.score_sequence(current_tokens, prompt_len)
        gen_mask = generated_region(tokens, prompt_len, mask_id)
        lp_initial = scores_before[gen_mask].mean().item()
        lp_final = scores_final[gen_mask].mean().item()

        return current_tokens, {
            "corrected": total_replaced > 0,
            "total_replaced": total_replaced,
            "rounds": len(round_stats),
            "round_details": round_stats,
            "mean_lp_before": lp_initial,
            "mean_lp_after": lp_final,
            "lp_improvement": lp_final - lp_initial,
        }

    @torch.no_grad()
    def _get_oracle_replacements(
        self,
        tokens: torch.Tensor,
        incoherent_mask: torch.Tensor,
        oracle_log_probs: torch.Tensor,
        prompt_len: int,
        temperature: float,
    ) -> Dict[int, str]:
        """Get oracle's replacement text for each bad position.

        For each incoherent MDLM position:
        1. Find the corresponding oracle position
        2. Get oracle's top token at that position
        3. Decode to text
        """
        replacements = {}

        all_ids = tokens[0].cpu().tolist()
        mask_id = self.mdlm_tok.mask_id
        pad_id = self.mdlm_tok.pad_id

        clean_ids = [t for t in all_ids if t not in (mask_id, pad_id)]
        _, full_text = build_token_alignment(
            clean_ids, self.mdlm_tok, [], self.oracle_tok
        )
        if not full_text.strip():
            return replacements

        enc = self.oracle_tok(full_text, return_offsets_mapping=True,
                              add_special_tokens=False)
        oracle_offsets = enc["offset_mapping"]

        # Walk through MDLM tokens, tracking oracle position alignment
        current_char = 0
        oracle_idx = 0
        incoherent_positions = set(
            incoherent_mask.cpu().nonzero(as_tuple=True)[0].tolist()
        )

        for seq_idx, tid in enumerate(all_ids):
            if tid in (mask_id, pad_id):
                continue

            tok_text = self.mdlm_tok.decode([tid])
            char_start = current_char
            char_end = current_char + len(tok_text)
            current_char = char_end

            # Advance oracle index to match this MDLM token's char range
            while oracle_idx < len(oracle_offsets):
                o_start, o_end = oracle_offsets[oracle_idx]
                if o_start >= char_end:
                    break
                oracle_idx += 1

            oracle_pos = oracle_idx - 1 if oracle_idx > 0 else 0

            if seq_idx not in incoherent_positions:
                continue
            if oracle_pos >= oracle_log_probs.shape[0]:
                continue

            # Get oracle's prediction at this position
            pos_lp = oracle_log_probs[oracle_pos]  # [oracle_vocab]

            if temperature == 0.0:
                # Greedy: argmax
                best_id = pos_lp.argmax().item()
            else:
                # Sample from oracle distribution
                probs = F.softmax(pos_lp / temperature, dim=-1)
                best_id = torch.multinomial(probs, 1).item()

            replacement_text = self.oracle_tok.decode([best_id]).strip()
            if replacement_text:
                replacements[seq_idx] = replacement_text

        return replacements
