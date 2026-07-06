"""
AR-Oracle HRM — Long-distance coherence via autoregressive verification.

PROBLEM:
  The MDLM (201M params) generates text that is locally coherent but
  lacks long-distance semantic coherence. The small model simply doesn't
  have enough capacity to track topic across 50+ tokens.

  Previous attempt (SemanticCoherenceHRM) used the MDLM's own embeddings
  for drift detection — but those embeddings are too weak (PPL=102 model).
  The detection worked but the correction degraded text.

SOLUTION:
  Use a REAL language model (Qwen3-0.6B, 596M params, trained on trillions
  of tokens) as an ORACLE to verify and refine the MDLM's output.

PIPELINE:
  1. MDLM generates text via semi-AR (fast, parallel, 60K TPS forward)
  2. Qwen3-0.6B scores each token via teacher-forcing log-probs
  3. Positions with low log-prob (surprising to the oracle) = incoherent
  4. For incoherent positions:
     a. Get Qwen3's top-k candidate tokens at that position
     b. Decode them back to MDLM token space
     c. Use as positive bias in MDLM regeneration (oracle-guided logits)
  5. MDLM regenerates only those positions with oracle guidance

ARCHITECTURE:

  ┌─────────┐     ┌──────────────┐     ┌─────────────┐
  │  MDLM   │────▶│  Qwen3-0.6B  │────▶│   Refine    │
  │ (201M)  │ text│   (596M)     │score│  positions  │
  │ semi-AR │     │   teacher    │     │  w/ oracle  │
  └─────────┘     │   forcing    │     │   logits    │
   fast gen       └──────────────┘     └─────────────┘
                   accurate eval         targeted fix

  The two models have DIFFERENT tokenizers:
  - MDLM: custom BPE (10K vocab)
  - Qwen3: tiktoken-based (151K vocab)

  We bridge this by working at the TEXT level:
  - MDLM decodes tokens → text
  - Qwen3 tokenizes text → its own tokens
  - Score at character level, map back to MDLM tokens

PARALLELISM:
  Qwen3 processes the full sequence in ONE forward pass (teacher forcing).
  No autoregressive loop needed for scoring — it's a single matrix mul.
  This makes it fast enough to run alongside MDLM generation.
"""
import sys
import time
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenizers: MDLM BPE and Qwen3
from transformers import AutoModelForCausalLM, AutoTokenizer as HFTokenizer

# Lazy-loaded global instances
_oracle_model = None
_oracle_tokenizer = None


def load_oracle(model_name: str = "Qwen/Qwen3-0.6B"):
    """Load the AR oracle model (Qwen3-0.6B) into VRAM."""
    global _oracle_model, _oracle_tokenizer
    if _oracle_model is not None:
        return _oracle_model, _oracle_tokenizer

    print(f"  Loading AR oracle: {model_name}...", end=" ", flush=True)
    t0 = time.time()
    _oracle_tokenizer = HFTokenizer.from_pretrained(model_name)
    _oracle_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    _oracle_model.eval()
    elapsed = time.time() - t0
    params = sum(p.numel() for p in _oracle_model.parameters()) / 1e6
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"done ({params:.0f}M params, {vram:.1f}GB VRAM, {elapsed:.1f}s)")
    return _oracle_model, _oracle_tokenizer


# ═══════════════════════════════════════════════════════════════════
# CHARACTER-LEVEL TOKEN MAPPING
# ═══════════════════════════════════════════════════════════════════

def build_token_alignment(mdlm_tokens: List[int], mdlm_tokenizer,
                           oracle_tokens: List[int], oracle_tokenizer):
    """Map MDLM token positions to Oracle token positions via text.

    Since the two tokenizers split text differently, we need to know
    which Oracle tokens correspond to which MDLM token.

    Strategy: decode each MDLM token to text, track character offsets,
    then find which Oracle tokens fall within those offsets.

    Args:
        mdlm_tokens: list of MDLM token IDs
        mdlm_tokenizer: MDLM BPETokenizer
        oracle_tokens: list of Oracle token IDs
        oracle_tokenizer: HuggingFace tokenizer

    Returns:
        alignment: list of (mdlm_idx, [oracle_idx_start, oracle_idx_end])
                   mapping each MDLM token to a range of Oracle tokens
    """
    # Decode each MDLM token individually to get its text + char length
    mdlm_char_ranges = []  # [(start_char, end_char)]
    current_char = 0
    full_text_parts = []

    for tid in mdlm_tokens:
        text = mdlm_tokenizer.decode([tid])
        char_len = len(text)
        mdlm_char_ranges.append((current_char, current_char + char_len))
        full_text_parts.append(text)
        current_char += char_len

    full_text = "".join(full_text_parts)

    # Now find Oracle token char ranges
    # The oracle tokenizer doesn't give us per-token char offsets directly,
    # but we can get them via offset_mapping
    enc = oracle_tokenizer(full_text, return_offsets_mapping=True,
                           add_special_tokens=False)
    oracle_offsets = enc["offset_mapping"]  # [(start, end), ...]

    # Map: for each MDLM token, find oracle tokens within its char range
    alignment = []
    oracle_idx = 0

    for mdlm_idx, (char_start, char_end) in enumerate(mdlm_char_ranges):
        start_o = oracle_idx
        while oracle_idx < len(oracle_offsets):
            o_start, o_end = oracle_offsets[oracle_idx]
            if o_start >= char_end:
                break
            oracle_idx += 1
        end_o = oracle_idx
        if end_o == start_o and oracle_idx < len(oracle_offsets):
            # At least one oracle token per MDLM token
            end_o = start_o + 1
            oracle_idx = end_o
        alignment.append((mdlm_idx, list(range(start_o, min(end_o, len(oracle_offsets))))))

    return alignment, full_text


# ═══════════════════════════════════════════════════════════════════
# AR-ORACLE HRM
# ═══════════════════════════════════════════════════════════════════

class AROracleHRM:
    """AR-Oracle HRM: verify and refine MDLM output with a real LM.

    The oracle (Qwen3-0.6B) scores each token's log-probability via
    teacher forcing. Low-probability positions are flagged as incoherent.

    For correction, the oracle's top-k tokens at each position are used
    to bias the MDLM's regeneration logits.
    """

    def __init__(
        self,
        mdlm_model,
        mdlm_tokenizer,
        oracle_model_name: str = "Qwen/Qwen3-0.6B",
        # Tuning parameters
        surprise_threshold: float = 2.0,  # nats below mean = incoherent
        correction_strength: float = 2.0,  # logit boost for oracle tokens
        max_correction_rate: float = 0.3,  # max fraction of tokens to correct
    ):
        self.mdlm = mdlm_model
        self.mdlm_tok = mdlm_tokenizer
        self.oracle, self.oracle_tok = load_oracle(oracle_model_name)
        self.surprise_threshold = surprise_threshold
        self.correction_strength = correction_strength
        self.max_correction_rate = max_correction_rate

        self.oracle_vocab_size = self.oracle_tok.vocab_size
        self.oracle_eos = self.oracle_tok.eos_token_id

    @torch.no_grad()
    def score_sequence(self, tokens: torch.Tensor, prompt_len: int = 0
                       ) -> Tuple[torch.Tensor, dict]:
        """Score each MDLM token using the AR oracle.

        Args:
            tokens: [batch=1, seq_len] MDLM token IDs
            prompt_len: number of prompt tokens (not scored)

        Returns:
            token_scores: [seq_len] per-token log-prob (higher = better)
            stats: dict with alignment info
        """
        # 1. Decode MDLM tokens to text
        all_ids = tokens[0].cpu().tolist()
        # Remove special tokens
        mask_id = self.mdlm_tok.mask_id
        pad_id = self.mdlm_tok.pad_id
        clean_ids = [t for t in all_ids if t not in (mask_id, pad_id)]

        # 2. Build alignment: MDLM token → text → Oracle token
        alignment, full_text = build_token_alignment(
            clean_ids, self.mdlm_tok, [], self.oracle_tok
        )

        if not full_text.strip():
            return torch.ones(tokens.shape[1], device=DEVICE), {"error": "empty"}

        # 3. Tokenize with Oracle
        oracle_ids = self.oracle_tok.encode(full_text, add_special_tokens=False)
        if len(oracle_ids) < 2:
            return torch.ones(tokens.shape[1], device=DEVICE), {"error": "too_short"}

        # 4. Teacher forcing: get log-probs from oracle
        oracle_input = torch.tensor([oracle_ids], device=DEVICE)
        logits = self.oracle(oracle_input).logits  # [1, seq, vocab]
        log_probs = F.log_softmax(logits.float(), dim=-1)  # [1, seq, vocab]

        # Per-token log-prob of the ACTUAL next token
        # log_probs[0, i] predicts oracle_ids[i+1]
        actual_next = oracle_input[0, 1:]  # [seq-1]
        predicted = log_probs[0, :-1]  # [seq-1, vocab]
        token_log_probs = predicted.gather(1, actual_next.unsqueeze(1)).squeeze(1)  # [seq-1]

        # 5. Map Oracle token log-probs back to MDLM token positions
        # Rebuild alignment with actual oracle tokens
        enc = self.oracle_tok(full_text, return_offsets_mapping=True,
                              add_special_tokens=False)
        oracle_offsets = enc["offset_mapping"]

        # For each MDLM token, find its Oracle token range and average log-prob
        mdlm_scores = torch.full((tokens.shape[1],), -10.0, device=DEVICE)

        current_char = 0
        oracle_idx = 0

        clean_idx = 0  # index into clean_ids
        for seq_idx, tid in enumerate(all_ids):
            if tid in (mask_id, pad_id):
                mdlm_scores[seq_idx] = 0.0  # neutral for special
                continue

            # Get this MDLM token's text and char range
            tok_text = self.mdlm_tok.decode([tid])
            char_start = current_char
            char_end = current_char + len(tok_text)
            current_char = char_end

            # Find oracle tokens in this char range
            oracle_lps = []
            while oracle_idx < len(oracle_offsets):
                o_start, o_end = oracle_offsets[oracle_idx]
                if o_start >= char_end:
                    break
                if oracle_idx < len(token_log_probs):
                    oracle_lps.append(token_log_probs[oracle_idx].item())
                oracle_idx += 1

            if oracle_lps:
                mdlm_scores[seq_idx] = sum(oracle_lps) / len(oracle_lps)
            else:
                mdlm_scores[seq_idx] = 0.0

            clean_idx += 1

        stats = {
            "oracle_tokens": len(oracle_ids),
            "mean_log_prob": token_log_probs.mean().item(),
            "alignment": len(alignment),
        }
        return mdlm_scores, stats

    @torch.no_grad()
    def detect_incoherent(
        self,
        tokens: torch.Tensor,
        prompt_len: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """Detect incoherent positions using the AR oracle.

        Args:
            tokens: [1, seq_len] MDLM token IDs
            prompt_len: prompt tokens are not flagged

        Returns:
            incoherent_mask: [seq_len] bool — True = needs regeneration
            stats: dict
        """
        scores, score_stats = self.score_sequence(tokens, prompt_len)

        # Only look at generated positions (after prompt, not special)
        mask_id = self.mdlm_tok.mask_id
        pad_id = self.mdlm_tok.pad_id
        generated_mask = torch.ones(tokens.shape[1], device=DEVICE, dtype=torch.bool)
        generated_mask[:prompt_len] = False
        generated_mask |= (tokens[0] == mask_id) | (tokens[0] == pad_id)

        # Compute threshold: positions significantly below mean
        gen_scores = scores[generated_mask]
        if gen_scores.numel() == 0:
            return torch.zeros_like(generated_mask), score_stats

        mean_lp = gen_scores.mean()
        std_lp = gen_scores.std()

        # Surprise = how many nats below mean
        # Threshold: mean - surprise_threshold * std (or absolute if std small)
        threshold = mean_lp - self.surprise_threshold
        if std_lp > 0.01:
            threshold = mean_lp - self.surprise_threshold * std_lp

        incoherent = (scores < threshold) & generated_mask

        # Cap correction rate
        n_incoherent = incoherent.sum().item()
        max_correct = int(self.max_correction_rate * generated_mask.sum().item())
        if n_incoherent > max_correct:
            # Keep only the worst positions
            gen_scores_np = scores.cpu().numpy()
            gen_indices = generated_mask.cpu().nonzero().squeeze(-1)
            # Get scores of incoherent positions
            incoh_indices = incoherent.cpu().nonzero().squeeze(-1)
            incoh_scores = gen_scores_np[incoh_indices]
            # Sort by score (worst first)
            sorted_order = np.argsort(incoh_scores)
            keep = set(incoh_indices[sorted_order[:max_correct]].tolist())
            incoherent = torch.zeros_like(generated_mask)
            for idx in keep:
                incoherent[idx] = True
            n_incoherent = max_correct

        stats = {**score_stats,
                 "n_incoherent": n_incoherent,
                 "mean_lp": mean_lp.item(),
                 "threshold": threshold.item()}
        return incoherent, stats

    @torch.no_grad()
    def get_oracle_suggestions(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> Dict[int, List[str]]:
        """Get oracle's top-k suggested token texts at given positions.

        For each flagged position, decode the MDLM text up to that point,
        feed to the oracle, and get its top-k predictions for the next token.

        Args:
            tokens: [1, seq_len] MDLM tokens
            positions: indices to get suggestions for

        Returns:
            suggestions: {position: [list of top-k token texts]}
        """
        mask_id = self.mdlm_tok.mask_id
        pad_id = self.mdlm_tok.pad_id
        suggestions = {}

        for pos in positions:
            pos = int(pos)
            # Get text up to this position
            prefix_ids = tokens[0, :pos].cpu().tolist()
            clean = [t for t in prefix_ids if t not in (mask_id, pad_id)]
            prefix_text = self.mdlm_tok.decode(clean)

            # Tokenize with oracle and get next-token distribution
            oracle_ids = self.oracle_tok.encode(prefix_text, add_special_tokens=False)
            if len(oracle_ids) == 0:
                continue

            oracle_input = torch.tensor([oracle_ids], device=DEVICE)
            logits = self.oracle(oracle_input).logits
            last_logits = logits[0, -1].float()
            log_probs = F.log_softmax(last_logits, dim=-1)

            # Top-5 suggestions
            top_k = log_probs.topk(5)
            top_texts = []
            for i in range(5):
                tid = top_k.indices[i].item()
                text = self.oracle_tok.decode([tid])
                top_texts.append(text.strip())

            suggestions[pos] = top_texts

        return suggestions

    @torch.no_grad()
    def refine(
        self,
        tokens: torch.Tensor,
        prompt_len: int = 0,
        n_steps: int = 6,
        temperature: float = 0.5,
        max_rounds: int = 2,
    ) -> Tuple[torch.Tensor, dict]:
        """Full refinement: detect incoherent → oracle-biased regeneration.

        OPTIMIZED: Uses a SINGLE Qwen3 forward pass to score ALL positions
        simultaneously, then builds a bias vector at the MDLM token level
        using the oracle's per-position log-prob distribution.

        Args:
            tokens: [1, seq_len] MDLM tokens
            prompt_len: prompt tokens are preserved
            n_steps: regeneration diffusion steps per round
            temperature: regeneration temperature
            max_rounds: number of detect→refine iterations

        Returns:
            refined_tokens, stats
        """
        mask_id = self.mdlm_tok.mask_id
        pad_id = self.mdlm_tok.pad_id
        vocab_size = self.mdlm_tok.vocab_size
        seq_len = tokens.shape[1]

        # Score before
        scores_before, _ = self.score_sequence(tokens, prompt_len)

        current_tokens = tokens.clone()
        total_corrected = 0
        round_stats = []

        for round_num in range(max_rounds):
            # ── DETECT: single oracle pass ──
            scores, oracle_logits = self.score_sequence_with_logits(
                current_tokens, prompt_len
            )

            incoherent_mask, detect_stats = self._threshold_incoherent(
                scores, current_tokens, prompt_len
            )
            n_incoherent = incoherent_mask.sum().item()

            if n_incoherent == 0:
                round_stats.append({
                    "round": round_num, "n_incoherent": 0, "improved": 0
                })
                break

            # ── BUILD ORACLE BIAS (single pass, vectorized) ──
            # For each incoherent position, build a bias vector over MDLM vocab
            # using the oracle's log-prob distribution at the corresponding
            # oracle position.
            oracle_bias = self._build_mdlm_bias_from_oracle(
                current_tokens, incoherent_mask, oracle_logits, prompt_len
            )

            # ── MASK AND REGENERATE ──
            masked_tokens = torch.where(
                incoherent_mask.unsqueeze(0), mask_id, current_tokens
            )

            for step in range(n_steps):
                t_val = max(1.0 - step / n_steps, 0.01)
                t = torch.full((1,), t_val, device=DEVICE)

                logits = self.mdlm(masked_tokens, t)

                # Apply oracle bias at masked positions
                current_mask = (masked_tokens == mask_id)
                logits = logits + oracle_bias.unsqueeze(0)

                # Sample at masked positions
                temp_logits = logits / max(temperature, 0.01)
                probs = F.softmax(temp_logits.float(), dim=-1)
                confidence = probs.max(dim=-1)[0]
                confidence[~current_mask] = -1.0

                n_masked = current_mask.sum(dim=1)
                n_to_unmask = torch.clamp(
                    n_masked // max(n_steps - step, 1), min=1
                )

                for b in range(1):
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
                        pos_probs = F.softmax(pos_logits.float(), dim=-1)
                        sampled = torch.multinomial(pos_probs, 1).squeeze(-1)
                        masked_tokens[b, positions] = sampled

            # Score this round's improvement
            scores_after, _ = self.score_sequence(masked_tokens, prompt_len)
            gen_mask = generated_region(current_tokens, prompt_len, mask_id)
            lp_before = scores[gen_mask].mean().item()
            lp_after = scores_after[gen_mask].mean().item()

            round_stats.append({
                "round": round_num,
                "n_incoherent": n_incoherent,
                "lp_before": lp_before,
                "lp_after": lp_after,
                "improved": lp_after - lp_before,
            })

            current_tokens = masked_tokens
            total_corrected += n_incoherent

        # Final scores
        scores_final, _ = self.score_sequence(current_tokens, prompt_len)
        gen_mask = generated_region(tokens, prompt_len, mask_id)
        lp_initial = scores_before[gen_mask].mean().item()
        lp_final = scores_final[gen_mask].mean().item()

        return current_tokens, {
            "corrected": total_corrected > 0,
            "total_corrected": total_corrected,
            "rounds": len(round_stats),
            "round_details": round_stats,
            "mean_lp_before": lp_initial,
            "mean_lp_after": lp_final,
            "lp_improvement": lp_final - lp_initial,
        }

    @torch.no_grad()
    def score_sequence_with_logits(
        self, tokens: torch.Tensor, prompt_len: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score sequence and return oracle logits for bias building.

        Returns:
            token_scores: [seq_len] per-token log-prob
            oracle_log_probs: [oracle_seq_len-1, oracle_vocab] full distribution
        """
        all_ids = tokens[0].cpu().tolist()
        mask_id = self.mdlm_tok.mask_id
        pad_id = self.mdlm_tok.pad_id
        clean_ids = [t for t in all_ids if t not in (mask_id, pad_id)]

        _, full_text = build_token_alignment(
            clean_ids, self.mdlm_tok, [], self.oracle_tok
        )

        if not full_text.strip():
            return torch.ones(tokens.shape[1], device=DEVICE), None

        oracle_ids = self.oracle_tok.encode(full_text, add_special_tokens=False)
        if len(oracle_ids) < 2:
            return torch.ones(tokens.shape[1], device=DEVICE), None

        oracle_input = torch.tensor([oracle_ids], device=DEVICE)
        logits = self.oracle(oracle_input).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)

        actual_next = oracle_input[0, 1:]
        predicted = log_probs[0, :-1]
        token_log_probs = predicted.gather(1, actual_next.unsqueeze(1)).squeeze(1)

        # Map back to MDLM positions (same as score_sequence)
        enc = self.oracle_tok(full_text, return_offsets_mapping=True,
                              add_special_tokens=False)
        oracle_offsets = enc["offset_mapping"]

        mdlm_scores = torch.full((tokens.shape[1],), -10.0, device=DEVICE)
        current_char = 0
        oracle_idx = 0

        for seq_idx, tid in enumerate(all_ids):
            if tid in (mask_id, pad_id):
                mdlm_scores[seq_idx] = 0.0
                continue
            tok_text = self.mdlm_tok.decode([tid])
            char_start = current_char
            char_end = current_char + len(tok_text)
            current_char = char_end

            oracle_lps = []
            while oracle_idx < len(oracle_offsets):
                o_start, o_end = oracle_offsets[oracle_idx]
                if o_start >= char_end:
                    break
                if oracle_idx < len(token_log_probs):
                    oracle_lps.append(token_log_probs[oracle_idx].item())
                oracle_idx += 1

            if oracle_lps:
                mdlm_scores[seq_idx] = sum(oracle_lps) / len(oracle_lps)
            else:
                mdlm_scores[seq_idx] = 0.0

        return mdlm_scores, log_probs[0, :-1]  # [oracle_seq-1, oracle_vocab]

    def _threshold_incoherent(
        self, scores, tokens, prompt_len
    ) -> Tuple[torch.Tensor, dict]:
        """Threshold-based detection of incoherent positions."""
        mask_id = self.mdlm_tok.mask_id
        pad_id = self.mdlm_tok.pad_id
        generated_mask = torch.ones(tokens.shape[1], device=DEVICE, dtype=torch.bool)
        generated_mask[:prompt_len] = False
        generated_mask |= (tokens[0] == mask_id) | (tokens[0] == pad_id)

        gen_scores = scores[generated_mask]
        if gen_scores.numel() == 0:
            return torch.zeros_like(generated_mask), {"n_incoherent": 0}

        mean_lp = gen_scores.mean()
        std_lp = gen_scores.std()
        threshold = mean_lp - self.surprise_threshold
        if std_lp > 0.01:
            threshold = mean_lp - self.surprise_threshold * std_lp

        incoherent = (scores < threshold) & generated_mask

        n_incoherent = incoherent.sum().item()
        max_correct = int(self.max_correction_rate * generated_mask.sum().item())
        if n_incoherent > max_correct:
            scores_np = scores.cpu().numpy()
            incoh_indices = incoherent.cpu().nonzero().squeeze(-1)
            incoh_scores = scores_np[incoh_indices]
            sorted_order = np.argsort(incoh_scores)
            keep = set(incoh_indices[sorted_order[:max_correct]].tolist())
            incoherent = torch.zeros_like(generated_mask)
            for idx in keep:
                incoherent[idx] = True
            n_incoherent = max_correct

        return incoherent, {
            "n_incoherent": n_incoherent,
            "mean_lp": mean_lp.item(),
            "threshold": threshold.item()
        }

    @torch.no_grad()
    def _build_mdlm_bias_from_oracle(
        self, tokens, incoherent_mask, oracle_log_probs, prompt_len
    ) -> torch.Tensor:
        """Build MDLM-vocab bias vector from oracle's log-prob distribution.

        For each incoherent position, the oracle's prediction distribution
        at the corresponding oracle position is mapped to MDLM tokens.

        Strategy: for each incoherent MDLM position, decode all MDLM vocab
        tokens and compute their oracle log-prob as the oracle's prediction
        for what comes at this point. This is expensive if done naively,
        so we use a pre-computed mapping.

        SIMPLIFIED (fast): Use oracle's top-k tokens, decode to text,
        encode in MDLM vocab, and boost those.
        """
        seq_len = tokens.shape[1]
        vocab_size = self.mdlm_tok.vocab_size
        bias = torch.zeros(seq_len, vocab_size, device=DEVICE)

        if oracle_log_probs is None:
            return bias

        # Get oracle's top-k tokens at each position
        k = 20
        topk_vals, topk_ids = oracle_log_probs.topk(k, dim=-1)  # [oracle_seq, k]

        # Decode MDLM text to find oracle position alignment
        all_ids = tokens[0].cpu().tolist()
        mask_id = self.mdlm_tok.mask_id
        pad_id = self.mdlm_tok.pad_id

        # Rebuild offset mapping
        clean_ids = [t for t in all_ids if t not in (mask_id, pad_id)]
        _, full_text = build_token_alignment(
            clean_ids, self.mdlm_tok, [], self.oracle_tok
        )
        if not full_text.strip():
            return bias

        enc = self.oracle_tok(full_text, return_offsets_mapping=True,
                              add_special_tokens=False)
        oracle_offsets = enc["offset_mapping"]

        # For each incoherent MDLM position, find the oracle position and
        # boost MDLM tokens that match oracle's top-k
        current_char = 0
        oracle_idx = 0
        incoherent_positions = incoherent_mask.cpu().nonzero(as_tuple=True)[0].tolist()

        for seq_idx, tid in enumerate(all_ids):
            if tid in (mask_id, pad_id):
                continue
            tok_text = self.mdlm_tok.decode([tid])
            char_start = current_char
            char_end = current_char + len(tok_text)
            current_char = char_end

            # Find oracle position for this MDLM token
            while oracle_idx < len(oracle_offsets):
                o_start, o_end = oracle_offsets[oracle_idx]
                if o_start >= char_end:
                    break
                oracle_idx += 1

            oracle_pos = oracle_idx - 1 if oracle_idx > 0 else 0

            if seq_idx not in incoherent_positions:
                continue
            if oracle_pos >= topk_ids.shape[0]:
                continue

            # Get oracle's top-k token texts at this position
            for ki in range(k):
                oracle_tok_id = topk_ids[oracle_pos, ki].item()
                oracle_lp = topk_vals[oracle_pos, ki].item()
                oracle_text = self.oracle_tok.decode([oracle_tok_id]).strip()
                if not oracle_text:
                    continue

                # Encode in MDLM vocab
                try:
                    mdlm_ids = self.mdlm_tok.encode(oracle_text, add_special=False)
                    for mid in mdlm_ids:
                        if 0 <= mid < vocab_size:
                            # Boost proportional to oracle's log-prob
                            boost = self.correction_strength * max(oracle_lp, -5) / (-5)
                            bias[seq_idx, mid] += boost
                except Exception:
                    pass

        return bias


def generated_region(tokens, prompt_len, mask_id):
    """Helper: return mask of generated (non-prompt, non-special) positions."""
    pad_id = 0
    mask = torch.ones(tokens.shape[1], device=tokens.device, dtype=torch.bool)
    mask[:prompt_len] = False
    mask |= (tokens[0] == mask_id) | (tokens[0] == pad_id)
    return mask
