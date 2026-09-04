"""Smoke test for fix 2 (off-by-one log-prob attribution, sev 4).

The oracle stack (ar_oracle_hrm.score_sequence, score_sequence_with_logits,
hybrid_speculative.verify_and_refine) gathers teacher-forcing log-probs and
then maps them back to character ranges. The raw gather

    shifted[i] = log_probs[0, i, oracle_ids[i+1]]

is the log-prob of oracle token i+1 stored at index i, but the char-range
mapping that follows reads index j as the score of oracle token j — so every
token was attributed the log-prob of the token AFTER it.

This test builds a tiny fake oracle whose logits make the off-by-one visible
and verifies the fixed attribution: the log-prob assigned to the char range
of token j must equal the oracle's log-prob of token j (given tokens 0..j-1).

CPU-only, no models downloaded. Run: python3 tests/smoke_off_by_one.py
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ar_oracle_hrm import AROracleHRM  # noqa: E402

VOCAB = 128  # small vocab; covers ASCII text so the -100 distractors matter


def _target_lp(j: int) -> float:
    """Distinct per-position target log-prob: strictly decreasing in j."""
    return -0.5 * (j + 1)


class FakeOracleTok:
    """Byte-level 'tokenizer': each character is one oracle token."""

    vocab_size = VOCAB
    eos_token_id = VOCAB - 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def __call__(self, text, return_offsets_mapping=False,
                 add_special_tokens=False):
        ids = self.encode(text)
        offsets = [(i, i + 1) for i in range(len(text))]
        return {"input_ids": ids, "offset_mapping": offsets}


class FakeOracle(torch.nn.Module):
    """Oracle where the log-prob of oracle token j (given 0..j-1) is a
    known strictly-decreasing function _target_lp(j).

    Row i predicts token i+1; we set its target logit so that after
    log_softmax over [target, -100 distractors] the true lp of token j=i+1
    equals _target_lp(j).  Because the distractors are constant, we can
    solve exactly: lp = T - logsumexp([T, -100 x (V-1)]) ⇒ choose T
    numerically below.
    """

    def forward(self, input_ids):
        seq = input_ids.shape[1]
        logits = torch.full((1, seq, VOCAB), -100.0)
        for i in range(seq - 1):
            j = i + 1
            logits[0, i, input_ids[0, j]] = self._solve_T(_target_lp(j))
        # last row: predict the last token itself (never used after gather,
        # but keep the tensor well-formed)
        logits[0, seq - 1, input_ids[0, seq - 1]] = self._solve_T(
            _target_lp(seq)
        )
        return type("Out", (), {"logits": logits})()

    @staticmethod
    def _solve_T(desired_lp: float) -> float:
        if desired_lp >= -1e-9:
            # Cannot realize lp ~ 0 with one dominant logit among -100
            # distractors (best achievable ≈ -6.8e-86). Clamp to a tiny
            # negative value indistinguishable from 0 at test tolerance.
            desired_lp = -1e-6
        T = desired_lp
        for _ in range(200):
            terms = [T] + [-100.0] * (VOCAB - 1)
            m = max(terms)
            lse = math.log(sum(math.exp(t - m) for t in terms)) + m
            err = (T - lse) - desired_lp
            T -= err  # d(err)/dT ≈ 1 - softmax(target) ≈ 1
        return T


class FakeMdlmTok:
    mask_id = 0
    pad_id = 1
    bos_id = 2
    eos_id = 3

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def reference_lp(j: int):
    """Expected oracle log-prob of token j under FakeOracle."""
    if j < 1:
        return None  # token 0 has no conditioning context
    return _target_lp(j)


def make_hrm():
    hrm = AROracleHRM.__new__(AROracleHRM)  # skip __init__ (no model load)
    hrm.oracle = FakeOracle()
    hrm.oracle_tok = FakeOracleTok()
    hrm.mdlm_tok = FakeMdlmTok()
    hrm.surprise_threshold = 2.0
    hrm.correction_strength = 2.0
    hrm.max_correction_rate = 0.3
    return hrm


TOK6 = [ord("a"), ord("b"), ord("c"), ord("d"), ord("e"), ord("f")]


def check_score_sequence():
    """AROracleHRM.score_sequence must attribute token j's lp to token j."""
    hrm = make_hrm()
    tokens = torch.tensor([TOK6])
    scores, stats = hrm.score_sequence(tokens, prompt_len=0)

    # Oracle tokens are one per char (a..f); MDLM token k = 1 char, so
    # score[k] should equal ref_lp(k) for k >= 1.
    tol = 1e-4
    for k in range(1, 6):
        expected = reference_lp(k)
        got = scores[k].item()
        assert abs(got - expected) < tol, (
            f"MDLM token {k}: expected lp {expected:.4f}, got {got:.4f}"
        )
    # Token 0 gets neutral fill = shifted[0] = lp of token 1.
    assert abs(scores[0].item() - reference_lp(1)) < tol
    # Discriminating check: the BUGGY (unshifted) attribution would give
    # shifted[k] = lp of token k+1 — strictly different because
    # _target_lp is strictly decreasing.
    for k in range(1, 6):
        buggy_value = reference_lp(k + 1)
        assert abs(scores[k].item() - buggy_value) > 1e-3, (
            f"token {k}: fixed score matches buggy next-token attribution — "
            "test is not discriminating"
        )
    # token_log_probs must now cover ALL oracle tokens (old code dropped the
    # last one, leaving it at the -10.0 sentinel → score 0.0 in the mapping).
    assert all(s.item() != 0.0 for s in scores), (
        "last oracle token fell back to the 0.0 sentinel — length mismatch"
    )
    return stats


def check_score_sequence_with_logits():
    """Same attribution check for score_sequence_with_logits."""
    hrm = make_hrm()
    tokens = torch.tensor([TOK6])
    scores, oracle_log_probs = hrm.score_sequence_with_logits(
        tokens, prompt_len=0
    )
    assert oracle_log_probs is not None
    tol = 1e-4
    for k in range(1, 6):
        expected = reference_lp(k)
        got = scores[k].item()
        assert abs(got - expected) < tol, (
            f"with_logits MDLM token {k}: expected {expected:.4f}, "
            f"got {got:.4f}"
        )


def check_hybrid():
    """hybrid_speculative token_lps fix: verify the shifted expression used
    in verify_and_refine attributes token j's lp to index j, and that the
    pre-fix unshifted gather would not."""
    oracle_tok = FakeOracleTok()
    oracle = FakeOracle()
    text = "abcdef"
    oracle_ids = oracle_tok.encode(text)
    oracle_input = torch.tensor([oracle_ids])
    logits = oracle(oracle_input).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    actual_next = oracle_input[0, 1:]
    predicted = log_probs[0, :-1]
    shifted = predicted.gather(1, actual_next.unsqueeze(1)).squeeze(1)
    token_lps = torch.cat([shifted[:1], shifted])

    tol = 1e-4
    for j in range(1, len(text)):
        assert abs(token_lps[j].item() - reference_lp(j)) < tol, (
            f"hybrid token {j}: expected {reference_lp(j):.4f}, "
            f"got {token_lps[j].item():.4f}"
        )
        if j < len(shifted):  # shifted has seq-1 entries (no score for token 0)
            assert abs(shifted[j].item() - reference_lp(j)) > 1e-3, (
                "unshifted gather equals the true per-token lp — "
                "non-discriminating"
            )

    # Also verify the module source actually contains the shift (guards
    # against the test passing while src stays unfixed).
    src = (REPO / "src" / "hybrid_speculative.py").read_text()
    assert "torch.cat([shifted[:1], shifted])" in src, (
        "hybrid_speculative.py missing the alignment shift"
    )
    src2 = (REPO / "src" / "ar_oracle_hrm.py").read_text()
    assert src2.count("torch.cat([shifted[:1], shifted])") == 2, (
        "ar_oracle_hrm.py must fix both score_sequence and "
        "score_sequence_with_logits"
    )


if __name__ == "__main__":
    stats = check_score_sequence()
    print(f"[ok] score_sequence attribution correct "
          f"(mean_lp={stats['mean_log_prob']:.4f}, "
          f"oracle_tokens={stats['oracle_tokens']})")
    check_score_sequence_with_logits()
    print("[ok] score_sequence_with_logits attribution correct")
    check_hybrid()
    print("[ok] hybrid_speculative token_lps shift correct")
    print("SMOKE PASS: off-by-one log-prob attribution fixed (3 sites)")
