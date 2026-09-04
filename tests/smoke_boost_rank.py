"""Smoke test: oracle boost is aligned with real rank (best candidate = max boost).

Repro of FINDINGS.md fix/oracle-guidance item 1 (sev 7): synthetic logits with
20 candidates ordered by log-prob; after the boost, the BEST candidate must
receive the LARGEST increment (monotonically decreasing with rank).

Runs the REAL _build_mdlm_bias_from_oracle on CPU with a stub oracle/tokenizers
(no Qwen3 download, no MDLM checkpoint).
"""
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ar_oracle_hrm import AROracleHRM  # noqa: E402


class StubOracleTok:
    """Minimal HF-like tokenizer: identity char-level, no special handling."""
    vocab_size = 200

    def encode(self, text, add_special_tokens=False):
        return [min(ord(c), 199) for c in text]

    def decode(self, ids, **kw):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return "".join(chr(int(i)) for i in ids)

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class StubMDLMTok:
    """Minimal MDLM BPE-like tokenizer: 2 chars per token."""
    vocab_size = 100
    mask_id = 0
    pad_id = 1
    bos_id = 2
    eos_id = 3

    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return "".join(chr(int(i) + 40) for i in ids)  # printable, 1 char per id

    def encode(self, text, add_special=False):
        return [min(ord(c) - 40, 99) for c in text]


def main():
    # Build AROracleHRM without calling __init__ (no Qwen3 load)
    hrm = AROracleHRM.__new__(AROracleHRM)
    hrm.mdlm_tok = StubMDLMTok()
    hrm.oracle_tok = StubOracleTok()
    hrm.surprise_threshold = 2.0
    hrm.correction_strength = 2.0
    hrm.max_correction_rate = 0.3

    # Sequence of 6 MDLM tokens, 1 flagged incoherent position (index 4)
    tokens = torch.tensor([[10, 11, 12, 13, 14, 15]])
    incoherent = torch.zeros(6, dtype=torch.bool)
    incoherent[4] = True

    # Oracle log-probs: 20 ordered candidates at the aligned oracle position.
    # Candidate i has lp = -0.25*i  → rank 0 is best (lp=-0.0), rank 19 worst (lp=-4.75).
    # StubOracleTok maps 1 char -> 1 oracle token; StubMDLMTok maps 1 id -> 1 char,
    # so oracle position aligns 1:1 and each oracle token id c decodes to chr(c).
    n_cand = 20
    lps = [-0.25 * i for i in range(n_cand)]
    # place at the oracle position that MDLM token 4 aligns to (char index 4)
    oracle_lp = torch.full((10, 200), -20.0)  # 10 oracle positions, 200 vocab
    for i, lp in enumerate(lps):
        oracle_lp[4, 100 + i] = lp  # candidate token ids 100..119

    bias = hrm._build_mdlm_bias_from_oracle(tokens, incoherent, oracle_lp, prompt_len=0)

    # Collect boosts at the incoherent position for each candidate's MDLM id.
    # StubMDLMTok.encode(chr(100+i)) = [min(60+i,99)] (chr(c) -> c-40)
    boosts = []
    for i in range(n_cand):
        mid = min(100 + i - 40, 99)
        boosts.append((i, lps[i], bias[4, mid].item()))

    print(f"{'rank':>4} {'oracle_lp':>10} {'boost':>8}")
    for rank, lp, b in boosts:
        print(f"{rank:>4} {lp:>10.2f} {b:>8.3f}")

    best_boost = boosts[0][2]
    worst_boost = boosts[-1][2]
    # Monotonic non-increasing check
    mono = all(boosts[i][2] >= boosts[i + 1][2] - 1e-6 for i in range(len(boosts) - 1))

    print()
    print(f"best (lp={lps[0]:.2f})  boost = {best_boost:.3f}")
    print(f"worst (lp={lps[-1]:.2f}) boost = {worst_boost:.3f}")
    print(f"monotonically decreasing with rank: {mono}")

    # Assertions (test criteria from FINDINGS.md):
    # 1) best candidate receives the maximum boost of all candidates
    assert best_boost == max(b for _, _, b in boosts), "best candidate not max boost"
    # 2) best boost ~= strength * (1 + lp/5) = 2.0 * (1 - 0) = 2.0
    assert abs(best_boost - 2.0) < 1e-5, f"best boost {best_boost} != strength 2.0"
    # 3) strictly less for the worst
    assert worst_boost < best_boost - 0.5, "worst not meaningfully lower"
    # 4) monotone
    assert mono, "boost not monotonic in rank"

    print("\nSMOKE TEST PASSED: boost aligned with rank (best candidate gets max boost).")


if __name__ == "__main__":
    main()
