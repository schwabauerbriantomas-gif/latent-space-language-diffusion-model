"""Smoke test Fix 2: holdout leakage removed from train_ar.py.

Checks (CPU, no data files needed):
  A. The split logic used by train_ar builds a training dataset that does
     NOT contain any holdout sequence (by index and by content).
  B. The holdout size is capped at 1000 and >= 1 for any corpus size.
  C. train_ar.py still parses (compile) and the holdout block appears
     exactly once (no stale duplicate).
"""
import sys, math, re
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

REPO = Path("/home/schwabauer/swarm_lsd/wt_eval")
sys.path.insert(0, str(REPO / "scripts"))

torch.manual_seed(0)

# ── Replicate the fixed split logic exactly ──
N = 5000
tokens_tensor = torch.randint(4, 500, (N, 16))  # unique-ish sequences

n_holdout = min(1000, max(1, len(tokens_tensor) // 100))
holdout = tokens_tensor[:n_holdout].clone()
train_tensor = tokens_tensor[n_holdout:]
dataset = TensorDataset(train_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

# A. no overlap by index construction
assert n_holdout == 50, f"expected 50 (1% of 5000), got {n_holdout}"
assert train_tensor.shape[0] == N - n_holdout
# no overlap by content: every batch row must differ from every holdout row
hold_set = set(map(tuple, holdout.tolist()))
leaks = 0
seen = 0
for batch in loader:
    for row in batch[0].tolist():
        seen += 1
        if tuple(row) in hold_set:
            leaks += 1
assert leaks == 0, f"{leaks} holdout sequences leaked into training loader"
print(f"A. 0/{seen} training rows overlap the {n_holdout}-seq holdout OK")

# B. cap behavior
for n, expect in [(100, 1), (1000, 10), (5000, 50), (500000, 1000)]:
    tt = torch.randint(4, 500, (n, 4))
    nh = min(1000, max(1, len(tt) // 100))
    assert nh == expect, f"n={n}: holdout {nh} != {expect}"
    assert 1 <= nh < n, f"n={n}: holdout must be nonempty proper subset"
print("B. holdout size = min(1000, max(1, N//100)) across corpus sizes OK")

# C. source inspection: exactly one holdout block, and it feeds the dataset
src = (REPO / "scripts" / "train_ar.py").read_text()
occurrences = len(re.findall(r"holdout\s*=\s*tokens_tensor\[", src))
assert occurrences == 1, f"expected exactly 1 holdout assignment, found {occurrences}"
assert "dataset = TensorDataset(train_tensor)" in src, \
    "training dataset must be built from train_tensor, not tokens_tensor"
assert re.search(r"holdout\s*=\s*tokens_tensor\[:n_holdout\]", src)
assert re.search(r"train_tensor\s*=\s*tokens_tensor\[n_holdout:\]", src)
compile(src, "train_ar.py", "exec")  # still valid Python
print("C. train_ar.py: single holdout split, dataset excludes it, compiles OK")

print("\nALL FIX-2 SMOKE TESTS PASSED")
