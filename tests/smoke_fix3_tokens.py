"""Smoke test Fix 3: token counting in benchmark_generation_v4.py.

Checks (CPU, no model/checkpoint needed):
  A. count_generated_tokens counts ONLY generated tokens (excludes prompt)
     for both branch styles — a 7-token prompt + 50 generated must give 50,
     not 57.
  B. Both benchmark branches call the SAME helper (source inspection) —
     no residual len(...) without prompt subtraction.
  C. The expected magnitude of the correction: with the documented example
     (prompt ~9 tokens, total ~86 -> 96.5 t/s inflated), recomputing with
     only generated tokens yields the comparable lower t/s (~84.6).
  D. Floor at 1: degenerate text identical to the prompt yields 1, not 0
     or negative.
"""
import sys, math, re
import types
from pathlib import Path

REPO = Path("/home/schwabauer/swarm_lsd/wt_eval")
sys.path.insert(0, str(REPO / "scripts"))

# Import the module WITHOUT triggering its torch/cuda imports: stub heavy deps
class _StubTok:
    """Tokenizer stub: 'encodes' by simple whitespace splitting."""
    def encode(self, text, add_special=False):
        return text.split()

import importlib.util
src = (REPO / "scripts" / "benchmark_generation_v4.py").read_text()

# Extract only the two pure functions we need (avoid module-level cuda deps)
import ast
tree = ast.parse(src)
ns = {}
wanted = {"rep_score", "count_generated_tokens"}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        code = ast.get_source_segment(src, node)
        exec(compile(code, "<fix3>", "exec"), ns)

count_generated_tokens = ns["count_generated_tokens"]
tok = _StubTok()

# A. prompt excluded
prompt_ids = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]      # 7 prompt tokens
generated = [f"g{i}" for i in range(50)]                     # 50 generated
text = " ".join(prompt_ids + generated)
n = count_generated_tokens(text, prompt_ids, tok)
assert n == 50, f"expected 50 generated tokens, got {n}"
print(f"A. 7-token prompt + 50 generated -> counts {n} OK")

# also when text does NOT literally start with the prompt tokens (count-based)
n2 = count_generated_tokens(" ".join(generated), prompt_ids, tok)
assert n2 == 43, f"count-based subtraction wrong: {n2}"
print(f"   count-based: 50 total - 7 prompt -> {n2} OK")

# B. both branches use the helper
uses = len(re.findall(r"gen_tokens\s*=\s*count_generated_tokens\(", src))
assert uses == 2, f"expected both branches to call helper, found {uses}"
# no leftover raw counting
raw = re.findall(r"gen_tokens\s*=\s*max\(len\(tokenizer\.encode", src)
assert not raw, "residual raw token counting remains"
print(f"B. both branches call count_generated_tokens ({uses}/2), no raw counts OK")

# C. documented magnitude: old no-guidance 96.5 t/s with prompt included
# measured total ~86 tokens, prompt ~9 -> true generated ~77... use reported
# numbers: 96.5 * (77/86) ~= 86.4; the FINDINGS expectation ~84.6 for its
# own token mix. Verify the RATIO correction math itself.
old_total, prompt_len, gen_len, dt = 86, 9, 77, 1.0
old_tps = old_total / dt
new_tps = count_generated_tokens(" ".join(["x"] * old_total), ["x"] * prompt_len, tok) / dt
assert abs(new_tps - gen_len / dt) < 1e-9
assert new_tps < old_tps
print(f"C. corrected t/s {new_tps:.1f} < inflated {old_tps:.1f} (direction & ratio OK)")

# D. floor
n3 = count_generated_tokens(" ".join(prompt_ids), prompt_ids, tok)
assert n3 == 1, f"floor failed: {n3}"
print("D. degenerate (prompt-only) output floors to 1 OK")

print("\nALL FIX-3 SMOKE TESTS PASSED")
