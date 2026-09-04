"""Smoke test Fix 3 — repetition penalty no invierte el signo (sev 6).

Caso del finding: logit -3 con penalty 1.3 debe quedar MÁS bajo, no más alto.
(Con la versión divide-by-penalty: -3 / 1.3 = -2.31 → SUBIÓ.)
Fix: resta uniforme de (penalty-1) sobre tokens usados — siempre baja.
Se verifican las 3 copias: logit_guidance.py, logit_guidance_v4.py, inference_v4.py.
"""
import sys, torch
sys.path.insert(0, "src")
from logit_guidance import apply_repetition_penalty as rp_v3
from logit_guidance_v4 import apply_repetition_penalty as rp_v4
from inference_v4 import apply_repetition_penalty_batch as rp_inf

PEN = 1.3
PEN_SHIFT = abs(PEN - 1.0)  # 0.3

def check(name, fn, uses_special_ids=False):
    b, s, v = 1, 3, 6
    logits = torch.tensor([[[-3.0, -0.5, 0.0, 2.0, 5.0, -1.0]]])   # mezcla de signos
    tokens = torch.tensor([[5, 5, 5]])                              # token 5 = "usado"
    if uses_special_ids:
        out = fn(logits, tokens, penalty=PEN, special_ids={0, 1, 2, 3})
    else:
        out = fn(logits, tokens, penalty=PEN, special_mask=None)
    expected_used = logits.clone()
    expected_used[0, 0, 5] -= PEN_SHIFT
    err = (out - expected_used).abs().max().item()
    print(f"=== {name} (penalty={PEN}) ===")
    for i, (before, after) in enumerate(zip(logits[0, 0].tolist(), out[0, 0].tolist())):
        mark = "USADO" if i == 5 else "     "
        direction = "↓" if after < before else ("=" if after == before else "↑")
        print(f"  token {i} {mark}: logit {before:+.2f} → {after:+.2f} {direction}")
    assert out[0, 0, 5].item() < logits[0, 0, 5].item(), \
        f"{name}: logit usado no bajó ({logits[0,0,5]} → {out[0,0,5]})"
    assert err < 1e-6, f"{name}: shift inesperado (err {err:.2e})"
    print(f"  token usado baja exactamente {PEN_SHIFT}; el resto intacto ✓\n")

check("logit_guidance.apply_repetition_penalty", rp_v3, uses_special_ids=True)
check("logit_guidance_v4.apply_repetition_penalty", rp_v4)
check("inference_v4.apply_repetition_penalty_batch", rp_inf)

# Caso A/B del finding: -3 con penalty 1.3
# tokens=[[3,3,3,3]] → el token USADO es el 3 (todos los logits son -3.0).
# Resta uniforme pen=|1.3-1|=0.3: usado 3 → -3.3 (más bajo); no usado 0 → -3.0 intacto.
out = rp_v4(torch.full((1, 1, 4), -3.0), torch.tensor([[3, 3, 3, 3]]), penalty=PEN)
print(f"Caso del finding: logit -3, penalty {PEN} → token usado(3): {out[0,0,3].item():+.4f} "
      f"(antes del fix: {-3/1.3:+.4f}, SUBÍA)")
assert abs(out[0, 0, 3].item() - (-3.0 - PEN_SHIFT)) < 1e-6, \
    f"token usado: esperado {-3.0 - PEN_SHIFT}, got {out[0,0,3].item()}"
assert abs(out[0, 0, 0].item() - (-3.0)) < 1e-6, \
    f"token no usado debe quedar intacto: esperado -3.0, got {out[0,0,0].item()}"

# Invariante de probabilidad: softmax del token usado debe disminuir
scores = torch.tensor([2.0, 1.0])             # token 1 usado (1-D)
out2 = rp_v4(scores.view(1, 1, 2), torch.tensor([[1]]), penalty=PEN).view(2)
p_before = torch.softmax(scores, -1)[1].item()
p_after = torch.softmax(out2, -1)[1].item()
print(f"softmax(token usado): {p_before:.4f} → {p_after:.4f} (menor ✓)")
assert p_after < p_before

print("\n✓ SMOKE FIX 3 PASSED — penalización siempre reduce el logit usado")
