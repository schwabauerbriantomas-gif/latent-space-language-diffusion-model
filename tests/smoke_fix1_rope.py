"""Smoke test Fix 1 — RoPE tile/cat (sev 8).

Repite el test de r1_arch (propiedad de posición relativa con q,k FIJOS):
  A) Norma de q preservada tras apply_rope en TODAS las posiciones.
  B) <R_m q, R_n k> = q^T R_{n-m} k depende SOLO de (n-m) — spread 0.
  C) apply_rope coincide exactamente con la rotación 2D canónica por pares.
"""
import sys, torch
sys.path.insert(0, "src")
from mdlm_v4_optimized import precompute_rope, apply_rope, rotate_half

torch.manual_seed(0)
head_dim, seq_len = 64, 128
base = 10000.0
cos, sin = precompute_rope(head_dim, seq_len, base)
print(f"cos/sin shape: {tuple(cos.shape)} (esperado ({seq_len}, {head_dim}))")

# --- A) Preservación de norma ---
q = torch.randn(1, 1, seq_len, head_dim)
q_rot = apply_rope(q, cos, sin)
dev = (q_rot.norm(dim=-1) - q.norm(dim=-1)).abs().max().item()
print(f"[A] Desviación max de norma: {dev:.3e}  (r1 midió 26.6% con repeat_interleave)")
assert dev < 1e-4, "norma no preservada"

# --- B) Propiedad de posición relativa (mismo q y k en toda posición) ---
q0 = torch.randn(head_dim)
k0 = torch.randn(head_dim)
scores = {}
for m in range(seq_len):
    for n in range(seq_len):
        if n > m:  # solo triángulo superior basta
            scores[(m, n)] = torch.dot(apply_rope(q0.view(1,1,-1,head_dim)[:,:,0] , cos[m:m+1], sin[m:m+1])[0,0,0],
                                       apply_rope(k0.view(1,1,-1,head_dim)[:,:,0] , cos[n:n+1], sin[n:n+1])[0,0,0]).item()
# spread por diagonal: todos los pares con la misma diferencia n-m
worst = 0.0
for d in range(1, 60):
    vals = [scores[(m, m + d)] for m in range(0, seq_len - d)]
    spread = max(vals) - min(vals)
    worst = max(worst, spread)
print(f"[B] Spread max de <R_m q, R_n k> con n-m fijo: {worst:.3e}  (r1 midió 2.05 con repeat_interleave)")
assert worst < 1e-3, "propiedad de posición relativa violada"

# --- C) Igualdad con rotación 2D canónica por pares ---
inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
def rot_exact(x, m):
    theta = m * inv
    x1, x2 = x[: head_dim // 2], x[head_dim // 2:]
    return torch.cat([x1 * theta.cos() - x2 * theta.sin(),
                      x1 * theta.sin() + x2 * theta.cos()])
for m in (0, 1, 10, 127):
    q_tile = q0.view(1, 1, 1, head_dim).expand(1, 1, seq_len, head_dim).contiguous()
    d = (apply_rope(q_tile, cos, sin)[0, 0, m] - rot_exact(q0, m)).abs().max().item()
    assert d < 1e-5, f"apply_rope != rotación canónica @pos{m} (err {d:.2e})"
print("[C] apply_rope == rotación 2D canónica en posiciones 0,1,10,127 (err < 1e-5)")

print("\n✓ SMOKE FIX 1 PASSED — RoPE canónico (tile/cat)")
