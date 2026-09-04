"""Smoke test Fix 5 — docstrings de mdlm_v4_optimized.py (sev 1-2).

Dos correcciones documentales verificadas numéricamente:
  A) Schedule coseno: el docstring decía "more tokens early, fewer late"
     (INVERTIDO). cumfrac = 0.5*(1-cos(pi*s/S)) es sigmoide: sus deltas
     son mínimos en los extremos y máximos al medio. Se verifica con el
     array delta_frac real del código (recomputado acá).
  B) Header de parámetros: decía ~201M / vocab 16K. El modelo por defecto
     (vocab_size=10_000) tiene exactamente 182,795,264 params = 182.8M.
     Se instancia el modelo default y se cuenta.
Y se chequea que el archivo ya no contiene las afirmaciones incorrectas.
"""
import re
import sys
import torch

sys.path.insert(0, "src")

SRC = "src/mdlm_v4_optimized.py"
text = open(SRC).read()

# ── A) Schedule coseno: forma real de delta_s ─────────────────────────────
S = 8
steps = torch.arange(0, S + 1, dtype=torch.float32)
cumfrac = 0.5 * (1.0 - torch.cos(torch.pi * steps / S))
delta = (cumfrac[1:] - cumfrac[:-1]).tolist()

print(f"delta_frac (S={S}): {[round(d, 4) for d in delta]}")
assert delta[0] < delta[S // 2], "el primer paso debe desenmascarar MENOS que el central"
assert delta[-1] < delta[S // 2], "el último paso debe desenmascarar MENOS que el central"
assert abs(delta[S // 2] - max(delta)) < 1e-9, "el máximo está al medio, no al inicio"
print(f"  primero={delta[0]:.4f} < medio={delta[S//2]:.4f} > último={delta[-1]:.4f} ✓ (pocos temprano/tardío, más al medio)")

# El docstring ya no afirma lo invertido ni usa el valor viejo:
assert "more tokens early, fewer late" not in text, \
    "docstring aún contiene la descripción invertida del schedule"
assert re.search(r"[Ff][Ee][Ww] tokens unmasked early", text), \
    "docstring debe describir el schedule real (few early)"
print("  docstring sin la frase invertida ✓")

# ── B) Parámetros reales del modelo ──────────────────────────────────────
from mdlm_v4_optimized import MDLMConfig, MDLMBPEV4

# El modelo ENTRENADO usa el vocab del tokenizer (10K) → 182.8M (r8 midió exacto)
cfg10k = MDLMConfig(vocab_size=10_000)
model = MDLMBPEV4(cfg10k)
n10k = sum(p.numel() for p in model.parameters())
print(f"\nvocab=10K (modelo entrenado): {n10k:,} ({n10k/1e6:.1f}M)")
assert n10k == 182_795_264, f"esperado 182,795,264, got {n10k}"

# El default del config es 16K → 188.9M (no ~201M como decía el header)
cfg16k = MDLMConfig()
n16k = sum(p.numel() for p in MDLMBPEV4(cfg16k).parameters())
print(f"vocab=16K (default config):   {n16k:,} ({n16k/1e6:.1f}M)")
assert cfg16k.vocab_size == 16_000 and n16k == 188_939_264

assert "~201M" not in text, "header aún cita ~201M"
assert "182.8M" in text and "182,795,264" in text, "header debe citar 182.8M exacto"
print("  header cita 182.8M/10K (entrenado) y 188.9M/16K (default), sin ~201M ✓")

# Sanity: el módulo sigue importando y ejecutando forward tras la edición
tok = torch.randint(4, cfg10k.vocab_size, (1, 16))
t = torch.rand(1)
logits = model(tok, t)
assert logits.shape == (1, 16, cfg10k.vocab_size)
print("  forward post-edición OK ✓")

print("\n✓ SMOKE FIX 5 PASSED — docstrings corregidos y verificados numéricamente")
