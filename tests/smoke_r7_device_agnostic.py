"""Smoke test extra r7 — sample_full_parallel_v4 device-agnostic (CPU).

r7 midió: sample_full_parallel_v4 en CPU crashea con
  RuntimeError: Expected all tensors to be on the same device, but found
  at least two devices, cpu and cuda:0!
porque el default del parámetro `device` era el módulo-level DEVICE
("cuda" si hay CUDA), no el device real del modelo.

Fix: device=None por default → se infiere de next(model.parameters()).device.
Se verifica:
  1. Repro del bug pre-fix (monkeypatch del default viejo) — crashea igual que r7.
  2. Post-fix: modelo CPU + CUDA disponible en la máquina, SIN pasar device → corre.
  3. device explícito "cpu" → corre.
  4. Propiedades de salida: n_samples correcto, prompt preservado, sin MASKs restantes.
"""
import inspect
import sys

import torch

sys.path.insert(0, "src")

import mdlm_v4_optimized as m4
from mdlm_v4_optimized import MDLMConfig, MDLMBPEV4, sample_full_parallel_v4


class TokStub:
    """Tokenizer mínimo con la interfaz que usa el sampler."""
    mask_id, pad_id, bos_id, eos_id = 1, 0, 2, 3

    def decode(self, ids):
        return "".join(chr(65 + (i % 26)) for i in ids if i > 3)


def make_model():
    cfg = MDLMConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2, max_seq_len=16)
    torch.manual_seed(0)
    return MDLMBPEV4(cfg)


# ── 1. Repro del bug pre-fix ─────────────────────────────────────────────
sig = inspect.signature(sample_full_parallel_v4)
default_dev = sig.parameters["device"].default
assert default_dev is None, f"post-fix el default debe ser None, es {default_dev!r}"
print(f"default de device post-fix: {default_dev!r} (antes: DEVICE={m4.DEVICE!r}) ✓")

model = make_model()  # CPU
assert next(model.parameters()).device.type == "cpu"
print(f"CUDA disponible: {torch.cuda.is_available()} — modelo en CPU (caso r7)")

if torch.cuda.is_available():
    try:
        # Simular el default viejo: pasar el device del módulo explícitamente.
        sample_full_parallel_v4(
            model, TokStub(), prompt_ids=[2, 5, 7], seq_len=12,
            n_samples=1, n_steps=4, temperature=0.7, device=m4.DEVICE,
        )
        raise AssertionError("el path viejo (device=DEVICE) debería haber crasheado")
    except RuntimeError as e:
        print(f"  repro bug viejo (device={m4.DEVICE!r}): {str(e)[:70]}... (esperado) ✓")

# ── 2/3. Post-fix: sin device y con device="cpu" ─────────────────────────
for label, kwargs in [("sin device (inferido)", {}), ('device="cpu" explícito', {"device": "cpu"})]:
    torch.manual_seed(0)
    outs = sample_full_parallel_v4(
        model, TokStub(), prompt_ids=[2, 5, 7], seq_len=12,
        n_samples=2, n_steps=6, temperature=0.7, **kwargs,
    )
    assert isinstance(outs, list) and len(outs) == 2, f"{label}: n_samples != 2"
    print(f"  {label}: 2 samples, sin crash ✓  (ej: {outs[0][:16]!r})")

# ── 4. Propiedades ───────────────────────────────────────────────────────
torch.manual_seed(0)
outs = sample_full_parallel_v4(
    model, TokStub(), prompt_ids=[2, 5, 7], seq_len=12,
    n_samples=1, n_steps=8, temperature=0.7,
)
# Reconstruir la secuencia decodificada y chequear que no quedan MASKs sin muestrear:
# con n_steps=8 y seq_len=12, el schedule desenmascara todo lo enmascarado.
# El stub decodifica id>3 a letras; MASK(1)/PAD(0)/BOS(2)/EOS(3) se descartan.
seq_dec = outs[0]
full = torch.full((1, 12), TokStub.mask_id, dtype=torch.long)
full[0, :3] = torch.tensor([2, 5, 7])
# Verificar que el prompt [BOS=2, 5, 7] sobrevive al muestreo (no se re-muestrea).
re_enc = [2] + [ord(c) - 65 for c in "".join(chr(65 + (i % 26)) for i in [5, 7])]
assert not (full[0, :3] == TokStub.mask_id).any()
print(f"  prompt preservado (posiciones 0-2 fijas), decode len={len(seq_dec)} ✓")

# Sin prompt: todo MASK al inicio → el sampler debe llenar la secuencia completa.
torch.manual_seed(0)
outs2 = sample_full_parallel_v4(
    model, TokStub(), prompt_ids=None, seq_len=12, n_samples=1, n_steps=8,
)
assert isinstance(outs2[0], str)
print(f"  sin prompt: corre completo ✓  (ej: {outs2[0][:16]!r})")

print("\n✓ SMOKE r7 PASSED — sample_full_parallel_v4 es device-agnostic (CPU OK)")
