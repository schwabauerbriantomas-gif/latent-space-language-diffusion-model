"""Smoke test Fix 4 — NaN guard: softmax con fila all--inf (sev 5).

Caso del finding: el ban de n-gramas hace masked_fill(-inf); si TODOS los
logits de una posición quedan -inf, F.softmax devuelve NaN y multinomial
crashea / contamina la generación.
Fix: cap -inf → -1e9 antes del softmax (port del clamp de v3). Una fila
totalmente baneada degrada a (casi) uniforme en vez de NaN.

Se verifican las 2 rutas de sampling v4:
  - logit_guidance_v4.sample_with_guidance  (loop de sampling)
  - inference_v4.generate_full_parallel      (loop de sampling)
Además, test unitario directo de la transformación del guard.
"""
import math
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# ── 1. Unit test de la transformación (la misma línea que el fix) ──
logits = torch.full((1, 4), float("-inf"))
logits[0, 0] = 2.0  # una posición viva
guarded = torch.where(torch.isinf(logits) & (logits < 0), -1e9, logits)
probs = F.softmax(guarded, dim=-1)
assert not torch.isnan(probs).any(), "guard: aún hay NaN"
assert probs[0, 0].item() > 0.999, "el logit vivo debe dominar"
print(f"unit: fila mixta (1 vivo + 3 baneados) → p[0]={probs[0,0]:.6f}, sin NaN ✓")

# Peor caso: fila 100% baneada
all_banned = torch.full((1, 4), float("-inf"))
guarded2 = torch.where(torch.isinf(all_banned) & (all_banned < 0), -1e9, all_banned)
probs2 = F.softmax(guarded2, dim=-1)
assert not torch.isnan(probs2).any(), "guard: NaN en fila totalmente baneada"
assert torch.allclose(probs2, torch.full_like(probs2, 0.25), atol=1e-6), \
    "fila totalmente baneada debe degradar a uniforme"
print("unit: fila 100% baneada → uniforme exacto (0.25), sin NaN ✓")


# ── 2. Test end-to-end con stub de modelo que emite logits baneables ──
class BanBombModel(torch.nn.Module):
    """Modelo adversarial: en cada forward devuelve TODOS los logits en
    -inf (p. ej. salida numéricamente colapsada de un forward bf16 sin
    autocast). Sin el guard, softmax(all--inf) = NaN y multinomial
    crashea con 'probability tensor contains either inf, nan or <0'."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.dummy = torch.nn.Parameter(torch.zeros(1))  # para .parameters()

    def forward(self, full, t):
        b, s = full.shape
        return torch.full((b, s, self.vocab), float("-inf"), device=full.device)


class TokStub:
    """Tokenizer mínimo con la interfaz que usan los samplers."""

    class _inner:
        @staticmethod
        def token_to_id(name):
            return None

    tokenizer = _inner()
    vocab_size = 8
    mask_id = 0
    pad_id = 1
    bos_id = 2
    eos_id = 3

    def encode(self, text, add_special=False):
        return [4, 5]

    def decode(self, ids):
        return "".join(chr(ord("a") + (i % 26)) for i in ids)


import sys
sys.path.insert(0, "src")
from logit_guidance_v4 import sample_with_guidance
from inference_v4 import generate_full_parallel

model = BanBombModel(vocab=8)
tok = TokStub()

# Sin fix: NaN en probs → multinomial levanta RuntimeError o devuelve basura.
# Con fix: termina y devuelve strings.
outs = sample_with_guidance(
    model, tok, prompt_ids=[4, 4], seq_len=8, n_samples=2, n_steps=4,
    temperature=1.0, repetition_penalty=1.3, no_repeat_ngram=2,
    frequency_penalty=0.4, top_p_val=1.0, device="cpu")
print(f"sample_with_guidance: {len(outs)} outputs, sin crash ✓")
assert len(outs) == 2 and all(isinstance(o, str) for o in outs)

outs2 = generate_full_parallel(
    model, tok, prompt="x", max_new_tokens=6, n_samples=2, n_steps=4,
    temperature=1.0, use_guidance=True, device="cpu")
print(f"generate_full_parallel: {len(outs2)} outputs, sin crash ✓")
assert len(outs2) == 2 and all(isinstance(o, str) for o in outs2)

print("\n✓ SMOKE FIX 4 PASSED — softmax nunca produce NaN con filas all--inf")
