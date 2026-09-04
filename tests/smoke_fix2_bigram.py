"""Smoke test Fix 2 — ban de bigramas v4 no computa bigramas contra MASK (sev 6).

seq = [5, MASK, 7, MASK, 8, 5, 9]
bigramas REALES: (5,9) [pos5→6], (8,5) [pos4→5].
El prefix de la posición p es seq[p-1]: MASK-prefix en pos 2 y pos 4.

Antes del fix: (MASK,7) y (MASK,MASK) alimentaban el ban set → baneaba
bigramas que nunca existieron (p.ej. token 7 tras un MASK). Después:
todo bigrama baneado existe en la secuencia con tokens reales (paridad v3).
"""
import sys, torch
sys.path.insert(0, "src")
from logit_guidance_v4 import apply_no_repeat_ngram
from inference_v4 import apply_no_repeat_bigram_batch

MASK, PAD, EOS, UNK = 1, 0, 2, 3
seq = [5, MASK, 7, MASK, 8, 5, 9]
tokens = torch.tensor([seq])                    # [1, 7]
b, s, v = 1, len(seq), 12
logits = torch.zeros(b, s, v)

real_bigrams = {(seq[i], seq[i + 1]) for i in range(s - 1)
                if seq[i] not in (0, 1, 2, 3) and seq[i + 1] not in (0, 1, 2, 3)}

for name, fn in [
    ("logit_guidance_v4.apply_no_repeat_ngram", lambda lg: apply_no_repeat_ngram(lg, tokens, n=2)),
    ("inference_v4.apply_no_repeat_bigram_batch", lambda lg: apply_no_repeat_bigram_batch(lg, tokens)),
]:
    out = fn(logits.clone())
    banned = (out == float("-inf"))[0]          # [s, v] bool
    print(f"\n=== {name} ===")
    print(f"bigramas reales en la secuencia: {sorted(real_bigrams)}")
    errs = []
    for pos in range(1, s):
        prefix = seq[pos - 1]
        for tok in banned[pos].nonzero().flatten().tolist():
            ok = (prefix, tok) in real_bigrams
            print(f"  pos {pos} (prefix={prefix}): ban de token {tok} → {'OK (bigrama real)' if ok else 'BAN INEXISTENTE ✗'}")
            if not ok:
                errs.append((pos, prefix, tok))
    # 1) posiciones cuyo predecessor es MASK (pos 2 y 4) no banear nada
    for p in (2, 4):
        assert not banned[p].any(), (
            f"{name}: pos {p} (prefix=MASK) banea {banned[p].nonzero().flatten().tolist()} — bigrama contra MASK")
    # 2) ningún bigrama inexistente en el ban set (la propiedad clave del fix)
    assert not errs, f"{name}: bigramas baneados que nunca existieron: {errs}"
    # 3) los bigramas reales SÍ se banean donde corresponde
    assert banned[1, 9].item(), f"{name}: (5,9) real no baneado en pos 1 (prefix=5)"
    assert banned[5, 5].item(), f"{name}: (8,5) real no baneado en pos 5 (prefix=8)"
    assert banned[6, 9].item(), f"{name}: (5,9) real no baneado en pos 6 (prefix=5)"
    print(f"  pos 2 y 4 (prefix=MASK): sin bans ✓")

print("\n✓ SMOKE FIX 2 PASSED — bigramas solo contra tokens reales (v3-parity)")
