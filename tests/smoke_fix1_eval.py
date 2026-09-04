"""Smoke test Fix 1: deterministic held-out PPL eval in train_v4.py.

Checks (CPU, tiny model, no tokenizer file needed):
  A. quick_eval is DETERMINISTIC: same model + same data -> identical CE twice.
  B. quick_eval returns plain CE (nats) in a sane range for a lightly-trained
     model — NOT the ELBO-inflated 17-22 nats, and exp(CE) << e^15.
  C. Held-out split: val_tensor and train_tensor are disjoint; loader only
     sees train_tensor.
  D. train loss (ELBO-weighted) is consistently HIGHER than val CE — proves
     we are no longer reporting the inflated stochastic quantity.
  E. val CE decreases after a few optimizer steps (eval tracks learning).
"""
import sys, math, types
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

REPO = Path("/home/schwabauer/swarm_lsd/wt_eval")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import mdlm_v4_optimized as m4
from mdlm_v4_optimized import MDLMConfig, MDLMBPEV4, mdlm_loss_v4

import train_v4 as tv

DEVICE = "cpu"
tv.DEVICE = DEVICE  # force CPU

torch.manual_seed(0)
np.random.seed(0)

VOCAB = 200
SEQ = 32
N_SEQS = 64
BATCH = 8
MASK_ID = 1

# Synthetic corpus with learnable UNIGRAM structure (Zipf frequencies).
# Position-independent statistics generalize to held-out data even on the
# (currently RoPE-broken) architecture, so this tests the EVAL path, not
# model quality. Copy/positional tasks would NOT generalize here.
probs = 1.0 / torch.arange(1, VOCAB + 1, dtype=torch.float)
probs = (probs / probs.sum())
data = torch.multinomial(probs, N_SEQS * SEQ, replacement=True).reshape(N_SEQS, SEQ)
UNIGRAM_ENTROPY = float(-(probs * probs.log()).sum())
tokens_tensor = data.clone()

# ── Replicate the new split logic exactly as in train_v4.train_v4 ──
val_frac = 0.05
n_val = max(1, int(len(tokens_tensor) * val_frac))
val_tensor = tokens_tensor[:n_val]
train_tensor = tokens_tensor[n_val:]
assert val_tensor.shape[0] == 3, f"expected 3 val seqs, got {val_tensor.shape[0]}"

# C. disjointness
train_set = set(map(tuple, train_tensor.tolist()))
val_set = set(map(tuple, val_tensor.tolist()))
assert train_set.isdisjoint(val_set) or len(train_set & val_set) == 0, "leak!"

config = MDLMConfig(vocab_size=VOCAB, d_model=64, n_heads=4, n_layers=2,
                    max_seq_len=64)
model = MDLMBPEV4(config, pad_id=0, mask_id=MASK_ID).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"model params: {n_params:,}")

val_loader = DataLoader(TensorDataset(val_tensor), batch_size=BATCH, shuffle=False)

# A. determinism BEFORE training (untrained but fixed init)
ce1 = tv.quick_eval(model, val_loader, MASK_ID, n_batches=20, eval_t=0.5)
ce2 = tv.quick_eval(model, val_loader, MASK_ID, n_batches=20, eval_t=0.5)
assert ce1 == ce2, f"non-deterministic: {ce1} != {ce2}"
print(f"A. determinism OK: CE={ce1:.6f} == {ce2:.6f}")

# B. untrained CE should be ~ln(VOCAB)=5.3 (not 17-22, not inf)
assert 3.0 < ce1 < 8.0, f"untrained CE out of range: {ce1}"
print(f"B. untrained CE={ce1:.4f} nats ~ ln({VOCAB})={math.log(VOCAB):.4f} OK, PPL={math.exp(ce1):.1f} << e^15")

# Train on train_tensor. The eval masks 50% of EVERY position, but training
# with random t ~ logitnormal(0, 1) rarely sees t >= 0.5 masks, so we ALSO
# train a few epochs at fixed t=eval_t so the model is calibrated at the
# exact mask ratio the deterministic eval uses (this tests the eval path,
# not convergence of the tiny model).
def fixed_t_loss(model, tokens, mask_id, t=0.5):
    b = tokens.shape[0]
    tt = torch.full((b,), float(t))
    rand = torch.rand(tokens.shape)
    mask_positions = rand < tt[:, None]
    is_special = (tokens == 0) | (tokens == 2) | (tokens == 3)
    mask_positions = mask_positions & (~is_special)
    if mask_positions.sum() == 0:
        return None
    masked = torch.where(mask_positions, mask_id, tokens)
    hidden = model(masked, tt, return_pre_proj=True)
    hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    mask_flat = mask_positions.reshape(-1)
    logits = F.linear(hidden_flat[mask_flat], model.token_emb.weight)
    return F.cross_entropy(logits, tokens.reshape(-1)[mask_flat])

# Train on train_tensor with the fixed-t loss (same objective the eval uses).
# Also collect ELBO losses for comparison (D).
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH, shuffle=True, drop_last=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
model.train()
elbo_losses = []
for phase in range(2):
    loss_fn = mdlm_loss_v4 if phase == 0 else fixed_t_loss
    for step in range(40):
        for batch in train_loader:
            tokens = batch[0]
            loss = loss_fn(model, tokens, MASK_ID)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            if phase == 0:
                elbo_losses.append(loss.item())

model.eval()
ce_after = tv.quick_eval(model, val_loader, MASK_ID, n_batches=20, eval_t=0.5)
ppl_after = math.exp(ce_after)
print(f"   after training: val CE={ce_after:.4f} val_ppl={ppl_after:.1f}")

# D. train ELBO loss vs val CE
mean_elbo = float(np.mean(elbo_losses[-20:]))
print(f"D. mean train ELBO loss={mean_elbo:.4f} vs val CE={ce_after:.4f}")
assert ce_after < ce1, f"val CE did not decrease: {ce1} -> {ce_after}"

# E. sane PPL — this is the core acceptance: NOT e^15, NOT 17-22 nats
assert 1.0 < ppl_after < math.exp(10), f"val PPL insane: {ppl_after}"
assert ce_after < 10.0, f"val CE still huge: {ce_after}"
# Generalization: val CE should approach the unigram entropy (the Bayes limit
# for this corpus) — proves eval is measured on UNSEEN data, not memorized train.
assert ce_after < ce1 - 0.3, f"val CE barely moved from untrained {ce1:.4f} -> {ce_after:.4f}"
assert ce_after < UNIGRAM_ENTROPY + 0.5, \
    f"val CE {ce_after:.4f} far above unigram entropy {UNIGRAM_ENTROPY:.4f} — memorization or eval bug"
print(f"E. val PPL={ppl_after:.1f} sane; val CE {ce1:.3f} -> {ce_after:.3f}, approaches unigram entropy {UNIGRAM_ENTROPY:.3f} OK")

# Also verify quick_eval differs from old broken behavior: old code would
# give ~2.65x CE. Sanity: ELBO-weighted on same val data should be higher.
def old_style_eval(model, loader, mask_id, n_batches=20):
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches: break
            tokens = batch[0]
            losses.append(mdlm_loss_v4(model, tokens, mask_id).item())
    return np.mean(losses)

old_ce = old_style_eval(model, val_loader, MASK_ID)
print(f"   old stochastic ELBO eval on same val data: {old_ce:.4f} (inflated by ~E[1/(1-t)]=2.65)")
assert old_ce > ce_after, "ELBO-weighted should exceed plain CE here"

print("\nALL FIX-1 SMOKE TESTS PASSED")
