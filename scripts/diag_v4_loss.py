"""Diagnostico: ¿el divergence de v4 viene del ELBO weighting?
Run corto (800 steps) entrenando v4 con PLAIN CE (sin weight 1/(1-t)),
logueando ambas metricas sobre el mismo modelo.
Correr desde la raiz del repo con python de Windows (GPU).
"""
import sys, os, math, time, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from mdlm_v4_optimized import MDLMBPEV4, MDLMConfig, mdlm_loss_v4, forward_mask_bpe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = 'cuda'
STEPS, BATCH, SEQ = 800, 64, 128
LR, WARMUP = 3e-4, 100

def plain_ce(model, tokens, mask_id=1):
    batch = tokens.shape[0]
    t = torch.sigmoid(torch.randn(batch, device=tokens.device))
    masked, mask_pos = forward_mask_bpe(tokens, t, mask_id)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        hidden = model(masked, t, return_pre_proj=True)
    mask_flat = mask_pos.reshape(-1)
    hidden_masked = hidden.reshape(-1, hidden.shape[-1])[mask_flat]
    tokens_masked = tokens.reshape(-1)[mask_flat]
    logits = F.linear(hidden_masked.float(), model.token_emb.weight.float())
    return F.cross_entropy(logits, tokens_masked)

data = np.load(os.path.join(ROOT, 'data', 'train_tokens_v4_128.npy'), mmap_mode='r')
print(f'data: {data.shape}', flush=True)

cfg = MDLMConfig(vocab_size=10000, d_model=1024, n_heads=16, n_layers=10,
                 max_seq_len=256, d_ff=2816, dropout=0.1)
model = MDLMBPEV4(cfg).to(DEV)
model.train()
print(f'params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s+1)/WARMUP, 1.0) *
                                          (0.5 * (1 + math.cos(math.pi * min(s/STEPS, 1.0)))) if s >= WARMUP else (s+1)/WARMUP)
rng = np.random.default_rng(42)
t0 = time.time()
hist = []
for step in range(STEPS):
    idx = rng.integers(0, data.shape[0], BATCH)
    toks = torch.from_numpy(np.ascontiguousarray(data[idx])).long().to(DEV)
    loss = plain_ce(model, toks)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 50 == 0 or step == STEPS - 1:
        with torch.no_grad():
            elbo = mdlm_loss_v4(model, toks).item()
        pl = loss.item()
        hist.append({'step': step, 'plain_ce': round(pl, 3), 'elbo_weighted': round(elbo, 3)})
        print(f'[S{step}] plain_ce={pl:.3f} elbo_weighted={elbo:.3f} lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)', flush=True)

model.eval()
# eval plain CE sobre 256 seqs frescas
evals = []
with torch.no_grad():
    for _ in range(4):
        idx = rng.integers(0, data.shape[0], 64)
        toks = torch.from_numpy(np.ascontiguousarray(data[idx])).long().to(DEV)
        evals.append(plain_ce(model, toks).item())
final = float(np.mean(evals))
print(f'FINAL plain_ce (held-out batches) = {final:.3f} -> ppl={math.exp(min(final,20)):,.1f}')
json.dump({'steps': STEPS, 'batch': BATCH, 'final_plain_ce': final,
           'final_ppl': math.exp(min(final, 20)), 'history': hist},
          open(os.path.join(ROOT, 'results', 'diag_v4_loss.json'), 'w'), indent=1)
print('receipt -> results/diag_v4_loss.json')
