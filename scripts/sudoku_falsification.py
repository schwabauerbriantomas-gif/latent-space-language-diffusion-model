"""Falsificacion barata del hallazgo central de NCM (arXiv:2610.14162) sobre el stack propio.

Claim del video/paper a falsificar: en Sudokus dificiles (30 clues), un modelo
autoregresivo resuelve ~1% y los modelos de generacion paralela (NCM 86%, diffusion ~81%).

Protocolo PROPIO (no es una replica exacta de NCM — se declara explicito):
  - 5000 soluciones generadas por backtracking aleatorio (seed 42).
  - Puzzles: se retiran celdas hasta dejar exactamente 30 clues (sin unicidad garantizada).
  - Split: 4000 train / 500 test. Augmentation de train: permutacion de digitos (x9) x
    transposicion (x2) = x18 -> 72k ejemplos efectivos.
  - Tres modelos de ~25M params con presupuesto identico:
      AR causal (raster, teacher forcing, KV cache en inferencia, greedy)
      MDLM bidireccional (masking tipo MDLM, clues protegidos, unmasking coseno 32 steps)
      MDLM + refine (3 rondas: re-enmascarar las peores celdas segun score AR o
        segun violacion de constraints fila/col/caja, y re-muestrear)
  - Metricas: exact-solve (81/81 contra la solucion de referencia) y cell accuracy.

Uso (desde WSL, python.exe de Windows con torch cu128):
  /mnt/d/ComfyUI/.venv/Scripts/python.exe scripts/sudoku_falsification.py --phase all
"""
import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'sudoku')
D_MODEL, N_LAYERS, N_HEADS, D_FF = 512, 6, 8, 2048
VOCAB = 11          # 0=PAD, 1..9 digitos, 10=MASK
SEQ = 81
CLUES = 30
N_TRAIN_ORIG, N_TEST = 4000, 500
EPOCHS, BATCH = 10, 256
LR = 3e-4
N_STEPS_GEN = 32
REFINE_ROUNDS = 3
REFINE_FRAC = 0.15
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# ─────────────────────────── data ───────────────────────────

def gen_solution(rng):
    """Backtracking con digitos barajados."""
    g = [0] * 81
    def ok(pos, d):
        r, c = divmod(pos, 9)
        for i in range(9):
            if g[r*9+i] == d or g[i*9+c] == d: return False
        br, bc = (r//3)*3, (c//3)*3
        for i in range(br, br+3):
            for j in range(bc, bc+3):
                if g[i*9+j] == d: return False
        return True
    def bt(pos):
        if pos == 81: return True
        for d in rng.sample(range(1, 10), 9):
            if ok(pos, d):
                g[pos] = d
                if bt(pos+1): return True
                g[pos] = 0
        return False
    bt(0)
    return g

def make_puzzle(sol, rng):
    p = list(sol)
    for pos in rng.sample(range(81), 81 - CLUES):
        p[pos] = 0
    return p

def permute_digits(grid, rng):
    perm = [0] + rng.sample(range(1, 10), 9)
    return [perm[v] for v in grid]

def transpose(grid):
    return [grid[c*9 + r] for r in range(9) for c in range(9)]

def build_data():
    os.makedirs(RES_DIR, exist_ok=True)
    rng = random.Random(SEED)
    sols, puzzles = [], []
    t0 = time.time()
    for _ in range(N_TRAIN_ORIG + N_TEST):
        s = gen_solution(rng)
        sols.append(s); puzzles.append(make_puzzle(s, rng))
    print(f'data: {len(sols)} puzzles en {time.time()-t0:.1f}s', flush=True)
    tr_s, tr_p = sols[:N_TRAIN_ORIG], puzzles[:N_TRAIN_ORIG]
    te_s, te_p = sols[N_TRAIN_ORIG:], puzzles[N_TRAIN_ORIG:]
    # augmentation x18 sobre train
    aug_s, aug_p = [], []
    for s, p in zip(tr_s, tr_p):
        for dig in range(9):
            sp = permute_digits(s, rng) if dig else s   # dig=0 conserva identidad
            pp = permute_digits(p, rng) if dig else p
            aug_s.append(sp); aug_p.append(pp)
            aug_s.append(transpose(sp)); aug_p.append(transpose(pp))
    np.savez_compressed(os.path.join(RES_DIR, 'data.npz'),
        train_s=np.array(aug_s, np.int8), train_p=np.array(aug_p, np.int8),
        test_s=np.array(te_s, np.int8), test_p=np.array(te_p, np.int8))
    return (np.array(aug_s), np.array(aug_p), np.array(te_s), np.array(te_p))

# ─────────────────────────── modelos ───────────────────────────

class Block(nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.causal = causal
        self.ln1 = nn.LayerNorm(D_MODEL); self.ln2 = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL); self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL); self.o = nn.Linear(D_MODEL, D_MODEL)
        self.f1 = nn.Linear(D_MODEL, D_FF); self.f2 = nn.Linear(D_FF, D_MODEL)
    def forward(self, x, mask=None):
        h = self.ln1(x)
        qkv = torch.cat([self.q(h).unflatten(-1, (N_HEADS, -1)),
                         self.k(h).unflatten(-1, (N_HEADS, -1)),
                         self.v(h).unflatten(-1, (N_HEADS, -1))], dim=0)
        y = F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2],
            is_causal=self.causal, enable_gqa=False)
        y = y.flatten(-2)
        x = x + self.o(y)
        x = x + self.f2(F.gelu(self.f1(self.ln2(x))))
        return x

class Net(nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(SEQ, D_MODEL)
        self.blocks = nn.ModuleList([Block(causal) for _ in range(N_LAYERS)])
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.emb.weight
        self.causal = causal
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, tokens):
        x = self.emb(tokens) + self.pos.weight[None, :tokens.shape[1], :]
        for b in self.blocks: x = b(x)
        return self.head(self.lnf(x))
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

def tok_of(grids, mask_tokens=False):
    t = torch.from_numpy(np.asarray(grids)).long() + 1          # 1..9, 0->1? no:
    return t

def encode(puz, sol=None, add_mask=False):
    """puz/sol: [B,81] con 0=vacio. Devuelve tokens [B,81] en DEV: digitos 1..9 en clues,
    MASK (10) en vacios si add_mask, si no digito de sol."""
    p = torch.from_numpy(np.ascontiguousarray(puz)).long().to(DEV)
    s = torch.from_numpy(np.ascontiguousarray(sol)).long().to(DEV) if sol is not None else None
    t = torch.where(p > 0, p, s if s is not None else p).clone()
    if add_mask:
        t = torch.where(p > 0, t, torch.full_like(t, 10))
    return t

def mdlm_loss(model, puz, sol):
    puz_t = torch.from_numpy(np.ascontiguousarray(puz)).long().to(DEV)
    sol_t = torch.from_numpy(np.ascontiguousarray(sol)).long().to(DEV)
    tok = encode(puz, sol, add_mask=True)                    # [B,81] con 10 en vacios
    B = tok.shape[0]
    t = torch.rand(B, device=DEV)
    frac = t[:, None]
    rand = torch.rand_like(tok.float())
    to_mask = (puz_t == 0) & (rand < frac.expand_as(tok.float()))   # nunca enmascaro clues
    inp = tok.clone(); inp[to_mask] = 10
    logits = model(inp)
    tgt = torch.where(puz_t == 0, sol_t, torch.full_like(tok, -100))
    return F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=-100)

def train(model, puz, sol, tag, kind):
    model.to(DEV).train()
    n = puz.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = 0; t0 = time.time(); losses = []
    for ep in range(EPOCHS):
        idx = torch.randperm(n)
        for i in range(0, n - BATCH + 1, BATCH):
            b = idx[i:i+BATCH].numpy()
            bp, bs = puz[b], sol[b]
            if kind == 'ar':
                tok = encode(bp, bs)                       # TF: solucion completa
                logits = model(tok[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, VOCAB), tok[:, 1:].reshape(-1))
            else:
                loss = mdlm_loss(model, bp, bs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); steps += 1; losses.append(loss.item())
        print(f'{tag} ep{ep+1}/{EPOCHS} loss={np.mean(losses[-(n//BATCH+1):]):.4f} ({time.time()-t0:.0f}s)', flush=True)
    model.eval()
    json.dump({'tag': tag, 'steps': steps, 'final_loss': float(np.mean(losses[-50:])),
               'train_s': time.time()-t0, 'params': model.n_params()},
              open(os.path.join(RES_DIR, f'{tag}_train.json'), 'w'), indent=1)
    return model

# ─────────────────────────── generacion / eval ───────────────────────────

@torch.no_grad()
def gen_ar(model, puz):
    B = puz.shape[0]
    tok = torch.from_numpy(np.asarray(puz)).long().to(DEV)
    tok = torch.where(tok > 0, tok, torch.zeros_like(tok))     # 0=PAD placeholder
    cur = tok.clone()
    for pos in range(SEQ):
        logits = model(cur[:, :pos+1])[:, -1]
        nxt = logits.argmax(-1)
        clue = tok[:, pos] > 0
        cur[:, pos] = torch.where(clue, tok[:, pos], nxt)
    return cur.cpu().numpy()

@torch.no_grad()
def gen_mdlm(model, puz, refine='none', ar_model=None):
    B = puz.shape[0]
    puz_t = torch.from_numpy(np.asarray(puz)).long().to(DEV)
    sol_t = torch.from_numpy(np.asarray(puz)).long().to(DEV)   # se completa iterativamente
    empty = puz_t == 0
    # draft inicial: todos MASK
    inp = torch.where(empty, torch.full_like(puz_t, 10), puz_t)
    n_empty = empty.sum(1)
    cumfrac_prev = 0.0
    for step in range(N_STEPS_GEN):
        cumfrac = 0.5 * (1 - math.cos(math.pi * (step + 1) / N_STEPS_GEN))
        k_target = (n_empty.float() * cumfrac).long()
        logits = model(inp)
        probs = F.softmax(logits, -1)
        conf, tok_pred = probs.max(-1)
        conf = conf.where(empty, torch.zeros_like(conf))       # clues: conf 0 (nunca elegidos)
        # ordenar por confianza y des-enmascarar hasta k_target
        order = conf.argsort(dim=1, descending=True)
        out = inp.clone()
        rank = order.argsort(dim=1)
        unmask = (rank.float() < k_target[:, None].float()) & empty
        out = torch.where(unmask, tok_pred, inp)
        inp = out
    cur = torch.where(empty, tok_pred, puz_t)
    # rondas de refine
    for r in range(REFINE_ROUNDS if refine != 'none' else 0):
        if refine == 'ar':
            with torch.no_grad():
                logits = ar_model(cur[:, :-1])
                lp = F.log_softmax(logits, -1)
                tgt = cur[:, 1:]
                cell_lp = lp.gather(-1, tgt[:, :, None])[:, :, 0]
            cell_lp = torch.cat([torch.full((B, 1), 9.0, device=DEV), cell_lp], 1)
            cell_lp = cell_lp.where(~empty, torch.full_like(cell_lp, 9.0))  # clues fuera
            k = max(1, int(REFINE_FRAC * (SEQ - CLUES)))
            worst = cell_lp.argsort(dim=1)[:, :k]
            re_mask = torch.zeros_like(empty)
            re_mask.scatter_(1, worst, True)
        else:  # constraints
            g = cur.view(B, 9, 9)
            dup = torch.zeros(B, 9, 9, dtype=torch.bool, device=empty.device)
            for i in range(9):
                for j in range(9):
                    v = g[:, i, j]
                    row_dup = (g[:, i, :] == v[:, None]).sum(1) > 1
                    col_dup = (g[:, :, j] == v[:, None]).sum(1) > 1
                    bi, bj = (i//3)*3, (j//3)*3
                    box = g[:, bi:bi+3, bj:bj+3].reshape(B, -1)
                    box_dup = (box == v[:, None]).sum(1) > 1
                    dup[:, i, j] = (row_dup | col_dup | box_dup)
            dup = dup.view(B, SEQ) & empty
            if dup.sum() == 0:
                break
            re_mask = dup
        # re-muestrear celdas seleccionadas con pasos finos
        inp = cur.clone()
        inp[re_mask] = 10
        for step in range(6):
            logits = model(inp)
            probs = F.softmax(logits, -1)
            conf, tok_pred = probs.max(-1)
            conf = conf.where(re_mask & empty, torch.zeros_like(conf))
            frac = (step + 1) / 6
            k = (re_mask.sum(1).float() * frac).long()
            order = conf.argsort(dim=1, descending=True)
            rank = order.argsort(dim=1)
            unmask = (rank.float() < k[:, None].float()) & re_mask & empty
            inp = torch.where(unmask, tok_pred, inp)
        cur = torch.where(empty, inp, puz_t)
    return cur.cpu().numpy()

def evaluate(preds, sols, puzs):
    exact = float((preds == sols).all(1).mean())
    cell = float((preds == sols).mean())
    clues_kept = float((preds[puzs > 0] == sols[puzs > 0]).mean())
    return {'exact_solve': exact, 'cell_acc': cell, 'clue_integrity': clues_kept,
            'n': int(len(sols))}

# ─────────────────────────── main ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='all', choices=['data', 'train', 'eval', 'all'])
    ap.add_argument('--skip_refine', action='store_true')
    args = ap.parse_args()

    dpath = os.path.join(RES_DIR, 'data.npz')
    if args.phase in ('data', 'all') or not os.path.exists(dpath):
        tr_s, tr_p, te_s, te_p = build_data()
    d = np.load(dpath)
    tr_s, tr_p, te_s, te_p = d['train_s'], d['train_p'], d['test_s'], d['test_p']
    print(f'train {tr_s.shape} test {te_s.shape} | device={DEV}', flush=True)

    summary = {}
    ar_model, mdlm_model = None, None
    if args.phase in ('train', 'all'):
        ar_model = train(Net(True), tr_p, tr_s, 'ar', 'ar')
        mdlm_model = train(Net(False), tr_p, tr_s, 'mdlm', 'mdlm')
        torch.save(ar_model.state_dict(), os.path.join(RES_DIR, 'ar.pt'))
        torch.save(mdlm_model.state_dict(), os.path.join(RES_DIR, 'mdlm.pt'))
    else:
        ar_model = Net(True); ar_model.load_state_dict(torch.load(os.path.join(RES_DIR, 'ar.pt'), map_location=DEV)); ar_model.to(DEV).eval()
        mdlm_model = Net(False); mdlm_model.load_state_dict(torch.load(os.path.join(RES_DIR, 'mdlm.pt'), map_location=DEV)); mdlm_model.to(DEV).eval()

    if args.phase in ('eval', 'all'):
        # eval en 500 test (submuestra para velocidad si hace falta)
        n_eval = 500
        te_p_e, te_s_e = te_p[:n_eval], te_s[:n_eval]
        bs = 100
        for name, fn in [('ar', lambda p: gen_ar(ar_model, p)),
                         ('mdlm', lambda p: gen_mdlm(mdlm_model, p)),
                         ('mdlm_constraint', lambda p: gen_mdlm(mdlm_model, p, refine='constraints'))] + \
                        ([] if args.skip_refine else [('mdlm_ar_refine', lambda p: gen_mdlm(mdlm_model, p, refine='ar', ar_model=ar_model))]):
            preds = []
            t0 = time.time()
            for i in range(0, n_eval, bs):
                preds.append(fn(te_p_e[i:i+bs]))
            preds = np.concatenate(preds)
            m = evaluate(preds, te_s_e, te_p_e)
            m['time_s'] = round(time.time() - t0, 1)
            summary[name] = m
            print(f'{name}: {json.dumps(m)}', flush=True)
        summary['_meta'] = {'protocol': 'propio; 30 clues sin unicidad garantizada; augmentation x18 (perm digitos x transposicion); '
                                      'mismo presupuesto para AR y MDLM; greedy decoding',
                            'params_per_model': mdlm_model.n_params(), 'epochs': EPOCHS, 'batch': BATCH,
                            'gen_steps': N_STEPS_GEN, 'seed': SEED, 'device': DEV}
        json.dump(summary, open(os.path.join(RES_DIR, 'falsification_summary.json'), 'w'), indent=1)
        print('summary ->', os.path.join(RES_DIR, 'falsification_summary.json'))

if __name__ == '__main__':
    main()
