"""
COMPREHENSIVE BENCHMARK — Text quality + TPS across all methods.

Methods compared:
  1. MDLM v3 baseline (semi-AR, no guidance)
  2. MDLM v3 + adaptive guidance (cooling temp + adaptive penalties + top-p)
  3. MDLM v3 + guidance + Repetition HRM
  4. Hybrid Speculative (MDLM draft + Qwen3 segment regen)
  5. Qwen3-0.6B standalone (AR baseline)

Metrics:
  - Text quality: oracle log-prob (Qwen3 teacher forcing)
  - Repetition score: RepetitionReviewer
  - Throughput: tokens/second (generation, not forward)
  - Latency: wall-clock time per sequence
"""
import sys, math, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch
import torch.nn.functional as F
from mdlm_bpe_v3 import MDLMConfig, MDLMBPEV3, BPETokenizer
from hrm_refiner import RepetitionReviewer
from hybrid_speculative import HybridSpeculative
from ar_oracle_hrm import load_oracle, build_token_alignment

DEVICE = "cuda"
CKPT_DIR = REPO / "checkpoints"
ckpt = torch.load(CKPT_DIR / "mdlm_bpe_v3_best.pt",
                   map_location=DEVICE, weights_only=False)
config = MDLMConfig(**ckpt["config"])
model = MDLMBPEV3(config).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
tokenizer = BPETokenizer()
mask_id = tokenizer.mask_id

print("Loading Qwen3-0.6B oracle...")
oracle_model, oracle_tok = load_oracle("Qwen/Qwen3-0.6B")

hybrid = HybridSpeculative(
    model, tokenizer,
    oracle_model_name="Qwen/Qwen3-0.6B",
    segment_size=8, surprise_threshold=1.0,
    max_regen_fraction=0.6, oracle_temp=0.6,
)

reviewer = RepetitionReviewer(
    pad_id=tokenizer.pad_id, mask_id=tokenizer.mask_id,
    bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
)

# ═══════════════════════════════════════════════════════════════
# GUIDANCE HELPERS
# ═══════════════════════════════════════════════════════════════

def apply_freq(logits, tokens, penalty, special_ids):
    b,s,v = logits.shape
    counts = torch.zeros(b,v,device=tokens.device)
    counts.scatter_add_(1, tokens, torch.ones_like(tokens,dtype=torch.float))
    for sid in special_ids:
        if sid < v: counts[:, sid] = 0
    return logits - (penalty * torch.sqrt(counts.float())).unsqueeze(1)

def apply_rep(logits, tokens, penalty, special_ids):
    b,s,v = logits.shape
    used = torch.zeros(b,v,dtype=torch.bool,device=tokens.device)
    used.scatter_(1, tokens, True)
    for sid in special_ids:
        if sid < v: used[:, sid] = False
    factor = torch.where(used, torch.tensor(1/penalty,device=logits.device),
                         torch.tensor(1.0,device=logits.device))
    return logits * factor.unsqueeze(1)

# ═══════════════════════════════════════════════════════════════
# GENERATION METHODS
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def gen_mdlm_baseline(prompt_ids, seq_len=64, block_size=4):
    """Method 1: MDLM semi-AR baseline (no guidance)."""
    special_ids = {tokenizer.pad_id, mask_id, tokenizer.bos_id, tokenizer.eos_id}
    full = torch.full((1, seq_len), mask_id, device=DEVICE)
    plen = min(len(prompt_ids), seq_len)
    full[:, :plen] = torch.tensor(prompt_ids[:plen], device=DEVICE)
    n_steps = max(2, block_size)
    for bs in range(plen, seq_len, block_size):
        be = min(bs + block_size, seq_len)
        for step in range(n_steps):
            t_val = max(0.5 - step/(n_steps*2), 0.01)
            t = torch.full((1,), t_val, device=DEVICE)
            logits = model(full, t)
            mask_in = (full[0, bs:be] == mask_id)
            if not mask_in.any(): break
            idxs = mask_in.nonzero(as_tuple=True)[0]
            pl = logits[0, bs:be][idxs] / 0.7
            probs = F.softmax(pl, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            conf = probs.max(dim=-1)[0]
            n_unmask = max(1, len(idxs) // (n_steps - step))
            tc, ti = conf.topk(min(n_unmask, len(idxs)))
            full[0, bs + idxs[ti]] = sampled[ti]
    return full

@torch.no_grad()
def gen_mdlm_guided(prompt_ids, seq_len=64, block_size=4):
    """Method 2: MDLM + adaptive guidance (cooling + adaptive penalties + top-p)."""
    special_ids = {tokenizer.pad_id, mask_id, tokenizer.bos_id, tokenizer.eos_id}
    full = torch.full((1, seq_len), mask_id, device=DEVICE)
    plen = min(len(prompt_ids), seq_len)
    full[:, :plen] = torch.tensor(prompt_ids[:plen], device=DEVICE)
    n_steps = max(2, block_size)
    n_blocks = math.ceil((seq_len - plen) / block_size)
    bi = 0
    for bs in range(plen, seq_len, block_size):
        be = min(bs + block_size, seq_len)
        frac = bi / max(n_blocks - 1, 1)
        temp = 1.0 + frac * (0.5 - 1.0)
        rep_p = 1.2 + frac * (1.5 - 1.2)
        freq_p = 0.3 + frac * (0.6 - 0.3)
        for step in range(n_steps):
            t_val = max(0.5 - step/(n_steps*2), 0.01)
            t = torch.full((1,), t_val, device=DEVICE)
            logits = model(full, t)
            logits = apply_freq(logits, full, freq_p, special_ids)
            logits = apply_rep(logits, full, rep_p, special_ids)
            mask_in = (full[0, bs:be] == mask_id)
            if not mask_in.any(): break
            idxs = mask_in.nonzero(as_tuple=True)[0]
            pl = logits[0, bs:be][idxs] / max(temp, 0.01)
            probs = F.softmax(pl, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cs = torch.cumsum(sp, dim=-1)
            sm = cs - sp > 0.95
            sp[sm] = 0
            pn = torch.zeros_like(probs)
            pn.scatter_(1, si, sp)
            probs = pn / pn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            conf = probs.max(dim=-1)[0]
            n_unmask = max(1, len(idxs) // (n_steps - step))
            tc, ti = conf.topk(min(n_unmask, len(idxs)))
            full[0, bs + idxs[ti]] = sampled[ti]
        bi += 1
    return full

@torch.no_grad()
def gen_mdlm_guided_hrm(prompt_ids, seq_len=64, block_size=4):
    """Method 3: MDLM + guidance + Repetition HRM."""
    tokens = gen_mdlm_guided(prompt_ids, seq_len, block_size)
    # Repetition HRM pass
    for r in range(3):
        bad = reviewer.detect_bad_positions(tokens)
        if bad.sum() == 0: break
        tokens = torch.where(bad, mask_id, tokens)
        for step in range(6):
            t_val = max(1.0 - step/6, 0.01)
            t = torch.full((1,), t_val, device=DEVICE)
            logits = model(tokens, t)
            logits = apply_freq(logits, tokens, 0.4, {tokenizer.pad_id, mask_id, tokenizer.bos_id, tokenizer.eos_id})
            logits = apply_rep(logits, tokens, 1.3, {tokenizer.pad_id, mask_id, tokenizer.bos_id, tokenizer.eos_id})
            cm = (tokens == mask_id)
            if not cm.any(): break
            tl = logits / 0.5
            probs = F.softmax(tl.float(), dim=-1)
            conf = probs.max(dim=-1)[0]
            conf[~cm] = -1
            nm = cm.sum(dim=1)
            nu = torch.clamp(nm // max(6-step,1), min=1)
            k = min(int(nu[0].item()), int(nm[0].item()))
            if k <= 0: continue
            tc, ti = conf[0].topk(k)
            valid = ti[tc > 0]
            if valid.numel() > 0:
                pl = logits[0, valid] / 0.5
                pp = F.softmax(pl.float(), dim=-1)
                samp = torch.multinomial(pp, 1).squeeze(-1)
                tokens[0, valid] = samp
    return tokens

@torch.no_grad()
def gen_hybrid(prompt_ids, seq_len=64, block_size=4):
    """Method 4: Hybrid Speculative (MDLM draft + Qwen3 segment regen)."""
    tokens = gen_mdlm_guided(prompt_ids, seq_len, block_size)
    final_text, stats = hybrid.verify_and_refine(tokens, prompt_len=len(prompt_ids))
    return final_text, stats

@torch.no_grad()
def gen_qwen3(prompt_text, max_tokens=64):
    """Method 5: Qwen3-0.6B standalone (pure AR)."""
    ids = oracle_tok.encode(prompt_text, return_tensors="pt").to(DEVICE)
    attention_mask = torch.ones_like(ids)
    out = oracle_model.generate(
        ids, max_new_tokens=max_tokens, do_sample=True,
        temperature=0.6, top_p=0.9, attention_mask=attention_mask,
        pad_token_id=oracle_tok.eos_token_id,
    )
    return oracle_tok.decode(out[0], skip_special_tokens=True)

# ═══════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def oracle_score(text):
    """Score text with Qwen3 teacher forcing. Higher = more coherent."""
    ids = oracle_tok.encode(text, add_special_tokens=False)
    if len(ids) < 2: return 0.0, 0
    inp = torch.tensor([ids], device=DEVICE)
    logits = oracle_model(inp).logits
    lp = F.log_softmax(logits.float(), dim=-1)
    actual = inp[0, 1:]
    pred = lp[0, :-1]
    tlps = pred.gather(1, actual.unsqueeze(1)).squeeze(1)
    return tlps.mean().item(), len(ids)

def rep_score(text):
    ids = tokenizer.encode(text, add_special=False)
    if not ids: return 1.0
    t = torch.tensor([ids[:64]], device=DEVICE)
    if t.shape[1] < 64:
        t = torch.cat([t, torch.full((1, 64-t.shape[1]), tokenizer.pad_id, device=DEVICE)], 1)
    return reviewer.score_sequence(t[0])

# ═══════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════

prompts = [
    "The future of artificial intelligence",
    "Climate change is one of the biggest challenges",
    "Programming is a skill that requires",
]

print("\n" + "=" * 80)
print("TEXT QUALITY COMPARISON")
print("=" * 80)

results = {}

for prompt in prompts:
    print(f"\n{'─' * 80}")
    print(f"PROMPT: \"{prompt}\"")
    print(f"{'─' * 80}")

    ids = tokenizer.encode(prompt, add_special=False)
    prompt_results = {}

    # Method 1: MDLM baseline
    t0 = time.time()
    tok1 = gen_mdlm_baseline(ids)
    text1 = tokenizer.decode(tok1[0].cpu().tolist())
    dt1 = time.time() - t0
    lp1, nt1 = oracle_score(text1)
    rs1 = rep_score(text1)
    gen_tok1 = len(tokenizer.encode(text1, add_special=False)) - len(ids)
    tps1 = max(gen_tok1, 1) / dt1
    prompt_results["MDLM baseline"] = {
        "text": text1.strip()[:200], "lp": lp1, "rep": rs1,
        "time": dt1, "tps": tps1, "tokens": gen_tok1,
    }

    # Method 2: MDLM + guidance
    t0 = time.time()
    tok2 = gen_mdlm_guided(ids)
    text2 = tokenizer.decode(tok2[0].cpu().tolist())
    dt2 = time.time() - t0
    lp2, nt2 = oracle_score(text2)
    rs2 = rep_score(text2)
    gen_tok2 = len(tokenizer.encode(text2, add_special=False)) - len(ids)
    tps2 = max(gen_tok2, 1) / dt2
    prompt_results["MDLM + guidance"] = {
        "text": text2.strip()[:200], "lp": lp2, "rep": rs2,
        "time": dt2, "tps": tps2, "tokens": gen_tok2,
    }

    # Method 3: MDLM + guidance + Rep HRM
    t0 = time.time()
    tok3 = gen_mdlm_guided_hrm(ids)
    text3 = tokenizer.decode(tok3[0].cpu().tolist())
    dt3 = time.time() - t0
    lp3, nt3 = oracle_score(text3)
    rs3 = rep_score(text3)
    gen_tok3 = len(tokenizer.encode(text3, add_special=False)) - len(ids)
    tps3 = max(gen_tok3, 1) / dt3
    prompt_results["MDLM + guidance + RepHRM"] = {
        "text": text3.strip()[:200], "lp": lp3, "rep": rs3,
        "time": dt3, "tps": tps3, "tokens": gen_tok3,
    }

    # Method 4: Hybrid Speculative
    t0 = time.time()
    text4, stats4 = gen_hybrid(ids)
    dt4 = time.time() - t0
    lp4, nt4 = oracle_score(text4 if isinstance(text4, str) else tokenizer.decode(text4[0].cpu().tolist()))
    rs4 = rep_score(text4 if isinstance(text4, str) else tokenizer.decode(text4[0].cpu().tolist()))
    gen_tok4 = nt4 - len(ids)
    tps4 = max(gen_tok4, 1) / dt4
    prompt_results["Hybrid (MDLM+Qwen3)"] = {
        "text": (text4 if isinstance(text4, str) else "").strip()[:200],
        "lp": lp4, "rep": rs4, "time": dt4, "tps": tps4, "tokens": gen_tok4,
    }

    # Method 5: Qwen3 standalone
    t0 = time.time()
    text5 = gen_qwen3(prompt, max_tokens=64)
    dt5 = time.time() - t0
    lp5, nt5 = oracle_score(text5)
    rs5 = rep_score(text5)
    gen_tok5 = nt5 - len(oracle_tok.encode(prompt, add_special_tokens=False))
    tps5 = max(gen_tok5, 1) / dt5
    prompt_results["Qwen3-0.6B (pure AR)"] = {
        "text": text5.strip()[:200], "lp": lp5, "rep": rs5,
        "time": dt5, "tps": tps5, "tokens": gen_tok5,
    }

    # Print results for this prompt
    for method, r in prompt_results.items():
        print(f"\n  【{method}】")
        print(f"    Quality:  lp={r['lp']:.2f}  rep={r['rep']:.2f}")
        print(f"    Speed:    {r['tps']:.1f} tok/s  ({r['time']:.1f}s for ~{r['tokens']} tokens)")
        print(f"    Text:     {r['text']}")

    results[prompt] = prompt_results

# ═══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 80}")
print("BENCHMARK SUMMARY (averaged across prompts)")
print("=" * 80)

methods = ["MDLM baseline", "MDLM + guidance", "MDLM + guidance + RepHRM",
           "Hybrid (MDLM+Qwen3)", "Qwen3-0.6B (pure AR)"]

print(f"\n  {'Method':<30} {'Oracle LP':>10} {'Rep Score':>10} {'TPS':>10} {'Latency':>10}")
print(f"  {'-'*70}")

for method in methods:
    avg_lp = np.mean([results[p][method]["lp"] for p in prompts])
    avg_rep = np.mean([results[p][method]["rep"] for p in prompts])
    avg_tps = np.mean([results[p][method]["tps"] for p in prompts])
    avg_time = np.mean([results[p][method]["time"] for p in prompts])
    print(f"  {method:<30} {avg_lp:>10.2f} {avg_rep:>10.2f} {avg_tps:>9.1f} {avg_time:>9.1f}s")

# Also print raw TPS for forward-pass comparison
print(f"\n{'─' * 80}")
print("FORWARD-PASS THROUGHPUT (batch inference, no generation loop)")
print(f"{'─' * 80}")

for bs in [1, 8, 32]:
    # MDLM
    tokens = torch.full((bs, 128), mask_id, device=DEVICE)
    t = torch.full((bs,), 0.5, device=DEVICE)
    for _ in range(3):
        with torch.no_grad(): _ = model(tokens, t)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        with torch.no_grad(): _ = model(tokens, t)
    torch.cuda.synchronize()
    mdlm_tps = bs * 128 * 20 / (time.time() - t0)

    # Qwen3
    qtokens = torch.randint(0, 1000, (bs, 128), device=DEVICE)
    qmask = torch.ones_like(qtokens)
    for _ in range(3):
        with torch.no_grad(): _ = oracle_model(qtokens, attention_mask=qmask)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        with torch.no_grad(): _ = oracle_model(qtokens, attention_mask=qmask)
    torch.cuda.synchronize()
    qwen_tps = bs * 128 * 20 / (time.time() - t0)

    print(f"  Batch {bs:3d}:  MDLM={mdlm_tps:>10,.0f} TPS   Qwen3={qwen_tps:>10,.0f} TPS   "
          f"MDLM {mdlm_tps/qwen_tps:.1f}x faster")

print(f"\n{'=' * 80}")
print("MODELS")
print(f"{'=' * 80}")
mdlm_params = sum(p.numel() for p in model.parameters())
qwen_params = sum(p.numel() for p in oracle_model.parameters())
print(f"  MDLM v3:       {mdlm_params/1e6:.1f}M params (d_model=1024, 10 layers)")
print(f"  Qwen3-0.6B:    {qwen_params/1e6:.0f}M params (28 layers, trained on trillions)")
print(f"  VRAM total:    {torch.cuda.memory_allocated()/1e9:.1f} GB")
