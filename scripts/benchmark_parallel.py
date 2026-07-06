"""
Full-parallel MDLM + Qwen3 validation pipeline.

Generates ALL tokens simultaneously (no left-to-right blocks), then
validates the complete sequence with Qwen3 and regenerates bad segments.

This recovers the parallelism advantage of masked diffusion:
  - Semi-AR (current): 16 blocks × 4 steps = 64 sequential forward passes
  - Full parallel:     32 diffusion steps, all positions at once
"""
import sys, math, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F
from mdlm_bpe_v3 import MDLMConfig, MDLMBPEV3, BPETokenizer
from hrm_refiner import RepetitionReviewer
from hybrid_speculative import HybridSpeculative
from ar_oracle_hrm import load_oracle

DEVICE = "cuda"
REPO = Path(__file__).resolve().parent.parent
ckpt = torch.load(REPO / "checkpoints" / "mdlm_bpe_v3_best.pt",
                   map_location=DEVICE, weights_only=False)
config = MDLMConfig(**ckpt["config"])
model = MDLMBPEV3(config).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
tokenizer = BPETokenizer()
mask_id = tokenizer.mask_id

oracle_model, oracle_tok = load_oracle("Qwen/Qwen3-0.6B")
hybrid = HybridSpeculative(model, tokenizer, "Qwen/Qwen3-0.6B",
                           segment_size=8, surprise_threshold=1.0,
                           max_regen_fraction=0.6, oracle_temp=0.6)
reviewer = RepetitionReviewer(
    pad_id=tokenizer.pad_id, mask_id=tokenizer.mask_id,
    bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
)

special_ids = {tokenizer.pad_id, mask_id, tokenizer.bos_id, tokenizer.eos_id}

# ═══════════════════════════════════════════════════════════════
# GUIDANCE HELPERS
# ═══════════════════════════════════════════════════════════════

def apply_freq(logits, tokens, penalty):
    b,s,v = logits.shape
    counts = torch.zeros(b,v,device=tokens.device)
    counts.scatter_add_(1, tokens, torch.ones_like(tokens,dtype=torch.float))
    counts[:, list(special_ids)] = 0
    return logits - (penalty * torch.sqrt(counts.float())).unsqueeze(1)

def apply_rep(logits, tokens, penalty):
    b,s,v = logits.shape
    used = torch.zeros(b,v,dtype=torch.bool,device=tokens.device)
    used.scatter_(1, tokens, True)
    used[:, list(special_ids)] = False
    factor = torch.where(used, torch.tensor(1/penalty,device=logits.device),
                         torch.tensor(1.0,device=logits.device))
    return logits * factor.unsqueeze(1)

def apply_no_repeat_bigram(logits, tokens):
    """Ban tokens that would complete an already-seen bigram."""
    seq_len = tokens.shape[1]
    if seq_len < 2: return logits
    ban = torch.zeros_like(logits, dtype=torch.bool)
    spec_tensor = torch.tensor(list(special_ids), device=tokens.device)
    b = 0
    seq_t = tokens[b]
    prefixes = seq_t[:-1]
    nexts = seq_t[1:]
    valid = ~torch.isin(prefixes, spec_tensor) & ~torch.isin(nexts, spec_tensor)
    for pos in range(1, seq_len):
        cur = seq_t[pos-1]
        if cur.item() in special_ids: continue
        match = (prefixes == cur) & valid
        if match.any():
            banned = nexts[match].unique()
            ban[b, pos, banned] = True
    return logits.masked_fill(ban, float('-inf'))

def apply_top_p(probs, p=0.95):
    sp, si = torch.sort(probs, descending=True)
    cs = torch.cumsum(sp, dim=-1)
    sp[cs - sp > p] = 0
    new = torch.zeros_like(probs)
    new.scatter_(1, si, sp)
    return new / new.sum(dim=-1, keepdim=True).clamp(min=1e-8)


# ═══════════════════════════════════════════════════════════════
# FULL PARALLEL GENERATION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def gen_full_parallel(prompt_ids, seq_len=64, n_steps=32, temperature=0.7,
                      use_guidance=True):
    """Full-parallel diffusion: ALL positions predicted simultaneously.

    No left-to-right blocks. Every diffusion step processes all positions
    at once. This is the maximum-parallelism mode.

    With adaptive guidance, the repetition problem (v2's main failure)
    is handled during generation.
    """
    full = torch.full((1, seq_len), mask_id, device=DEVICE)
    plen = min(len(prompt_ids), seq_len)
    full[:, :plen] = torch.tensor(prompt_ids[:plen], device=DEVICE)

    for step in range(n_steps):
        t_val = max(1.0 - step / n_steps, 0.01)
        t = torch.full((1,), t_val, device=DEVICE)
        logits = model(full, t)

        if use_guidance:
            logits = apply_freq(logits, full, 0.4)
            logits = apply_rep(logits, full, 1.3)
            logits = apply_no_repeat_bigram(logits, full)

        mask_positions = (full[0] == mask_id)
        if not mask_positions.any(): break

        idxs = mask_positions.nonzero(as_tuple=True)[0]
        temp_logits = logits[0, idxs] / max(temperature, 0.01)
        probs = F.softmax(temp_logits, dim=-1)
        probs = apply_top_p(probs, 0.95)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        conf = probs.max(dim=-1)[0]

        # Unmask proportionally: more tokens early, fewer late
        n_masked = len(idxs)
        n_unmask = max(1, n_masked // max(n_steps - step, 1))
        tc, ti = conf.topk(min(n_unmask, n_masked))
        full[0, idxs[ti]] = sampled[ti]

    return full


@torch.no_grad()
def gen_semi_ar(prompt_ids, seq_len=64, block_size=4):
    """Semi-AR with adaptive guidance (current best)."""
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
            logits = apply_freq(logits, full, freq_p)
            logits = apply_rep(logits, full, rep_p)
            logits = apply_no_repeat_bigram(logits, full)

            mask_in = (full[0, bs:be] == mask_id)
            if not mask_in.any(): break
            idxs = mask_in.nonzero(as_tuple=True)[0]
            pl = logits[0, bs:be][idxs] / max(temp, 0.01)
            probs = F.softmax(pl, dim=-1)
            probs = apply_top_p(probs, 0.95)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            conf = probs.max(dim=-1)[0]
            n_unmask = max(1, len(idxs) // (n_steps - step))
            tc, ti = conf.topk(min(n_unmask, len(idxs)))
            full[0, bs + idxs[ti]] = sampled[ti]
        bi += 1
    return full


# ═══════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def oracle_score(text):
    ids = oracle_tok.encode(text, add_special_tokens=False)
    if len(ids) < 2: return 0.0
    inp = torch.tensor([ids], device=DEVICE)
    logits = oracle_model(inp).logits
    lp = F.log_softmax(logits.float(), dim=-1)
    actual = inp[0, 1:]
    pred = lp[0, :-1]
    tlps = pred.gather(1, actual.unsqueeze(1)).squeeze(1)
    return tlps.mean().item()

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
    "To build a reliable system, you need",
    "Climate change is one of the biggest challenges",
    "The key to success in any project is",
    "Education is important because",
    "Programming is a skill that requires",
]

configs = [
    ("Semi-AR + guidance (current)", "semi_ar"),
    ("Full-parallel + guidance",     "full_parallel"),
    ("Full-parallel + guidance + Qwen3 validate", "full_parallel_hybrid"),
    ("Semi-AR + guidance + Qwen3 validate",       "semi_ar_hybrid"),
]

print("=" * 80)
print("FULL PARALLEL vs SEMI-AR — SPEED + QUALITY + QWEN3 VALIDATION")
print("=" * 80)

all_results = {}

for config_name, mode in configs:
    lps = []
    reps = []
    tpss = []
    times = []

    print(f"\n{'─' * 80}")
    print(f"  {config_name}")
    print(f"{'─' * 80}")

    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special=False)

        t0 = time.time()

        if mode == "semi_ar":
            tokens = gen_semi_ar(ids, seq_len=64, block_size=4)
            text = tokenizer.decode(tokens[0].cpu().tolist())
        elif mode == "full_parallel":
            tokens = gen_full_parallel(ids, seq_len=64, n_steps=32,
                                       temperature=0.7, use_guidance=True)
            text = tokenizer.decode(tokens[0].cpu().tolist())
        elif mode == "full_parallel_hybrid":
            tokens = gen_full_parallel(ids, seq_len=64, n_steps=32,
                                       temperature=0.7, use_guidance=True)
            text, stats = hybrid.verify_and_refine(tokens, prompt_len=len(ids))
            if not isinstance(text, str):
                text = tokenizer.decode(text[0].cpu().tolist())
        elif mode == "semi_ar_hybrid":
            tokens = gen_semi_ar(ids, seq_len=64, block_size=4)
            text, stats = hybrid.verify_and_refine(tokens, prompt_len=len(ids))
            if not isinstance(text, str):
                text = tokenizer.decode(text[0].cpu().tolist())

        dt = time.time() - t0

        lp = oracle_score(text)
        rs = rep_score(text)
        gen_tokens = max(len(tokenizer.encode(text, add_special=False)) - len(ids), 1)
        tps = gen_tokens / dt

        lps.append(lp)
        reps.append(rs)
        tpss.append(tps)
        times.append(dt)

        print(f"  [{rs:.2f}] lp={lp:.2f} {tps:.1f}t/s ({dt:.1f}s)  {prompt}")
        print(f"         → {text.strip()[:120]}")

    avg_lp = np.mean(lps)
    avg_rep = np.mean(reps)
    avg_tps = np.mean(tpss)
    avg_time = np.mean(times)

    all_results[config_name] = {
        "lp": avg_lp, "rep": avg_rep, "tps": avg_tps, "time": avg_time
    }
    print(f"\n  >>> Avg: lp={avg_lp:.2f} rep={avg_rep:.2f} {avg_tps:.1f}t/s ({avg_time:.1f}s)")

# Summary
print(f"\n{'=' * 80}")
print("FINAL COMPARISON")
print(f"{'=' * 80}")
print(f"\n  {'Method':<45} {'LP':>6} {'Rep':>6} {'TPS':>8} {'Time':>7}")
print(f"  {'-'*72}")
for name, r in sorted(all_results.items(), key=lambda x: -x[1]["tps"]):
    print(f"  {name:<45} {r['lp']:>6.2f} {r['rep']:>6.2f} {r['tps']:>7.1f} {r['time']:>6.1f}s")

# Speedup calculation
fp_tps = all_results.get("Full-parallel + guidance", {}).get("tps", 0)
sa_tps = all_results.get("Semi-AR + guidance (current)", {}).get("tps", 0)
if sa_tps > 0:
    print(f"\n  Full-parallel speedup vs semi-AR: {fp_tps/sa_tps:.2f}x")

fph_lp = all_results.get("Full-parallel + guidance + Qwen3 validate", {}).get("lp", 0)
sah_lp = all_results.get("Semi-AR + guidance + Qwen3 validate", {}).get("lp", 0)
print(f"  Quality (hybrid): full-parallel lp={fph_lp:.2f} vs semi-AR lp={sah_lp:.2f}")
