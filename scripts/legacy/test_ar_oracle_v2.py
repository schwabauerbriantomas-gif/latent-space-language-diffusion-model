"""Compare AR-Oracle HRM v1 (guided regen) vs v2 (direct replacement)."""
import sys, math, time
import numpy as np
sys.path.insert(0, "/root/ff-splatdiffusion/src")

import torch
import torch.nn.functional as F
from mdlm_bpe_v3 import MDLMConfig, MDLMBPEV3, BPETokenizer
from hrm_refiner import RepetitionReviewer
from ar_oracle_hrm import AROracleHRM
from ar_oracle_hrm_v2 import AROracleHRMv2

DEVICE = "cuda"
ckpt = torch.load("/root/ff-splatdiffusion/checkpoints/mdlm_bpe_v3_best.pt",
                   map_location=DEVICE, weights_only=False)
config = MDLMConfig(**ckpt["config"])
model = MDLMBPEV3(config).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
tokenizer = BPETokenizer()

print("Loading AR Oracle HRM v2 (direct replacement)...")
oracle_v2 = AROracleHRMv2(
    model, tokenizer,
    oracle_model_name="Qwen/Qwen3-0.6B",
    surprise_threshold=1.5,
    correction_strength=4.0,
    max_correction_rate=0.3,
)

mask_id = tokenizer.mask_id

# Adaptive guidance
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

@torch.no_grad()
def generate_guided(prompt_ids, seq_len=64, block_size=4):
    model.eval()
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

prompts = [
    "The future of artificial intelligence",
    "To build a reliable system, you need",
    "Climate change is one of the biggest challenges",
    "The key to success in any project is",
    "Education is important because",
    "Programming is a skill that requires",
]

reviewer = RepetitionReviewer(
    pad_id=tokenizer.pad_id, mask_id=tokenizer.mask_id,
    bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
)

def rep_score(text):
    ids = tokenizer.encode(text, add_special=False)
    if not ids: return 1.0
    t = torch.tensor([ids[:64]], device=DEVICE)
    if t.shape[1] < 64:
        t = torch.cat([t, torch.full((1, 64-t.shape[1]), tokenizer.pad_id, device=DEVICE)], 1)
    return reviewer.score_sequence(t[0])

all_before_lp = []
all_after_lp = []
all_scores = []

print("=" * 70)
print("AR-ORACLE HRM v2: DIRECT REPLACEMENT")
print("=" * 70)

for prompt in prompts:
    print(f"\n{'─' * 70}")
    print(f"PROMPT: {prompt}")
    print(f"{'─' * 70}")

    ids = tokenizer.encode(prompt, add_special=False)

    # Generate
    t0 = time.time()
    tokens = generate_guided(ids, seq_len=64, block_size=4)
    text_guided = tokenizer.decode(tokens[0].cpu().tolist())
    t_gen = time.time() - t0

    # Score before
    scores_before, _ = oracle_v2.score_sequence(tokens, len(ids))

    # Refine with v2 (direct replacement)
    t0 = time.time()
    refined, stats = oracle_v2.refine(tokens, prompt_len=len(ids),
                                       temperature=0.0, max_rounds=3)
    t_refine = time.time() - t0
    text_refined = tokenizer.decode(refined[0].cpu().tolist())

    # Score after
    scores_after, _ = oracle_v2.score_sequence(refined, len(ids))

    gen_mask = torch.ones(64, device=DEVICE, dtype=torch.bool)
    gen_mask[:len(ids)] = False
    gen_mask |= (tokens[0] == mask_id)
    lp_before = scores_before[gen_mask].mean().item()
    lp_after = scores_after[gen_mask].mean().item()

    rs_before = rep_score(text_guided)
    rs_after = rep_score(text_refined)

    all_before_lp.append(lp_before)
    all_after_lp.append(lp_after)
    all_scores.append(rs_after)

    print(f"  GUIDED:  [{rs_before:.2f}] lp={lp_before:.2f} ({t_gen:.1f}s)")
    print(f"           → {text_guided.strip()[:140]}")
    print(f"  REFINED: [{rs_after:.2f}] lp={lp_after:.2f} ({t_refine:.1f}s) "
          f"replaced={stats['total_replaced']} rounds={stats['rounds']}")
    print(f"           → {text_refined.strip()[:140]}")
    for rd in stats.get("round_details", []):
        if rd.get("replaced", 0) > 0:
            print(f"           round {rd['round']}: {rd['replaced']} replaced, "
                  f"lp {rd['lp_before']:.2f}→{rd['lp_after']:.2f} "
                  f"(Δ={rd['improved']:+.2f})")

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"  Oracle log-prob:  {np.mean(all_before_lp):.2f} → {np.mean(all_after_lp):.2f} "
      f"(Δ={np.mean(all_after_lp)-np.mean(all_before_lp):+.2f})")
print(f"  Repetition score: {np.mean(all_scores):.3f}")
