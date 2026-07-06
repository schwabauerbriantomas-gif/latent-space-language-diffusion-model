"""Test Hybrid Speculative: MDLM drafts + Qwen3 regenerates bad segments."""
import sys, math, time
import numpy as np
sys.path.insert(0, "/root/ff-splatdiffusion/src")

import torch
import torch.nn.functional as F
from mdlm_bpe_v3 import MDLMConfig, MDLMBPEV3, BPETokenizer
from hrm_refiner import RepetitionReviewer
from hybrid_speculative import HybridSpeculative

DEVICE = "cuda"
ckpt = torch.load("/root/ff-splatdiffusion/checkpoints/mdlm_bpe_v3_best.pt",
                   map_location=DEVICE, weights_only=False)
config = MDLMConfig(**ckpt["config"])
model = MDLMBPEV3(config).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
tokenizer = BPETokenizer()

print("Loading Hybrid Speculative pipeline...")
hybrid = HybridSpeculative(
    model, tokenizer,
    oracle_model_name="Qwen/Qwen3-0.6B",
    segment_size=8,
    surprise_threshold=1.0,
    max_regen_fraction=0.6,
    oracle_temp=0.6,
)

mask_id = tokenizer.mask_id

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

print("=" * 70)
print("HYBRID SPECULATIVE: MDLM draft + Qwen3 segment regeneration")
print("=" * 70)

for prompt in prompts:
    print(f"\n{'─' * 70}")
    print(f"PROMPT: {prompt}")
    print(f"{'─' * 70}")

    ids = tokenizer.encode(prompt, add_special=False)

    # MDLM generates draft
    t0 = time.time()
    tokens = generate_guided(ids, seq_len=64, block_size=4)
    text_draft = tokenizer.decode(tokens[0].cpu().tolist())
    t_gen = time.time() - t0

    # Hybrid refinement
    t0 = time.time()
    final_text, stats = hybrid.verify_and_refine(tokens, prompt_len=len(ids))
    t_refine = time.time() - t0

    lp_before = stats.get("mean_lp_before", 0)
    lp_after = stats.get("mean_lp_after", 0)
    regen = stats.get("regenerated", 0)

    all_before_lp.append(lp_before)
    all_after_lp.append(lp_after)

    print(f"  DRAFT:   lp={lp_before:.2f} ({t_gen:.1f}s)")
    print(f"           → {text_draft.strip()[:140]}")
    print(f"  REFINED: lp={lp_after:.2f} ({t_refine:.1f}s) "
          f"regen={regen}/{stats.get('segments',0)} segments")
    print(f"           → {final_text.strip()[:200]}")
    if "bad_segments" in stats:
        for bs in stats["bad_segments"][:2]:
            print(f"           bad: '{bs['text']}' (lp={bs['lp']:.2f})")

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"  Oracle log-prob:  {np.mean(all_before_lp):.2f} → {np.mean(all_after_lp):.2f} "
      f"(Δ={np.mean(all_after_lp)-np.mean(all_before_lp):+.2f})")
