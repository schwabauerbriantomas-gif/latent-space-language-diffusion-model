"""Test top-p (nucleus) sampling vs current full-distribution sampling."""
import sys, math, time
import numpy as np
sys.path.insert(0, "/root/ff-splatdiffusion/src")

import torch
import torch.nn.functional as F
from mdlm_bpe_v3 import MDLMConfig, MDLMBPEV3, BPETokenizer
from hrm_refiner import RepetitionReviewer

DEVICE = "cuda"
ckpt = torch.load("/root/ff-splatdiffusion/checkpoints/mdlm_bpe_v3_best.pt",
                   map_location=DEVICE, weights_only=False)
config = MDLMConfig(**ckpt["config"])
model = MDLMBPEV3(config).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
tokenizer = BPETokenizer()
mask_id = tokenizer.mask_id

# Guidance functions (same as evaluate_v3_hrm.py)
def apply_freq(logits, tokens, penalty=0.4):
    b,s,v = logits.shape
    counts = torch.zeros(b,v,device=tokens.device)
    counts.scatter_add_(1, tokens, torch.ones_like(tokens,dtype=torch.float))
    counts[:, [0,1,2,3]] = 0
    return logits - (penalty * torch.sqrt(counts.float())).unsqueeze(1)

def apply_rep(logits, tokens, penalty=1.3):
    b,s,v = logits.shape
    used = torch.zeros(b,v,dtype=torch.bool,device=tokens.device)
    used.scatter_(1, tokens, True)
    used[:, [0,1,2,3]] = False
    factor = torch.where(used, torch.tensor(1/penalty,device=logits.device),
                         torch.tensor(1.0,device=logits.device))
    return logits * factor.unsqueeze(1)

@torch.no_grad()
def sample_v3(model, tokenizer, prompt_ids, seq_len=64, block_size=4,
              temp=0.7, top_p=1.0, use_guidance=True):
    """Semi-AR sampling with optional top-p and guidance."""
    model.eval()
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

            if use_guidance:
                logits = apply_freq(logits, full, 0.4)
                logits = apply_rep(logits, full, 1.3)

            mask_in = (full[0, bs:be] == mask_id)
            if not mask_in.any():
                break
            idxs = mask_in.nonzero(as_tuple=True)[0]
            pl = logits[0, bs:be][idxs] / max(temp, 0.01)
            probs = F.softmax(pl, dim=-1)

            # TOP-P FILTERING
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                # Remove tokens above threshold (keep first token that exceeds)
                sorted_mask = cumsum - sorted_probs > top_p
                sorted_probs[sorted_mask] = 0
                # Re-normalize
                probs_new = torch.zeros_like(probs)
                probs_new.scatter_(1, sorted_idx, sorted_probs)
                probs_new = probs_new / probs_new.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                probs = probs_new

            probs = probs.clamp(min=0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            conf = probs.max(dim=-1)[0]
            n_unmask = max(1, len(idxs) // (n_steps - step))
            tc, ti = conf.topk(min(n_unmask, len(idxs)))
            full[0, bs + idxs[ti]] = sampled[ti]

    return tokenizer.decode(full[0].cpu().tolist())


reviewer = RepetitionReviewer(
    pad_id=tokenizer.pad_id, mask_id=tokenizer.mask_id,
    bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
)

def score(text):
    ids = tokenizer.encode(text, add_special=False)
    if not ids: return 1.0
    t = torch.tensor([ids[:64]], device=DEVICE)
    if t.shape[1] < 64:
        t = torch.cat([t, torch.full((1, 64-t.shape[1]), tokenizer.pad_id, device=DEVICE)], 1)
    return reviewer.score_sequence(t[0])

prompts = [
    "The future of artificial intelligence",
    "To build a reliable system, you need",
    "Climate change is one of the biggest challenges",
    "The key to success in any project is",
    "Education is important because",
    "Programming is a skill that requires",
]

print("=" * 70)
print("TOP-P SAMPLING COMPARISON")
print("=" * 70)

for top_p_val in [1.0, 0.95, 0.9]:
    scores = []
    print(f"\n{'─'*70}")
    print(f"top_p = {top_p_val}")
    print(f"{'─'*70}")
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special=False)
        text = sample_v3(model, tokenizer, ids, seq_len=64, block_size=4,
                         temp=0.7, top_p=top_p_val, use_guidance=True)
        s = score(text)
        scores.append(s)
        print(f"  [{s:.2f}] {prompt}")
        print(f"         → {text.strip()[:140]}")
    print(f"\n  Avg score: {np.mean(scores):.3f}")
