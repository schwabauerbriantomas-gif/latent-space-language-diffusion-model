"""Analyze logit distribution to find guidance improvements."""
import sys, math
sys.path.insert(0, "/root/ff-splatdiffusion/src")

import torch
import torch.nn.functional as F
from mdlm_bpe_v3 import MDLMConfig, MDLMBPEV3, BPETokenizer

DEVICE = "cuda"
ckpt = torch.load("/root/ff-splatdiffusion/checkpoints/mdlm_bpe_v3_best.pt",
                   map_location=DEVICE, weights_only=False)
config = MDLMConfig(**ckpt["config"])
model = MDLMBPEV3(config).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
tokenizer = BPETokenizer()

mask_id = tokenizer.mask_id

prompts = [
    "The future of artificial intelligence",
    "To build a reliable system, you need",
    "Climate change is one of the biggest challenges",
]

print("=" * 70)
print("LOGIT DISTRIBUTION ANALYSIS")
print("=" * 70)

for prompt in prompts:
    ids = tokenizer.encode(prompt, add_special=False)
    full = torch.full((1, 64), mask_id, device=DEVICE)
    full[:, :len(ids)] = torch.tensor(ids, device=DEVICE)

    with torch.no_grad():
        t = torch.full((1,), 0.3, device=DEVICE)
        logits = model(full, t)

    pos = len(ids)
    pos_logits = logits[0, pos]
    probs = F.softmax(pos_logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)

    entropy = -(probs * (probs + 1e-10).log()).sum().item()

    print(f"\nPrompt: '{prompt}'")
    print(f"  Position: {pos}")
    print(f"  Top-1: p={sorted_probs[0]:.4f} → '{tokenizer.decode([sorted_indices[0].item()])}'")
    print(f"  Entropy: {entropy:.3f} nats (max={math.log(tokenizer.vocab_size):.3f})")

    for p_thresh in [0.5, 0.8, 0.9]:
        n_tokens = (cumsum < p_thresh).sum().item() + 1
        print(f"  top-p={p_thresh}: {n_tokens} tokens")

    print(f"  Top-10:")
    for i in range(10):
        tok = tokenizer.decode([sorted_indices[i].item()])
        print(f"    {i+1:2d}. p={sorted_probs[i]:.4f}  '{tok}'")
