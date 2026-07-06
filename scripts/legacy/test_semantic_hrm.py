"""Test Semantic Coherence HRM: drift detection and correction."""
import sys, math, time
import numpy as np
sys.path.insert(0, "/root/ff-splatdiffusion/src")

import torch
import torch.nn.functional as F
from mdlm_bpe_v3 import MDLMConfig, MDLMBPEV3, BPETokenizer
from hrm_refiner import RepetitionReviewer
from semantic_hrm import SemanticCoherenceHRM, SemanticState

DEVICE = "cuda"
ckpt = torch.load("/root/ff-splatdiffusion/checkpoints/mdlm_bpe_v3_best.pt",
                   map_location=DEVICE, weights_only=False)
config = MDLMConfig(**ckpt["config"])
model = MDLMBPEV3(config).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
tokenizer = BPETokenizer()

# ═══════════════════════════════════════════════════════════════
# TEST 1: Does the model produce semantically distinguishable
# embeddings for different topics?
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Embedding separability (can we distinguish topics?)")
print("=" * 70)

topics = [
    "Machine learning is a subset of artificial intelligence",
    "The mitochondria is the powerhouse of the cell",
    "Climate change affects global weather patterns",
    "Python is a popular programming language",
    "The Roman Empire fell in 476 AD",
]

# Get embeddings
embs = []
for text in topics:
    ids = tokenizer.encode(text, add_special=True)
    t = torch.tensor([ids], device=DEVICE)
    with torch.no_grad():
        hidden = model.get_embeddings(t)
    emb = hidden[0].mean(dim=0)
    embs.append(emb)

embs = torch.stack(embs)  # [5, D]

# Compute pairwise cosine similarities
embs_norm = F.normalize(embs, dim=-1)
sim_matrix = embs_norm @ embs_norm.T

print("\n  Cosine similarity matrix:")
print(f"  {'':>40}", end="")
for i in range(5):
    print(f"  T{i}", end="")
print()
for i in range(5):
    label = topics[i][:38]
    print(f"  {label:>40}", end="")
    for j in range(5):
        val = sim_matrix[i][j].item()
        if i == j:
            print(f"  --", end="")
        else:
            print(f" {val:.2f}", end="")
    print()

# Expected: diagonal=1.0, off-diagonal < 1.0, related topics higher
off_diag = sim_matrix[~torch.eye(5, dtype=torch.bool, device=DEVICE)]
print(f"\n  Off-diagonal mean: {off_diag.mean():.3f}")
print(f"  Off-diagonal std:  {off_diag.std():.3f}")
print(f"  Range: [{off_diag.min():.3f}, {off_diag.max():.3f}]")

if off_diag.std() > 0.02:
    print("  ✓ Embeddings ARE semantically distinguishable — HRM can work")
else:
    print("  ✗ Embeddings are NOT distinguishable — HRM won't work well")


# ═══════════════════════════════════════════════════════════════
# TEST 2: Drift detection on generated text
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("TEST 2: Drift detection on generated text")
print("=" * 70)

semantic_hrm = SemanticCoherenceHRM(
    model, tokenizer,
    drift_threshold=0.65,
    guidance_strength=0.3,
    ema_decay=0.85,
)

mask_id = tokenizer.mask_id

# Generate text using baseline semi-AR (no guidance)
@torch.no_grad()
def generate_baseline(prompt_ids, seq_len=64, block_size=4, temp=0.7):
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
            pl = logits[0, bs:be][idxs] / max(temp, 0.01)
            probs = F.softmax(pl, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            conf = probs.max(dim=-1)[0]
            n_unmask = max(1, len(idxs) // (n_steps - step))
            tc, ti = conf.topk(min(n_unmask, len(idxs)))
            full[0, bs + idxs[ti]] = sampled[ti]
    return full

prompts_test = [
    "The future of artificial intelligence",
    "Climate change is one of the biggest challenges",
]

for prompt_text in prompts_test:
    print(f"\n  Prompt: '{prompt_text}'")
    ids = tokenizer.encode(prompt_text, add_special=False)

    # Generate baseline
    tokens = generate_baseline(ids, seq_len=64, block_size=4)
    text = tokenizer.decode(tokens[0].cpu().tolist())
    print(f"  Generated: {text.strip()[:150]}")

    # Initialize semantic state from prompt
    state = semantic_hrm.init_state(tokens, len(ids))

    # Check drift in each block
    print(f"  Block-by-block drift analysis:")
    for bs in range(len(ids), 64, 4):
        be = min(bs + 4, 64)
        drift_mask, avg_sim = semantic_hrm.detect_drift(tokens, bs, be, state)
        n_drift = drift_mask.sum().item()
        block_text = tokenizer.decode(tokens[0, bs:be].cpu().tolist())
        marker = "⚠ DRIFT" if n_drift > 0 else "✓"
        print(f"    [{bs:2d}-{be:2d}] sim={avg_sim:.3f} drift={n_drift} {marker}  '{block_text.strip()}'")


# ═══════════════════════════════════════════════════════════════
# TEST 3: Correction — does semantic guidance improve drift?
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("TEST 3: Semantic correction")
print("=" * 70)

for prompt_text in prompts_test:
    print(f"\n  Prompt: '{prompt_text}'")
    ids = tokenizer.encode(prompt_text, add_special=False)

    # Generate baseline
    tokens = generate_baseline(ids, seq_len=64, block_size=4)
    text_before = tokenizer.decode(tokens[0].cpu().tolist())

    # Init state
    state = semantic_hrm.init_state(tokens, len(ids))

    # Refine each block
    total_drifted = 0
    total_corrected = 0
    for bs in range(len(ids), 64, 4):
        be = min(bs + 4, 64)
        # Check drift before correction
        drift_before, sim_before = semantic_hrm.detect_drift(tokens, bs, be, state)
        n_before = drift_before.sum().item()
        total_drifted += n_before

        if n_before > 0:
            # Refine this block
            tokens, stats = semantic_hrm.refine_block(
                tokens, bs, be, state, n_steps=6, temperature=0.5
            )
            total_corrected += stats.get("corrected", False)
            print(f"    Block [{bs:2d}-{be:2d}]: drifted={n_before} "
                  f"sim {stats['pre_sim']:.3f}→{stats['post_sim']:.3f}")

    text_after = tokenizer.decode(tokens[0].cpu().tolist())
    print(f"\n  BEFORE: {text_before.strip()[:150]}")
    print(f"  AFTER:  {text_after.strip()[:150]}")
    print(f"  Total drifted positions: {total_drifted}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
