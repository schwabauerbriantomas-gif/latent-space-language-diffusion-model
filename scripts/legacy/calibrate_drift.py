"""Calibrate drift threshold by measuring similarity distribution
on coherent vs incoherent text."""
import sys, math
import numpy as np
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

# ═══════════════════════════════════════════════════════════════
# Measure similarity of each token to the prompt's anchor
# in WELL-WRITTEN human text vs model-generated text
# ═══════════════════════════════════════════════════════════════

# Human-written paragraphs (coherent, on-topic)
human_texts = [
    "Machine learning is a subset of artificial intelligence that enables "
    "systems to learn from data. It uses statistical techniques to find "
    "patterns in large datasets. Common applications include image recognition, "
    "natural language processing, and recommendation systems. Deep learning, "
    "a subfield of machine learning, uses neural networks with many layers.",

    "Climate change refers to long-term shifts in global temperatures and "
    "weather patterns. The primary cause is the burning of fossil fuels, "
    "which releases greenhouse gases into the atmosphere. These gases trap "
    "heat, leading to rising sea levels and extreme weather events. "
    "Scientists warn that urgent action is needed to limit warming.",

    "Python is a high-level programming language known for its simplicity. "
    "It supports multiple programming paradigms including object-oriented "
    "and functional programming. Python's extensive standard library makes "
    "it suitable for web development, data analysis, and automation. "
    "The language emphasizes code readability and clean syntax.",
]

print("=" * 70)
print("CALIBRATION: Token-level similarity to prompt anchor")
print("=" * 70)

for text in human_texts:
    ids = tokenizer.encode(text, add_special=False)
    if len(ids) < 40:
        continue

    # Use first 8 tokens as anchor (prompt region)
    anchor_len = 8
    t = torch.tensor([ids], device=DEVICE)
    with torch.no_grad():
        hidden = model.get_embeddings(t)

    anchor = hidden[0, :anchor_len].mean(dim=0)  # [D]
    anchor = F.normalize(anchor.unsqueeze(0), dim=-1).squeeze(0)

    # Per-token similarity
    token_embs = F.normalize(hidden[0], dim=-1)  # [seq, D]
    sims = F.cosine_similarity(token_embs, anchor.unsqueeze(0), dim=-1)  # [seq]

    # Stats per region
    region1 = sims[:anchor_len]
    region2 = sims[anchor_len:anchor_len+15]
    region3 = sims[anchor_len+15:anchor_len+30]
    region4 = sims[anchor_len+30:]

    text_short = text[:50]
    print(f"\n  Text: '{text_short}...'")
    print(f"  Anchor (tokens 0-{anchor_len}):       mean={region1.mean():.3f}")
    if region2.numel() > 0:
        print(f"  Early generation ({anchor_len}-{anchor_len+15}):    mean={region2.mean():.3f} min={region2.min():.3f}")
    if region3.numel() > 0:
        print(f"  Mid generation ({anchor_len+15}-{anchor_len+30}):     mean={region3.mean():.3f} min={region3.min():.3f}")
    if region4.numel() > 0:
        print(f"  Late generation ({anchor_len+30}+):        mean={region4.mean():.3f} min={region4.min():.3f}")

# Now measure on MODEL-GENERATED text (from previous test)
print(f"\n{'─' * 70}")
print("MODEL-GENERATED TEXT (known drift)")
print(f"{'─' * 70}")

model_texts = [
    "The future of artificial intelligence and machine learning already emerged. "
    "The future of artificial intelligence has changed in the early 1980s to "
    "the 1970s. This has led to the rapid development of artificial intelligence "
    "and machine learning in the context of healthcare.",

    "Climate change is one of the biggest challenges facing the world. "
    "The impacts of climate change are the face of climate change. "
    "It is a must-minded path and the environment and environment are "
    "learning in it. It is a safe and effective way to reduce the environmental "
    "impact of climate change and to stop it.",
]

for text in model_texts:
    ids = tokenizer.encode(text, add_special=False)
    if len(ids) < 40:
        continue

    anchor_len = min(8, len(ids) // 4)
    t = torch.tensor([ids], device=DEVICE)
    with torch.no_grad():
        hidden = model.get_embeddings(t)

    anchor = hidden[0, :anchor_len].mean(dim=0)
    anchor = F.normalize(anchor.unsqueeze(0), dim=-1).squeeze(0)

    token_embs = F.normalize(hidden[0], dim=-1)
    sims = F.cosine_similarity(token_embs, anchor.unsqueeze(0), dim=-1)

    region1 = sims[:anchor_len]
    region2 = sims[anchor_len:anchor_len+15]
    region3 = sims[anchor_len+15:anchor_len+30]
    region4 = sims[anchor_len+30:]

    text_short = text[:50]
    print(f"\n  Text: '{text_short}...'")
    print(f"  Anchor (tokens 0-{anchor_len}):       mean={region1.mean():.3f}")
    if region2.numel() > 0:
        print(f"  Early generation ({anchor_len}-{anchor_len+15}):    mean={region2.mean():.3f} min={region2.min():.3f}")
    if region3.numel() > 0:
        print(f"  Mid generation ({anchor_len+15}-{anchor_len+30}):     mean={region3.mean():.3f} min={region3.min():.3f}")
    if region4.numel() > 0:
        print(f"  Late generation ({anchor_len+30}+):        mean={region4.mean():.3f} min={region4.min():.3f}")

print(f"\n{'=' * 70}")
print("CALIBRATION RESULT")
print(f"{'=' * 70}")
print("Compare human vs model means to set drift_threshold.")
print("Good threshold = midpoint between human-low and model-low.")
