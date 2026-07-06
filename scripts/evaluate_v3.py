"""
Evaluate MDLM-BPE v3 with semi-AR sampling and logit guidance.
"""
import json
import sys
import time
import math
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = REPO / "results"
CHECKPOINT_DIR = REPO / "checkpoints"

from mdlm_bpe_v3 import (
    MDLMConfig, MDLMBPEV3, BPETokenizer, sample_semi_ar,
    generate_response_semi_ar,
)
from hrm_refiner import RepetitionReviewer


def load_model():
    ckpt_path = CHECKPOINT_DIR / "mdlm_bpe_v3_best.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "mdlm_bpe_v3_final.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = MDLMConfig(**ckpt["config"])
    model = MDLMBPEV3(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokenizer = BPETokenizer()
    print(f"Loaded: {ckpt_path.name} (step {ckpt.get('step', '?')}, "
          f"PPL={ckpt.get('ppl', '?')})")
    return model, tokenizer, config


def main():
    print("=" * 70)
    print("MDLM-BPE v3 EVALUATION")
    print("=" * 70)

    model, tokenizer, config = load_model()
    reviewer = RepetitionReviewer()

    gpu = torch.cuda.get_device_name(0)
    params = sum(p.numel() for p in model.parameters())
    print(f"  GPU: {gpu}")
    print(f"  Params: {params:,} ({params/1e6:.1f}M)")

    # Test all block sizes
    print("\n── Semi-AR Sampling (different block sizes) ──")
    prompts = [
        "The future of artificial intelligence",
        "def fibonacci(n):",
        "To build a reliable system, you need",
    ]

    for block_size in [1, 2, 4, 8, 16]:
        print(f"\n  Block size = {block_size}:")
        for prompt in prompts:
            ids = tokenizer.encode(prompt, add_special=False)
            result = sample_semi_ar(
                model, tokenizer, prompt_ids=ids, seq_len=64,
                n_samples=1, block_size=block_size, temperature=0.7,
            )
            print(f"    '{prompt}'")
            print(f"      → {result[0].strip()[:120]}")

    # Chatbot responses
    print("\n── Chatbot Responses ──")
    chat_prompts = [
        "What is machine learning?",
        "How do I write a Python function?",
        "Explain neural networks simply",
        "What is Python?",
    ]
    scores = []
    for prompt in chat_prompts:
        resp = generate_response_semi_ar(
            model, tokenizer, prompt, max_len=64, block_size=4, temperature=0.6,
        )
        # Score
        ids = tokenizer.encode(resp, add_special=False)
        if ids:
            t = torch.tensor([ids[:64]], device=DEVICE)
            if t.shape[1] < 64:
                t = torch.cat([t, torch.full((1, 64-t.shape[1]), tokenizer.pad_id, device=DEVICE)], 1)
            score = reviewer.score_sequence(t[0])
        else:
            score = 1.0
        scores.append(score)
        print(f"  Q: {prompt}")
        print(f"  A [{score:.2f}]: {resp.strip()[:150]}")
        print()

    # TPS
    print("── Throughput ──")
    for bs in [1, 10, 50]:
        tokens = torch.full((bs, 128), tokenizer.mask_id, device=DEVICE)
        t = torch.full((bs,), 0.5, device=DEVICE)
        for _ in range(3):
            with torch.no_grad():
                _ = model(tokens, t)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = model(tokens, t)
        torch.cuda.synchronize()
        tps = bs * 128 * 10 / (time.time() - start)
        print(f"  Batch {bs:3d}: {tps:,.0f} TPS")

    print(f"\n  Avg repetition score: {np.mean(scores):.2f}")


if __name__ == "__main__":
    main()
