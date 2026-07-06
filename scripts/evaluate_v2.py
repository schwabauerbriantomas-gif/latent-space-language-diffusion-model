"""
Evaluate MDLM-BPE v2 — comprehensive quality and speed assessment.
"""
import json
import sys
import time
import math
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = REPO / "results"
CHECKPOINT_DIR = REPO / "checkpoints"
DATA_FILE = REPO / "data" / "train_tokens_large.npy"

from mdlm_bpe_v2 import (
    MDLMConfig, MDLMBPEV2, BPETokenizer,
    forward_mask_bpe, mdlm_bpe_loss_v2, sample_mdlm_v2,
)


def load_model():
    ckpt_path = CHECKPOINT_DIR / "mdlm_bpe_v2_best.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "mdlm_bpe_v2_final.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = MDLMConfig(**ckpt["config"])
    model = MDLMBPEV2(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokenizer = BPETokenizer()
    print(f"Loaded: {ckpt_path.name} (step {ckpt.get('step', '?')}, "
          f"PPL={ckpt.get('ppl', '?')})")
    return model, tokenizer, config


def measure_perplexity(model, tokenizer, n=2000):
    print("\n── Perplexity ──")
    tokens = np.load(DATA_FILE)
    holdout = tokens[-len(tokens)//10:]
    if len(holdout) > n:
        idx = np.random.choice(len(holdout), n, replace=False)
        holdout = holdout[idx]

    total_loss = 0.0
    total_masked = 0
    bs = 64
    with torch.no_grad():
        for i in range(0, len(holdout), bs):
            batch = torch.from_numpy(holdout[i:i+bs].astype(np.int64)).to(DEVICE)
            t = torch.full((len(batch),), 0.5, device=DEVICE)
            masked, mask_pos = forward_mask_bpe(batch, t, tokenizer.mask_id)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(masked, t)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1])[mask_pos.reshape(-1)],
                batch.reshape(-1)[mask_pos.reshape(-1)],
                reduction='sum',
            )
            total_loss += loss.item()
            total_masked += mask_pos.sum().item()

    avg = total_loss / total_masked
    ppl = math.exp(avg)
    print(f"  CE loss: {avg:.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    return avg, ppl


def measure_tps(model, tokenizer, seq_len=64):
    print("\n── Throughput ──")
    results = {}
    for bs, steps in [(1, 20), (10, 20), (50, 20), (100, 16), (200, 12)]:
        try:
            # Warmup
            for _ in range(2):
                tokens = torch.full((bs, seq_len), tokenizer.mask_id, device=DEVICE)
                t = torch.full((bs,), 0.5, device=DEVICE)
                with torch.no_grad():
                    _ = model(tokens, t)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(5):
                tokens = torch.full((bs, seq_len), tokenizer.mask_id, device=DEVICE)
                for step in range(steps):
                    t_val = max(1.0 - step / steps, 0.01)
                    t = torch.full((bs,), t_val, device=DEVICE)
                    with torch.no_grad():
                        logits = model(tokens, t)
                    mask_pos = (tokens == tokenizer.mask_id)
                    if not mask_pos.any():
                        break
                    probs = F.softmax(logits / 0.7, dim=-1)
                    conf = probs.max(dim=-1)[0]
                    conf[~mask_pos] = -1
                    sampled = torch.multinomial(
                        probs.reshape(-1, probs.shape[-1]), 1
                    ).reshape(bs, seq_len)
                    k = max(1, int(mask_pos.sum().item()) // (steps - step))
                    for b in range(bs):
                        top_idx = conf[b].topk(min(k, int(mask_pos[b].sum().item()))).indices
                        tokens[b, top_idx] = sampled[b, top_idx]
            torch.cuda.synchronize()
            elapsed = time.time() - start
            tps = bs * seq_len * 5 / elapsed
            results[f"batch_{bs}"] = {"tps": tps, "latency_ms": elapsed/5*1000}
            print(f"  Batch {bs:3d}: {tps:,.0f} TPS | {elapsed/5*1000:.0f} ms")
        except Exception as e:
            print(f"  Batch {bs}: {e}")
            results[f"batch_{bs}"] = {"error": str(e)}
    return results


def measure_quality(model, tokenizer, n=200, seq_len=64):
    print("\n── Quality ──")
    samples = sample_mdlm_v2(model, tokenizer, seq_len=seq_len,
                             n_samples=n, n_steps=20, temperature=0.7)
    unique = len(set(samples))
    all_words = []
    for s in samples:
        all_words.extend(s.split())
    d1 = len(set(all_words)) / max(len(all_words), 1)
    bigrams = list(zip(all_words[:-1], all_words[1:]))
    d2 = len(set(bigrams)) / max(len(bigrams), 1)
    print(f"  Unique: {unique}/{n} ({100*unique/n:.1f}%)")
    print(f"  Distinct-1: {d1:.4f}")
    print(f"  Distinct-2: {d2:.4f}")

    # Show samples
    print(f"\n  Qualitative samples:")
    for i, s in enumerate(samples[:8]):
        print(f"    [{i}] {s.strip()[:150]}")

    # Prompt completion
    print(f"\n  Prompt completion:")
    prompts = [
        "The future of artificial intelligence",
        "def fibonacci(n):",
        "To build a reliable system",
    ]
    for p in prompts:
        ids = tokenizer.encode(p, add_special=False)
        result = sample_mdlm_v2(model, tokenizer, prompt_ids=ids,
                                seq_len=seq_len, n_samples=1,
                                n_steps=32, temperature=0.7)
        print(f"    '{p}'")
        print(f"      → {result[0].strip()[:150]}")

    return {"distinct_1": d1, "distinct_2": d2, "uniqueness": unique/n}


def main():
    print("=" * 70)
    print("MDLM-BPE v2 EVALUATION")
    print("=" * 70)
    model, tokenizer, config = load_model()
    gpu = torch.cuda.get_device_name(0)
    params = sum(p.numel() for p in model.parameters())
    print(f"  GPU: {gpu}")
    print(f"  Params: {params:,} ({params/1e6:.1f}M)")

    ce, ppl = measure_perplexity(model, tokenizer)
    tps = measure_tps(model, tokenizer)
    quality = measure_quality(model, tokenizer)

    results = {
        "model": "MDLM-BPE v2",
        "gpu": gpu,
        "params": params,
        "perplexity": ppl,
        "ce_loss": ce,
        "tps": tps,
        "quality": quality,
    }
    out = RESULTS_DIR / "mdlm_bpe_v2_eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results: {out}")


if __name__ == "__main__":
    main()
