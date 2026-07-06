"""
Evaluate MDLM-BPE quality after training.

Measures:
  1. Perplexity on held-out data
  2. Generation quality (free generation, prompt completion)
  3. TPS (tokens per second) at different batch sizes
  4. Diversity metrics (distinct-n, self-BLEU)
  5. Qualitative samples
"""
import json
import sys
import time
import math
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = REPO / "results"
DATA_FILE = REPO / "data" / "train_tokens.npy"

from mdlm_bpe import (
    MDLMConfig, MDLMBPETransformer, BPETokenizer,
    forward_mask_bpe, sample_mdlm_bpe,
)


def load_model(checkpoint_path=None):
    """Load trained model."""
    if checkpoint_path is None:
        checkpoint_path = REPO / "checkpoints" / "mdlm_bpe_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = REPO / "checkpoints" / "mdlm_bpe_final.pt"

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = MDLMConfig(**ckpt["config"])
    model = MDLMBPETransformer(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokenizer = BPETokenizer()
    print(f"Loaded: {checkpoint_path.name} (step {ckpt.get('step', '?')})")
    return model, tokenizer, config


def measure_perplexity(model, tokenizer, n_samples=1000):
    """Measure perplexity on held-out data."""
    print("\n── Perplexity ──")
    tokens = np.load(DATA_FILE)
    # Use last 10% as held-out
    n_holdout = len(tokens) // 10
    holdout = tokens[-n_holdout:]
    print(f"  Held-out sequences: {len(holdout):,}")

    # Sample subset
    if len(holdout) > n_samples:
        idx = np.random.choice(len(holdout), n_samples, replace=False)
        holdout = holdout[idx]

    total_loss = 0.0
    total_masked = 0
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(holdout), batch_size):
            batch = holdout[i:i+batch_size]
            batch_t = torch.from_numpy(batch.astype(np.int64)).to(DEVICE)

            # Use t=0.5 (half masked) for evaluation
            t = torch.full((len(batch_t),), 0.5, device=DEVICE)
            masked, mask_pos = forward_mask_bpe(batch_t, t, tokenizer.mask_id)
            logits = model(masked, t)

            mask_flat = mask_pos.reshape(-1)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            tokens_flat = batch_t.reshape(-1)

            loss = F.cross_entropy(
                logits_flat[mask_flat], tokens_flat[mask_flat],
                reduction='sum',
            )
            total_loss += loss.item()
            total_masked += mask_flat.sum().item()

    avg_loss = total_loss / total_masked
    ppl = math.exp(avg_loss)
    print(f"  Avg CE loss: {avg_loss:.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    return avg_loss, ppl


def measure_tps(model, tokenizer, seq_len=64):
    """Measure tokens per second at various batch sizes."""
    print("\n── Throughput (TPS) ──")
    results = {}

    configs = [
        (1, 20),
        (10, 20),
        (50, 20),
        (100, 16),
        (200, 12),
    ]

    for batch_size, n_steps in configs:
        try:
            # Warmup
            for _ in range(2):
                tokens = torch.full(
                    (batch_size, seq_len), tokenizer.mask_id, device=DEVICE,
                )
                for step in range(min(3, n_steps)):
                    t = torch.full((batch_size,), 0.5, device=DEVICE)
                    with torch.no_grad():
                        _ = model(tokens, t)

            torch.cuda.synchronize()
            start = time.time()

            # Actual measurement
            for _ in range(5):
                tokens = torch.full(
                    (batch_size, seq_len), tokenizer.mask_id, device=DEVICE,
                )
                for step in range(n_steps):
                    t_val = 1.0 - step / n_steps
                    t = torch.full((batch_size,), max(t_val, 0.01), device=DEVICE)
                    with torch.no_grad():
                        logits = model(tokens, t)

                    # Simple unmasking (most confident)
                    mask_pos = (tokens == tokenizer.mask_id)
                    if not mask_pos.any():
                        break
                    probs = F.softmax(logits / 0.7, dim=-1)
                    confidence = probs.max(dim=-1)[0]
                    confidence[~mask_pos] = -1
                    sampled = torch.multinomial(
                        probs.reshape(-1, probs.shape[-1]),
                        num_samples=1,
                    ).reshape(batch_size, seq_len)

                    n_to_unmask = max(1, mask_pos.sum().item() // (n_steps - step))
                    for b in range(batch_size):
                        conf_b = confidence[b]
                        top_idx = conf_b.topk(min(n_to_unmask, int(mask_pos[b].sum().item()))).indices
                        tokens[b, top_idx] = sampled[b, top_idx]

            torch.cuda.synchronize()
            elapsed = time.time() - start

            total_tokens = batch_size * seq_len * 5
            tps = total_tokens / elapsed
            results[f"batch_{batch_size}"] = {
                "tps": tps,
                "latency_ms": elapsed / 5 * 1000,
                "steps": n_steps,
            }
            print(f"  Batch {batch_size:3d} ({n_steps} steps): "
                  f"{tps:,.0f} TPS | {elapsed/5*1000:.0f} ms/run")
        except Exception as e:
            print(f"  Batch {batch_size}: FAILED ({e})")
            results[f"batch_{batch_size}"] = {"error": str(e)}

    return results


def measure_diversity(model, tokenizer, n_samples=200, seq_len=64):
    """Measure diversity metrics on generated text."""
    print("\n── Diversity ──")
    samples = sample_mdlm_bpe(
        model, tokenizer, seq_len=seq_len,
        n_samples=n_samples, n_steps=20, temperature=0.7,
    )

    # Distinct-1 and Distinct-2
    all_tokens = []
    for s in samples:
        toks = s.split()
        all_tokens.extend(toks)

    unigrams = set(all_tokens)
    bigrams = set(zip(all_tokens[:-1], all_tokens[1:]))

    distinct1 = len(unigrams) / max(len(all_tokens), 1)
    distinct2 = len(bigrams) / max(len(all_tokens) - 1, 1)

    # Unique samples
    unique = len(set(samples))
    uniqueness = unique / len(samples)

    print(f"  Generated: {n_samples} samples")
    print(f"  Unique: {unique}/{n_samples} ({uniqueness*100:.1f}%)")
    print(f"  Distinct-1: {distinct1:.4f}")
    print(f"  Distinct-2: {distinct2:.4f}")

    return {
        "distinct_1": distinct1,
        "distinct_2": distinct2,
        "uniqueness": uniqueness,
        "n_unique": unique,
    }


def qualitative_samples(model, tokenizer, seq_len=64):
    """Generate and show qualitative samples."""
    print("\n── Qualitative Samples ──")

    # Free generation
    print("\n  Free generation (no prompt):")
    samples = sample_mdlm_bpe(
        model, tokenizer, seq_len=seq_len,
        n_samples=5, n_steps=32, temperature=0.7,
    )
    for i, s in enumerate(samples):
        clean = s.strip().replace("  ", " ")
        print(f"    [{i}] {clean[:150]}")

    # Prompt completion
    test_prompts = [
        "The future of artificial intelligence",
        "To build a reliable system, you need",
        "Python is a programming language",
    ]

    print("\n  Prompt completion:")
    for prompt in test_prompts:
        prompt_ids = tokenizer.encode(prompt, add_special=False)
        samples = sample_mdlm_bpe(
            model, tokenizer,
            prompt_ids=prompt_ids, seq_len=seq_len,
            n_samples=1, n_steps=32, temperature=0.7,
        )
        result = samples[0].strip() if samples else "(failed)"
        print(f"    '{prompt}...'")
        print(f"      → {result[:150]}")

    return samples


def run_full_evaluation():
    """Run complete evaluation suite."""
    print("=" * 70)
    print("MDLM-BPE EVALUATION")
    print("=" * 70)

    model, tokenizer, config = load_model()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  GPU: {gpu_name}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Run all evaluations
    ppl_loss, ppl = measure_perplexity(model, tokenizer)
    tps_results = measure_tps(model, tokenizer)
    diversity = measure_diversity(model, tokenizer)
    samples = qualitative_samples(model, tokenizer)

    # Save results
    results = {
        "gpu": gpu_name,
        "model_params": sum(p.numel() for p in model.parameters()),
        "perplexity": ppl,
        "ce_loss": ppl_loss,
        "tps": tps_results,
        "diversity": diversity,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_path = RESULTS_DIR / "mdlm_bpe_eval.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")

    return results


if __name__ == "__main__":
    run_full_evaluation()
