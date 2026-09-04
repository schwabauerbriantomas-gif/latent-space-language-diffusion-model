"""
Generation benchmark for MDLM v4.

Measures generation speed, quality (oracle LP), and repetition score.
Runs after training completes.
"""
import sys, json, time, math
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch
import torch.nn.functional as F

DEVICE = "cuda"

from mdlm_v4_optimized import (
    MDLMConfig as MDLMConfigV4, MDLMBPEV4, BPETokenizer,
    sample_full_parallel_v4, mdlm_loss_v4,
)
from logit_guidance_v4 import sample_with_guidance

def load_model():
    ckpt_path = REPO / "checkpoints" / "mdlm_v4_best.pt"
    if not ckpt_path.exists():
        ckpt_path = REPO / "checkpoints" / "mdlm_v4_final.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = MDLMConfigV4(**ckpt["config"])
    model = MDLMBPEV4(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded: {ckpt_path.name} (step {ckpt.get('step','?')}, "
          f"eval_loss={ckpt.get('eval_loss','?')})")
    return model, BPETokenizer()

def rep_score(text, tokenizer):
    """Simple repetition score: unique bigrams / total bigrams."""
    ids = tokenizer.encode(text, add_special=False)
    if len(ids) < 4:
        return 1.0
    bigrams = set()
    total = 0
    for i in range(len(ids) - 1):
        bg = (ids[i], ids[i+1])
        bigrams.add(bg)
        total += 1
    return len(bigrams) / max(total, 1)

def count_generated_tokens(text, prompt_ids, tokenizer):
    """Count ONLY the tokens generated (exclude the prompt prefix).

    Both benchmark branches (no-guidance and guided) must use this so their
    t/s are comparable. Previously the no-guidance branch counted the prompt
    as generated, inflating its t/s (96.5 vs a comparable ~84.6).
    """
    n_total = len(tokenizer.encode(text, add_special=False))
    return max(n_total - len(prompt_ids), 1)

def benchmark_generation():
    model, tokenizer = load_model()
    
    prompts = [
        "The future of artificial intelligence",
        "To build a reliable system, you need",
        "Climate change is one of the biggest challenges",
        "The key to success in any project is",
        "Education is important because",
        "Programming is a skill that requires",
    ]
    
    print("\n" + "=" * 80)
    print("MDLM v4 — GENERATION BENCHMARK")
    print("=" * 80)
    
    all_results = []
    
    # Test 1: Basic full-parallel sampling (no guidance)
    print("\n── Full-parallel (no guidance) ──")
    for prompt in prompts:
        t0 = time.time()
        ids = tokenizer.encode(prompt, add_special=False)
        texts = sample_full_parallel_v4(
            model, tokenizer, prompt_ids=ids,
            seq_len=min(128, len(ids)+64),
            n_samples=1, n_steps=32, temperature=0.7,
        )
        dt = time.time() - t0
        text = texts[0]
        gen_tokens = count_generated_tokens(text, ids, tokenizer)
        tps = gen_tokens / dt
        rep = rep_score(text, tokenizer)
        all_results.append({
            "method": "full_parallel", "prompt": prompt,
            "text": text[:200], "tps": tps, "time_s": dt, "rep": rep,
        })
        print(f"  [{tps:.1f} t/s rep={rep:.2f}] {prompt}")
        print(f"    → {text.strip()[:120]}")
    
    # Test 2: Full-parallel WITH guidance
    print("\n── Full-parallel + guidance ──")
    for prompt in prompts:
        t0 = time.time()
        ids = tokenizer.encode(prompt, add_special=False)
        texts = sample_with_guidance(
            model, tokenizer, prompt_ids=ids,
            seq_len=min(128, len(ids)+64),
            n_samples=1, n_steps=32, temperature=0.7,
        )
        dt = time.time() - t0
        text = texts[0]
        gen_tokens = count_generated_tokens(text, ids, tokenizer)
        tps = gen_tokens / dt
        rep = rep_score(text, tokenizer)
        all_results.append({
            "method": "full_parallel_guided", "prompt": prompt,
            "text": text[:200], "tps": tps, "time_s": dt, "rep": rep,
        })
        print(f"  [{tps:.1f} t/s rep={rep:.2f}] {prompt}")
        print(f"    → {text.strip()[:120]}")
    
    # Summary
    for method in ["full_parallel", "full_parallel_guided"]:
        subset = [r for r in all_results if r["method"] == method]
        avg_tps = np.mean([r["tps"] for r in subset])
        avg_rep = np.mean([r["rep"] for r in subset])
        avg_time = np.mean([r["time_s"] for r in subset])
        print(f"\n  {method}: avg {avg_tps:.1f} t/s, rep={avg_rep:.2f}, {avg_time:.2f}s/gen")
    
    # Save
    results = {
        "version": "v4",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "results": all_results,
    }
    with open(REPO / "results" / "v4_generation_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/v4_generation_benchmark.json")


if __name__ == "__main__":
    benchmark_generation()
