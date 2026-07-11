"""
Head-to-head benchmark: MDLM v3 vs AR Control Model.

Compares:
  1. Perplexity (held-out, same data)
  2. Forward-pass throughput (TPS at batch 1/8/32)
  3. Generation speed (tokens/sec, latency)
  4. Text quality (oracle log-prob via Qwen3, repetition score)
  5. Sample outputs (side-by-side)

Both models use the same tokenizer, same data, same parameter budget (~201M).
Only architecture differs: masked diffusion (bidirectional) vs autoregressive (causal).
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
CHECKPOINT_DIR = REPO / "checkpoints"
RESULTS_DIR = REPO / "results"
DATA_DIR = REPO / "data"

from mdlm_bpe_v3 import (
    MDLMConfig, MDLMBPEV3, BPETokenizer as MDLMTokenizer,
    forward_mask_bpe, mdlm_loss, sample_semi_ar,
)
from ar_control import (
    ARConfig, ARControlModel, BPETokenizer as ARTokenizer,
    ar_loss, sample_ar, measure_perplexity as ar_measure_ppl,
)


def load_mdlm():
    """Load trained MDLM v3."""
    tok = MDLMTokenizer()
    config = MDLMConfig(
        vocab_size=tok.vocab_size,
        d_model=1024, n_heads=16, n_layers=10, max_seq_len=256,
    )
    model = MDLMBPEV3(config, pad_id=tok.pad_id, mask_id=tok.mask_id).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "mdlm_bpe_v3_best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok, ckpt


def load_ar():
    """Load trained AR Control."""
    tok = ARTokenizer()
    config = ARConfig(
        vocab_size=tok.vocab_size,
        d_model=1024, n_heads=16, n_layers=15, max_seq_len=256,
    )
    model = ARControlModel(config, pad_id=tok.pad_id).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "ar_control_best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok, ckpt


# ═══════════════════════════════════════════════════════════════════════════
# 1. Perplexity
# ═══════════════════════════════════════════════════════════════════════════

def eval_mdlm_perplexity(model, tokens, tokenizer, batch_size=32):
    """MDLM perplexity via masked CE loss.

    MDLM's loss is computed only on masked positions. For fair comparison,
    we use the same eval procedure as the original MDLM training: sample
    random timesteps, mask tokens, compute CE on masked positions only.
    """
    model.eval()
    total_loss = 0.0
    total_masked = 0
    mask_id = tokenizer.mask_id

    n = tokens.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = tokens[i:i+batch_size].to(DEVICE)
            bsz = batch.shape[0]
            t = torch.rand(bsz, device=DEVICE)
            masked, mask_pos = forward_mask_bpe(batch, t, mask_id)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(masked, t)

            mask_flat = mask_pos.reshape(-1)
            if mask_flat.sum() == 0:
                continue
            logits_flat = logits.reshape(-1, logits.shape[-1])
            tokens_flat = batch.reshape(-1)

            loss = F.cross_entropy(
                logits_flat[mask_flat], tokens_flat[mask_flat],
                reduction='sum',
            )
            total_loss += loss.item()
            total_masked += mask_flat.sum().item()

    avg_loss = total_loss / max(total_masked, 1)
    ppl = math.exp(min(avg_loss, 15))
    return avg_loss, ppl


# ═══════════════════════════════════════════════════════════════════════════
# 2. Forward-pass throughput
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_throughput_mdlm(model, vocab_size, seq_len=128, batches=[1, 8, 32]):
    """Measure MDLM forward-pass TPS."""
    model.eval()
    results = {}
    mask_id = 1

    with torch.no_grad():
        for bs in batches:
            tokens = torch.randint(0, vocab_size, (bs, seq_len), device=DEVICE)
            t = torch.rand(bs, device=DEVICE)

            # Warmup
            for _ in range(3):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _ = model(tokens, t)
            torch.cuda.synchronize()

            n_iters = 20
            start = time.time()
            for _ in range(n_iters):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _ = model(tokens, t)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            tps = bs * seq_len * n_iters / elapsed
            latency_ms = elapsed / n_iters * 1000
            results[f"batch_{bs}"] = {"tps": tps, "latency_ms": latency_ms, "steps": n_iters}

    return results


def benchmark_throughput_ar(model, vocab_size, seq_len=128, batches=[1, 8, 32]):
    """Measure AR forward-pass TPS (full sequence, teacher forcing)."""
    model.eval()
    results = {}

    with torch.no_grad():
        for bs in batches:
            tokens = torch.randint(0, vocab_size, (bs, seq_len), device=DEVICE)
            inp = tokens[:, :-1]

            # Warmup
            for _ in range(3):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _ = model(inp)
            torch.cuda.synchronize()

            n_iters = 20
            start = time.time()
            for _ in range(n_iters):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _ = model(inp)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            tps = bs * (seq_len - 1) * n_iters / elapsed
            latency_ms = elapsed / n_iters * 1000
            results[f"batch_{bs}"] = {"tps": tps, "latency_ms": latency_ms, "steps": n_iters}

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. Generation speed
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_generation_mdlm(model, tokenizer, n_samples=5, seq_len=128):
    """Measure MDLM generation speed (full-parallel + semi-AR)."""
    model.eval()
    results = {"full_parallel": [], "semi_ar": []}

    prompts = [
        "Climate change is one of the biggest challenges",
        "The future of artificial intelligence depends on",
        "Education systems around the world need to",
        " Renewable energy sources such as solar and wind",
        "The economic impact of the pandemic has",
    ]

    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]
        prompt_ids = tokenizer.encode(prompt, add_special=False)

        # Full-parallel (block_size = seq_len)
        torch.cuda.synchronize()
        start = time.time()
        text = sample_semi_ar(model, tokenizer, prompt_ids=prompt_ids,
                              seq_len=seq_len, n_samples=1,
                              block_size=seq_len, temperature=0.7)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        n_tokens = len(tokenizer.encode(text[0], add_special=False))
        results["full_parallel"].append({
            "text": text[0][:200],
            "tokens": n_tokens,
            "time_s": elapsed,
            "tps": n_tokens / max(elapsed, 0.001),
        })

        # Semi-AR (block_size = 4)
        torch.cuda.synchronize()
        start = time.time()
        text = sample_semi_ar(model, tokenizer, prompt_ids=prompt_ids,
                              seq_len=seq_len, n_samples=1,
                              block_size=4, temperature=0.7)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        n_tokens = len(tokenizer.encode(text[0], add_special=False))
        results["semi_ar"].append({
            "text": text[0][:200],
            "tokens": n_tokens,
            "time_s": elapsed,
            "tps": n_tokens / max(elapsed, 0.001),
        })

    return results


def benchmark_generation_ar(model, tokenizer, n_samples=5, max_new_tokens=64):
    """Measure AR generation speed (one token at a time with KV cache)."""
    model.eval()
    results = []

    prompts = [
        "Climate change is one of the biggest challenges",
        "The future of artificial intelligence depends on",
        "Education systems around the world need to",
        " Renewable energy sources such as solar and wind",
        "The economic impact of the pandemic has",
    ]

    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]
        prompt_ids = tokenizer.encode(prompt, add_special=False)

        torch.cuda.synchronize()
        start = time.time()
        text = sample_ar(model, tokenizer, prompt_ids=prompt_ids,
                         max_new_tokens=max_new_tokens,
                         temperature=0.7, top_p=0.95)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        n_tokens = len(tokenizer.encode(text, add_special=False))
        results.append({
            "text": text[:200],
            "tokens": n_tokens,
            "time_s": elapsed,
            "tps": n_tokens / max(elapsed, 0.001),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. Quality metrics (repetition score, diversity)
# ═══════════════════════════════════════════════════════════════════════════

def repetition_score(text):
    """1.0 = no repetition, 0.0 = heavy repetition.

    Computes ratio of unique bigrams to total bigrams.
    """
    words = text.split()
    if len(words) < 2:
        return 1.0
    bigrams = list(zip(words[:-1], words[1:]))
    unique = len(set(bigrams))
    total = len(bigrams)
    return unique / total if total > 0 else 1.0


def distinct_n(text, n=1):
    """Distinct-N metric: unique n-grams / total n-grams."""
    words = text.split()
    if len(words) < n:
        return 1.0
    ngrams = list(zip(*[words[i:] for i in range(n)]))
    unique = len(set(ngrams))
    total = len(ngrams)
    return unique / total if total > 0 else 1.0


def compute_quality_metrics(text):
    """Compute text quality metrics."""
    return {
        "repetition_score": round(repetition_score(text), 3),
        "distinct_1": round(distinct_n(text, 1), 3),
        "distinct_2": round(distinct_n(text, 2), 3),
        "n_words": len(text.split()),
        "n_chars": len(text),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Oracle log-prob (Qwen3 teacher forcing)
# ═══════════════════════════════════════════════════════════════════════════

def oracle_log_prob(text, qwen_model, qwen_tokenizer, device=DEVICE):
    """Compute mean per-token log-probability under Qwen3-0.6B.

    Higher (closer to 0) = more coherent text.
    """
    inputs = qwen_tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=256).to(device)

    with torch.no_grad():
        outputs = qwen_model(**inputs)
        logits = outputs.logits[:, :-1, :]  # predict token t+1
        target = inputs.input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.mean().item()


# ═══════════════════════════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════════════════════════

def run_benchmark(include_oracle=False):
    print("=" * 70)
    print("HEAD-TO-HEAD BENCHMARK: MDLM v3 vs AR CONTROL")
    print("=" * 70)

    # Load models
    print("\nLoading MDLM v3...")
    mdlm_model, mdlm_tok, mdlm_ckpt = load_mdlm()
    mdlm_params = sum(p.numel() for p in mdlm_model.parameters())
    print(f"  {mdlm_params:,} params ({mdlm_params/1e6:.1f}M)")

    print("Loading AR Control...")
    ar_model, ar_tok, ar_ckpt = load_ar()
    ar_params = sum(p.numel() for p in ar_model.parameters())
    print(f"  {ar_params:,} params ({ar_params/1e6:.1f}M)")

    # Load holdout data
    print("\nLoading holdout data...")
    tokens = np.load(DATA_DIR / "train_tokens_v3_128.npy", mmap_mode='r')
    holdout = torch.from_numpy(np.array(tokens[:1000], dtype=np.int16)).long()
    print(f"  {len(holdout)} sequences for perplexity eval")

    # 1. Perplexity
    print("\n--- 1. PERPLEXITY ---")
    print("  MDLM (masked CE)...")
    mdlm_loss_val, mdlm_ppl = eval_mdlm_perplexity(mdlm_model, holdout, mdlm_tok)
    print(f"  MDLM: loss={mdlm_loss_val:.4f} PPL={mdlm_ppl:.1f}")

    print("  AR (next-token CE)...")
    ar_loss_val, ar_ppl = ar_measure_ppl(ar_model, holdout)
    print(f"  AR:   loss={ar_loss_val:.4f} PPL={ar_ppl:.1f}")

    # 2. Forward throughput
    print("\n--- 2. FORWARD-PASS THROUGHPUT ---")
    print("  MDLM...")
    mdlm_tps = benchmark_throughput_mdlm(mdlm_model, mdlm_tok.vocab_size)
    for k, v in mdlm_tps.items():
        print(f"    {k}: {v['tps']:,.0f} TPS ({v['latency_ms']:.1f}ms)")

    print("  AR...")
    ar_tps = benchmark_throughput_ar(ar_model, ar_tok.vocab_size)
    for k, v in ar_tps.items():
        print(f"    {k}: {v['tps']:,.0f} TPS ({v['latency_ms']:.1f}ms)")

    # 3. Generation speed
    print("\n--- 3. GENERATION SPEED ---")
    print("  MDLM (full-parallel + semi-AR)...")
    mdlm_gen = benchmark_generation_mdlm(mdlm_model, mdlm_tok, n_samples=5)

    fp_tps = [r["tps"] for r in mdlm_gen["full_parallel"]]
    sar_tps = [r["tps"] for r in mdlm_gen["semi_ar"]]
    print(f"    Full-parallel: {np.mean(fp_tps):.1f} tok/s avg")
    print(f"    Semi-AR:       {np.mean(sar_tps):.1f} tok/s avg")

    print("  AR (sequential with KV cache)...")
    ar_gen = benchmark_generation_ar(ar_model, ar_tok, n_samples=5)
    ar_gen_tps = [r["tps"] for r in ar_gen]
    print(f"    AR:            {np.mean(ar_gen_tps):.1f} tok/s avg")

    # 4. Quality
    print("\n--- 4. TEXT QUALITY ---")
    prompts = [
        "Climate change is one of the biggest challenges",
        "The future of artificial intelligence depends on",
        "Education systems around the world need to",
    ]

    mdlm_samples = []
    ar_samples = []

    for prompt in prompts:
        prompt_ids = mdlm_tok.encode(prompt, add_special=False)

        mdlm_text = sample_semi_ar(mdlm_model, mdlm_tok, prompt_ids=prompt_ids,
                                    seq_len=128, n_samples=1,
                                    block_size=128, temperature=0.7)[0]
        ar_text = sample_ar(ar_model, ar_tok, prompt_ids=prompt_ids,
                            max_new_tokens=64, temperature=0.7, top_p=0.95)

        mdlm_q = compute_quality_metrics(mdlm_text)
        ar_q = compute_quality_metrics(ar_text)

        mdlm_samples.append({"prompt": prompt, "text": mdlm_text[:300], "quality": mdlm_q})
        ar_samples.append({"prompt": prompt, "text": ar_text[:300], "quality": ar_q})

        print(f"\n  Prompt: {prompt}")
        print(f"  MDLM:  {mdlm_text[:150]}")
        print(f"         rep={mdlm_q['repetition_score']} d1={mdlm_q['distinct_1']} d2={mdlm_q['distinct_2']}")
        print(f"  AR:    {ar_text[:150]}")
        print(f"         rep={ar_q['repetition_score']} d1={ar_q['distinct_1']} d2={ar_q['distinct_2']}")

    # 5. Oracle (optional)
    oracle_results = None
    if include_oracle:
        print("\n--- 5. ORACLE LOG-PROB (Qwen3-0.6B) ---")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer as HFTokenizer
            print("  Loading Qwen3-0.6B...")
            qwen_tok = HFTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            qwen_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16
            ).to(DEVICE).eval()

            mdlm_oracle = []
            ar_oracle = []

            for s in mdlm_samples:
                lp = oracle_log_prob(s["text"], qwen_model, qwen_tok)
                mdlm_oracle.append(lp)
            for s in ar_samples:
                lp = oracle_log_prob(s["text"], qwen_model, qwen_tok)
                ar_oracle.append(lp)

            print(f"  MDLM oracle LP: {np.mean(mdlm_oracle):.3f}")
            print(f"  AR oracle LP:   {np.mean(ar_oracle):.3f}")
            oracle_results = {
                "mdlm": mdlm_oracle, "ar": ar_oracle,
                "mdlm_mean": float(np.mean(mdlm_oracle)),
                "ar_mean": float(np.mean(ar_oracle)),
            }
        except Exception as e:
            print(f"  Oracle skipped: {e}")

    # Compile results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "mdlm": {
            "params": mdlm_params,
            "config": mdlm_ckpt.get("config", {}),
            "perplexity": mdlm_ppl,
            "eval_loss": mdlm_loss_val,
            "throughput": mdlm_tps,
            "generation": {
                "full_parallel_tps": float(np.mean(fp_tps)),
                "semi_ar_tps": float(np.mean(sar_tps)),
            },
            "samples": mdlm_samples,
        },
        "ar": {
            "params": ar_params,
            "config": ar_ckpt.get("config", {}),
            "perplexity": ar_ppl,
            "eval_loss": ar_loss_val,
            "throughput": ar_tps,
            "generation": {
                "tps": float(np.mean(ar_gen_tps)),
            },
            "samples": ar_samples,
        },
        "oracle": oracle_results,
    }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'MDLM v3':>15} {'AR Control':>15}")
    print("-" * 62)
    print(f"{'Parameters':<30} {mdlm_params/1e6:>14.1f}M {ar_params/1e6:>14.1f}M")
    print(f"{'Layers':<30} {'10 (+AdaLN)':>15} {'15':>15}")
    print(f"{'Perplexity':<30} {mdlm_ppl:>15.1f} {ar_ppl:>15.1f}")
    print(f"{'Eval loss':<30} {mdlm_loss_val:>15.4f} {ar_loss_val:>15.4f}")
    for bs in [1, 8, 32]:
        k = f"batch_{bs}"
        m_tps = mdlm_tps[k]["tps"]
        a_tps = ar_tps[k]["tps"]
        speedup = m_tps / a_tps if a_tps > 0 else 0
        print(f"{'Fwd TPS (batch=' + str(bs) + ')':<30} {m_tps:>15,.0f} {a_tps:>15,.0f}  ({speedup:.1f}x)")
    print(f"{'Gen TPS (best mode)':<30} {max(np.mean(fp_tps), np.mean(sar_tps)):>15.1f} {np.mean(ar_gen_tps):>15.1f}")
    if oracle_results:
        print(f"{'Oracle log-prob':<30} {oracle_results['mdlm_mean']:>15.3f} {oracle_results['ar_mean']:>15.3f}")

    # Save
    output_path = RESULTS_DIR / "comparison_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true", help="Include Qwen3 oracle scoring")
    args = parser.parse_args()
    run_benchmark(include_oracle=args.oracle)
