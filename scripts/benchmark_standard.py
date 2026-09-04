#!/usr/bin/env python3
"""
benchmark_standard.py — Standardized MDLM benchmark protocol (r9-benchdesign).

Why: the repo's benchmark*.py scripts mix inconsistent token numerators (prompt
counted in one branch, not another), lack torch.cuda.synchronize in generation
timing, lack warmup (first-measured config pays CUDA init), and report single
20-iter means with no dispersion. This script fixes the protocol:

  1. Warmup:              --warmup forwards (default 10) discarded per shape.
  2. Sync:                torch.cuda.synchronize() BEFORE and AFTER every timed
                          region (forward and generation).
  3. Repeats:             --repeats independent timing blocks (default 30).
  4. Statistics:          median + IQR (p75-p25) + min/max across repeats.
  5. Explicit TPS:        forward  TPS = batch * seq / dt          (dt = median
                          per-forward seconds). Positions processed per forward
                          pass — NOT generated tokens.
                          generation TPS = (seq_len - prompt_len) / dt  — only
                          tokens actually unmasked by the sampler, prompt
                          excluded by construction, never re-encoded.
  6. Conditions in JSON:  GPU, dtype, batch, seq, warmup, repeats, weights
                          (checkpoint vs random), timestamp, torch version.

Runs on CUDA only (refuses CPU unless --allow-cpu, because timing is
meaningless across devices).

Usage (GPU wave):
  python3 benchmark_standard.py --model v4 --batch 1,8,32 --seq 128 \
      --dtype bfloat16 --repeats 30 --warmup 10 --out /tmp/bench.json
  python3 benchmark_standard.py --model v3 --mode gen --seq 64 --repeats 20
"""
import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", choices=["v3", "v4"], default="v4",
                   help="Which architecture to benchmark")
    p.add_argument("--mode", choices=["fwd", "gen", "both"], default="fwd",
                   help="fwd = forward-pass TPS; gen = full-parallel generation TPS")
    p.add_argument("--batch", type=str, default="1,8,32",
                   help="Comma-separated batch sizes for fwd mode (e.g. 1,8,32)")
    p.add_argument("--seq", type=int, default=128,
                   help="Sequence length (fixed, unlike repo scripts)")
    p.add_argument("--dtype", choices=list(DTYPES), default="bfloat16",
                   help="Compute dtype: real cast of model + inputs (autocast "
                        "in repo scripts was undeclared in their JSON output)")
    p.add_argument("--repeats", type=int, default=30,
                   help="Independent timed blocks; median+IQR reported")
    p.add_argument("--warmup", type=int, default=10,
                   help="Discarded forwards per shape before timing")
    p.add_argument("--steps-per-block", type=int, default=20,
                   help="Forward passes averaged inside each timed block")
    p.add_argument("--gen-steps", type=int, default=32,
                   help="Diffusion steps for generation mode")
    p.add_argument("--gen-repeats", type=int, default=20,
                   help="Independent generation runs in gen mode")
    p.add_argument("--prompt", type=str, default="The future of artificial intelligence")
    p.add_argument("--ckpt", type=str, default="",
                   help="Explicit checkpoint path; default = repo convention, "
                        "falls back to random weights (flagged in JSON)")
    p.add_argument("--out", type=str, default="",
                   help="Output JSON path (default prints to stdout only)")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Permit CPU execution (recorded as device=cpu)")
    return p.parse_args()


def load_model(which, dtype, ckpt_arg):
    """Load v3/v4 with real weights when available; else random (flagged)."""
    if which == "v3":
        from mdlm_bpe_v3 import MDLMConfig, MDLMBPEV3, BPETokenizer
        default_ckpt = REPO / "checkpoints" / "mdlm_bpe_v3_best.pt"
        path = Path(ckpt_arg) if ckpt_arg else default_ckpt
        tok = BPETokenizer()
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            cfg = MDLMConfig(**ckpt["config"])
            model = MDLMBPEV3(cfg, pad_id=tok.pad_id, mask_id=tok.mask_id)
            model.load_state_dict(ckpt["model_state"])
            weights = f"checkpoint:{path.name}"
        else:
            cfg = MDLMConfig(vocab_size=tok.vocab_size, d_model=1024,
                             n_heads=16, n_layers=10, max_seq_len=256)
            model = MDLMBPEV3(cfg, pad_id=tok.pad_id, mask_id=tok.mask_id)
            weights = "random (checkpoint NOT found — shape-equivalent only)"
    else:
        from mdlm_v4_optimized import (MDLMConfig as C4, MDLMBPEV4,
                                       BPETokenizer as Tok4)
        tok = Tok4()
        default_ckpt = REPO / "checkpoints" / "mdlm_v4_best.pt"
        path = Path(ckpt_arg) if ckpt_arg else default_ckpt
        if not path.exists():
            path = REPO / "checkpoints" / "mdlm_v4_final.pt"
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            cfg = C4(**ckpt["config"])
            model = MDLMBPEV4(cfg, pad_id=tok.pad_id, mask_id=tok.mask_id)
            model.load_state_dict(ckpt["model_state"])
            weights = f"checkpoint:{path.name}"
        else:
            cfg = C4(vocab_size=tok.vocab_size, d_model=1024, n_heads=16,
                     n_layers=10, max_seq_len=256)
            model = MDLMBPEV4(cfg, pad_id=tok.pad_id, mask_id=tok.mask_id)
            weights = "random (checkpoint NOT found — shape-equivalent only)"
    model = model.to(dtype=dtype)
    model.eval()
    return model, tok, weights


def median_iqr(samples):
    """Median + interquartile range + extremes for a list of floats.
    Quartiles use statistics.quantiles (exclusive method, default)."""
    s = sorted(samples)
    n = len(s)
    med = statistics.median(s)
    if n >= 4:
        q1, _, q3 = statistics.quantiles(s, n=4)  # exclusive method
    else:
        q1, q3 = min(s), max(s)
    return {
        "median": med,
        "iqr": q3 - q1,
        "q1": q1, "q3": q3,
        "min": min(s), "max": max(s),
        "n": n,
        "cv_pct": (statistics.stdev(s) / statistics.mean(s) * 100) if n > 1 else 0.0,
    }


def bench_forward(model, vocab_size, mask_id, batch, seq, dtype_name,
                  warmup, repeats, steps_per_block, device):
    """Forward-pass TPS. TPS = batch * seq / dt — positions processed per
    forward pass (all positions, NOT generated tokens). Sync-bracketed."""
    tokens = torch.randint(0, vocab_size, (batch, seq), device=device)
    t = torch.rand(batch, device=device)
    out = {}
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(tokens, t)
        torch.cuda.synchronize() if device.type == "cuda" else None
        block_times = []
        for _ in range(repeats):
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()
            for _ in range(steps_per_block):
                _ = model(tokens, t)
            torch.cuda.synchronize() if device.type == "cuda" else None
            block_times.append((time.perf_counter() - t0) / steps_per_block)
    stats = median_iqr(block_times)
    dt = stats["median"]
    tps = batch * seq / dt  # EXPLICIT formula: positions/sec through the model
    out = {
        "batch": batch, "seq": seq, "dtype": dtype_name,
        "warmup_iters": warmup, "repeats": repeats,
        "steps_per_block": steps_per_block,
        "dt_s_per_forward": stats,
        "tps_formula": "batch*seq/dt_median",
        "tps": tps,
        "latency_ms": dt * 1000.0,
        "numerator_semantics": "all seq positions per forward pass",
    }
    return out


@torch.no_grad()
def gen_full_parallel(model, prompt_ids, mask_id, seq_len, n_steps,
                      temperature, device):
    """Self-contained full-parallel sampler (no guidance, no extra deps).
    Returns the token tensor. Numerator for TPS is computed by the caller as
    seq_len - prompt_len (positions the sampler had to fill), never by
    re-encoding decoded text."""
    full = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    plen = min(len(prompt_ids), seq_len)
    full[0, :plen] = torch.tensor(prompt_ids[:plen], device=device)
    for step in range(n_steps):
        t_val = max(1.0 - step / n_steps, 0.01)
        t = torch.full((1,), float(t_val), device=device)
        logits = model(full, t)
        mask_pos = full[0] == mask_id
        if not mask_pos.any():
            break
        idxs = mask_pos.nonzero(as_tuple=True)[0]
        probs = torch.softmax(logits[0, idxs] / temperature, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        conf = probs.max(dim=-1)[0]
        n_masked = int(idxs.numel())
        n_unmask = max(1, n_masked // max(n_steps - step, 1))
        _, ti = conf.topk(min(n_unmask, n_masked))
        full[0, idxs[ti]] = sampled[ti]
    return full


def bench_generation(model, tok, prompt, seq, gen_steps, gen_repeats,
                     dtype_name, device):
    """Generation TPS with sync brackets and a fixed numerator:
    generated_tokens = seq - prompt_len (excludes prompt by construction)."""
    # Tokenize OUTSIDE the timed region (repo scripts sometimes included it).
    prompt_ids = tok.encode(prompt, add_special=False)
    plen = min(len(prompt_ids), seq)
    gen_tokens = seq - plen  # positions the sampler must fill
    with torch.no_grad():
        # Warmup: one full generation, discarded.
        _ = gen_full_parallel(model, prompt_ids, tok.mask_id, seq, gen_steps,
                              0.7, device)
        times = []
        for _ in range(gen_repeats):
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()
            _ = gen_full_parallel(model, prompt_ids, tok.mask_id, seq,
                                  gen_steps, 0.7, device)
            torch.cuda.synchronize() if device.type == "cuda" else None
            times.append(time.perf_counter() - t0)
    stats = median_iqr(times)
    dt = stats["median"]
    return {
        "prompt": prompt, "prompt_len_tokens": plen, "seq": seq,
        "gen_steps": gen_steps, "gen_repeats": gen_repeats,
        "dtype": dtype_name,
        "generated_tokens_numerator": gen_tokens,
        "numerator_semantics": "seq_len - prompt_len (prompt excluded by construction)",
        "dt_s_per_generation": stats,
        "tps_formula": "(seq_len - prompt_len)/dt_median",
        "tps": gen_tokens / dt,
        "latency_ms": dt * 1000.0,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        print(json.dumps({"error": "CUDA unavailable; use --allow-cpu to force"}))
        sys.exit(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[args.dtype]
    torch.manual_seed(42)  # reproducible inputs/sampling

    model, tok, weights = load_model(args.model, dtype, args.ckpt)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    conditions = {
        "script": "benchmark_standard.py (r9 protocol)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch": torch.__version__,
        "model_version": args.model,
        "params": n_params,
        "weights": weights,
        "dtype": args.dtype,
        "seed": 42,
    }
    result = {"conditions": conditions, "mode": args.mode, "results": {}}

    if args.mode in ("fwd", "both"):
        batches = [int(b) for b in args.batch.split(",")]
        fwd = {}
        for b in batches:
            fwd[f"batch_{b}"] = bench_forward(
                model, tok.vocab_size, tok.mask_id, b, args.seq,
                args.dtype, args.warmup, args.repeats, args.steps_per_block,
                device)
            r = fwd[f"batch_{b}"]
            print(f"[fwd] b={b} seq={args.seq} {args.dtype}: "
                  f"TPS={r['tps']:,.0f} (IQR {r['dt_s_per_forward']['iqr']*1e3:.2f}ms, "
                  f"cv {r['dt_s_per_forward']['cv_pct']:.1f}%)")
        result["results"]["forward"] = fwd

    if args.mode in ("gen", "both"):
        gen = bench_generation(model, tok, args.prompt, args.seq,
                               args.gen_steps, args.gen_repeats,
                               args.dtype, device)
        result["results"]["generation"] = gen
        print(f"[gen] seq={args.seq} steps={args.gen_steps} {args.dtype}: "
              f"TPS={gen['tps']:.1f} tok/s "
              f"(median {gen['dt_s_per_generation']['median']:.3f}s, "
              f"IQR {gen['dt_s_per_generation']['iqr']:.3f}s)")

    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"Saved: {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
