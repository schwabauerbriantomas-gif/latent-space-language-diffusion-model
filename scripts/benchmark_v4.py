"""
Benchmark v4: Full benchmark comparing v3 vs v4 architecture.

Measures:
  1. Forward-pass throughput (batch 1, 8, 32)
  2. Training throughput (tok/s)
  3. Generation speed (full-parallel, semi-AR)
  4. Memory usage (peak VRAM)
  5. Model quality (oracle LP, repetition score)
"""
import sys, json, time, math, gc
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

DEVICE = "cuda"


def measure_forward_throughput(model, seq_len=128, warmup=3, steps=20):
    """Measure raw forward-pass throughput."""
    model.eval()
    # Warmup
    dummy_tokens = torch.randint(0, 10000, (32, seq_len), device=DEVICE)
    dummy_t = torch.rand(32, device=DEVICE)
    for _ in range(warmup):
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = model(dummy_tokens, dummy_t)
    torch.cuda.synchronize()

    results = {}
    for batch_size in [1, 8, 32]:
        tokens = torch.randint(0, 10000, (batch_size, seq_len), device=DEVICE)
        t = torch.rand(batch_size, device=DEVICE)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(tokens, t)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / steps
        tps = batch_size * seq_len / dt
        results[batch_size] = {"tps": tps, "latency_ms": dt * 1000, "steps": steps}
    
    return results


def measure_vram(model, seq_len=128, batch_size=32):
    """Measure peak VRAM during forward+backward."""
    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    tokens = torch.randint(0, 10000, (batch_size, seq_len), device=DEVICE)
    t = torch.rand(batch_size, device=DEVICE)
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(tokens, t)
        # Simulate loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.reshape(-1),
        )
    loss.backward()
    
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.empty_cache()
    return peak_gb


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def benchmark_v4():
    """Run complete v4 benchmark."""
    from mdlm_v4_optimized import MDLMConfig as MDLMConfigV4, MDLMBPEV4, BPETokenizer
    print("=" * 80)
    print("MDLM v4 BENCHMARK")
    print("=" * 80)
    
    tokenizer = BPETokenizer()
    config = MDLMConfigV4(
        vocab_size=tokenizer.vocab_size,
        d_model=1024,
        n_heads=16,
        n_layers=10,
        max_seq_len=256,
    )
    model = MDLMBPEV4(config, pad_id=tokenizer.pad_id,
                      mask_id=tokenizer.mask_id).to(DEVICE)
    model.eval()
    
    # Params
    params = count_params(model)
    print(f"\n  Parameters: {params['total']:,} ({params['total']/1e6:.1f}M)")
    
    # ── Forward throughput ──
    print("\n  Measuring forward throughput...")
    fwd = measure_forward_throughput(model, seq_len=128)
    print(f"    batch=1:  {fwd[1]['tps']:>10,.0f} TPS  ({fwd[1]['latency_ms']:.1f}ms)")
    print(f"    batch=8:  {fwd[8]['tps']:>10,.0f} TPS  ({fwd[8]['latency_ms']:.1f}ms)")
    print(f"    batch=32: {fwd[32]['tps']:>10,.0f} TPS  ({fwd[32]['latency_ms']:.1f}ms)")
    
    # ── VRAM ──
    print("\n  Measuring peak VRAM (forward+backward)...")
    vram = measure_vram(model)
    print(f"    Peak VRAM: {vram:.2f} GB (batch=32, seq=128)")
    
    # ── Save ──
    results = {
        "version": "v4",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "params": params,
        "throughput": fwd,
        "peak_vram_gb": vram,
        "config": config.to_dict(),
    }
    
    with open(REPO / "results" / "v4_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Saved to results/v4_benchmark.json")
    return results


def benchmark_v3_vs_v4():
    """Head-to-head comparison."""
    from mdlm_v4_optimized import MDLMConfig as MDLMConfigV4, MDLMBPEV4, BPETokenizer
    from mdlm_bpe_v3 import MDLMConfig, MDLMBPEV3
    
    print("\n" + "=" * 80)
    print("V3 vs V4 HEAD-TO-HEAD")
    print("=" * 80)
    
    tokenizer = BPETokenizer()
    
    # v3
    config3 = MDLMConfig(vocab_size=tokenizer.vocab_size, d_model=1024,
                          n_heads=16, n_layers=10, max_seq_len=256)
    model3 = MDLMBPEV3(config3, pad_id=tokenizer.pad_id,
                        mask_id=tokenizer.mask_id).to(DEVICE)
    model3.eval()
    params3 = count_params(model3)
    fwd3 = measure_forward_throughput(model3, seq_len=128)
    vram3 = measure_vram(model3)
    del model3
    torch.cuda.empty_cache()
    gc.collect()
    
    # v4
    config4 = MDLMConfigV4(vocab_size=tokenizer.vocab_size, d_model=1024,
                            n_heads=16, n_layers=10, max_seq_len=256)
    model4 = MDLMBPEV4(config4, pad_id=tokenizer.pad_id,
                        mask_id=tokenizer.mask_id).to(DEVICE)
    model4.eval()
    params4 = count_params(model4)
    fwd4 = measure_forward_throughput(model4, seq_len=128)
    vram4 = measure_vram(model4)
    del model4
    torch.cuda.empty_cache()
    gc.collect()
    
    # Comparison table
    print(f"\n  {'Metric':<35} {'v3':>15} {'v4':>15} {'Δ':>10}")
    print(f"  {'-'*80}")
    print(f"  {'Parameters':<35} {params3['total']:>15,} {params4['total']:>15,} {(params4['total']-params3['total']):>+10,}")
    print(f"  {'Params (M)':<35} {params3['total']/1e6:>14.1f}M {params4['total']/1e6:>14.1f}M {(params4['total']-params3['total'])/1e6:>+9.1f}M")
    
    for bs in [1, 8, 32]:
        t3 = fwd3[bs]['tps']
        t4 = fwd4[bs]['tps']
        delta = (t4 - t3) / t3 * 100
        print(f"  {'Fwd TPS batch=' + str(bs):<35} {t3:>12,.0f} TPS {t4:>12,.0f} TPS {delta:>+9.1f}%")
    
    print(f"  {'Peak VRAM (GB)':<35} {vram3:>15.2f} {vram4:>15.2f} {(vram4-vram3):>+9.2f}")
    
    results = {
        "v3": {"params": params3, "throughput": fwd3, "vram": vram3},
        "v4": {"params": params4, "throughput": fwd4, "vram": vram4},
    }
    
    with open(REPO / "results" / "v3_vs_v4_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Saved to results/v3_vs_v4_comparison.json")


if __name__ == "__main__":
    benchmark_v4()
    benchmark_v3_vs_v4()
