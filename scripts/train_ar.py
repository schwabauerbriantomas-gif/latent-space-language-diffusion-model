"""
Train AR Control Model with identical hyperparameters to MDLM-BPE v3.

Model: 199M params, d_model=1024, 15 layers, 16 heads, seq_len=128
Data: SAME Ultra-FineWeb 1M docs, SAME tokenizer, SAME tokenization
Training: 3 epochs, bf16, gradient accumulation, identical scheduler

This produces a controlled comparison: same data, same compute budget,
same parameter count — only the architecture differs (AR vs MDLM).
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
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = REPO / "checkpoints"
RESULTS_DIR = REPO / "results"
DATA_DIR = REPO / "data"

from ar_control import (
    ARConfig, ARControlModel, BPETokenizer,
    ar_loss, sample_ar, measure_perplexity,
)


def prepare_data(seq_len=128):
    """Tokenize 1M docs and pack into 128-token sequences.

    Reuses the SAME tokenized data as MDLM v3 (train_tokens_v3_128.npy)
    to guarantee identical training data.
    """
    output_file = DATA_DIR / f"train_tokens_v3_{seq_len}.npy"

    if output_file.exists():
        arr = np.load(output_file, mmap_mode='r')
        print(f"  Cached: {output_file} ({len(arr):,} seqs, mmap)")
        return np.array(arr), len(arr)

    raise FileNotFoundError(
        f"Training data not found: {output_file}\n"
        f"Run scripts/train.py first to tokenize data for MDLM v3."
    )


def train_ar(epochs=3, batch_size=32, lr=3e-4, seq_len=128,
             warmup_ratio=0.05, eval_every=500, gradient_accumulation=4,
             n_layers=15):
    """Train AR Control Model.

    Hyperparameters intentionally identical to MDLM v3 train.py:
    - Same lr, same scheduler (OneCycleLR), same batch size
    - Same gradient accumulation, same warmup ratio
    - Same seq_len, same data file
    """
    print("=" * 70)
    print("TRAINING AR CONTROL MODEL (199M PARAMS)")
    print("=" * 70)

    # Data — IDENTICAL to MDLM training
    print("Loading data...")
    tokens, n_seqs = prepare_data(seq_len=seq_len)
    print(f"  Total sequences: {n_seqs:,}")

    tokens_int16 = np.array(tokens, dtype=np.int16)
    del tokens
    tokens_tensor = torch.from_numpy(tokens_int16).long()
    dataset = TensorDataset(tokens_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=2, pin_memory=True)

    # Model
    tokenizer = BPETokenizer()
    config = ARConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=1024,
        n_heads=16,
        n_layers=n_layers,
        max_seq_len=256,
    )
    model = ARControlModel(config, pad_id=tokenizer.pad_id).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  Model: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Layers: {n_layers} (MDLM v3 has 10 + AdaLN)")
    print(f"  Data: {len(tokens_tensor):,} seqs × {seq_len} tokens = {len(tokens_tensor)*seq_len:,} tokens")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Accum: {gradient_accumulation}")
    print(f"  Effective batch: {batch_size * gradient_accumulation}")
    print(f"  Opt steps/epoch: {len(loader)//gradient_accumulation:,}")
    print()

    # Optimizer — identical to MDLM v3
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    optimizer_steps_per_epoch = len(loader) // gradient_accumulation
    total_optimizer_steps = optimizer_steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_optimizer_steps,
        pct_start=warmup_ratio,
    )

    # Training loop — mirrors MDLM v3 train.py structure
    model.train()
    micro_step = 0
    opt_step = 0
    best_eval = float('inf')
    losses = []
    start = time.time()
    accum_loss = 0.0

    # Hold out 1000 sequences for perplexity eval
    n_holdout = 1000
    holdout = tokens_tensor[:n_holdout].clone()

    for epoch in range(epochs):
        ep_loss = 0
        ep_count = 0

        for batch in loader:
            micro_step += 1
            batch_tokens = batch[0].to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = ar_loss(model, batch_tokens, pad_id=tokenizer.pad_id)
                loss = loss / gradient_accumulation

            loss.backward()
            accum_loss += loss.item()

            if micro_step % gradient_accumulation == 0:
                opt_step += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if opt_step <= total_optimizer_steps:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                ep_loss += accum_loss
                ep_count += 1
                losses.append(accum_loss)
                accum_loss = 0.0

                if opt_step % 100 == 0:
                    elapsed = time.time() - start
                    tps = micro_step * batch_size * seq_len / elapsed
                    lr_cur = optimizer.param_groups[0]['lr']
                    print(f"  [E{epoch+1} O{opt_step:,}] loss={losses[-1]:.4f} "
                          f"avg={ep_loss/ep_count:.4f} lr={lr_cur:.2e} "
                          f"{opt_step/elapsed:.1f} opt/s {tps:,.0f} tok/s")

                if opt_step % eval_every == 0:
                    model.eval()
                    eval_loss = quick_eval(model, loader, tokenizer.pad_id)
                    model.train()

                    ppl = math.exp(min(eval_loss, 15))
                    print(f"    → eval_loss={eval_loss:.4f} PPL={ppl:.1f}")

                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        torch.save({
                            "model_state": model.state_dict(),
                            "config": config.to_dict(),
                            "step": opt_step,
                            "eval_loss": eval_loss,
                            "ppl": ppl,
                        }, CHECKPOINT_DIR / "ar_control_best.pt")

                        samples = sample_ar(
                            model, tokenizer, max_new_tokens=64,
                            temperature=0.7, device=DEVICE,
                        )
                        print(f"    → Best sample (AR):")
                        print(f"      {samples.strip()[:150]}")

    # Final eval with proper perplexity on holdout
    model.eval()
    final_loss, final_ppl = measure_perplexity(model, holdout, batch_size=32)

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Optimizer steps: {opt_step}")
    print(f"  Best eval loss (training): {best_eval:.4f} (PPL={math.exp(min(best_eval,15)):.1f})")
    print(f"  Final holdout PPL: {final_ppl:.1f} (loss={final_loss:.4f})")

    torch.save({
        "model_state": model.state_dict(),
        "config": config.to_dict(),
        "step": opt_step,
        "losses": losses[-1000:],
        "holdout_ppl": final_ppl,
        "holdout_loss": final_loss,
    }, CHECKPOINT_DIR / "ar_control_final.pt")

    results = {
        "optimizer_steps": opt_step,
        "time": elapsed,
        "best_eval_loss": best_eval,
        "best_ppl": math.exp(min(best_eval, 15)),
        "final_holdout_ppl": final_ppl,
        "final_holdout_loss": final_loss,
        "final_loss": losses[-1],
        "tokens_trained": micro_step * batch_size * seq_len,
        "n_params": n_params,
        "n_layers": n_layers,
        "model_type": "autoregressive_control",
    }
    with open(RESULTS_DIR / "ar_control_training.json", "w") as f:
        json.dump(results, f, indent=2)

    return model


def quick_eval(model, loader, pad_id, n_batches=20):
    """Quick eval on training data batches (same as MDLM's quick_eval)."""
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            tokens = batch[0].to(DEVICE)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = ar_loss(model, tokens, pad_id=pad_id)
            losses.append(loss.item())
    return np.mean(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--layers", type=int, default=15)
    args = parser.parse_args()
    train_ar(epochs=args.epochs, batch_size=args.batch_size,
             lr=args.lr, seq_len=args.seq_len, gradient_accumulation=args.accum,
             n_layers=args.layers)
