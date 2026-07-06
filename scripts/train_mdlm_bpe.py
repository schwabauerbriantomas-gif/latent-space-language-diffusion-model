"""
Train MDLM-BPE on Ultra-FineWeb data.

Trains the masked diffusion model on real text.
Saves checkpoints, logs training loss.
"""
import math
import json
import sys
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = REPO / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = REPO / "results"
DATA_FILE = REPO / "data" / "train_tokens.npy"

from mdlm_bpe import (
    MDLMConfig, MDLMBPETransformer, BPETokenizer,
    forward_mask_bpe, sample_mdlm_bpe,
)


def train_mdlm_bpe(
    epochs=3,
    batch_size=256,
    lr=3e-4,
    warmup_steps=500,
    eval_every=500,
    save_every=2000,
    max_steps=None,
    seq_len=64,
):
    """Train MDLM-BPE."""
    print("=" * 70)
    print("TRAINING MDLM-BPE on Ultra-FineWeb")
    print("=" * 70)

    # Load data
    print("Loading data...")
    tokens = np.load(DATA_FILE)
    print(f"  Shape: {tokens.shape}")
    print(f"  Sequences: {len(tokens):,}")

    tokens_tensor = torch.from_numpy(tokens.astype(np.int64))
    dataset = TensorDataset(tokens_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=4, pin_memory=True)

    # Model
    tokenizer = BPETokenizer()
    config = MDLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        max_seq_len=seq_len,
    )
    model = MDLMBPETransformer(
        config, pad_id=tokenizer.pad_id, mask_id=tokenizer.mask_id,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  Seq len: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Steps/epoch: {len(loader):,}")
    print()

    # Optimizer with cosine schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                                   betas=(0.9, 0.95))
    total_steps = len(loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=warmup_steps/total_steps,
    )

    # Training loop
    model.train()
    step = 0
    best_loss = float('inf')
    train_losses = []
    eval_losses = []
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in loader:
            step += 1
            if max_steps and step > max_steps:
                break

            batch_tokens = batch[0].to(DEVICE, non_blocking=True)

            # Forward
            optimizer.zero_grad()
            loss = _compute_loss(model, batch_tokens, tokenizer.mask_id)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            train_losses.append(loss.item())

            if step % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                lr_cur = optimizer.param_groups[0]['lr']
                avg_loss = epoch_loss / epoch_steps
                print(f"  [E{epoch+1} S{step:,}] loss={loss.item():.4f} "
                      f"avg={avg_loss:.4f} lr={lr_cur:.2e} "
                      f"{steps_per_sec:.1f} steps/s "
                      f"tokens/s={batch_size*seq_len*steps_per_sec:,.0f}")

            if step % eval_every == 0:
                model.eval()
                eval_loss = _evaluate(model, loader, tokenizer.mask_id, n_batches=20)
                eval_losses.append({"step": step, "loss": eval_loss})
                model.train()
                print(f"    → eval_loss={eval_loss:.4f}")

                # Generate samples
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    samples = sample_mdlm_bpe(
                        model, tokenizer, seq_len=seq_len,
                        n_samples=3, n_steps=20, temperature=0.7,
                    )
                    print(f"    → Samples (best so far):")
                    for i, s in enumerate(samples):
                        print(f"      [{i}] {s[:120]}")

                # Save best checkpoint
                ckpt_path = CHECKPOINT_DIR / "mdlm_bpe_best.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config.__dict__,
                    "step": step,
                    "loss": eval_loss,
                }, ckpt_path)

            if step % save_every == 0:
                ckpt_path = CHECKPOINT_DIR / f"mdlm_bpe_step{step}.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config.__dict__,
                    "step": step,
                }, ckpt_path)

        avg = epoch_loss / max(epoch_steps, 1)
        print(f"\n  Epoch {epoch+1} avg loss: {avg:.4f}\n")

        if max_steps and step >= max_steps:
            break

    # Final save
    elapsed = time.time() - start_time
    final_path = CHECKPOINT_DIR / "mdlm_bpe_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": config.__dict__,
        "step": step,
        "train_losses": train_losses,
        "eval_losses": eval_losses,
    }, final_path)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total steps: {step:,}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Best eval loss: {best_loss:.4f}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Checkpoint: {final_path}")

    # Save results
    results = {
        "total_steps": step,
        "time_seconds": elapsed,
        "best_eval_loss": best_loss,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "model_params": n_params,
        "vocab_size": tokenizer.vocab_size,
        "seq_len": seq_len,
        "n_sequences": len(tokens),
        "tokens_trained": step * batch_size * seq_len,
    }
    with open(RESULTS_DIR / "mdlm_bpe_training.json", "w") as f:
        json.dump(results, f, indent=2)

    return model, train_losses


def _compute_loss(model, tokens, mask_id):
    """Compute MDLM loss for a batch."""
    batch = tokens.shape[0]
    t = torch.rand(batch, device=tokens.device)
    masked, mask_pos = forward_mask_bpe(tokens, t, mask_id)
    logits = model(masked, t)

    mask_flat = mask_pos.reshape(-1)
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=tokens.device)

    logits_flat = logits.reshape(-1, logits.shape[-1])
    tokens_flat = tokens.reshape(-1)
    return F.cross_entropy(logits_flat[mask_flat], tokens_flat[mask_flat])


def _evaluate(model, loader, mask_id, n_batches=20):
    """Quick eval on n_batches."""
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch_tokens = batch[0].to(DEVICE)
            loss = _compute_loss(model, batch_tokens, mask_id)
            losses.append(loss.item())
    return np.mean(losses)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    train_mdlm_bpe(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        seq_len=args.seq_len,
    )
