"""
Train MDLM-BPE v2 (scaled model) on larger dataset.

Model: 93.9M params, d_model=768, 8 layers, 12 heads
Data: 500K Ultra-FineWeb documents (~250M chars, ~70M tokens)
Training: 5 epochs, cosine schedule, gradient clipping
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

from mdlm_bpe_v2 import (
    MDLMConfig, MDLMBPEV2, BPETokenizer,
    forward_mask_bpe, mdlm_bpe_loss_v2, sample_mdlm_v2,
)


def prepare_data(seq_len=64):
    """Tokenize the large dataset and pack into sequences.

    STREAMING version: processes in chunks to avoid OOM on low-RAM systems.
    Peak memory ≈ chunk_size × avg_doc_chars (not full dataset).
    """
    input_file = DATA_DIR / "ultra_fineweb_en_large.jsonl"
    output_file = DATA_DIR / "train_tokens_large.npy"

    if output_file.exists():
        # Load with mmap to avoid loading entire array into RAM
        arr = np.load(output_file, mmap_mode='r')
        print(f"  Cached: {output_file} ({len(arr):,} seqs, mmap)")
        return np.array(arr)  # copy out of mmap for DataLoader

    from tokenizers import Tokenizer
    tok_path = REPO / "tokenizer" / "bpe_tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    # Count lines first (fast, no content load)
    print(f"  Counting documents in {input_file.name}...")
    total_docs = sum(1 for _ in open(input_file))
    print(f"  {total_docs:,} documents")

    # Process in STREAMING chunks: read → tokenize → pack → write to temp
    # Peak RAM: CHUNK × ~500 chars/doc ≈ 5MB per chunk (negligible)
    CHUNK = 2000
    packed_seqs = []     # only holds packed seq_len arrays (int16, small)
    current = []          # rolling buffer for cross-doc packing
    processed = 0

    print(f"  Streaming tokenize + pack (chunk={CHUNK})...")

    with open(input_file) as f:
        chunk_texts = []
        for line in f:
            chunk_texts.append(json.loads(line)["content"])

            if len(chunk_texts) >= CHUNK:
                # Tokenize this chunk only (not the whole dataset)
                encoded = tokenizer.encode_batch(chunk_texts)
                chunk_texts = []  # free memory

                for enc in encoded:
                    ids = enc.ids
                    if len(ids) > seq_len * 4:
                        ids = ids[:seq_len * 2]
                    doc = [bos_id] + ids + [eos_id]
                    for tok in doc:
                        current.append(tok)
                        if len(current) >= seq_len:
                            packed_seqs.append(current[:seq_len])
                            current = []

                processed += CHUNK
                if processed % 20000 == 0:
                    print(f"    [{processed:,}/{total_docs:,}] "
                          f"seqs={len(packed_seqs):,} "
                          f"RAM≈{len(packed_seqs)*seq_len*2/1e6:.0f}MB")

        # Flush remaining chunk
        if chunk_texts:
            encoded = tokenizer.encode_batch(chunk_texts)
            for enc in encoded:
                ids = enc.ids
                if len(ids) > seq_len * 4:
                    ids = ids[:seq_len * 2]
                doc = [bos_id] + ids + [eos_id]
                for tok in doc:
                    current.append(tok)
                    if len(current) >= seq_len:
                        packed_seqs.append(current[:seq_len])
                        current = []

    # Handle remainder
    if current:
        while len(current) < seq_len:
            current.append(pad_id)
        packed_seqs.append(current)

    arr = np.array(packed_seqs, dtype=np.int16)
    np.save(output_file, arr)
    # Free the list immediately
    del packed_seqs, current
    print(f"  Saved: {len(arr):,} sequences ({arr.nbytes/1e6:.1f} MB)")
    return arr


def train_v2(epochs=5, batch_size=64, lr=3e-4, seq_len=64,
             warmup_ratio=0.05, eval_every=500, gradient_accumulation=3):
    """Train MDLM-BPE v2 on scaled data.

    Memory-conscious: batch_size=64 + gradient_accumulation=3 = effective batch 192.
    num_workers=2 (not 4) to avoid RAM multiplication from worker prefetch.
    """
    print("=" * 70)
    print("TRAINING MDLM-BPE v2 (SCALED — memory-optimized)")
    print("=" * 70)

    # Data
    print("Loading data...")
    tokens = prepare_data(seq_len=seq_len)
    # Use int32 (not int64) to halve memory: 1M seqs × 64 × 4B = 256MB
    tokens_tensor = torch.from_numpy(tokens.astype(np.int32)).long()
    del tokens  # free numpy array
    dataset = TensorDataset(tokens_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=2, pin_memory=True)

    # Model
    tokenizer = BPETokenizer()
    config = MDLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=8,
        max_seq_len=seq_len,
    )
    model = MDLMBPEV2(config, pad_id=tokenizer.pad_id,
                      mask_id=tokenizer.mask_id).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  Model: {n_params:,} params ({n_params/1e6:.1f}M)")
    print(f"  Data: {len(tokens_tensor):,} sequences × {seq_len} tokens")
    print(f"  Total tokens: {len(tokens_tensor)*seq_len:,}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Grad accum: {gradient_accumulation}")
    print(f"  Effective batch: {batch_size * gradient_accumulation}")
    print(f"  Steps/epoch: {len(loader):,} (optimizer steps: {len(loader)//gradient_accumulation:,})")
    print(f"  LR: {lr}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    optimizer_steps_per_epoch = len(loader) // gradient_accumulation
    total_optimizer_steps = optimizer_steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_optimizer_steps,
        pct_start=warmup_ratio,
    )

    # Mixed precision (bfloat16 doesn't need GradScaler)
    use_amp = True

    # Training
    model.train()
    micro_step = 0       # every forward pass
    opt_step = 0          # every optimizer.step()
    best_eval = float('inf')
    losses = []
    start = time.time()
    accum_loss = 0.0

    for epoch in range(epochs):
        ep_loss = 0
        ep_count = 0

        for batch in loader:
            micro_step += 1
            batch_tokens = batch[0].to(DEVICE, non_blocking=True)

            # Mixed precision forward + backward (no accumulation of grads yet)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = mdlm_bpe_loss_v2(model, batch_tokens, tokenizer.mask_id)
                loss = loss / gradient_accumulation  # scale for accumulation

            loss.backward()
            accum_loss += loss.item()

            # Only step every gradient_accumulation micro-batches
            if micro_step % gradient_accumulation == 0:
                opt_step += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # Guard against scheduler off-by-one on last epoch boundary
                if opt_step <= total_optimizer_steps:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)  # free grad memory

                ep_loss += accum_loss / gradient_accumulation
                ep_count += 1
                losses.append(accum_loss / gradient_accumulation)
                accum_loss = 0.0

                if opt_step % 100 == 0:
                    elapsed = time.time() - start
                    tps = micro_step * batch_size * seq_len / elapsed
                    lr_cur = optimizer.param_groups[0]['lr']
                    avg_loss = ep_loss / ep_count
                    print(f"  [E{epoch+1} O{opt_step:,}] loss={losses[-1]:.4f} "
                          f"avg={avg_loss:.4f} lr={lr_cur:.2e} "
                          f"{opt_step/elapsed:.1f} opt_step/s {tps:,.0f} tok/s")

                # Periodic eval
                if opt_step % eval_every == 0:
                    model.eval()
                    eval_loss = quick_eval(model, loader, tokenizer.mask_id)
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
                        }, CHECKPOINT_DIR / "mdlm_bpe_v2_best.pt")

                        samples = sample_mdlm_v2(
                            model, tokenizer, seq_len=seq_len,
                            n_samples=3, n_steps=20, temperature=0.7,
                        )
                        print(f"    → Best samples:")
                        for i, s in enumerate(samples):
                            print(f"      [{i}] {s.strip()[:120]}")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Optimizer steps: {opt_step:,}")
    print(f"  Micro steps: {micro_step:,}")
    print(f"  Best eval loss: {best_eval:.4f} (PPL={math.exp(min(best_eval,15)):.1f})")
    print(f"  Final loss: {losses[-1]:.4f}")

    # Save final
    torch.save({
        "model_state": model.state_dict(),
        "config": config.to_dict(),
        "step": opt_step,
        "losses": losses[-1000:],
    }, CHECKPOINT_DIR / "mdlm_bpe_v2_final.pt")

    results = {
        "total_optimizer_steps": opt_step,
        "total_micro_steps": micro_step,
        "time_seconds": elapsed,
        "best_eval_loss": best_eval,
        "best_ppl": math.exp(min(best_eval, 15)),
        "final_train_loss": losses[-1],
        "model_params": n_params,
        "tokens_trained": micro_step * batch_size * seq_len,
    }
    with open(RESULTS_DIR / "mdlm_bpe_v2_training.json", "w") as f:
        json.dump(results, f, indent=2)

    return model


def quick_eval(model, loader, mask_id, n_batches=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            tokens = batch[0].to(DEVICE)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = mdlm_bpe_loss_v2(model, tokens, mask_id)
            losses.append(loss.item())
    return np.mean(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()
    train_v2(epochs=args.epochs, batch_size=args.batch_size,
             lr=args.lr, seq_len=args.seq_len)
