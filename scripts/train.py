"""
Train MDLM-BPE v3 (scaled model) on 2M Ultra-FineWeb documents.

Model: 207M params, d_model=1024, 10 layers, 16 heads, seq_len=128
Data: 2M docs → ~4M sequences of 128 tokens → ~500M tokens
Training: 3 epochs, bf16, gradient accumulation, streaming data
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

from mdlm_bpe_v3 import (
    MDLMConfig, MDLMBPEV3, BPETokenizer,
    forward_mask_bpe, mdlm_loss, sample_semi_ar,
)


def prepare_data(seq_len=128):
    """Tokenize 2M docs and pack into 128-token sequences.

    DISK-STREAMING: writes directly to numpy memmap, never holds
    all sequences in Python lists. Peak RAM = chunk_size only.
    """
    input_file = DATA_DIR / "ultra_fineweb_en_1m.jsonl"
    output_file = DATA_DIR / f"train_tokens_v3_{seq_len}.npy"

    if output_file.exists():
        arr = np.load(output_file, mmap_mode='r')
        print(f"  Cached: {output_file} ({len(arr):,} seqs, mmap)")
        return np.array(arr)

    from tokenizers import Tokenizer
    tok_path = REPO / "tokenizer" / "bpe_tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    # Count lines
    print(f"  Counting docs...")
    total_docs = sum(1 for _ in open(input_file))
    print(f"  {total_docs:,} documents")

    # Estimate max sequences: each doc produces ~3 seqs of 128 tokens
    # Use a generous pre-allocated memmap, truncate after
    max_seqs = total_docs * 4  # generous upper bound
    tmp_file = DATA_DIR / "train_tokens_v3_tmp.npy"
    arr = np.memmap(str(tmp_file), dtype=np.int16, mode='w+',
                    shape=(max_seqs, seq_len))

    CHUNK = 1000  # smaller chunks = less peak RAM
    n_seqs = 0
    current = []
    processed = 0

    print(f"  Streaming tokenize → memmap (chunk={CHUNK})...")

    with open(input_file) as f:
        chunk_texts = []
        for line in f:
            chunk_texts.append(json.loads(line)["content"])

            if len(chunk_texts) >= CHUNK:
                encoded = tokenizer.encode_batch(chunk_texts)
                chunk_texts = []  # free immediately

                for enc in encoded:
                    ids = enc.ids
                    if len(ids) > seq_len * 4:
                        ids = ids[:seq_len * 2]
                    doc = [bos_id] + ids + [eos_id]
                    for tok in doc:
                        current.append(tok)
                        if len(current) >= seq_len:
                            arr[n_seqs] = current[:seq_len]
                            n_seqs += 1
                            current = []

                processed += CHUNK
                if processed % 50000 == 0:
                    print(f"    [{processed:,}/{total_docs:,}] "
                          f"seqs={n_seqs:,} RAM≈free")

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
                        arr[n_seqs] = current[:seq_len]
                        n_seqs += 1
                        current = []

    # Handle remainder
    if current:
        while len(current) < seq_len:
            current.append(pad_id)
        arr[n_seqs] = current[:seq_len]
        n_seqs += 1

    # Truncate to actual size and save properly
    del arr
    arr_final = np.memmap(str(tmp_file), dtype=np.int16, mode='r',
                          shape=(n_seqs, seq_len))
    final = np.array(arr_final[:n_seqs])
    np.save(output_file, final)
    del arr_final, final
    tmp_file.unlink()  # cleanup temp

    print(f"  Saved: {n_seqs:,} sequences ({n_seqs*seq_len*2/1e6:.1f} MB)")
    # Return mmap view to avoid loading entire array into RAM
    return np.load(output_file, mmap_mode='r'), n_seqs


def train_v3(epochs=3, batch_size=32, lr=3e-4, seq_len=128,
             warmup_ratio=0.05, eval_every=500, gradient_accumulation=4):
    """Train MDLM-BPE v3."""
    print("=" * 70)
    print("TRAINING MDLM-BPE v3 (207M PARAMS)")
    print("=" * 70)

    # Data
    print("Loading data...")
    result = prepare_data(seq_len=seq_len)
    if isinstance(result, tuple):
        tokens, n_seqs = result
    else:
        tokens = result
        n_seqs = len(tokens)
    print(f"  Total sequences: {n_seqs:,}")

    # Use int16 (not int32) to halve memory, convert to long on GPU per-batch
    tokens_int16 = np.array(tokens, dtype=np.int16)
    del tokens
    tokens_tensor = torch.from_numpy(tokens_int16).long()
    dataset = TensorDataset(tokens_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=2, pin_memory=True)

    # Model
    tokenizer = BPETokenizer()
    config = MDLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=1024,
        n_heads=16,
        n_layers=10,
        max_seq_len=256,  # support both training (128) and SFT (256)
    )
    model = MDLMBPEV3(config, pad_id=tokenizer.pad_id,
                      mask_id=tokenizer.mask_id).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  Model: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Data: {len(tokens_tensor):,} seqs × {seq_len} tokens = {len(tokens_tensor)*seq_len:,} tokens")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Accum: {gradient_accumulation}")
    print(f"  Effective batch: {batch_size * gradient_accumulation}")
    print(f"  Opt steps/epoch: {len(loader)//gradient_accumulation:,}")
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

    # Training
    model.train()
    micro_step = 0
    opt_step = 0
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

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = mdlm_loss(model, batch_tokens, tokenizer.mask_id)
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
                        }, CHECKPOINT_DIR / "mdlm_bpe_v3_best.pt")

                        samples = sample_semi_ar(
                            model, tokenizer, seq_len=seq_len,
                            n_samples=3, block_size=4, temperature=0.7,
                        )
                        print(f"    → Best samples (semi-AR):")
                        for i, s in enumerate(samples):
                            print(f"      [{i}] {s.strip()[:120]}")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Optimizer steps: {opt_step}")
    print(f"  Best eval loss: {best_eval:.4f} (PPL={math.exp(min(best_eval,15)):.1f})")

    torch.save({
        "model_state": model.state_dict(),
        "config": config.to_dict(),
        "step": opt_step,
        "losses": losses[-1000:],
    }, CHECKPOINT_DIR / "mdlm_bpe_v3_final.pt")

    results = {
        "optimizer_steps": opt_step,
        "time": elapsed,
        "best_eval_loss": best_eval,
        "best_ppl": math.exp(min(best_eval, 15)),
        "final_loss": losses[-1],
        "tokens_trained": micro_step * batch_size * seq_len,
    }
    with open(RESULTS_DIR / "mdlm_bpe_v3_training.json", "w") as f:
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
                loss = mdlm_loss(model, tokens, mask_id)
            losses.append(loss.item())
    return np.mean(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--accum", type=int, default=4)
    args = parser.parse_args()
    train_v3(epochs=args.epochs, batch_size=args.batch_size,
             lr=args.lr, seq_len=args.seq_len, gradient_accumulation=args.accum)
