"""
Train MDLM v4 (OPTIMIZED) — all v3 improvements + new architecture.

Key optimizations over v3:
  1. torch.compile (15-30% speedup on Ampere)
  2. Logit-normal timestep sampling (better gradient signal)
  3. ELBO-weighted loss (variational bound)
  4. Efficient loss: compute logits only on masked positions
  5. SwiGLU FFN + RMSNorm + simplified AdaLN (faster, fewer params)
  6. Cosine LR schedule with warmup
  7. Mixed precision bf16 throughout
  8. Larger effective batch via gradient accumulation
  9. Streaming data pipeline (no OOM)
"""
import json
import sys
import time
import math
import argparse
import random
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

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def prepare_data(seq_len=128):
    """Tokenize Ultra-FineWeb and pack into seq_len sequences.
    Disk-streaming via memmap — peak RAM = chunk_size only.
    """
    input_file = DATA_DIR / "ultra_fineweb_en_1m.jsonl"
    if not input_file.exists():
        input_file = DATA_DIR / "ultra_fineweb_en.jsonl"
    
    output_file = DATA_DIR / f"train_tokens_v4_{seq_len}.npy"

    if output_file.exists():
        arr = np.load(output_file, mmap_mode='r')
        print(f"  Cached: {output_file} ({len(arr):,} seqs, mmap)")
        return np.array(arr), len(arr)

    from tokenizers import Tokenizer
    tok_path = REPO / "tokenizer" / "bpe_tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    total_docs = sum(1 for _ in open(input_file))
    print(f"  {total_docs:,} documents")

    max_seqs = total_docs * 4
    tmp_file = DATA_DIR / "train_tokens_v4_tmp.npy"
    arr = np.memmap(str(tmp_file), dtype=np.int16, mode='w+',
                    shape=(max_seqs, seq_len))

    CHUNK = 1000
    n_seqs = 0
    current = []
    processed = 0

    print(f"  Streaming tokenize → memmap (chunk={CHUNK})...")

    with open(input_file) as f:
        chunk_texts = []
        for line in f:
            try:
                chunk_texts.append(json.loads(line)["content"])
            except json.JSONDecodeError:
                continue

            if len(chunk_texts) >= CHUNK:
                encoded = tokenizer.encode_batch(chunk_texts)
                chunk_texts = []

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
                    print(f"    [{processed:,}/{total_docs:,}] seqs={n_seqs:,}")

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

    if current:
        while len(current) < seq_len:
            current.append(pad_id)
        arr[n_seqs] = current[:seq_len]
        n_seqs += 1

    del arr
    arr_final = np.memmap(str(tmp_file), dtype=np.int16, mode='r',
                          shape=(n_seqs, seq_len))
    final = np.array(arr_final[:n_seqs])
    np.save(output_file, final)
    del arr_final, final
    tmp_file.unlink()

    print(f"  Saved: {n_seqs:,} sequences ({n_seqs*seq_len*2/1e6:.1f} MB)")
    return np.load(output_file, mmap_mode='r'), n_seqs


def train_v4(epochs=3, batch_size=32, lr=3e-4, seq_len=128,
             warmup_ratio=0.05, eval_every=500, gradient_accumulation=4,
             use_compile=True, use_checkpointing=False, val_frac=0.05,
             eval_t=0.5):
    """Train MDLM v4 with all optimizations."""
    
    from mdlm_v4_optimized import (
        MDLMConfig as MDLMConfigV4, MDLMBPEV4, BPETokenizer,
        mdlm_loss_v4, sample_full_parallel_v4,
    )

    print("=" * 70)
    print("TRAINING MDLM v4 (OPTIMIZED)")
    print("=" * 70)

    # ── Data ──
    print("Loading data...")
    tokens, n_seqs = prepare_data(seq_len=seq_len)
    print(f"  Total sequences: {n_seqs:,}")

    tokens_int16 = np.array(tokens, dtype=np.int16)
    del tokens
    tokens_tensor = torch.from_numpy(tokens_int16).long()

    # ── Held-out split (5%) — separated BEFORE the train shuffle ──
    n_val = max(1, int(len(tokens_tensor) * val_frac))
    val_tensor = tokens_tensor[:n_val]                     # fixed indices, no shuffle
    train_tensor = tokens_tensor[n_val:]
    dataset = TensorDataset(train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=4, pin_memory=True,
                        persistent_workers=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size,
                            shuffle=False, drop_last=False)
    print(f"  Held-out: {len(val_tensor):,} seqs ({val_frac:.0%}, deterministic eval)")

    # ── Model ──
    tokenizer = BPETokenizer()
    config = MDLMConfigV4(
        vocab_size=tokenizer.vocab_size,
        d_model=1024,
        n_heads=16,
        n_layers=10,
        max_seq_len=256,
        use_checkpointing=use_checkpointing,
    )
    model = MDLMBPEV4(config, pad_id=tokenizer.pad_id,
                      mask_id=tokenizer.mask_id).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    # ── Scheduler math (before compile, needed for display) ──
    batches_per_epoch = len(loader)
    total_micro_steps = batches_per_epoch * epochs
    total_optimizer_steps = max(1, total_micro_steps // gradient_accumulation)

    print(f"  Model: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Data: {len(tokens_tensor):,} seqs × {seq_len} tokens = {len(tokens_tensor)*seq_len:,} tokens")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Accum: {gradient_accumulation}")
    print(f"  Effective batch: {batch_size * gradient_accumulation}")
    print(f"  Total opt steps: {total_optimizer_steps:,}")

    # ── torch.compile ──
    if use_compile:
        print("  Compiling model with torch.compile...")
        compile_start = time.time()
        model = torch.compile(model, mode="max-autotune")
        print(f"    Compiled in {time.time()-compile_start:.1f}s")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_optimizer_steps,
        pct_start=warmup_ratio,
    )

    # ── Training loop ──
    model.train()
    micro_step = 0
    opt_step = 0
    best_eval = float('inf')
    did_eval = False
    losses = []
    start = time.time()
    accum_loss = 0.0

    # VRAM tracking
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        ep_loss = 0
        ep_count = 0

        for batch in loader:
            micro_step += 1
            batch_tokens = batch[0].to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = mdlm_loss_v4(model, batch_tokens, tokenizer.mask_id)
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

                if opt_step == 1:
                    # Seed smoke check: first opt-step loss depends only on
                    # init weights + dropout masks (both seeded) — must be
                    # identical across runs with the same --seed.
                    print(f"  [seed-check] first opt-step loss = {losses[0]:.6f}")

                if opt_step % 100 == 0:
                    elapsed = time.time() - start
                    tps = micro_step * batch_size * seq_len / elapsed
                    lr_cur = optimizer.param_groups[0]['lr']
                    peak_vram = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  [E{epoch+1} O{opt_step:,}] loss={losses[-1]:.4f} "
                          f"avg={ep_loss/ep_count:.4f} lr={lr_cur:.2e} "
                          f"{opt_step/elapsed:.1f} opt/s {tps:,.0f} tok/s "
                          f"VRAM={peak_vram:.1f}GB")

                if opt_step % eval_every == 0:
                    did_eval = True
                    model.eval()
                    eval_loss = quick_eval(model, val_loader, tokenizer.mask_id,
                                           eval_t=eval_t)
                    model.train()

                    # val_ppl = exp(plain CE, deterministic) — no clamp, no ELBO weight
                    ppl = math.exp(eval_loss)
                    print(f"    → val_ppl={ppl:.1f} (CE={eval_loss:.4f}, held-out, t={eval_t})")

                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        # Save uncompiled state dict
                        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                        torch.save({
                            "model_state": raw_model.state_dict(),
                            "config": config.to_dict(),
                            "step": opt_step,
                            "eval_loss": eval_loss,
                            "ppl": ppl,
                            "version": "v4",
                        }, CHECKPOINT_DIR / "mdlm_v4_best.pt")

                        samples = sample_full_parallel_v4(
                            raw_model, tokenizer, seq_len=seq_len,
                            n_samples=3, n_steps=32, temperature=0.7,
                        )
                        print(f"    → Best samples (full-parallel):")
                        for i, s in enumerate(samples):
                            print(f"      [{i}] {s.strip()[:120]}")

    elapsed = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    # Final eval: guarantee at least one val_ppl measurement even when the run
    # is shorter than eval_every (short/1-epoch runs previously left best_eval
    # at inf and never wrote mdlm_v4_best.pt).
    if did_eval is False or best_eval == float('inf'):
        model.eval()
        eval_loss = quick_eval(model, val_loader, tokenizer.mask_id, eval_t=eval_t)
        model.train()
        ppl = math.exp(eval_loss)
        print(f"  → final eval: val_ppl={ppl:.1f} (CE={eval_loss:.4f}, held-out, t={eval_t})")
        if eval_loss < best_eval:
            best_eval = eval_loss
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                "model_state": raw_model.state_dict(),
                "config": config.to_dict(),
                "step": opt_step,
                "eval_loss": eval_loss,
                "ppl": ppl,
                "version": "v4",
            }, CHECKPOINT_DIR / "mdlm_v4_best.pt")

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Optimizer steps: {opt_step}")
    print(f"  Best val CE: {best_eval:.4f} (val_ppl={math.exp(best_eval):.1f}, held-out, no clamp)")
    print(f"  Peak VRAM: {peak_vram:.1f} GB")

    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        "model_state": raw_model.state_dict(),
        "config": config.to_dict(),
        "step": opt_step,
        "losses": losses[-1000:],
        "version": "v4",
    }, CHECKPOINT_DIR / "mdlm_v4_final.pt")

    results = {
        "version": "v4",
        "optimizer_steps": opt_step,
        "time_s": elapsed,
        "time_min": elapsed / 60,
        "best_val_ce": best_eval,
        "best_val_ppl": math.exp(best_eval),
        "final_loss": losses[-1],
        "tokens_trained": micro_step * batch_size * seq_len,
        "params": n_params,
        "peak_vram_gb": peak_vram,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": gradient_accumulation,
            "lr": lr,
            "seq_len": seq_len,
            "use_compile": use_compile,
            "use_checkpointing": use_checkpointing,
            "val_frac": val_frac,
            "eval_t": eval_t,
        },
    }
    with open(RESULTS_DIR / "mdlm_v4_training.json", "w") as f:
        json.dump(results, f, indent=2)

    return model


def quick_eval(model, val_loader, mask_id, n_batches=20, eval_t=0.5):
    """Deterministic held-out eval: plain CE on masked positions.

    Unlike training loss (stochastic ELBO-weighted, mean weight E[1/(1-t)]
    ~= 2.65), this fixes t=eval_t (default 0.5) and the mask via a local RNG
    generator, so val_ppl is reproducible run-to-run and directly comparable
    across checkpoints. Returns mean CE in nats.
    """
    model.eval()
    losses = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            tokens = batch[0].to(DEVICE)
            batch_size = tokens.shape[0]
            # Fixed t (no logit-normal sampling, no ELBO weighting)
            t = torch.full((batch_size,), float(eval_t), device=tokens.device)
            # Deterministic mask with local generator (protect specials as in train)
            rand = torch.rand(tokens.shape, generator=gen).to(tokens.device)
            mask_positions = rand < t[:, None]
            is_special = (tokens == 0) | (tokens == 2) | (tokens == 3)
            mask_positions = mask_positions & (~is_special)
            if mask_positions.sum() == 0:
                continue
            masked = torch.where(mask_positions, mask_id, tokens)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                hidden = model(masked, t, return_pre_proj=True)
                hidden_flat = hidden.reshape(-1, hidden.shape[-1])
                mask_flat = mask_positions.reshape(-1)
                hidden_masked = hidden_flat[mask_flat]
                tokens_masked = tokens.reshape(-1)[mask_flat]
                logits = F.linear(hidden_masked.float(),
                                  model.token_emb.weight.float())
                ce = F.cross_entropy(logits, tokens_masked)
            losses.append(ce.item())
    return np.mean(losses) if losses else float("nan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="held-out fraction for deterministic val PPL")
    parser.add_argument("--eval-t", type=float, default=0.5,
                        help="fixed mask fraction t for deterministic eval")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for torch/random/numpy (reproducibility)")
    args = parser.parse_args()

    # Reproducibility: seed all RNGs before any data/model init
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"  Seed: {args.seed} (torch/random/numpy)")

    train_v4(epochs=args.epochs, batch_size=args.batch_size,
             lr=args.lr, seq_len=args.seq_len, gradient_accumulation=args.accum,
             use_compile=not args.no_compile, use_checkpointing=args.checkpointing,
             val_frac=args.val_frac, eval_t=args.eval_t)
