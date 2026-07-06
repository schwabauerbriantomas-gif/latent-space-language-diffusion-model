"""
SFT Fine-tune MDLM-BPE v3 for chatbot.
Uses semi-AR unmasking for coherent generation.
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
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = REPO / "checkpoints"
RESULTS_DIR = REPO / "results"
TOKENIZER_PATH = REPO / "tokenizer" / "bpe_tokenizer.json"
SFT_FILE = REPO / "data" / "sft_ultrachat.jsonl"
MAX_SEQ_LEN = 256

from mdlm_bpe_v3 import (
    MDLMConfig, MDLMBPEV3, BPETokenizer,
    forward_mask_bpe, generate_response_semi_ar,
)
from tokenizers import Tokenizer


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, raw_tokenizer, max_seq_len=MAX_SEQ_LEN):
        self.tok = raw_tokenizer
        self.max_len = max_seq_len
        self.bos = raw_tokenizer.token_to_id("<bos>")
        self.eos = raw_tokenizer.token_to_id("<eos>")
        self.pad = raw_tokenizer.token_to_id("<pad>")
        self.user_tok = raw_tokenizer.token_to_id("<|user|>")
        self.asst_tok = raw_tokenizer.token_to_id("<|assistant|>")
        self.samples = []
        self._load(jsonl_path)

    def _load(self, path):
        with open(path) as f:
            for line in f:
                conv = json.loads(line)
                messages = conv["messages"]
                for i in range(1, len(messages), 2):
                    if i >= len(messages):
                        break
                    user_msg = messages[i-1]
                    asst_msg = messages[i]
                    if user_msg["role"] != "user":
                        continue
                    if asst_msg["role"] != "assistant":
                        continue

                    ctx_ids = self.tok.encode(user_msg["content"]).ids
                    resp_ids = self.tok.encode(asst_msg["content"]).ids

                    prefix = [self.bos, self.user_tok] + ctx_ids + [self.asst_tok]
                    full = prefix + resp_ids + [self.eos]

                    if len(full) > self.max_len:
                        keep_prefix = min(len(prefix), self.max_len // 2)
                        keep_suffix = self.max_len - keep_prefix
                        prefix = prefix[:keep_prefix]
                        full = prefix + resp_ids[:keep_suffix - 1] + [self.eos]

                    if len(full) < 10 or len(resp_ids) < 3:
                        continue

                    response_start = len(prefix)
                    padded = full + [self.pad] * (self.max_len - len(full))
                    self.samples.append({
                        "tokens": padded[:self.max_len],
                        "resp_start": response_start,
                        "resp_end": min(len(full), self.max_len),
                    })
        print(f"  SFT samples: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "tokens": torch.tensor(s["tokens"], dtype=torch.long),
            "resp_start": s["resp_start"],
            "resp_end": s["resp_end"],
        }


def sft_collate(batch):
    return {
        "tokens": torch.stack([b["tokens"] for b in batch]),
        "resp_starts": [b["resp_start"] for b in batch],
        "resp_ends": [b["resp_end"] for b in batch],
    }


def sft_loss(model, batch_tokens, resp_starts, resp_ends, mask_id):
    """Vectorized infilling loss on response span only."""
    batch, seq = batch_tokens.shape
    t = torch.rand(batch, device=batch_tokens.device)

    positions = torch.arange(seq, device=batch_tokens.device).unsqueeze(0)
    resp_starts_t = torch.tensor(resp_starts, device=batch_tokens.device).unsqueeze(1)
    resp_ends_t = torch.tensor(resp_ends, device=batch_tokens.device).unsqueeze(1)
    is_response = (positions >= resp_starts_t) & (positions < resp_ends_t)

    rand = torch.rand_like(batch_tokens.float())
    mask_positions = is_response & (rand < t[:, None])
    masked = torch.where(mask_positions, mask_id, batch_tokens)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(masked, t)

    mask_flat = mask_positions.reshape(-1)
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=batch_tokens.device, requires_grad=True)

    logits_flat = logits.reshape(-1, logits.shape[-1])
    tokens_flat = batch_tokens.reshape(-1)
    return F.cross_entropy(logits_flat[mask_flat], tokens_flat[mask_flat])


def finetune_sft(epochs=3, batch_size=16, lr=5e-5, gradient_accumulation=4):
    print("=" * 70)
    print("SFT FINE-TUNING — MDLM-BPE v3 → Chatbot")
    print("=" * 70)

    pretrain_path = CHECKPOINT_DIR / "mdlm_bpe_v3_best.pt"
    if not pretrain_path.exists():
        pretrain_path = CHECKPOINT_DIR / "mdlm_bpe_v3_final.pt"

    ckpt = torch.load(pretrain_path, map_location=DEVICE, weights_only=False)
    config = MDLMConfig(**ckpt["config"])
    model = MDLMBPEV3(config, pad_id=0, mask_id=1).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  Base: {pretrain_path.name}")
    print(f"  Params: {n_params:,} ({n_params/1e6:.1f}M)")

    raw_tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    dataset = SFTDataset(SFT_FILE, raw_tok, max_seq_len=MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=sft_collate, drop_last=True,
                        num_workers=2, pin_memory=True)

    mask_id = raw_tok.token_to_id("<mask>")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(loader) // gradient_accumulation) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.05,
    )

    model.train()
    micro_step = 0
    opt_step = 0
    best_loss = float('inf')
    losses = []
    start = time.time()

    for epoch in range(epochs):
        ep_loss = 0
        ep_count = 0
        accum = 0.0

        for batch in loader:
            micro_step += 1
            tokens = batch["tokens"].to(DEVICE, non_blocking=True)

            loss = sft_loss(model, tokens, batch["resp_starts"],
                           batch["resp_ends"], mask_id)
            loss = loss / gradient_accumulation
            loss.backward()
            accum += loss.item()

            if micro_step % gradient_accumulation == 0:
                opt_step += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if opt_step <= total_steps:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                ep_loss += accum
                ep_count += 1
                losses.append(accum)
                accum = 0.0

                if opt_step % 50 == 0:
                    elapsed = time.time() - start
                    print(f"  [E{epoch+1} O{opt_step:,}] loss={losses[-1]:.4f} "
                          f"avg={ep_loss/ep_count:.4f} "
                          f"{opt_step/elapsed:.1f} opt/s")

                if opt_step % 200 == 0:
                    model.eval()
                    bpe_tok = BPETokenizer()
                    tests = ["What is machine learning?", "How do I write a function?"]
                    print(f"    Generation test (semi-AR):")
                    for q in tests:
                        resp = generate_response_semi_ar(
                            model, bpe_tok, q, max_len=64,
                            block_size=4, temperature=0.6,
                        )
                        print(f"      Q: {q}")
                        print(f"      A: {resp.strip()[:120]}")
                    model.train()

                    if losses[-1] < best_loss:
                        best_loss = losses[-1]
                        torch.save({
                            "model_state": model.state_dict(),
                            "config": config.to_dict(),
                            "step": opt_step,
                            "loss": best_loss,
                        }, CHECKPOINT_DIR / "mdlm_bpe_v3_sft.pt")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"SFT COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Steps: {opt_step}, Best loss: {best_loss:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "config": config.to_dict(),
        "step": opt_step,
        "losses": losses[-500:],
    }, CHECKPOINT_DIR / "mdlm_bpe_v3_sft_final.pt")

    results = {
        "sft_steps": opt_step,
        "time": elapsed,
        "best_loss": best_loss,
        "final_loss": losses[-1] if losses else None,
    }
    with open(RESULTS_DIR / "mdlm_bpe_v3_sft.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--accum", type=int, default=4)
    args = parser.parse_args()
    finetune_sft(epochs=args.epochs, batch_size=args.batch_size,
                 lr=args.lr, gradient_accumulation=args.accum)
