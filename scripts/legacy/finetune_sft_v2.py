"""
SFT Fine-tune MDLM-BPE v2 for chatbot functionality.

Approach: infilling-style training
  - Context (user message) is kept fixed
  - Only the assistant response span is masked/predicted
  - This leverages MDLM's native masked prediction capability

Memory-conscious: streaming, small batch, gradient accumulation.
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

from mdlm_bpe_v2 import (
    MDLMConfig, MDLMBPEV2, BPETokenizer,
    forward_mask_bpe, sample_mdlm_v2,
)
from tokenizers import Tokenizer


MAX_SEQ_LEN = 128


class SFTDataset(Dataset):
    """Dataset for SFT: formats conversations, marks response spans."""

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
    tokens = torch.stack([b["tokens"] for b in batch])
    return {
        "tokens": tokens,
        "resp_starts": [b["resp_start"] for b in batch],
        "resp_ends": [b["resp_end"] for b in batch],
    }


def sft_loss(model, batch_tokens, resp_starts, resp_ends, mask_id):
    """MDLM loss only on response tokens (infilling).

    Vectorized: builds a response mask, masks tokens there, computes CE.
    """
    batch_size, seq_len = batch_tokens.shape
    t = torch.rand(batch_size, device=batch_tokens.device)

    # Build position indices [batch, seq]
    positions = torch.arange(seq_len, device=batch_tokens.device).unsqueeze(0)

    # Response span mask: True where position is in [resp_start, resp_end)
    resp_starts_t = torch.tensor(resp_starts, device=batch_tokens.device).unsqueeze(1)
    resp_ends_t = torch.tensor(resp_ends, device=batch_tokens.device).unsqueeze(1)
    is_response = (positions >= resp_starts_t) & (positions < resp_ends_t)

    # Random masking only within response span
    rand = torch.rand_like(batch_tokens.float())
    mask_positions = is_response & (rand < t[:, None])

    # Apply mask
    masked = torch.where(mask_positions, mask_id, batch_tokens)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(masked, t)

    mask_flat = mask_positions.reshape(-1)
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=batch_tokens.device, requires_grad=True)

    logits_flat = logits.reshape(-1, logits.shape[-1])
    tokens_flat = batch_tokens.reshape(-1)
    return F.cross_entropy(logits_flat[mask_flat], tokens_flat[mask_flat])


@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_len=48, n_steps=24,
                      temperature=0.7):
    """Generate response via infilling (mask response span, unmask)."""
    model.eval()
    mask_id = tokenizer.mask_id
    pad_id = tokenizer.pad_id
    user_tok = tokenizer.tokenizer.token_to_id("<|user|>")
    asst_tok = tokenizer.tokenizer.token_to_id("<|assistant|>")

    ctx_ids = tokenizer.tokenizer.encode(prompt).ids
    prefix = [tokenizer.bos_id, user_tok] + ctx_ids + [asst_tok]
    response_start = len(prefix)

    seq = prefix + [mask_id] * max_len
    seq = seq[:MAX_SEQ_LEN]
    while len(seq) < MAX_SEQ_LEN:
        seq.append(pad_id)

    tokens = torch.tensor([seq], device=DEVICE)

    for step in range(n_steps):
        t_val = max(1.0 - step / n_steps, 0.01)
        t = torch.full((1,), t_val, device=DEVICE)
        logits = model(tokens, t)

        mask_pos = (tokens[0] == mask_id)
        mask_pos[:response_start] = False
        masked_idx = mask_pos.nonzero(as_tuple=True)[0]
        if len(masked_idx) == 0:
            break

        pos_logits = logits[0, masked_idx] / max(temperature, 0.01)
        probs = F.softmax(pos_logits, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        confidence = probs.max(dim=-1)[0]

        n_unmask = max(1, len(masked_idx) // (n_steps - step))
        top_conf, top_idx = confidence.topk(min(n_unmask, len(masked_idx)))
        tokens[0, masked_idx[top_idx]] = sampled[top_idx]

    resp_ids = tokens[0, response_start:].cpu().tolist()
    if tokenizer.eos_id in resp_ids:
        resp_ids = resp_ids[:resp_ids.index(tokenizer.eos_id)]
    resp_ids = [i for i in resp_ids if i not in (pad_id, mask_id)]
    return tokenizer.decode(resp_ids)


def finetune_sft(epochs=2, batch_size=16, lr=5e-5, gradient_accumulation=4):
    """Fine-tune on UltraChat SFT data."""
    print("=" * 70)
    print("SFT FINE-TUNING — MDLM-BPE v2 → Chatbot")
    print("=" * 70)

    # Load pretrained
    pretrain_path = CHECKPOINT_DIR / "mdlm_bpe_v2_best.pt"
    ckpt = torch.load(pretrain_path, map_location=DEVICE, weights_only=False)
    # Model config — use max_seq_len=128 to support SFT (longer sequences)
    config_overrides = ckpt["config"].copy()
    config_overrides["max_seq_len"] = MAX_SEQ_LEN
    config = MDLMConfig(**config_overrides)
    model = MDLMBPEV2(config, pad_id=0, mask_id=1).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  Base: {pretrain_path.name} (step {ckpt.get('step', '?')})")
    print(f"  Params: {n_params:,}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Grad accum: {gradient_accumulation}")
    print(f"  Effective batch: {batch_size * gradient_accumulation}")
    print(f"  LR: {lr}")

    raw_tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    dataset = SFTDataset(SFT_FILE, raw_tok, max_seq_len=MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=sft_collate, drop_last=True,
                        num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(loader) // gradient_accumulation) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.05,
    )

    mask_id = raw_tok.token_to_id("<mask>")
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

            # zero_grad happens after optimizer.step(), not here
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
                    print(f"    Generation test:")
                    for q in tests:
                        resp = generate_response(model, bpe_tok, q, max_len=48, n_steps=20)
                        print(f"      Q: {q}")
                        print(f"      A: {resp[:120]}")
                    model.train()

                    if losses[-1] < best_loss:
                        best_loss = losses[-1]
                        torch.save({
                            "model_state": model.state_dict(),
                            "config": config.to_dict(),
                            "step": opt_step,
                            "loss": best_loss,
                        }, CHECKPOINT_DIR / "mdlm_bpe_v2_sft.pt")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"SFT COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Optimizer steps: {opt_step}")
    print(f"  Best loss: {best_loss:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "config": config.to_dict(),
        "step": opt_step,
        "losses": losses[-500:],
    }, CHECKPOINT_DIR / "mdlm_bpe_v2_sft_final.pt")

    results = {
        "sft_steps": opt_step,
        "time": elapsed,
        "best_loss": best_loss,
        "final_loss": losses[-1] if losses else None,
    }
    with open(RESULTS_DIR / "mdlm_bpe_v2_sft.json", "w") as f:
        json.dump(results, f, indent=2)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--accum", type=int, default=4)
    args = parser.parse_args()
    finetune_sft(epochs=args.epochs, batch_size=args.batch_size,
                 lr=args.lr, gradient_accumulation=args.accum)
