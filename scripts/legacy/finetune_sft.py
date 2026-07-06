"""
Fine-tune MDLM-BPE for chatbot functionality.

Since MDLM is NOT autoregressive (it generates all positions in parallel),
we adapt SFT data to the masked diffusion paradigm:

  Approach: "Fill-in-the-blank" dialog training
  - Format: <|user|> prompt <|assistant|> response
  - During training: mask only the assistant response portion
  - The model learns to generate responses given the full context

  This is analogous to infilling / masked span prediction, which is
  exactly what MDLM does natively.

For tool-use and code generation, we add special format markers.
"""
import json
import sys
import time
import math
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = REPO / "checkpoints"
RESULTS_DIR = REPO / "results"
TOKENIZER_PATH = REPO / "tokenizer" / "bpe_tokenizer.json"
SFT_FILE = REPO / "data" / "sft_ultrachat.jsonl"

from mdlm_bpe import (
    MDLMConfig, MDLMBPETransformer, BPETokenizer,
    forward_mask_bpe, sample_mdlm_bpe,
)

MAX_SEQ_LEN = 128


class SFTDataset(Dataset):
    """Dataset for SFT: formats conversations, marks response spans."""

    def __init__(self, jsonl_path, tokenizer, max_seq_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        self._load(jsonl_path)

    def _load(self, path):
        with open(path) as f:
            for line in f:
                conv = json.loads(line)
                messages = conv["messages"]

                # Create training samples: for each assistant response,
                # create a context (all prior turns) + target (the response)
                for i in range(1, len(messages), 2):
                    if i >= len(messages):
                        break
                    user_msg = messages[i-1]
                    asst_msg = messages[i]
                    if user_msg["role"] != "user" or asst_msg["role"] != "assistant":
                        continue

                    context = user_msg["content"]
                    response = asst_msg["content"]

                    # Tokenize
                    ctx_ids = self.tokenizer.encode(context, add_special=False)
                    resp_ids = self.tokenizer.encode(response, add_special=False)

                    # Construct: <bos> <|user|> ctx <|assistant|> resp <eos>
                    user_tok = self.tokenizer.token_to_id("<|user|>")
                    asst_tok = self.tokenizer.token_to_id("<|assistant|>")
                    bos = self.tokenizer.token_to_id("<bos>")
                    eos = self.tokenizer.token_to_id("<eos>")

                    # Build sequence
                    prefix = [bos, user_tok] + ctx_ids + [asst_tok]
                    full = prefix + resp_ids + [eos]

                    # Truncate if too long (keep the end of context + full response)
                    if len(full) > self.max_seq_len:
                        # Keep last max_seq_len tokens but ensure prefix is included
                        keep_prefix = min(len(prefix), self.max_seq_len // 2)
                        keep_suffix = self.max_seq_len - keep_prefix
                        prefix = prefix[:keep_prefix]
                        # Need to rejoin properly
                        full = prefix + resp_ids[:keep_suffix-1] + [eos]

                    if len(full) < 10 or len(resp_ids) < 3:
                        continue

                    # Pad to max_seq_len
                    pad_id = self.tokenizer.token_to_id("<pad>")
                    response_start = len(prefix)
                    padded = full + [pad_id] * (self.max_seq_len - len(full))

                    self.samples.append({
                        "tokens": padded[:self.max_seq_len],
                        "response_start": response_start,
                        "response_end": min(len(full), self.max_seq_len),
                    })

        print(f"  Loaded {len(self.samples):,} SFT samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "tokens": torch.tensor(s["tokens"], dtype=torch.long),
            "resp_start": s["response_start"],
            "resp_end": s["response_end"],
        }


def sft_collate(batch):
    tokens = torch.stack([b["tokens"] for b in batch])
    return {
        "tokens": tokens,
        "resp_starts": [b["resp_start"] for b in batch],
        "resp_ends": [b["resp_end"] for b in batch],
    }


def sft_loss(model, batch_tokens, resp_starts, resp_ends, mask_id):
    """MDLM loss but only on response tokens.

    The model sees the full sequence (context + masked response),
    and is trained to predict the response tokens.
    """
    batch_size, seq_len = batch_tokens.shape
    t = torch.rand(batch_size, device=batch_tokens.device)

    # Forward mask: only apply to response span
    masked = batch_tokens.clone()
    mask_positions = torch.zeros_like(batch_tokens, dtype=torch.bool)

    for b in range(batch_size):
        rs, re = resp_starts[b], resp_ends[b]
        # Mask tokens in the response span with probability t
        n_resp = re - rs
        if n_resp <= 0:
            continue
        rand = torch.rand(n_resp, device=batch_tokens.device)
        apply_mask = rand < t[b]
        mask_positions[b, rs:re] = apply_mask
        masked[b, rs:re][apply_mask] = mask_id

    logits = model(masked, t)

    # Loss only on masked response positions
    mask_flat = mask_positions.reshape(-1)
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=batch_tokens.device)

    logits_flat = logits.reshape(-1, logits.shape[-1])
    tokens_flat = batch_tokens.reshape(-1)
    return F.cross_entropy(logits_flat[mask_flat], tokens_flat[mask_flat])


@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_len=64, n_steps=32,
                      temperature=0.7):
    """Generate a response to a prompt.

    Uses the infilling approach: context is fixed, response span is
    iteratively unmasked.
    """
    model.eval()

    user_tok = tokenizer.token_to_id("<|user|>")
    asst_tok = tokenizer.token_to_id("<|assistant|>")
    bos = tokenizer.bos_id
    eos = tokenizer.eos_id
    mask_id = tokenizer.mask_id
    pad_id = tokenizer.pad_id

    ctx_ids = tokenizer.encode(prompt, add_special=False)
    prefix = [bos, user_tok] + ctx_ids + [asst_tok]
    response_start = len(prefix)

    # Create sequence
    seq = prefix + [mask_id] * max_len
    seq = seq[:MAX_SEQ_LEN]

    # Pad if needed
    while len(seq) < MAX_SEQ_LEN:
        seq.append(pad_id)

    tokens = torch.tensor([seq], device=DEVICE)
    response_end = MAX_SEQ_LEN

    # Iterative unmasking (only in response span)
    for step in range(n_steps):
        t_val = 1.0 - step / n_steps
        t_val = max(t_val, 0.01)
        t = torch.full((1,), t_val, device=DEVICE)

        logits = model(tokens, t)

        # Only look at masked positions in response span
        for b in range(1):
            mask_pos = (tokens[b] == mask_id)
            mask_pos[:response_start] = False  # Don't touch context

            masked_idx = mask_pos.nonzero(as_tuple=True)[0]
            if len(masked_idx) == 0:
                break

            pos_logits = logits[b, masked_idx] / max(temperature, 0.01)
            probs = F.softmax(pos_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            confidence = probs.max(dim=-1)[0]

            n_to_unmask = max(1, len(masked_idx) // (n_steps - step))
            top_conf, top_idx = confidence.topk(min(n_to_unmask, len(masked_idx)))
            positions_to_fill = masked_idx[top_idx]
            tokens[b, positions_to_fill] = sampled[top_idx]

    # Decode response
    resp_ids = tokens[0, response_start:response_end].cpu().tolist()
    if eos in resp_ids:
        resp_ids = resp_ids[:resp_ids.index(eos)]
    # Remove pad and mask
    resp_ids = [i for i in resp_ids if i not in (pad_id, mask_id)]

    return tokenizer.decode(resp_ids)


def finetune_sft(epochs=2, batch_size=64, lr=1e-4):
    """Fine-tune on SFT data."""
    print("=" * 70)
    print("FINE-TUNING MDLM-BPE for Chatbot (UltraChat SFT)")
    print("=" * 70)

    # Load pretrained model
    pretrain_path = CHECKPOINT_DIR / "mdlm_bpe_best.pt"
    if not pretrain_path.exists():
        pretrain_path = CHECKPOINT_DIR / "mdlm_bpe_final.pt"

    ckpt = torch.load(pretrain_path, map_location=DEVICE, weights_only=False)
    config = MDLMConfig(**ckpt["config"])
    model = MDLMBPETransformer(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded pretrained: {pretrain_path.name}")
    print(f"  Params: {n_params:,}")
    print(f"  SFT epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR: {lr}")

    # Load SFT data
    raw_tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    dataset = SFTDataset(SFT_FILE, raw_tokenizer, max_seq_len=MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=sft_collate, drop_last=True,
                        num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(loader))

    model.train()
    step = 0
    start = time.time()
    losses = []

    for epoch in range(epochs):
        ep_loss = 0
        ep_steps = 0

        for batch in loader:
            step += 1
            tokens = batch["tokens"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            loss = sft_loss(model, tokens, batch["resp_starts"],
                           batch["resp_ends"], config)
            # Get mask_id from the tokenizer
            mask_id = raw_tokenizer.token_to_id("<mask>")
            loss = sft_loss(model, tokens, batch["resp_starts"],
                           batch["resp_ends"], mask_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            ep_steps += 1
            losses.append(loss.item())

            if step % 100 == 0:
                elapsed = time.time() - start
                print(f"  [E{epoch+1} S{step:,}] loss={loss.item():.4f} "
                      f"avg={ep_loss/ep_steps:.4f} "
                      f"{step/elapsed:.1f} steps/s")

        print(f"  Epoch {epoch+1} avg: {ep_loss/max(ep_steps,1):.4f}")

        # Test generation
        model.eval()
        test_prompts = [
            "What is machine learning?",
            "How do I write a Python function?",
        ]
        print(f"\n  Generation test:")
        bpe_tok = BPETokenizer()
        for prompt in test_prompts:
            response = generate_response(model, bpe_tok, prompt, max_len=48, n_steps=20)
            print(f"    Q: {prompt}")
            print(f"    A: {response[:150]}")
        model.train()

    # Save
    elapsed = time.time() - start
    sft_path = CHECKPOINT_DIR / "mdlm_bpe_sft.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": config.__dict__,
        "step": step,
        "sft_losses": losses,
    }, sft_path)

    print(f"\n✓ SFT complete in {elapsed:.1f}s")
    print(f"  Saved: {sft_path}")

    results = {
        "sft_steps": step,
        "sft_time": elapsed,
        "final_loss": losses[-1] if losses else None,
        "sft_samples": len(dataset),
    }
    with open(RESULTS_DIR / "mdlm_bpe_sft.json", "w") as f:
        json.dump(results, f, indent=2)

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    finetune_sft(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
