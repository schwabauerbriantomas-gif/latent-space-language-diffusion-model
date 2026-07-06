"""
Download and prepare SFT (Supervised Fine-Tuning) data for chatbot/tool-use.

Uses HuggingFaceH4/ultrachat_200k (open alternative to UltraData-SFT-2605).
Formats conversations as chat templates compatible with our BPE tokenizer.
"""
import json
import time
from pathlib import Path

from datasets import load_dataset

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

N_SFT_SAMPLES = 20_000     # Enough for chatbot fine-tuning


def download_sft_data():
    """Download UltraChat samples for SFT."""
    print("=" * 70)
    print("DOWNLOADING UltraChat 200K (SFT data)")
    print("=" * 70)
    print(f"  Target: {N_SFT_SAMPLES:,} conversations")

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True,
    )

    output_file = DATA_DIR / "sft_ultrachat.jsonl"
    collected = 0
    start = time.time()

    with open(output_file, "w") as f:
        for item in ds:
            if collected >= N_SFT_SAMPLES:
                break

            messages = item.get("messages", [])
            if len(messages) < 2:
                continue

            # Format as chat turns
            formatted = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "").strip()
                if not content:
                    continue
                if role == "user":
                    formatted.append({"role": "user", "content": content})
                elif role == "assistant":
                    formatted.append({"role": "assistant", "content": content})

            if len(formatted) < 2:
                continue

            # Keep conversations reasonable length
            total_chars = sum(len(m["content"]) for m in formatted)
            if total_chars > 4000:
                # Truncate to first few turns
                acc = 0
                kept = []
                for m in formatted:
                    if acc + len(m["content"]) > 4000:
                        break
                    kept.append(m)
                    acc += len(m["content"])
                formatted = kept if len(kept) >= 2 else formatted[:2]

            record = {
                "messages": formatted,
                "prompt_id": item.get("prompt_id", ""),
            }
            f.write(json.dumps(record) + "\n")
            collected += 1

            if collected % 5000 == 0:
                elapsed = time.time() - start
                rate = collected / elapsed if elapsed > 0 else 0
                print(f"  [{collected:,}/{N_SFT_SAMPLES:,}] {rate:.0f} conv/s")

    elapsed = time.time() - start
    print(f"\n✓ Downloaded {collected:,} conversations in {elapsed:.1f}s")
    print(f"  Saved to: {output_file}")

    # Stats
    n_turns = []
    n_chars = []
    with open(output_file) as f:
        for line in f:
            r = json.loads(line)
            n_turns.append(len(r["messages"]))
            n_chars.append(sum(len(m["content"]) for m in r["messages"]))

    import numpy as np
    turns = np.array(n_turns)
    chars = np.array(n_chars)
    print(f"\n  Turns per conversation:")
    print(f"    Mean: {turns.mean():.1f}, Median: {np.median(turns):.0f}")
    print(f"  Chars per conversation:")
    print(f"    Mean: {chars.mean():.0f}, Median: {np.median(chars):.0f}")
    print(f"  Total chars: {chars.sum():,}")


if __name__ == "__main__":
    download_sft_data()
