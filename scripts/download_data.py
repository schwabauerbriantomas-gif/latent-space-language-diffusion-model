"""
Download and prepare Ultra-FineWeb samples for MDLM training.

Streams from HuggingFace, filters by quality score, saves to disk.
Does NOT download the full 1.29B rows — samples a meaningful subset.
"""
import json
import time
import itertools
from pathlib import Path

from datasets import load_dataset

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Config ──
N_SAMPLES = 100_000       # 100K documents — enough for BPE + pretraining
MIN_SCORE = 0.5           # Quality filter
MIN_CHARS = 200           # Skip very short docs
MAX_CHARS = 5000          # Skip very long docs (for manageable seq lens)
BATCH_SAVE = 10_000       # Save every N docs


def download_ultra_fineweb():
    """Stream Ultra-FineWeb (en), filter, save."""
    print("=" * 70)
    print("DOWNLOADING Ultra-FineWeb (en) — streaming sample")
    print("=" * 70)
    print(f"  Target: {N_SAMPLES:,} documents")
    print(f"  Filters: score >= {MIN_SCORE}, {MIN_CHARS}-{MAX_CHARS} chars")
    print()

    ds = load_dataset("openbmb/Ultra-FineWeb", split="en", streaming=True)

    output_file = DATA_DIR / "ultra_fineweb_en.jsonl"
    collected = 0
    rejected_score = 0
    rejected_len = 0
    start = time.time()
    batch = []

    with open(output_file, "w") as f:
        for doc in ds:
            if collected >= N_SAMPLES:
                break

            content = doc.get("content", "")
            score = doc.get("score", 0.0)
            source = doc.get("source", "unknown")

            # Filters
            if score < MIN_SCORE:
                rejected_score += 1
                continue
            if len(content) < MIN_CHARS or len(content) > MAX_CHARS:
                rejected_len += 1
                continue

            batch.append({
                "content": content,
                "score": score,
                "source": source,
            })
            collected += 1

            if len(batch) >= BATCH_SAVE:
                for item in batch:
                    f.write(json.dumps(item) + "\n")
                elapsed = time.time() - start
                rate = collected / elapsed if elapsed > 0 else 0
                print(f"  [{collected:,}/{N_SAMPLES:,}] "
                      f"{rate:.0f} docs/s | "
                      f"rejected: score={rejected_score:,} len={rejected_len:,}")
                batch = []

    # Flush remaining
    for item in batch:
        f.write(json.dumps(item) + "\n")

    elapsed = time.time() - start
    print()
    print(f"✓ Downloaded {collected:,} documents in {elapsed:.1f}s")
    print(f"  Rejected: score={rejected_score:,} len={rejected_len:,}")
    print(f"  Saved to: {output_file}")

    # Stats
    total_chars = sum(len(json.loads(line)["content"])
                      for line in open(output_file))
    total_words = total_chars // 5  # rough estimate
    total_tokens_est = int(total_words * 1.3)  # rough BPE estimate

    print(f"\n  Total chars: {total_chars:,}")
    print(f"  Estimated words: {total_words:,}")
    print(f"  Estimated tokens: {total_tokens_est:,}")

    return output_file


if __name__ == "__main__":
    download_ultra_fineweb()
