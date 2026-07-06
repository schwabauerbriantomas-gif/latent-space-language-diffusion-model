"""
Download MORE Ultra-FineWeb data for scaled training.
Target: 500K documents (5x more than before).
"""
import json
import time
import itertools
from pathlib import Path
from datasets import load_dataset

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

N_SAMPLES = 500_000
MIN_SCORE = 0.5
MIN_CHARS = 200
MAX_CHARS = 5000
BATCH_SAVE = 10_000


def download_more():
    print(f"Downloading {N_SAMPLES:,} documents...")
    ds = load_dataset("openbmb/Ultra-FineWeb", split="en", streaming=True)
    output = DATA_DIR / "ultra_fineweb_en_large.jsonl"
    collected = 0
    rejected = 0
    start = time.time()
    batch = []

    with open(output, "w") as f:
        for doc in ds:
            if collected >= N_SAMPLES:
                break
            content = doc.get("content", "")
            score = doc.get("score", 0.0)
            if score < MIN_SCORE:
                continue
            if len(content) < MIN_CHARS or len(content) > MAX_CHARS:
                rejected += 1
                continue
            batch.append({"content": content, "score": score, "source": doc.get("source", "")})
            collected += 1
            if len(batch) >= BATCH_SAVE:
                for item in batch:
                    f.write(json.dumps(item) + "\n")
                elapsed = time.time() - start
                print(f"  [{collected:,}/{N_SAMPLES:,}] {collected/elapsed:.0f} docs/s rejected={rejected:,}")
                batch = []
    for item in batch:
        f.write(json.dumps(item) + "\n")
    print(f"Done: {collected:,} in {time.time()-start:.1f}s")


if __name__ == "__main__":
    download_more()
