"""
Prepare tokenized training data from Ultra-FineWeb.

Reads raw JSONL, tokenizes with BPE, packs into fixed-length sequences,
saves as binary int16 arrays for fast loading.
"""
import json
import time
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
TOKENIZER_PATH = REPO / "tokenizer" / "bpe_tokenizer.json"

from tokenizers import Tokenizer

SEQ_LEN = 64              # Fixed sequence length for training
PACK_MULTIPLE = 4         # Pack 4 documents per sequence (with separators)


def prepare_training_data():
    """Tokenize and pack documents into training sequences."""
    print("=" * 70)
    print("PREPARING TRAINING DATA")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    sep_id = tokenizer.token_to_id("<eos>")  # Use EOS as separator

    input_file = DATA_DIR / "ultra_fineweb_en.jsonl"
    output_file = DATA_DIR / "train_tokens.npy"

    # Read all texts
    print("Reading documents...")
    texts = []
    with open(input_file) as f:
        for line in f:
            doc = json.loads(line)
            texts.append(doc["content"])

    print(f"  {len(texts):,} documents")

    # Tokenize all texts
    print("Tokenizing (batch)...")
    start = time.time()

    # Batch tokenize for speed
    BATCH_SIZE = 5000
    all_token_ids = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        encoded = tokenizer.encode_batch(batch)
        for enc in encoded:
            all_token_ids.append(enc.ids)
        if (i // BATCH_SIZE) % 5 == 0:
            elapsed = time.time() - start
            print(f"  [{i+len(batch):,}/{len(texts):,}] {len(all_token_ids):,} docs tokenized")

    elapsed = time.time() - start
    total_tokens = sum(len(ids) for ids in all_token_ids)
    print(f"  ✓ Tokenized in {elapsed:.1f}s")
    print(f"  Total tokens: {total_tokens:,}")

    # Token length distribution
    lens = [len(ids) for ids in all_token_ids]
    lens_arr = np.array(lens)
    print(f"\n  Token length stats:")
    print(f"    Mean: {lens_arr.mean():.1f}")
    print(f"    Median: {np.median(lens_arr):.0f}")
    print(f"    Min: {lens_arr.min()}, Max: {lens_arr.max()}")
    print(f"    <{SEQ_LEN}: {(lens_arr < SEQ_LEN).sum():,} ({100*(lens_arr < SEQ_LEN).mean():.1f}%)")
    print(f"    <{SEQ_LEN*2}: {(lens_arr < SEQ_LEN*2).sum():,} ({100*(lens_arr < SEQ_LEN*2).mean():.1f}%)")

    # Strategy: pack documents into SEQ_LEN chunks
    # Concatenate all tokens with EOS separators, then chunk
    print(f"\nPacking into {SEQ_LEN}-token sequences...")

    packed_tokens = []
    current_seq = []
    n_sequences = 0

    for ids in all_token_ids:
        if len(ids) > SEQ_LEN * 4:
            # Too long, truncate to reasonable size
            ids = ids[:SEQ_LEN * 2]

        # Add BOS + tokens + EOS
        doc_tokens = [bos_id] + ids + [eos_id]

        for tok in doc_tokens:
            current_seq.append(tok)
            if len(current_seq) >= SEQ_LEN:
                packed_tokens.append(current_seq[:SEQ_LEN])
                current_seq = []
                n_sequences += 1

    # Handle remainder
    if current_seq:
        # Pad to SEQ_LEN with pad tokens
        pad_id = tokenizer.token_to_id("<pad>")
        while len(current_seq) < SEQ_LEN:
            current_seq.append(pad_id)
        packed_tokens.append(current_seq)
        n_sequences += 1

    print(f"  ✓ Packed {n_sequences:,} sequences of {SEQ_LEN} tokens each")
    print(f"  Total training tokens: {n_sequences * SEQ_LEN:,}")

    # Save as int16 numpy array
    arr = np.array(packed_tokens, dtype=np.int16)
    np.save(output_file, arr)
    print(f"  Saved to: {output_file}")
    print(f"  Size: {arr.nbytes / 1e6:.1f} MB")

    return arr


if __name__ == "__main__":
    arr = prepare_training_data()
    print(f"\nSample sequence (first):")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    sample = arr[0].tolist()
    decoded = tok.decode(sample)
    print(f"  Tokens: {sample[:20]}...")
    print(f"  Text: {decoded[:200]}...")
