"""
Train a BPE tokenizer on Ultra-FineWeb data.

Output: 10K vocab BPE tokenizer (HuggingFace tokenizers format)
"""
import json
import time
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.normalizers import NFKC, Sequence as NormalizerSequence

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
TOKENIZER_DIR = REPO / "tokenizer"
TOKENIZER_DIR.mkdir(exist_ok=True)

# ── Config ──
VOCAB_SIZE = 10_000      # Target BPE vocab
SPECIAL_TOKENS = ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>"]
# Reserve range for user-defined tokens (code, tool-call markers, etc.)
ADDITIONAL_TOKENS = [
    "<|user|>", "<|assistant|>", "<|system|>",
    "<|think|>", "<|/think|>",
    "<|tool_call|>", "<|/tool_call|>",
    "<|code|>", "<|/code|>",
    "<newline>", "<tab>",
]

ALL_SPECIAL = SPECIAL_TOKENS + ADDITIONAL_TOKENS


def train_bpe_tokenizer():
    """Train BPE on Ultra-FineWeb data."""
    data_file = DATA_DIR / "ultra_fineweb_en.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            "Run scripts/download_data.py first."
        )

    print("=" * 70)
    print("TRAINING BPE TOKENIZER")
    print("=" * 70)
    print(f"  Data: {data_file}")
    print(f"  Target vocab: {VOCAB_SIZE:,}")
    print(f"  Special tokens: {len(ALL_SPECIAL)}")
    print()

    # Extract text to temp files for training
    texts = []
    total_chars = 0
    with open(data_file) as f:
        for line in f:
            doc = json.loads(line)
            texts.append(doc["content"])
            total_chars += len(doc["content"])

    print(f"  Documents: {len(texts):,}")
    print(f"  Total chars: {total_chars:,}")
    print()

    # Configure BPE
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NormalizerSequence([NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=True
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=ALL_SPECIAL,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print("Training BPE...")
    start = time.time()
    tokenizer.train_from_iterator(texts, trainer=trainer)
    elapsed = time.time() - start

    actual_vocab = tokenizer.get_vocab_size()
    print(f"\n✓ Trained in {elapsed:.1f}s")
    print(f"  Actual vocab size: {actual_vocab:,}")

    # Test
    test_texts = [
        "The cat sat on the mat.",
        "def hello_world():\n    print('Hello, World!')",
        "function add(a, b) { return a + b; }",
    ]
    print("\n  Tokenization tests:")
    for text in test_texts:
        enc = tokenizer.encode(text)
        dec = tokenizer.decode(enc.ids)
        print(f"    '{text[:50]}...'")
        print(f"      → {len(enc.tokens)} tokens: {enc.tokens[:10]}...")
        print(f"      ← decode: '{dec[:60]}...'")

    # Save
    tokenizer_path = TOKENIZER_DIR / "bpe_tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"\n  Saved to: {tokenizer_path}")

    # Verify special token IDs
    print("\n  Special token IDs:")
    for tok in ALL_SPECIAL:
        tid = tokenizer.token_to_id(tok)
        print(f"    {tok:20s} → {tid}")

    return tokenizer, actual_vocab


if __name__ == "__main__":
    train_bpe_tokenizer()
