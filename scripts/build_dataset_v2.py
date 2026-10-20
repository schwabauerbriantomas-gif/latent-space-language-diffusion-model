"""build_dataset_v2 — windows STRICTLY inside documents (anti-drift fix, swarm G).

Old prepare_data: cross-doc packing (current stream persists across docs) ->
windows mix end-of-doc-A + BOS-of-doc-B, teaching 'any topic jump is plausible'.

v2: window 1 = [BOS] + ids[:seq_len-1]; subsequent windows = ids[k:k+seq_len].
Tail < seq_len discarded. Output: data/train_tokens_v5_{seq_len}.npy (int16).

Usage: python scripts/build_dataset_v2.py [--seq-len 128]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
TOK_PATH = REPO / "tokenizer" / "bpe_tokenizer.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()
    L = args.seq_len

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOK_PATH))
    bos_id = tok.token_to_id("<bos>")
    eos_id = tok.token_to_id("<eos>")

    input_file = DATA_DIR / "ultra_fineweb_en_1m.jsonl"
    if not input_file.exists():
        input_file = DATA_DIR / "ultra_fineweb_en.jsonl"
    output_file = DATA_DIR / f"train_tokens_v5_{L}.npy"
    tmp_file = DATA_DIR / "train_tokens_v5_tmp.npy"

    total_docs = sum(1 for _ in open(input_file, encoding="utf-8", errors="ignore"))
    print(f"{total_docs:,} docs -> intra-doc windows of {L}")

    max_seqs = total_docs * 4
    arr = np.memmap(str(tmp_file), dtype=np.int16, mode="w+", shape=(max_seqs, L))
    n_seqs = 0
    total_tok_kept = 0
    total_tok_seen = 0
    t0 = time.time()
    CHUNK = 2000
    chunk, processed = [], 0

    def flush(texts):
        nonlocal n_seqs, total_tok_kept, total_tok_seen
        for enc in tok.encode_batch(texts):
            ids = enc.ids
            total_tok_seen += len(ids)
            if len(ids) < 8:  # junk
                continue
            # window 1: BOS + first L-1 tokens, EOS-padded if short
            arr[n_seqs, 0] = bos_id
            take = min(L - 1, len(ids))
            arr[n_seqs, 1:1 + take] = ids[:take]
            if take < L - 1:
                arr[n_seqs, 1 + take:] = eos_id
            n_seqs += 1
            total_tok_kept += 1 + take
            # full windows from the rest
            k = L - 1
            while k + L <= len(ids):
                arr[n_seqs] = ids[k:k + L]
                n_seqs += 1
                total_tok_kept += L
                k += L
            # remaining tail < L: skip (anti-cross-doc guarantee)

    with open(input_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                chunk.append(json.loads(line)["content"])
            except Exception:
                continue
            if len(chunk) >= CHUNK:
                flush(chunk); chunk = []
                processed += CHUNK
                if processed % 50000 == 0:
                    print(f"  [{processed:,}/{total_docs:,}] seqs={n_seqs:,} ({time.time()-t0:.0f}s)", flush=True)
        if chunk:
            flush(chunk)

    out = np.array(arr[:n_seqs])
    np.save(str(output_file).replace(".npy", "") + ".npy", out)
    tmp = Path(tmp_file)
    if tmp.exists(): tmp.unlink()
    print(f"DONE: {n_seqs:,} seqs ({n_seqs*L:,} tokens kept / {total_tok_seen:,} seen, "
          f"{100*total_tok_kept/max(total_tok_seen,1):.0f}% kept) -> {output_file}")
    print(f"file: {output_file.stat().st_size/1e6:.0f} MB, elapsed {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
