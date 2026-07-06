"""
End-to-end test with real bge-m3 embeddings.

Uses the SplatDBConnector to:
  1. Embed real words/sentences with bge-m3
  2. Train the latent diffusion model on real embeddings
  3. Generate new latent samples
  4. Decode samples back to text via nearest-neighbor (HNSW equivalent)

This validates the full pipeline on real data distribution.
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from latent_diffusion_pipeline import LatentDiffusionModel
from splatdb_connector import SplatDBConnector

RESULTS_DIR = REPO / "results"


# A curated set of short phrases organized in semantic clusters
CORPUS = {
    "animals": [
        "cat", "dog", "bird", "fish", "horse", "cow", "sheep", "chicken",
        "rabbit", "mouse", "bear", "lion", "tiger", "wolf", "deer", "elephant",
        "monkey", "snake", "turtle", "duck",
    ],
    "colors": [
        "red", "blue", "green", "yellow", "white", "black", "orange", "purple",
        "pink", "brown", "gray", "silver", "gold", "cyan", "magenta", "crimson",
        "scarlet", "azure", "emerald", "amber",
    ],
    "food": [
        "bread", "milk", "cheese", "apple", "orange", "banana", "rice", "pasta",
        "meat", "fish", "egg", "sugar", "salt", "honey", "coffee", "tea",
        "wine", "beer", "soup", "salad",
    ],
    "emotion": [
        "happy", "sad", "angry", "calm", "excited", "afraid", "brave", "tired",
        "bored", "amazed", "proud", "ashamed", "grateful", "lonely", "hopeful",
        "worried", "relaxed", "confused", "surprised", "disgusted",
    ],
    "nature": [
        "sun", "moon", "star", "fire", "earth", "sky", "cloud", "rain", "snow",
        "wind", "storm", "mountain", "river", "lake", "ocean", "forest",
        "desert", "valley", "island", "cave",
    ],
}


def build_dataset(connector):
    """Embed all corpus words and create labeled dataset."""
    print("Embedding corpus with bge-m3...")
    all_words = []
    all_embeddings = []
    all_labels = []
    cluster_names = list(CORPUS.keys())

    for cluster_idx, (cluster_name, words) in enumerate(CORPUS.items()):
        for word in words:
            emb = connector.embed_text(word)
            all_embeddings.append(emb.squeeze(0).cpu())
            all_words.append(word)
            all_labels.append(cluster_idx)

    embeddings = torch.stack(all_embeddings)
    print(f"Embedded {len(all_words)} words → {embeddings.shape}")
    print(f"Clusters: {len(cluster_names)} ({', '.join(cluster_names)})")

    # Verify embedding quality
    emb_norm = embeddings.norm(dim=-1).mean()
    print(f"Embedding norm: {emb_norm:.4f} (should be ~1.0 for bge-m3)")

    return embeddings, all_words, all_labels, cluster_names


def decode_samples(samples, connector):
    """Decode latent samples back to text via nearest-neighbor."""
    print(f"\nDecoding {samples.shape[0]} samples...")
    tokens = []
    distances = []
    for i in range(samples.shape[0]):
        tok, dist = connector.decode_latent(samples[i:i+1])
        tokens.append(tok[0])
        distances.append(dist[0].item())
    return tokens, distances


def run_e2e_test():
    print("=" * 70)
    print("END-TO-END: Real bge-m3 embeddings → diffusion → text generation")
    print("=" * 70)

    # Build connector
    connector = SplatDBConnector(vocab_size=5000)

    # Build dataset
    embeddings, words, labels, cluster_names = build_dataset(connector)

    # Split train/test
    n = len(words)
    perm = torch.randperm(n)
    train_idx = perm[:int(0.8 * n)]
    test_idx = perm[int(0.8 * n):]

    train_emb = embeddings[train_idx]
    test_emb = embeddings[test_idx]
    train_words = [words[i] for i in train_idx]
    test_words = [words[i] for i in test_idx]

    print(f"\nTrain: {len(train_words)} words")
    print(f"Test: {len(test_words)} words")

    # Fit model
    print(f"\n--- Fitting LatentDiffusionModel ---")
    model = LatentDiffusionModel(
        variance_threshold=0.80,  # lower k for better score matching
        score_hidden=256, score_blocks=6, score_epochs=3000,
        svgd_iters=600, svgd_particles=200,
    )
    model.fit(train_emb, verbose=True)

    # Sample
    print(f"\n--- Generating samples ---")
    samples = model.sample(verbose=True, real_ref_1024=train_emb[:50])

    # Evaluate in embedding space
    metrics = model.evaluate(samples, train_emb)
    print(f"\n--- Quality metrics ---")
    print(f"  Near data (<0.3): {metrics['near_03']:.2%}")
    print(f"  Near data (<0.1): {metrics['near_01']:.2%}")
    print(f"  Diversity: {metrics['diversity']:.4f}")

    # Decode to text
    tokens, dists = decode_samples(samples, connector)

    print(f"\n--- Generated samples (decoded to nearest word) ---")
    for i in range(0, min(30, len(tokens))):
        print(f"  sample {i:3d}: '{tokens[i]}'  (dist={dists[i]:.4f})")

    # Semantic coherence: do decoded words belong to coherent clusters?
    from collections import Counter
    token_counts = Counter(tokens)
    print(f"\n--- Token distribution ---")
    print(f"  Unique tokens: {len(token_counts)} / {len(tokens)} samples")
    for word, count in token_counts.most_common(10):
        cluster = [k for k, v in CORPUS.items() if word in v]
        cl_name = cluster[0] if cluster else "?"
        print(f"  '{word}' ({cl_name}): {count}x")

    result = {
        "experiment": "e2e_bge_m3",
        "timestamp": datetime.now().isoformat(),
        "n_train": len(train_words),
        "n_test": len(test_words),
        "k": model.k,
        "metrics": metrics,
        "sample_tokens": tokens[:50],
        "unique_tokens": len(token_counts),
        "top_tokens": dict(token_counts.most_common(20)),
    }

    out = RESULTS_DIR / "e2e_bge_m3_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run_e2e_test()
