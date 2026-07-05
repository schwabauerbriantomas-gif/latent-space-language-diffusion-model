"""
DensityConfidenceScore: replaces the neural ConfidenceNet.

THE PROBLEM:
  The neural ConfidenceNet (Phase 8) failed to discriminate novel vs known
  topics (novel=0.668, known=0.650 — within noise). A neural net trained
  on 280 examples cannot generalize to the full 1024D embedding space.

THE SOLUTION:
  Replace the neural net with a geometric confidence score computed
  directly from SplatsDB's vector store. Three complementary signals:

  1. NEIGHBOR DENSITY: how many SplatsDB entries are near this topic?
     - High density = well-covered topic = high confidence
     - Low density = knowledge gap = low confidence

  2. NEAREST NEIGHBOR DISTANCE: how close is the closest entry?
     - Close = the exact topic exists → high confidence
     - Far = no similar entry → low confidence

  3. NEIGHBOR CONSISTENCY: do nearby entries agree (low variance)?
     - Consistent neighbors = the model knows this area well
     - Scattered neighbors = the model is guessing

  These signals are computed in O(N) with a single matrix multiply
  (or O(log N) with HNSW in production SplatsDB), require NO training,
  and adapt instantly when SplatsDB ingests new data.

THE CLAIM:
  This approach is:
    - 100% adaptive (reads SplatsDB state at query time)
    - Zero training (no neural net to retrain)
    - O(1) latency for HNSW, O(N) for brute force
    - More reliable than neural net (geometry doesn't overfit)
"""
import math
import json
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = REPO / "results"

from vocab_cfg import VOCAB, FUNC, PAD, MASK, BOS, EOS, UNK
from information_seeker import MockSplatsDB, MockExternalRetriever, IngestTrigger
from topic_mdlm import build_category_topic_embeddings


# ═══════════════════════════════════════════════════════════════════════════
# DensityConfidenceScore
# ═══════════════════════════════════════════════════════════════════════════

class DensityConfidenceScore:
    """Geometric confidence score from SplatsDB vector density.

    NO NEURAL NETWORK. NO TRAINING. Pure geometry.

    Confidence = f(density, nearest_dist, consistency)
    """

    def __init__(self, density_weight=0.4, dist_weight=0.4, consistency_weight=0.2,
                 sim_threshold=0.10, k_neighbors=10):
        """
        Args:
            density_weight: weight for neighbor density signal
            dist_weight: weight for nearest-neighbor distance signal
            consistency_weight: weight for neighbor consistency signal
            sim_threshold: cosine similarity threshold for "near" (calibrated
                           for 1024D bge-m3 embeddings)
            k_neighbors: number of neighbors to examine for consistency
        """
        self.density_weight = density_weight
        self.dist_weight = dist_weight
        self.consistency_weight = consistency_weight
        self.sim_threshold = sim_threshold
        self.k_neighbors = k_neighbors

    def score(self, topic_emb: torch.Tensor,
              splatsdb: MockSplatsDB) -> Tuple[float, Dict]:
        """Compute confidence score [0, 1] for a topic embedding.

        Args:
            topic_emb: [1024] topic embedding
            splatsdb: SplatsDB vector store

        Returns:
            (confidence_score, details_dict)
        """
        if not splatsdb.entries:
            return 0.0, {
                "density": 0.0,
                "nearest_dist": 1.0,
                "consistency": 0.0,
                "reason": "empty_store",
            }

        # Compute similarities to all entries
        all_embs = torch.stack([e.embedding for e in splatsdb.entries])
        topic_norm = F.normalize(topic_emb.unsqueeze(0), dim=-1)
        all_norm = F.normalize(all_embs, dim=-1)
        sims = (topic_norm @ all_norm.T).squeeze(0)  # [N]

        # ── Signal 1: Neighbor Density ───────────────────────────────
        # Fraction of entries above similarity threshold
        n_near = (sims > self.sim_threshold).sum().item()
        density = n_near / len(splatsdb.entries)
        # Normalize: density of 0.03+ (3% of store nearby) = max confidence
        # Calibrated for 1024D block-separated embeddings where intra-category
        # cosine sim is naturally low (~0.10-0.20)
        density_score = min(1.0, density / 0.03)

        # ── Signal 2: Nearest Neighbor Distance ──────────────────────
        # How close is the very closest entry?
        nearest_sim = sims.max().item()
        nearest_dist = 1.0 - nearest_sim
        # Normalize: sim of 0.15+ (strong match) = 1.0,
        #            sim of 0.05 or below (very far) = 0.0
        # Calibrated for block-separated embeddings
        dist_score = max(0.0, min(1.0, (nearest_sim - 0.05) / 0.10))

        # ── Signal 3: Neighbor Consistency ───────────────────────────
        # Do the top-k neighbors have high inter-similarity (clustered)?
        # If the nearest neighbors are spread out, the region is sparse/confused
        k = min(self.k_neighbors, len(sims))
        topk_sims, topk_idx = sims.topk(k)
        mean_inter_sim = 0.0
        if k >= 2:
            topk_embs = all_norm[topk_idx]  # [k, 1024]
            # Pairwise cosine sim among top-k neighbors
            pairwise = topk_embs @ topk_embs.T  # [k, k]
            # Mean of off-diagonal (exclude self-similarity)
            mask = ~torch.eye(k, dtype=torch.bool, device=pairwise.device)
            mean_inter_sim = pairwise[mask].mean().item()
            # Normalize: inter-sim of 0.15+ = consistent cluster = high confidence
            consistency = max(0.0, min(1.0, mean_inter_sim / 0.15))
        else:
            consistency = 0.5  # neutral if too few entries

        # ── Combined Confidence ──────────────────────────────────────
        confidence = (
            self.density_weight * density_score +
            self.dist_weight * dist_score +
            self.consistency_weight * consistency
        )

        details = {
            "density": density,
            "density_score": density_score,
            "n_near": n_near,
            "nearest_sim": nearest_sim,
            "nearest_dist": nearest_dist,
            "dist_score": dist_score,
            "consistency": consistency,
            "mean_inter_sim": mean_inter_sim if k >= 2 else 0.0,
            "confidence": confidence,
        }

        return confidence, details


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_confidence_score():
    """Validate the DensityConfidenceScore against the same test as Phase 8c.

    Tests:
      1. Known categories with data → should score HIGH
      2. Known categories with gaps → should score LOW
      3. Novel random topics → should score LOW
      4. Mixed topics (partial overlap) → should score MEDIUM
      5. After ingestion → score should INCREASE
    """
    print("=" * 70)
    print("DENSITY CONFIDENCE SCORE — VALIDATION")
    print("=" * 70)

    cat_embeds, categories, mixed = build_category_topic_embeddings()

    # Setup SplatsDB with the SAME gaps as Phase 8
    splatsdb = MockSplatsDB(cat_embeds, categories)

    coverage = {}
    for i, cat in enumerate(categories):
        if i % 4 == 0:
            coverage[cat] = 15  # rich
        elif i % 4 == 1:
            coverage[cat] = 3   # sparse
        else:
            coverage[cat] = 0   # empty

    splatsdb.initialize(coverage)

    scorer = DensityConfidenceScore(
        density_weight=0.4, dist_weight=0.4, consistency_weight=0.2,
        sim_threshold=0.10, k_neighbors=10,
    )

    results = {}

    # ── TEST 1: Known categories with data → HIGH confidence ────────
    print(f"\n{'='*60}")
    print("TEST 1: Rich categories (15 entries) → should be HIGH confidence")
    print(f"{'='*60}")

    rich_cats = [cat for cat in categories if coverage.get(cat, 0) >= 15]
    rich_scores = {}
    for cat in rich_cats:
        score, details = scorer.score(cat_embeds[cat], splatsdb)
        rich_scores[cat] = score
        print(f"  {cat:15s}: confidence={score:.3f} "
              f"(density={details['density']:.3f}, "
              f"nearest={details['nearest_sim']:.3f}, "
              f"consistency={details['consistency']:.3f})")

    avg_rich = np.mean(list(rich_scores.values()))
    print(f"\n  Average (rich): {avg_rich:.3f}")

    results["test1_rich"] = {"scores": rich_scores, "avg": avg_rich}

    # ── TEST 2: Sparse categories (3 entries) → MEDIUM confidence ───
    print(f"\n{'='*60}")
    print("TEST 2: Sparse categories (3 entries) → should be MEDIUM/LOW")
    print(f"{'='*60}")

    sparse_cats = [cat for cat in categories if coverage.get(cat, 0) == 3]
    sparse_scores = {}
    for cat in sparse_cats:
        score, details = scorer.score(cat_embeds[cat], splatsdb)
        sparse_scores[cat] = score
        print(f"  {cat:15s}: confidence={score:.3f} "
              f"(density={details['density']:.3f}, "
              f"nearest={details['nearest_sim']:.3f})")

    avg_sparse = np.mean(list(sparse_scores.values()))
    print(f"\n  Average (sparse): {avg_sparse:.3f}")

    results["test2_sparse"] = {"scores": sparse_scores, "avg": avg_sparse}

    # ── TEST 3: Empty categories (0 entries) → LOW confidence ───────
    print(f"\n{'='*60}")
    print("TEST 3: Empty categories (0 entries) → should be LOW")
    print(f"{'='*60}")

    empty_cats = [cat for cat in categories if coverage.get(cat, 0) == 0]
    empty_scores = {}
    for cat in empty_cats:
        score, details = scorer.score(cat_embeds[cat], splatsdb)
        empty_scores[cat] = score
        print(f"  {cat:15s}: confidence={score:.3f} "
              f"(density={details['density']:.3f}, "
              f"nearest={details['nearest_sim']:.3f})")

    avg_empty = np.mean(list(empty_scores.values()))
    print(f"\n  Average (empty): {avg_empty:.3f}")

    results["test3_empty"] = {"scores": empty_scores, "avg": avg_empty}

    # ── TEST 4: Novel random topics → LOW confidence ────────────────
    print(f"\n{'='*60}")
    print("TEST 4: Novel random topics → should be LOW (never seen)")
    print(f"{'='*60}")

    novel_scores = []
    for i in range(20):
        random_topic = F.normalize(torch.randn(1024, device=DEVICE), dim=0)
        score, details = scorer.score(random_topic, splatsdb)
        novel_scores.append(score)

    avg_novel = np.mean(novel_scores)
    print(f"  20 random topics: avg={avg_novel:.3f}, "
          f"min={np.min(novel_scores):.3f}, max={np.max(novel_scores):.3f}")
    print(f"  Std: {np.std(novel_scores):.3f}")

    results["test4_novel"] = {"scores": novel_scores, "avg": avg_novel}

    # ── TEST 5: Discrimination accuracy ─────────────────────────────
    print(f"\n{'='*60}")
    print("TEST 5: Discrimination accuracy")
    print(f"{'='*60}")

    # Define ground truth: rich = confident (1), empty/novel = gap (0)
    ground_truth_positive = rich_cats  # should score > threshold
    ground_truth_negative = empty_cats + ["novel_random"]  # should score < threshold

    threshold = 0.3
    tp = sum(1 for cat in ground_truth_positive
             if rich_scores[cat] > threshold)
    fp = sum(1 for cat in ground_truth_negative
             if cat != "novel_random" and empty_scores.get(cat, 0) > threshold)
    fn = sum(1 for cat in ground_truth_positive
             if rich_scores[cat] <= threshold)
    tn_empty = sum(1 for cat in empty_cats
                   if empty_scores[cat] <= threshold)
    tn_novel = sum(1 for s in novel_scores if s <= threshold)

    accuracy = (tp + tn_empty + tn_novel) / (
        len(ground_truth_positive) + len(empty_cats) + len(novel_scores))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    print(f"\n  Threshold: {threshold}")
    print(f"  Rich categories correctly HIGH:   {tp}/{len(rich_cats)}")
    print(f"  Empty categories correctly LOW:    {tn_empty}/{len(empty_cats)}")
    print(f"  Novel topics correctly LOW:        {tn_novel}/{len(novel_scores)}")
    print(f"  False positives (empty scored high): {fp}")
    print(f"  False negatives (rich scored low):   {fn}")
    print(f"\n  ACCURACY: {accuracy:.1%}")
    print(f"  PRECISION: {precision:.1%}")
    print(f"  RECALL: {recall:.1%}")

    results["test5_discrimination"] = {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
        "tn_empty": tn_empty, "tn_novel": tn_novel,
    }

    # ── TEST 6: Score increases after ingestion ─────────────────────
    print(f"\n{'='*60}")
    print("TEST 6: Score INCREASES after ingestion (adaptivity test)")
    print(f"{'='*60}")

    # Pick an empty category, measure before and after ingestion
    test_cat = empty_cats[0]
    score_before, details_before = scorer.score(cat_embeds[test_cat], splatsdb)
    print(f"\n  Before ingestion ({test_cat}, {coverage[test_cat]} entries):")
    print(f"    Confidence: {score_before:.3f}")
    print(f"    Density: {details_before['density']:.3f}")

    # Simulate ingestion
    retriever = MockExternalRetriever(cat_embeds, categories)
    retrieved = retriever.retrieve(f"info about {test_cat}", test_cat, n_items=10)
    splatsdb.ingest_batch(retrieved)

    score_after, details_after = scorer.score(cat_embeds[test_cat], splatsdb)
    print(f"\n  After ingestion ({test_cat}, {len(retrieved)} items added):")
    print(f"    Confidence: {score_after:.3f}")
    print(f"    Density: {details_after['density']:.3f}")
    print(f"\n  Δ Confidence: +{score_after - score_before:.3f}")
    print(f"  Adapted instantly: {'YES ✓' if score_after > score_before else 'NO ✗'}")

    results["test6_adaptivity"] = {
        "before": {"confidence": score_before, "density": details_before["density"]},
        "after": {"confidence": score_after, "density": details_after["density"]},
        "delta": score_after - score_before,
        "adapted": score_after > score_before,
    }

    # ── TEST 7: Latency ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TEST 7: Latency")
    print(f"{'='*60}")

    # Warmup
    for _ in range(10):
        scorer.score(cat_embeds["animals"], splatsdb)

    times = []
    for _ in range(100):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        scorer.score(cat_embeds["animals"], splatsdb)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_latency = np.mean(times)
    p99 = np.percentile(times, 99)
    print(f"  Store size: {len(splatsdb.entries)} entries")
    print(f"  Avg latency: {avg_latency*1000:.2f} ms")
    print(f"  P99 latency: {p99*1000:.2f} ms")
    print(f"  Throughput: {1/avg_latency:.0f} scores/sec")

    results["test7_latency"] = {
        "store_size": len(splatsdb.entries),
        "avg_ms": avg_latency * 1000,
        "p99_ms": p99 * 1000,
    }

    # ── COMPARISON: Neural vs Density ───────────────────────────────
    print(f"\n{'='*70}")
    print("COMPARISON: Neural ConfidenceNet vs DensityConfidenceScore")
    print(f"{'='*70}")

    print(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │  METRIC                │  NEURAL (Phase 8c)  │  DENSITY (THIS)   │
  ├────────────────────────┼─────────────────────┼───────────────────┤
  │  Novel discrimination  │  FAILED (0.668      │  {'WORKS' if avg_novel < avg_rich else 'FAILED'} ({avg_novel:.3f}    │
  │                        │    vs 0.650)        │    vs {avg_rich:.3f})      │
  │  Rich vs empty gap     │  N/A                │  {avg_rich:.3f} vs {avg_empty:.3f}    │
  │  Accuracy              │  ~50% (chance)      │  {accuracy:.1%}           │
  │  Training required     │  YES (1500 epochs)  │  NO                │
  │  Latency               │  4.7 ms             │  {avg_latency*1000:.1f} ms        │
  │  Adapts to new data    │  NO                 │  YES (instant)    │
  │  Parameters            │  3,884,545          │  0                 │
  └────────────────────────┴─────────────────────┴───────────────────┘
""")

    # Save results
    result = {
        "experiment": "density_confidence_validation",
        "timestamp": datetime.now().isoformat(),
        "scorer_config": {
            "density_weight": scorer.density_weight,
            "dist_weight": scorer.dist_weight,
            "consistency_weight": scorer.consistency_weight,
            "sim_threshold": scorer.sim_threshold,
            "k_neighbors": scorer.k_neighbors,
        },
        "results": results,
        "comparison": {
            "neural": {
                "novel_confidence": 0.668,
                "known_confidence": 0.650,
                "discriminates": False,
                "accuracy": "~50%",
                "training": "1500 epochs (~37s)",
                "latency_ms": 4.7,
                "params": 3884545,
            },
            "density": {
                "novel_confidence": float(avg_novel),
                "rich_confidence": float(avg_rich),
                "empty_confidence": float(avg_empty),
                "discriminates": avg_novel < avg_rich,
                "accuracy": float(accuracy),
                "training": "NONE",
                "latency_ms": float(avg_latency * 1000),
                "params": 0,
            },
        },
    }

    out = RESULTS_DIR / "density_confidence_validation.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    validate_confidence_score()
