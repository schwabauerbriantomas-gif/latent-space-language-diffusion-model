"""
TPS Benchmark & Quality Evaluation.

Measures:
  1. TOKENS PER SECOND at different batch sizes and step counts
  2. QUALITY metrics: grammaticality, diversity, novelty, on-topic, reviewer score
  3. OPTIMIZATION: find the Pareto frontier of quality vs speed

MDLM generates tokens via PARALLEL iterative unmasking (unlike autoregressive
which generates one token at a time). So TPS = (batch_size × seq_len) / latency.
At each diffusion step, ALL masked positions are predicted simultaneously.

This means MDLM can achieve much higher TPS than autoregressive models
of similar parameter count, at the cost of iterative refinement.
"""
import math
import json
import sys
import time
import random
import string
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = REPO / "results"

from vocab_cfg import (
    build_vocab, CFGGenerator, VOCAB, FUNC,
    PAD, MASK, BOS, EOS, UNK,
)
from topic_mdlm import (
    TopicMDLMTransformer, TopicReviewer, sample_topic_mdLM,
    build_category_topic_embeddings, editor_refine_topic,
)
from mdlm import decode_tokens, encode_sequences
from density_confidence import DensityConfidenceScore
from information_seeker import MockSplatsDB


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════════════════
# Quality Metrics
# ═══════════════════════════════════════════════════════════════════════════

def n_gram_diversity(sequences: List[List[str]], n: int = 2) -> float:
    """Fraction of unique n-grams across all sequences."""
    all_ngrams = []
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            all_ngrams.append(tuple(seq[i:i+n]))
    if not all_ngrams:
        return 0.0
    unique = len(set(all_ngrams))
    return unique / len(all_ngrams)


def self_bleu(sequences: List[str], max_n: int = 4) -> float:
    """Self-BLEU: lower = more diverse. Measures how much generated
    sequences resemble each other (1.0 = identical, 0.0 = totally different)."""
    if len(sequences) < 2:
        return 0.0

    # Tokenize
    tokenized = [seq.split() for seq in sequences]
    scores = []

    for i in range(len(tokenized)):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        hyp = tokenized[i]

        # Simple BLEU-n approximation
        precisions = []
        for n in range(1, max_n + 1):
            hyp_ngrams = Counter(tuple(hyp[k:k+n]) for k in range(len(hyp)-n+1))
            ref_ngram_count = Counter()
            for ref in refs:
                for k in range(len(ref)-n+1):
                    ref_ngram_count[tuple(ref[k:k+n])] += 1

            matches = sum(min(hyp_ngrams[ng], ref_ngram_count[ng])
                         for ng in hyp_ngrams)
            total = max(1, sum(hyp_ngrams.values()))
            precisions.append(matches / total)

        # Geometric mean
        if all(p > 0 for p in precisions):
            bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            bleu = 0.0
        scores.append(bleu)

    return np.mean(scores)


def distinct_n(sequences: List[str], n: int = 1) -> float:
    """Distinct-n: unique n-grams / total n-grams. Higher = more diverse."""
    all_ngrams = []
    for seq in sequences:
        words = seq.split()
        for i in range(len(words) - n + 1):
            all_ngrams.append(tuple(words[i:i+n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def evaluate_quality(decoded_seqs: List[str], tok2id, id2tok,
                     reviewer, topic_emb, target_cat: str = None,
                     train_seqs: set = None) -> Dict:
    """Comprehensive quality evaluation of generated sequences."""
    from vocab_cfg import check_grammar, tag_sequence

    n = len(decoded_seqs)

    # Tokenize (remove special tokens)
    tokenized = []
    for seq in decoded_seqs:
        words = [w for w in seq.split() if w != "[M]"]
        tokenized.append(words)

    # 1. Grammaticality
    grammatical = 0
    for words in tokenized:
        ok, _ = check_grammar(words)
        if ok:
            grammatical += 1
    grammar_rate = grammatical / n

    # 2. Unique sequences
    unique_strs = set(" ".join(w) for w in tokenized)
    uniqueness = len(unique_strs) / n

    # 3. Novelty (not in training)
    if train_seqs:
        novel = 0
        for words in tokenized:
            if " ".join(words) not in train_seqs:
                novel += 1
        novelty = novel / n
    else:
        novelty = 1.0

    # 4. Diversity metrics
    distinct_1 = distinct_n([" ".join(w) for w in tokenized], n=1)
    distinct_2 = distinct_n([" ".join(w) for w in tokenized], n=2)
    ngram_div = n_gram_diversity(tokenized, n=2)
    sbleu = self_bleu([" ".join(w) for w in tokenized])

    # 5. Reviewer scores
    # Encode for reviewer
    encoded = encode_sequences(tokenized, tok2id, 20).to(DEVICE)
    topic_batch = topic_emb.unsqueeze(0).expand(n, -1).to(DEVICE)
    with torch.no_grad():
        scores = torch.sigmoid(reviewer(encoded, topic_batch))
    avg_score = scores.mean().item()
    score_std = scores.std().item()

    # 6. On-topic rate (if target category specified)
    on_topic_rate = None
    if target_cat and target_cat in VOCAB:
        target_words = set(VOCAB[target_cat])
        on_count = 0
        total_content = 0
        all_content_words = set()
        for cat_words in VOCAB.values():
            all_content_words.update(cat_words)
        for words in tokenized:
            for w in words:
                if w in all_content_words:
                    total_content += 1
                    if w in target_words:
                        on_count += 1
        on_topic_rate = on_count / max(1, total_content)

    # 7. Average sequence length
    avg_len = np.mean([len(w) for w in tokenized])

    return {
        "n": n,
        "grammar_rate": grammar_rate,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "ngram_diversity": ngram_div,
        "self_bleu": sbleu,
        "reviewer_score": avg_score,
        "reviewer_score_std": score_std,
        "on_topic_rate": on_topic_rate,
        "avg_seq_len": avg_len,
        "pass_rate": (scores > 0.5).float().mean().item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# TPS Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_tps(generator, tok2id, id2tok, reviewer, cat_embeds, categories):
    """Measure tokens per second at various configurations."""
    print(f"\n{'='*70}")
    print("TOKENS PER SECOND BENCHMARK")
    print(f"{'='*70}")

    topic = cat_embeds["animals"]
    seq_len = 20  # BOS + 18 content + EOS

    results = []

    # Test configurations: (batch_size, n_steps)
    configs = [
        (1, 10), (1, 20), (1, 40),
        (10, 10), (10, 20), (10, 40),
        (50, 10), (50, 20),
        (100, 10), (100, 20),
        (200, 10),
        (500, 10),
    ]

    print(f"\n  {'BATCH':>6s} {'STEPS':>6s} {'LATENCY':>10s} {'TOKENS':>8s} "
          f"{'TPS':>10s} {'TPS/BATCH':>10s}")
    print(f"  {'─'*6} {'─'*6} {'─'*10} {'─'*8} {'─'*10} {'─'*10}")

    for batch_size, n_steps in configs:
        # Warmup
        for _ in range(2):
            _ = sample_topic_mdLM(generator, seq_len, topic, tok2id,
                                   n_samples=min(batch_size, 10),
                                   n_steps=n_steps, temperature=0.7)

        # Measure
        n_trials = 3
        latencies = []
        for _ in range(n_trials):
            sync()
            t0 = time.perf_counter()
            _ = sample_topic_mdLM(generator, seq_len, topic, tok2id,
                                   n_samples=batch_size,
                                   n_steps=n_steps, temperature=0.7)
            sync()
            latencies.append(time.perf_counter() - t0)

        avg_latency = np.mean(latencies)
        total_tokens = batch_size * seq_len
        tps = total_tokens / avg_latency

        config_result = {
            "batch_size": batch_size,
            "n_steps": n_steps,
            "latency_ms": avg_latency * 1000,
            "total_tokens": total_tokens,
            "tps": tps,
        }
        results.append(config_result)

        print(f"  {batch_size:>6d} {n_steps:>6d} {avg_latency*1000:>8.1f}ms "
              f"{total_tokens:>8d} {tps:>8.1f} {tps:>10.1f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Quality Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_generation_quality(generator, reviewer, editor, tok2id, id2tok,
                                 cat_embeds, categories):
    """Evaluate quality across all categories with standard metrics."""
    print(f"\n{'='*70}")
    print("QUALITY EVALUATION")
    print(f"{'='*70}")

    # Build training set for novelty check
    cfg_gen = CFGGenerator(seed=42)
    train_seqs = cfg_gen.generate_dataset(n=2000, seed=42)
    train_set = set(" ".join(s) for s in train_seqs)

    seq_len = 20
    n_samples = 50

    all_results = {}
    all_decoded = []

    for cat in categories:
        topic = cat_embeds[cat]
        samples = sample_topic_mdLM(generator, seq_len, topic, tok2id,
                                     n_samples=n_samples, n_steps=30,
                                     temperature=0.7)
        decoded = decode_tokens(samples, id2tok)

        quality = evaluate_quality(decoded, tok2id, id2tok, reviewer,
                                    topic, target_cat=cat, train_seqs=train_set)
        all_results[cat] = quality
        all_decoded.extend(decoded)

    # Aggregate
    avg_grammar = np.mean([r["grammar_rate"] for r in all_results.values()])
    avg_unique = np.mean([r["uniqueness"] for r in all_results.values()])
    avg_novel = np.mean([r["novelty"] for r in all_results.values()])
    avg_dist1 = np.mean([r["distinct_1"] for r in all_results.values()])
    avg_dist2 = np.mean([r["distinct_2"] for r in all_results.values()])
    avg_sbleu = np.mean([r["self_bleu"] for r in all_results.values()])
    avg_score = np.mean([r["reviewer_score"] for r in all_results.values()])
    avg_pass = np.mean([r["pass_rate"] for r in all_results.values()])
    avg_ontopic = np.mean([r["on_topic_rate"] for r in all_results.values()])

    print(f"\n  Per-category results:")
    print(f"  {'CATEGORY':>15s} {'GRAMMAR':>8s} {'UNIQUE':>7s} {'NOVEL':>7s} "
          f"{'DIST-1':>7s} {'DIST-2':>7s} {'S-BLEU':>7s} {'SCORE':>7s} "
          f"{'ON-TOPIC':>9s}")
    print(f"  {'─'*15} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*9}")

    for cat in categories:
        r = all_results[cat]
        print(f"  {cat:>15s} {r['grammar_rate']:>7.1%} {r['uniqueness']:>7.1%} "
              f"{r['novelty']:>7.1%} {r['distinct_1']:>7.2f} {r['distinct_2']:>7.2f} "
              f"{r['self_bleu']:>7.3f} {r['reviewer_score']:>7.3f} "
              f"{r['on_topic_rate']:>8.1%}")

    print(f"\n  {'AVERAGE':>15s} {avg_grammar:>7.1%} {avg_unique:>7.1%} "
          f"{avg_novel:>7.1%} {avg_dist1:>7.2f} {avg_dist2:>7.2f} "
          f"{avg_sbleu:>7.3f} {avg_score:>7.3f} {avg_ontopic:>8.1%}")

    # Overall diversity
    overall_dist1 = distinct_n(all_decoded, n=1)
    overall_dist2 = distinct_n(all_decoded, n=2)
    overall_sbleu = self_bleu(all_decoded)

    print(f"\n  Overall (all {len(all_decoded)} sequences combined):")
    print(f"    Distinct-1: {overall_dist1:.3f}")
    print(f"    Distinct-2: {overall_dist2:.3f}")
    print(f"    Self-BLEU:  {overall_sbleu:.3f} (lower = more diverse)")

    return {
        "per_category": all_results,
        "averages": {
            "grammar_rate": avg_grammar,
            "uniqueness": avg_unique,
            "novelty": avg_novel,
            "distinct_1": avg_dist1,
            "distinct_2": avg_dist2,
            "self_bleu": avg_sbleu,
            "reviewer_score": avg_score,
            "pass_rate": avg_pass,
            "on_topic_rate": avg_ontopic,
        },
        "overall": {
            "distinct_1": overall_dist1,
            "distinct_2": overall_dist2,
            "self_bleu": overall_sbleu,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Quality Improvement: Temperature Schedule + Editor + More Steps
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_with_schedule(model, seq_len, topic_emb, tok2id,
                          n_samples=20, n_steps=40, device=DEVICE,
                          temp_schedule="cosine"):
    """Sampling with temperature schedule instead of fixed temperature.

    temp_schedule:
      "fixed"  → 0.7 throughout
      "cosine" → starts high (0.9), decreases to 0.3 (explore→exploit)
      "linear" → linearly decreases from 0.9 to 0.3
      "low"    → 0.3 throughout (more deterministic)
    """
    model.eval()
    batch = n_samples

    if topic_emb.dim() == 1:
        topic_emb = topic_emb.unsqueeze(0)
    topic_batch = topic_emb.expand(batch, -1).to(device)

    tokens = torch.full((batch, seq_len), MASK, device=device)
    tokens_per_step = max(1, seq_len // n_steps)

    for step in range(n_steps):
        # Temperature schedule
        progress = step / n_steps
        if temp_schedule == "fixed":
            temp = 0.7
        elif temp_schedule == "cosine":
            temp = 0.3 + 0.6 * (1 + math.cos(math.pi * progress)) / 2
        elif temp_schedule == "linear":
            temp = 0.9 - 0.6 * progress
        elif temp_schedule == "low":
            temp = 0.3
        else:
            temp = 0.7

        t_val = max(0.01, 1.0 - step / n_steps)
        t = torch.full((batch,), t_val, device=device)

        logits = model(tokens, t, topic_batch)

        mask_positions = (tokens == MASK)

        for b in range(batch):
            masked_idx = mask_positions[b].nonzero(as_tuple=True)[0]
            if len(masked_idx) == 0:
                continue

            pos_logits = logits[b, masked_idx] / temp
            probs = F.softmax(pos_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

            confidence = probs.max(dim=-1)[0]
            n_keep = min(tokens_per_step, len(masked_idx))
            top_confident = confidence.topk(n_keep)[1]

            for idx in top_confident:
                pos = masked_idx[idx]
                tokens[b, pos] = sampled[idx]

    return tokens


def optimize_quality(generator, reviewer, editor, tok2id, id2tok,
                      cat_embeds, categories):
    """Test different quality improvement strategies."""
    print(f"\n{'='*70}")
    print("QUALITY OPTIMIZATION")
    print(f"{'='*70}")

    cfg_gen = CFGGenerator(seed=42)
    train_seqs = cfg_gen.generate_dataset(n=2000, seed=42)
    train_set = set(" ".join(s) for s in train_seqs)

    seq_len = 20
    n_samples = 50
    test_cats = categories[:5]  # subset for speed

    strategies = {
        "baseline_fixed_0.7_30steps": {"schedule": "fixed", "steps": 30, "editor": False},
        "cosine_schedule_40steps": {"schedule": "cosine", "steps": 40, "editor": False},
        "linear_schedule_40steps": {"schedule": "linear", "steps": 40, "editor": False},
        "low_temp_0.3_30steps": {"schedule": "low", "steps": 30, "editor": False},
        "cosine_40steps+editor": {"schedule": "cosine", "steps": 40, "editor": True},
        "baseline_50steps": {"schedule": "fixed", "steps": 50, "editor": False},
    }

    results = {}

    for name, config in strategies.items():
        print(f"\n  ── {name} ──")

        cat_results = []
        for cat in test_cats:
            topic = cat_embeds[cat]

            # Generate
            samples = sample_with_schedule(
                generator, seq_len, topic, tok2id,
                n_samples=n_samples, n_steps=config["steps"],
                temp_schedule=config["schedule"],
            )

            # Editor refinement
            if config["editor"]:
                samples, _ = editor_refine_topic(
                    editor, samples, topic, reviewer,
                    tok2id, id2tok, n_steps=3, temperature=0.5,
                )

            decoded = decode_tokens(samples, id2tok)
            quality = evaluate_quality(decoded, tok2id, id2tok, reviewer,
                                        topic, target_cat=cat, train_seqs=train_set)
            cat_results.append(quality)

        # Aggregate
        avg_grammar = np.mean([r["grammar_rate"] for r in cat_results])
        avg_unique = np.mean([r["uniqueness"] for r in cat_results])
        avg_novel = np.mean([r["novelty"] for r in cat_results])
        avg_dist1 = np.mean([r["distinct_1"] for r in cat_results])
        avg_sbleu = np.mean([r["self_bleu"] for r in cat_results])
        avg_score = np.mean([r["reviewer_score"] for r in cat_results])
        avg_pass = np.mean([r["pass_rate"] for r in cat_results])
        avg_ontopic = np.mean([r["on_topic_rate"] for r in cat_results])

        # Measure latency
        sync()
        t0 = time.perf_counter()
        _ = sample_with_schedule(
            generator, seq_len, cat_embeds[test_cats[0]], tok2id,
            n_samples=n_samples, n_steps=config["steps"],
            temp_schedule=config["schedule"],
        )
        sync()
        latency = time.perf_counter() - t0
        tps = (n_samples * seq_len) / latency

        results[name] = {
            "grammar_rate": avg_grammar,
            "uniqueness": avg_unique,
            "novelty": avg_novel,
            "distinct_1": avg_dist1,
            "self_bleu": avg_sbleu,
            "reviewer_score": avg_score,
            "pass_rate": avg_pass,
            "on_topic_rate": avg_ontopic,
            "latency_ms": latency * 1000,
            "tps": tps,
        }

        print(f"    Grammar: {avg_grammar:.1%}  Unique: {avg_unique:.1%}  "
              f"Novel: {avg_novel:.1%}")
        print(f"    Dist-1: {avg_dist1:.3f}  S-BLEU: {avg_sbleu:.3f}  "
              f"Score: {avg_score:.3f}")
        print(f"    Pass: {avg_pass:.1%}  On-topic: {avg_ontopic:.1%}")
        print(f"    Latency: {latency*1000:.0f}ms  TPS: {tps:.1f}")

    # Find best strategy
    print(f"\n  {'='*60}")
    print(f"  STRATEGY COMPARISON")
    print(f"  {'='*60}")
    print(f"  {'STRATEGY':>30s} {'GRAMMAR':>8s} {'SCORE':>7s} {'PASS':>6s} "
          f"{'S-BLEU':>7s} {'TPS':>8s}")
    print(f"  {'─'*30} {'─'*8} {'─'*7} {'─'*6} {'─'*7} {'─'*8}")

    for name, r in sorted(results.items(), key=lambda x: -x[1]["grammar_rate"]):
        print(f"  {name:>30s} {r['grammar_rate']:>7.1%} {r['reviewer_score']:>7.3f} "
              f"{r['pass_rate']:>6.1%} {r['self_bleu']:>7.3f} {r['tps']:>8.1f}")

    # Pareto front: best grammar at each TPS level
    best_grammar = max(results.items(), key=lambda x: x[1]["grammar_rate"])
    best_score = max(results.items(), key=lambda x: x[1]["reviewer_score"])
    best_diversity = min(results.items(), key=lambda x: x[1]["self_bleu"])
    best_speed = max(results.items(), key=lambda x: x[1]["tps"])

    print(f"\n  Best grammar:     {best_grammar[0]} ({best_grammar[1]['grammar_rate']:.1%})")
    print(f"  Best score:       {best_score[0]} ({best_score[1]['reviewer_score']:.3f})")
    print(f"  Best diversity:   {best_diversity[0]} (S-BLEU={best_diversity[1]['self_bleu']:.3f})")
    print(f"  Best speed:       {best_speed[0]} ({best_speed[1]['tps']:.1f} TPS)")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_tps_quality_benchmark():
    print("=" * 70)
    print("TPS & QUALITY BENCHMARK")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    ckpt = torch.load(RESULTS_DIR / "topic_conditioned_models.pt",
                      map_location=DEVICE, weights_only=False)
    tok2id = ckpt["tok2id"]
    id2tok = ckpt["id2tok"]
    vocab_size = len(tok2id)
    cat_embeds = ckpt["cat_embeds"]
    categories = ckpt["categories"]

    generator = TopicMDLMTransformer(
        vocab_size, topic_dim=1024, d_model=384,
        n_heads=6, n_layers=6, max_seq_len=20, dropout=0.1,
    ).to(DEVICE)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    reviewer = TopicReviewer(
        vocab_size, topic_dim=1024, d_model=256,
        n_heads=4, n_layers=4, max_seq_len=20, dropout=0.1,
    ).to(DEVICE)
    reviewer.load_state_dict(ckpt["reviewer"])
    reviewer.eval()

    editor = TopicMDLMTransformer(
        vocab_size, topic_dim=1024, d_model=384,
        n_heads=6, n_layers=6, max_seq_len=20, dropout=0.1,
    ).to(DEVICE)
    editor.load_state_dict(ckpt["generator"])
    editor.eval()

    print(f"  Generator: {sum(p.numel() for p in generator.parameters()):,} params")
    print(f"  Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {total_mem:.1f} GB")

    # ── TPS Benchmark ───────────────────────────────────────────────
    tps_results = benchmark_tps(generator, tok2id, id2tok, reviewer,
                                 cat_embeds, categories)

    # ── Quality Evaluation ──────────────────────────────────────────
    quality_results = evaluate_generation_quality(
        generator, reviewer, editor, tok2id, id2tok,
        cat_embeds, categories,
    )

    # ── Quality Optimization ────────────────────────────────────────
    optimization_results = optimize_quality(
        generator, reviewer, editor, tok2id, id2tok,
        cat_embeds, categories,
    )

    # ── Final Summary ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    # Best TPS
    best_tps = max(tps_results, key=lambda x: x["tps"])
    # TPS at common configs
    tps_batch1_20 = next(r for r in tps_results if r["batch_size"] == 1 and r["n_steps"] == 20)
    tps_batch10_20 = next(r for r in tps_results if r["batch_size"] == 10 and r["n_steps"] == 20)
    tps_batch100_10 = next(r for r in tps_results if r["batch_size"] == 100 and r["n_steps"] == 10)

    q_avg = quality_results["averages"]

    print(f"""
  ┌──────────────────────────────────────────────────────────────────────┐
  │  THROUGHPUT                                                          │
  ├──────────────────────────────────────────────────────────────────────┤
  │  Single sequence (batch=1, 20 steps):   {tps_batch1_20['tps']:>7.1f} TPS          │
  │  Batch of 10 (20 steps):                {tps_batch10_20['tps']:>7.1f} TPS          │
  │  Batch of 100 (10 steps):               {tps_batch100_10['tps']:>7.1f} TPS          │
  │  Peak throughput (batch={best_tps['batch_size']}, {best_tps['n_steps']} steps):        {best_tps['tps']:>7.1f} TPS          │
  ├──────────────────────────────────────────────────────────────────────┤
  │  QUALITY (50 samples × {len(categories)} categories = {50*len(categories)} total)                   │
  ├──────────────────────────────────────────────────────────────────────┤
  │  Grammar rate:          {q_avg['grammar_rate']:>6.1%}                                     │
  │  Uniqueness:            {q_avg['uniqueness']:>6.1%}                                     │
  │  Novelty:               {q_avg['novelty']:>6.1%}                                     │
  │  Distinct-1 (unigram):  {q_avg['distinct_1']:>6.3f}                                     │
  │  Distinct-2 (bigram):   {q_avg['distinct_2']:>6.3f}                                     │
  │  Self-BLEU (↓ better):  {q_avg['self_bleu']:>6.3f}                                     │
  │  Reviewer score:        {q_avg['reviewer_score']:>6.3f}                                     │
  │  Pass rate (≥0.5):      {q_avg['pass_rate']:>6.1%}                                     │
  │  On-topic rate:         {q_avg['on_topic_rate']:>6.1%}                                     │
  └──────────────────────────────────────────────────────────────────────┘

  NOTE: MDLM generates ALL positions in parallel at each diffusion step.
  Unlike autoregressive (1 token/step), MDLM predicts seq_len tokens/step.
  This gives massive throughput advantage at batch scale.
""")

    # Save results
    result = {
        "experiment": "tps_quality_benchmark",
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        "tps_results": tps_results,
        "quality_results": quality_results,
        "optimization_results": optimization_results,
    }

    out = RESULTS_DIR / "tps_quality_benchmark.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    run_tps_quality_benchmark()
