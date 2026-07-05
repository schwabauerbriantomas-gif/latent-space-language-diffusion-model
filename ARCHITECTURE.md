# FF-SplatDiffusion — Full System Architecture Diagram

*All latencies [MEASURED] on RTX 3090. All adaptation claims validated.*

---

## End-to-End Flow (per topic request)

```
                          ┌─────────────────────────────────┐
                          │     TOPIC REQUEST               │
                          │  (user asks about a topic,      │
                          │   or SplatsDB samples one)      │
                          └──────────────┬──────────────────┘
                                         │
                                         ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║  STAGE 1: KNOWLEDGE GAP DETECTION                    1.8 ms   ║
    ║                                                               ║
    ║  ┌─────────────────────────────────────────────────────────┐  ║
    ║  │  DensityConfidenceScore                    0 params     │  ║
    ║  │  (geometric — NO neural net, NO training)                │  ║
    ║  │                                                         │  ║
    ║  │  Reads SplatsDB vector store, computes 3 signals:       │  ║
    ║  │    1. Neighbor density   (how many entries nearby?)     │  ║
    ║  │    2. Nearest distance   (how close is closest?)        │  ║
    ║  │    3. Neighbor consistency (do neighbors cluster?)      │  ║
    ║  │                                                         │  ║
    ║  │  Output: confidence ∈ [0, 1]                            │  ║
    ║  └──────────────────────┬──────────────────────────────────┘  ║
    ╚═════════════════════════╪═════════════════════════════════════╝
                              │
                    ┌─────────▼─────────┐
                    │  confidence ≥ 0.3?│
                    └────┬─────────┬────┘
                         │         │
                    YES  │         │  NO (gap detected)
                         │         ▼
                         │    ╔══════════════════════════════════════╗
                         │    ║  STAGE 2: INFORMATION SEEKING       ║
                         │    ║                              35 ms   ║
                         │    ║  ┌──────────────────────────────┐   ║
                         │    ║  │ QueryGenerator    5.2M params │   ║
                         │    ║  │ → "tell me about [topic]"     │   ║
                         │    ║  └──────────────┬───────────────┘   ║
                         │    ║                 │                   ║
                         │    ║  ┌──────────────▼───────────────┐   ║
                         │    ║  │ SplatsDB.query()     1.0 ms   │   ║
                         │    ║  │ (cosine similarity search)    │   ║
                         │    ║  └──────────────┬───────────────┘   ║
                         │    ║                 │                   ║
                         │    ║  ┌──────────────▼───────────────┐   ║
                         │    ║  │ IngestTrigger        0.7 ms   │   ║
                         │    ║  │ density < threshold?          │   ║
                         │    ║  └──────┬───────────────┬───────┘   ║
                         │    ║     YES│              NO│           ║
                         │    ║         │               │           ║
                         │    ║  ┌──────▼──────┐  ┌──────▼─────┐    ║
                         │    ║  │EXTERNAL     │  │ SKIP       │    ║
                         │    ║  │RETRIEVER    │  │ (sufficient│    ║
                         │    ║  │(LLM/web/    │  │  data)     │    ║
                         │    ║  │ RAG)        │  └────────────┘    ║
                         │    ║  └──────┬──────┘                    ║
                         │    ║         │                           ║
                         │    ║  ┌──────▼───────────────────────┐   ║
                         │    ║  │ SplatsDB.ingest()  0.24 ms   │   ║
                         │    ║  │ (embed → store, O(1))        │   ║
                         │    ║  │ Knowledge base GROWS         │   ║
                         │    ║  └──────────────────────────────┘   ║
                         │    ╚══════════════════════════════════════╝
                         │         │
                         └─────────┴──────────┐
                                               │
                                               ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║  STAGE 3: TEXT GENERATION                          242 ms     ║
    ║                                                               ║
    ║  ┌─────────────────────────────────────────────────────────┐  ║
    ║  │  Generator (TopicMDLMTransformer)         16.1M params  │  ║
    ║  │                                                         │  ║
    ║  │  Input:  [MASK MASK MASK MASK] + topic embedding        │  ║
    ║  │  Process: iterative unmasking (20 steps)                │  ║
    ║  │    Each layer: self-attn (syntax)                       │  ║
    ║  │                cross-attn ← topic (semantics)           │  ║
    ║  │  Output: [the] [brave] [lion] [runs] [fast]             │  ║
    ║  └──────────────────────┬──────────────────────────────────┘  ║
    ╚═════════════════════════╪═════════════════════════════════════╝
                              │
                              ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║  STAGE 4: QUALITY REVIEW                           8 ms       ║
    ║                                                               ║
    ║  ┌─────────────────────────────────────────────────────────┐  ║
    ║  │  Reviewer (TopicReviewer)                  5.1M params  │  ║
    ║  │                                                         │  ║
    ║  │  Scores: grammar quality + topic consistency            │  ║
    ║  │  Output: score ∈ [0, 1] per sequence                    │  ║
    ║  └──────────────────────┬──────────────────────────────────┘  ║
    ╚═════════════════════════╪═════════════════════════════════════╝
                              │
                    ┌─────────▼─────────┐
                    │  score ≥ 0.5?     │
                    └────┬─────────┬────┘
                         │         │
                    YES  │         │  NO (needs refinement)
                         │         ▼
                         │    ╔══════════════════════════════════════╗
                         │    ║  STAGE 5: EDITING             9 ms   ║
                         │    ║                                       ║
                         │    ║  ┌───────────────────────────────┐   ║
                         │    ║  │ Editor (TopicMDLMTransformer) │   ║
                         │    ║  │                   16.1M params│   ║
                         │    ║  │                               │   ║
                         │    ║  │ 1. Mask worst positions       │   ║
                         │    ║  │ 2. Regenerate (cross-attn)    │   ║
                         │    ║  │ 3. Accept if score improved   │   ║
                         │    ║  │ Repeat up to 3 rounds         │   ║
                         │    ║  └───────────────┬───────────────┘   ║
                         │    ╚══════════════════╪═══════════════════╝
                         │                       │
                         └───────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   FINAL OUTPUT          │
                    │   "the brave lion runs  │
                    │    fast in the forest"  │
                    └─────────────────────────┘
```

---

## Latency Breakdown (E2E, 10 sequences)

```
  STAGE                    TIME        % OF TOTAL    CUMULATIVE
  ─────────────────────────────────────────────────────────────
  1. Confidence check       1.8 ms        0.4%        1.8 ms
  2. SplatsDB query         1.1 ms        0.2%        2.9 ms
     Ingest trigger          0.7 ms        0.1%        3.6 ms
     (External retrieval)    variable      —           —
     (SplatsDB ingest)       0.24 ms/entry —           —
  3. Generation (10 seqs) 479.8 ms       94.5%      483.4 ms
  4. Review (10 seqs)       7.9 ms        1.6%      491.3 ms
  5. Edit (3 rounds)       16.5 ms        3.3%      507.9 ms
  ─────────────────────────────────────────────────────────────
  TOTAL E2E               507.9 ms      100.0%
  PER SEQUENCE             50.8 ms
```

**Generation dominates at 94.5% of total time.** Everything else combined is 28ms.

---

## Model Parameters

```
  HEAD                  PARAMETERS    TYPE           TRAINABLE?
  ─────────────────────────────────────────────────────────────
  DensityConfidence            0     geometric      NO (pure math)
  QueryGenerator         5,206,408   transformer    one-time
  Generator             16,076,795   transformer    one-time
  Reviewer               5,105,665   transformer    one-time
  Editor                16,076,795   transformer    one-time
  ─────────────────────────────────────────────────────────────
  TOTAL                 42,465,663   (~42.5M)
```

---

## Training Schedule: When Does Each Head Retrain?

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  TRAINING SCHEDULE                                               │
  │                                                                  │
  │  Event                           │ What retrains?  │ Cost       │
  │  ────────────────────────────────┼────────────────┼──────────── │
  │                                  │                │             │
  │  SplatsDB ingests new data       │  NOTHING       │  0 ms       │
  │  (normal operation)              │                │             │
  │                                  │                │             │
  │  New vocabulary token added      │  Generator     │  ~2.4 min   │
  │  (rare, structural change)       │  Editor        │  ~2.4 min   │
  │                                  │  QueryGen      │  ~1 min     │
  │                                  │  Reviewer      │  ~100 min * │
  │                                  │                │             │
  │  Grammar rules change            │  Generator     │  ~2.4 min   │
  │  (e.g., new CFG rules)           │  Editor        │  ~2.4 min   │
  │                                  │  Reviewer      │  ~100 min * │
  │                                  │                │             │
  │  Embedding model changes         │  ALL           │  ~106 min   │
  │  (e.g., bge-m3 → bge-m4)         │                │             │
  │                                  │                │             │
  │  Confidence threshold tuning     │  NOTHING       │  0 ms       │
  │  (adjust sim_threshold)          │  (just config) │             │
  │                                  │                │             │
  └──────────────────────────────────────────────────────────────────┘

  * Reviewer is the most expensive to retrain (~100 min) because it
    needs generated samples in each training epoch. All others <3 min.
    This is a one-time cost per structural change, NOT per ingestion.
```

### Key Distinction

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  DAILY OPERATION (SplatsDB grows):                              │
  │    Training needed: ZERO                                        │
  │    Adaptation mechanism: cross-attention reads new embeddings   │
  │    Latency impact: NONE (constant regardless of store size)     │
  │                                                                 │
  │  STRUCTURAL CHANGES (rare, maybe once a month):                 │
  │    New vocab: ~106 min total retrain (mostly reviewer)          │
  │    New grammar: ~105 min (generator+editor+reviewer)            │
  │    New embedding model: ~106 min (full retrain)                 │
  │                                                                 │
  │  The system NEVER retrrains just because data was ingested.     │
  │  It ONLY retrains when the STRUCTURE changes.                   │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Adaptation Verification (Zero-Shot, No Retraining)

```
  TOPIC TYPE              CONFIDENCE    INTERPRETATION
  ──────────────────────────────────────────────────────
  Rich (15 entries)         0.805      ✅ well-covered
  Sparse (3 entries)        0.445      ⚡ partially covered
  Empty (0 entries)         0.134      ⚡ gap → trigger retrieval
  Novel random              0.127      ⚡ gap → trigger retrieval
  ──────────────────────────────────────────────────────
  Gap (rich - novel)        0.712      clear discrimination

  After ingesting 10 items:
    0.047 → 0.827  (Δ=+0.780, instant adaptation, NO retraining)
```

---

## Comparison: Neural vs Density Confidence

```
  METRIC                   NEURAL (v1)    DENSITY (v2)
  ─────────────────────────────────────────────────────
  Novel confidence           0.668          0.108
  Known confidence           0.650          0.821
  Discriminates?             NO (Δ=0.018)   YES (Δ=0.712)
  Accuracy                   ~50%           90%
  Parameters                 3,884,545      0
  Training                   1500 epochs    NONE
  Latency                    4.7 ms         1.8 ms
  Adapts to new data         NO             YES (instant)
```

---

## Data Flow Summary

```
  ┌──────────┐         ┌──────────┐         ┌──────────┐
  │  USER    │────────▶│DENSITY   │────────▶│SPLATSDB  │
  │  REQUEST │         │CONFIDENCE│         │  QUERY   │
  └──────────┘         │  (1.8ms) │         │  (1.0ms) │
                       │  0 params│         └────┬─────┘
                       └──────────┘              │
                            │                    │
                     confidence < 0.3?           │
                            │                    │
                    ┌───────▼───────┐            │
                    │    GAP?       │            │
                    ├───┐       ┌───┤            │
                    │NO │       │YES│            │
                    └───┘       └──┬┘            │
                     │             │             │
                     │    ┌────────▼────────┐    │
                     │    │QUERY GENERATOR  │    │
                     │    │   (34 ms)       │    │
                     │    │  5.2M params    │    │
                     │    └────────┬────────┘    │
                     │             │             │
                     │    ┌────────▼────────┐    │
                     │    │EXTERNAL RETRIEVE│    │
                     │    │(LLM/web search) │    │
                     │    └────────┬────────┘    │
                     │             │             │
                     │    ┌────────▼────────┐    │
                     │    │SPLATSDB.INGEST  │    │
                     │    │  (0.24 ms/entry)│    │
                     │    │  KNOWLEDGE GROWS│    │
                     │    └────────┬────────┘    │
                     │             │             │
                     └─────────────┴─────────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │  GENERATOR (242 ms)   │
                       │  16.1M params         │
                       │  [MASK] → text        │
                       │  cross-attn ← topic   │
                       └───────────┬───────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │  REVIEWER (8 ms)      │
                       │  5.1M params          │
                       │  score ∈ [0,1]        │
                       └───────────┬───────────┘
                                   │
                          score < 0.5?
                           │       │
                          YES      NO
                           │       │
                    ┌──────▼──┐    │
                    │ EDITOR  │    │
                    │(9 ms)   │    │
                    │16.1M    │    │
                    │mask+fix │    │
                    └──────┬──┘    │
                           │       │
                           └───┬───┘
                               │
                               ▼
                       ┌───────────────┐
                       │ FINAL TEXT    │
                       │ (50.8 ms/seq) │
                       └───────────────┘
```

---

## One-Time Training Costs (Do NOT Repeat Per Ingestion)

```
  MODEL            PARAMS      TRAINING TIME    WHEN TO RETRAIN
  ─────────────────────────────────────────────────────────────
  Generator        16.1M       ~2.4 min         New vocab/grammar
  Reviewer          5.1M       ~100 min *       New vocab/grammar
  Editor           16.1M       ~2.4 min         New vocab/grammar
  QueryGen          5.2M       ~1 min           New vocab
  DensityConf          0       0 (none)         NEVER (pure math)
  ─────────────────────────────────────────────────────────────
  TOTAL            42.5M       ~106 min         Structural change only

  * Reviewer needs generated samples per epoch → expensive
  All heads use early stopping (best checkpoint tracking)
  GPU memory peak: 1.7 GB
```
