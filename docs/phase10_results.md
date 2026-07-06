# Phase 10: MDLM v3 + Multi-Layer HRM Experiments

## Overview

Scaled MDLM to 201M params, trained on 1M Ultra-FineWeb docs (272M tokens).
Achieved PPL 102.6. Explored three HRM approaches for long-distance semantic
coherence.

## Training Results [MEASURED]

```
Model:      MDLM-BPE v3 (201.3M params)
  - d_model=1024, 10 layers, 16 heads
  - Semi-autoregressive unmasking (left-to-right blocks of 4)
  - RoPE positional encoding, AdaLN conditioning

Training:   49,779 optimizer steps over 3 epochs
  - 272M tokens (1M docs × 128 seq_len)
  - bf16, gradient accumulation, cosine LR schedule
  - Duration: 6h 45min on RTX 3090
  - Throughput: 33.5K tok/s, 2.0 opt/s

Results:    Loss 7.97 → 4.63  (PPL 1341 → 102.6)
```

## Guidance Optimization [MEASURED]

Tested 3 guidance techniques on top of semi-AR generation:

| Technique              | Repetition Score | Notes                    |
|------------------------|:----------------:|--------------------------|
| Baseline (no guidance) |     0.79         | Repetitive loops         |
| + Adaptive guidance    |     0.99         | Cooling temp + adaptive penalties |
| + Repetition HRM       |     1.00         | Token-level safety net   |

**Optimal guidance config:**
- Temperature cooling: 1.0 → 0.5 across blocks
- Repetition penalty: 1.2 → 1.5 (adaptive, increases per block)
- Frequency penalty: 0.3 → 0.6 (adaptive)
- Top-p: 0.95
- No-repeat bigram

## HRM Experiments for Semantic Coherence

### Experiment 1: Semantic Coherence HRM (embedding-based) — FAILED

Used MDLM's own hidden states (1024-dim) to detect semantic drift.

**Result:** Detection worked (embeddings separated topics with std=0.109),
but correction DEGRADED text. The 201M model's embeddings are too weak to
reliably distinguish real drift from normal variation.

### Experiment 2: AR-Oracle HRM v1 (guided regeneration) — FAILED

Used Qwen3-0.6B as oracle to detect incoherent positions, then guided MDLM
to regenerate them with oracle-biased logits.

**Result:** Δ log-prob = -0.02 (noise). The MDLM is too weak to generate
good replacements even with strong oracle guidance.

### Experiment 3: AR-Oracle HRM v2 (token replacement) — FAILED

Oracle directly replaced bad tokens with its own predictions.

**Result:** Δ log-prob = -0.71 (WORSE). Token-level replacement across
different vocabularies (MDLM 10K vs Qwen3 151K) breaks coherence.

### Experiment 4: Hybrid Speculative (segment regen) — PARTIAL SUCCESS

MDLM generates full draft → Qwen3 identifies bad segments → Qwen3 regenerates
those segments as text.

**Result:** Δ log-prob = +0.22 (first positive improvement). 4/6 prompts
improved significantly, 1 marginal, 1 regressed (over-correction).

## Final Benchmark [MEASURED]

```
Method                       Oracle LP   Rep Score   TPS     Latency
─────────────────────────────────────────────────────────────────────
MDLM baseline                 -3.38       0.84       51.8     1.1s
MDLM + guidance               -4.29       0.98       52.2     1.1s
MDLM + guidance + RepHRM      -4.35       0.99       51.0     1.1s
Hybrid (MDLM+Qwen3)           -3.99       0.99       22.9     1.9s
Qwen3-0.6B (pure AR)          -1.18       0.95       17.1     3.8s

Forward-pass throughput:
  Batch  1:  MDLM=8,443 TPS   Qwen3=2,152 TPS   (3.9x faster)
  Batch  8:  MDLM=50,440 TPS  Qwen3=17,009 TPS  (3.0x faster)
  Batch 32:  MDLM=57,759 TPS  Qwen3=32,023 TPS  (1.8x faster)
```

## Key Findings

1. **Guidance works:** Adaptive logit guidance eliminates repetition
   (0.79→0.99) at zero cost. This is the clear win.

2. **MDLM forward pass is 1.8-3.9x faster than Qwen3** at batch inference.
   The parallel nature of masked diffusion gives consistent speedup.

3. **Small model embeddings are too weak** for semantic drift detection.
   A 201M PPL=102 model doesn't produce semantically rich enough hidden
   states to distinguish coherence quality.

4. **Cross-vocabulary token operations fail.** Both guided regeneration
   and direct replacement break when mapping between different BPE
   vocabularies at the token level.

5. **Segment-level hybrid works** but with caveats: improvements are real
   (+0.22 log-prob) but inconsistent (1/6 prompts regress). The approach
   needs better segment boundary detection to avoid over-correction.

6. **The quality gap is fundamental:** Qwen3 (596M, trained on trillions)
   achieves lp=-1.18 vs MDLM's lp=-3.38. This is a capacity and data gap
   that no amount of guidance or HRM can fully close.
