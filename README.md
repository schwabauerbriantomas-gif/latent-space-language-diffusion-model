# FF-SplatDiffusion

Forward-Forward energy training + continuous-space diffusion, using SplatsDB's latent space as the energy backbone.

## The Question

Can Hinton's Forward-Forward (FF) algorithm train an energy-based model (EBM) that, combined with continuous diffusion in SplatsDB's latent space, generates coherent text — without backpropagation and without extra compute beyond what SplatsDB already provides?

## Architecture

```
                    TRAINING (no backprop)
                    =====================
   clean text ──┐
                ├─► embed ──► [FF layer 1] ──► [FF layer 2] ──► ... ──► E(x)
   noisy text ──┘     (SplatsDB          each layer learns local goodness(x)
                       bge-m3 1024d)       positive: E(data) < threshold
                                          negative: E(noise) > threshold

                    INFERENCE (diffusion sampling)
                    ==============================
        ε ~ N(0,I)
         │
         ▼  x_T = ε
    ┌────────────────────────────────────────┐
    │  for t = T ... 0:                      │
    │    score = ∇_x E(x_t)  [autodiff]      │  ← FF layers frozen, grad OK
    │    x_{t-1} = denoise(x_t, score, t)    │
    │    decode: nearest splat → token        │  ← SplatsDB HNSW
    └────────────────────────────────────────┘
         │
         ▼
     coherent text
```

## Why Each Component

| Component | Role | From |
|-----------|------|------|
| SplatsDB splats (μ, α, κ) | Latent space structure | Existing — no change |
| bge-m3 1024d embeddings | Token → latent | Existing SplatsDB config |
| FF layers | Energy E(x) over sequences | **NEW** — trains without backprop |
| Diffusion sampler | Score-based generation | **NEW** — uses ∇E at inference |
| HNSW decode | Latent → token | Existing SplatsDB |

## What We Reuse vs Build

### Reused from SplatsDB (zero new compute)
- Token embeddings (bge-m3)
- Splat parameters (μ, α, κ) — the latent manifold
- HNSW index for latent→token decoding
- Energy formula: E(x) = -log Σ α_i exp(-κ_i ||x-μ_i||²)

### New (the research contribution)
- FF energy head: 2-4 layer MLP learning E(x) from sequence embeddings
- Continuous diffusion sampler: Langevin/DDIM in latent space guided by ∇E
- Training loop: FF positive/negative batches

## Empirical Results

### Phase 1 — Measured 2026-07-05 on RTX 3090, synthetic SplatsDB latent space

| Test | Metric | Baseline (80 ep) | After Sweep (500 ep) | Verdict |
|------|--------|:---:|:---:|:---:|
| T1a — point discrimination | AUROC | 1.000 | 1.000 | ✅ PASS |
| T1b — sequence discrimination | AUROC | 0.663 | **0.999** | ✅ PASS |
| T2 — score gradient direction | dist decrease | -12% | -12% | ❌ FAIL |
| T3 — sampling near data | ratio <1.0 | 0.000 | 0.000 | ❌ FAIL |

### Key Finding

**FF learns to discriminate, but not to generate.**

The autoresearch sweep (47 configs) found that FF achieves perfect sequence discrimination (AUROC=1.0) when trained for 500+ epochs — the baseline failure was **undertraining**, not a fundamental limitation. The dominant factor was epochs (dose-response: 50ep→0.62, 200ep→0.78, 500ep→0.98, 1000ep→1.0).

However, **T2 and T3 still fail**. The learned energy separates data/noise by magnitude (E(real)≈0.6 vs E(noise)≈3.4), but the **gradient ∇E does not point toward the data manifold**. Langevin sampling drives samples *away* from real data (+12% distance).

This reveals a fundamental gap between **discriminative energy** and **generative energy**:
- FF's local goodness objective optimizes for *separation* (good for classification)
- Score-based diffusion needs the gradient to *point toward high-density regions* (different requirement)
- A good discriminator is not necessarily a good generative model

### What Would Be Needed

- **Score matching** (not goodness separation) as the FF training objective, OR
- **Annealed Langevin** with multi-scale noise, OR
- **A different sampler** that uses the energy landscape differently than naive gradient descent

### Autoresearch Sweep Details

- 47 configurations tested (`results/autoresearch_sweep.jsonl`)
- 19/47 pass T1b (>0.85)
- Winning config: `hidden=256, n_layers=3, epochs=500, thr=(0.05,0.2), lr=0.5, seq_len=32`
- Robust across 5 seeds (42, 123, 7, 999, 2024 — all AUROC=1.0)
- **Mean-pooling is the best aggregation** (0.993 vs sum/max/var 0.53-0.57)

## Honest Constraints (stated upfront)

1. **SplatsDB is bag-of-tokens by design** — no sequence model. The FF energy head MUST supply the sequential structure. If it can't, the approach fails. This is the core empirical question.

2. **No claims until measured.** Every number in results carries [MEASURED] provenance.

3. **Scale**: experiments on TinyStories (small vocabulary, short sequences). Scaling to full language is Phase 3+, not this repo.

## Reproduce

```bash
# Phase 1 tests (each ~5-15 min on RTX 3090)
python src/phase1_test.py --test t1  # FF energy discrimination
python src/phase1_test.py --test t2  # score gradient direction
python src/phase1_test.py --test t3  # sampling degeneracy
```

## References

- Hinton, G. (2022). *The Forward-Forward Algorithm*. [arXiv:2212.13345](https://arxiv.org/abs/2212.13345)
- Austin et al. (2021). *Structured Denoising Diffusion in Discrete State-Spaces* (D3PM). [arXiv:2107.03006](https://arxiv.org/abs/2107.03006)
- Sahoo et al. (2024). *MDLM: Masked Diffusion Language Model*. [arXiv:2406.03709](https://arxiv.org/abs/2406.03709)
- Du & Mordatch (2019). *Implicit Generation and Modeling with EBMs*. [arXiv:1903.08689](https://arxiv.org/abs/1903.08689)
