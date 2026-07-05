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

### Final Verdict — Measured 2026-07-05

**FF + diffusion in SplatsDB latent space is empirically refuted for text generation.**

Three independent training methods all fail to produce a usable generative energy:

| Method | T1b Discrimination | T2 Gradient | T3 Sampling | Generative? |
|--------|:--:|:--:|:--:|:--:|
| FF (80 ep) | 0.663 FAIL | -12% FAIL | 0.000 FAIL | ❌ |
| FF (500 ep) | **0.999 PASS** | -12% FAIL | 0.000 FAIL | ❌ |
| CD (200 ep) | **1.000 PASS** | -10% FAIL | 0.000 FAIL | ❌ |
| CD + R1 penalty (λ=1) | **1.000 PASS** | -7% FAIL | 0.000 FAIL | ❌ |
| CD + R1 penalty (λ=10) | **1.000 PASS** | -5% FAIL | 0.000 FAIL | ❌ |
| CD + R1 penalty (λ=100) | 0.500 FAIL | -6% FAIL | 0.000 FAIL | ❌ |

### Root Cause: Energy Collapse

Diagnostic measurements revealed WHY every method fails:

| Metric | At real data | At noise | Ratio |
|--------|:--:|:--:|:--:|
| \|∇E\| (gradient magnitude) | 1752 | 10 | **175×** |
| Gradient direction at real | — | — | Away from center (23→78 distance) |
| E vs \|x\| correlation | Spearman -0.083 | — | E has angular structure |

**The energy forms sharp spikes on training data** (175× steeper gradient at data than at noise). Langevin sampling that approaches data hits these walls and gets repelled. The energy is a perfect **classifier** (AUROC=1.0) but a terrible **generative model** because its gradient field pushes samples away from the data manifold.

This is not fixable by:
- More training (FF 80→500 epochs: fixed T1b, did NOT fix T2/T3)
- Better negatives (CD with Langevin negatives: fixed stability, did NOT fix T2/T3)
- Gradient regularization (R1 penalty λ=1..100: did NOT fix T2/T3 at any value)

### What This Confirms

The theoretical concern was correct: **Hinton's FF goodness objective optimizes for separation (discrimination), not for smooth density (generation)**. The local goodness function creates energy landscapes that are excellent classifiers but whose gradients are adversarial to sampling. No amount of CD, R1 regularization, or hyperparameter tuning within the FF energy framework resolves this.

### What Would Actually Work (not in this repo)

The fix is NOT in the training method — it's in the **energy parameterization**:
1. **Score matching** (Hyvärinen): train ∇E to match the data score directly. The gradient IS the target, not a side-effect.
2. **Spectral-normalized networks**: bound the Lipschitz constant → smooth E by construction.
3. **Diffusion-native objectives**: DDPM/SEDD-style losses that parameterize the score directly.

These abandon the FF premise (local goodness). They're known to work (that's how real diffusion LMs are trained). FF's value is elsewhere (efficient discriminative training, biological plausibility), not in generative modeling.

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
