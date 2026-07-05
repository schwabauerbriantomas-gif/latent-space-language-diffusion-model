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

## Empirical Validation Plan

### Phase 1 — Fast tests (<30 min each, pass/fail)

**T1: FF learns meaningful energy.**
- Hypothesis: E(clean_text) < E(random_token_sequence) with margin > 0.5
- Setup: 1000 TinyStories, bge-m3 embed, 2-layer FF energy head
- Pass: AUROC(data vs noise) > 0.85
- Fail: AUROC < 0.70 → FF cannot separate clean/noise in this space

**T2: ∇E points toward data manifold.**
- Hypothesis: stepping along -∇E from noise moves toward nearest real sequence
- Setup: 500 noisy samples, 50 Langevin steps, measure distance to nearest real
- Pass: median distance decreases monotonically
- Fail: no decrease → score not useful for sampling

**T3: Diffusion sampling produces non-degenerate output.**
- Hypothesis: sampled latents decode to non-repetitive token sequences
- Setup: generate 100 samples, measure repetition ratio
- Pass: <30% repetition (non_rep > 0.7)
- Fail: >50% repetition → mode collapse

If all 3 fail → FF+diffusion in this space is empirically refuted (with data, not theory).
If T1+T2 pass, T3 fails → energy is learnable but sampling dynamics are wrong.
If all pass → proceed to Phase 2 (quality/coherence).

### Phase 2 — Quality (only if Phase 1 passes)
- Coherence: do samples read as language? (human eval + split-half similarity)
- Coherence vs baseline: random token selection, nearest-neighbor retrieval
- Autoresearch loop: systematically improve via guidance hyperparameters

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
