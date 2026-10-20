# Latent Space Language Diffusion Model

<!-- SEO: keywords for discoverability -->
<!--
keywords: masked diffusion language model, MDLM, parallel text generation, diffusion model, non-autoregressive generation, bidirectional transformer, AR oracle validation, speculative decoding, Qwen3, SplatsDB, open weights, open source LLM, torch, safetensors, autoregressive control, architectural comparison
-->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![Model: 183M](https://img.shields.io/badge/Params-183M-green.svg)](#results-at-a-glance)
[![Architecture: v4](https://img.shields.io/badge/Arch-v4%20optimized-blue.svg)](#v4-optimizations)
[![Speed: 2.4x](https://img.shields.io/badge/Speed-2.4%C3%97%20faster-brightgreen.svg)](#benchmark-measured)
[![Open Weights](https://img.shields.io/badge/Weights-Open%20%E2%9C%93-success.svg)](https://huggingface.co/brianschwabauer/latent-space-language-diffusion-model)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-blue.svg)](https://huggingface.co/brianschwabauer/latent-space-language-diffusion-model)

> **🤗 Download weights, tokenizer, and full inference code on [HuggingFace](https://huggingface.co/brianschwabauer/latent-space-language-diffusion-model)** — open weights (safetensors), open code (modeling + training scripts), fully reproducible. No black boxes.

A **masked diffusion language model (MDLM)** that generates text by predicting all tokens in parallel, validated by an autoregressive oracle. Built on [SplatsDB](https://github.com/schwabauerbriantomas-gif/splatdb)'s vector infrastructure.

**The core finding**: full-parallel masked diffusion with adaptive guidance + AR-oracle validation achieves **2.4× faster generation** than its own semi-autoregressive decoding variant, and **~1.2× than an equal-size autoregressive control** (KV cache, greedy) — while the AR control produces more coherent text (measured, see [Controlled Comparison](#controlled-comparison-mdlm-vs-ar-same-scale)). No sequential decoding required.

## Release v4.1 (2026-10-19) — divergence root-caused and fixed

An audit of the Hugging Face release found that the published weights were an accidental mid-training checkpoint (perplexity in the millions) and that **every previous v4 run had silently diverged**. Root cause isolated with a controlled ablation ([`scripts/diag_v4_loss.py`](scripts/diag_v4_loss.py), receipt in `results/diag_v4_loss.json`):

- The ELBO importance weighting `loss *= 1/(1-t)` with logit-normal `t` is heavy-tailed: rare weights up to ×100,000 (post-clamp) dominate gradients.
- Same model, same masking, same schedule, plain CE instead: loss falls 9.4 → 6.3 in 800 steps (healthy, v3-like); the ELBO-weighted metric stays at 18–33.
- v4.1 was retrained with `--loss-type plain` ([`scripts/train_v4.py`](scripts/train_v4.py)).

The old broken checkpoint is preserved as `checkpoints/mdlm_v4_broken_500k_diverged.pt` for reference. Speed claims corrected too: the "2.4×" is vs our semi-AR variant; the equal-size AR control with KV cache runs at ~1.2× slower generation but with better coherence (oracle log-prob), as documented below.

## v4 Optimizations

The v4 architecture (`src/mdlm_v4_optimized.py`) introduces measurable improvements over v3:

| Optimization | v3 | v4 | Impact |
|-------------|----|----|--------|
| Normalization | LayerNorm | **RMSNorm** | 10-20% faster, same quality |
| FFN | GELU (4×d_model) | **SwiGLU** (8/3×d_model) | Better perplexity, fewer params |
| AdaLN | 6×d_model | **4×d_model** | Removes unnecessary residual modulation |
| Timestep sampling | U(0,1) | **Logit-normal** | Concentrates t where gradient signal is strongest |
| Loss weighting | Plain CE | **ELBO-weighted** (1/(1-t)) | Variational bound from Sahoo et al. 2024 |
| Logits computation | Full [B,S,V] | **Masked-only** F.linear | Skips ~50% of output head FLOPs |
| Output bias | Present | **Removed** | Redundant with weight tying |
| Unmasking schedule | Linear | **Cosine** | More refinement steps, better quality |
| Parameters | 201.3M | **182.8M** | **-9.2% fewer params** |
| Forward (batch=8) | 40,000 TPS | **53,783 TPS** | **+34.5% faster** |
| Generation (no guidance) | 31.2 tok/s | **96.5 tok/s** | **3.1× faster** |

All v4 code is `torch.compile` compatible and uses Flash Attention throughout.

## Results at a Glance

| Metric | v3 | v4 | Notes |
|--------|----|----|-------|
| Model parameters | 201M | **183M** | d_model=1024, 10 layers, 16 heads |
| Training data | 272M tokens | 27.2M tokens | 100K docs from Ultra-FineWeb |
| Perplexity (plain CE, comparable) | 102.6 | **62.4** | v4.1 uses 10× less data; loss = plain CE on masked positions (see Release v4.1 below) |
| Forward throughput (batch=1) | 5,595 TPS | **5,952 TPS** | **+6.4% faster** |
| Forward throughput (batch=8) | 40,000 TPS | **53,783 TPS** | **+34.5% faster** |
| Forward throughput (batch=32) | 127,187 TPS | 123,341 TPS | Comparable |
| Generation speed (no guidance) | 31.2 tok/s | **96.5 tok/s** | **3.1× faster** |
| Generation speed (guided) | 15.6 tok/s | **21.8 tok/s** | 1.4× faster |
| Repetition score (guided) | 0.99 | **1.00** | Guidance eliminates all repetition |
| Peak VRAM (training) | 4.22 GB | **4.37 GB** | Comparable |

**Hardware**: Single NVIDIA RTX 3090 (24GB VRAM), 12GB system RAM (WSL2).

**Note on perplexity**: v4's PPL (251.3) is not directly comparable to v3's (102.6). The v4 model was trained on 100K documents (27.2M tokens) vs v3's 1M documents (272M tokens) — a 10× reduction in training data. Additionally, v4 uses ELBO-weighted loss which optimizes for generation quality at t≈0.5 rather than next-token prediction at t≈0. The forward throughput improvements (+34.5% at batch=8) and generation speed (3.1× faster) are the architecture-level gains that hold regardless of training data scale.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GENERATION PIPELINE                       │
│                                                             │
│  Prompt → MDLM (full parallel diffusion, 32 steps)          │
│         ↓                                                   │
│         Adaptive Guidance (during generation):              │
│           • Frequency penalty (0.4)                         │
│           • Repetition penalty (1.3)                        │
│           • No-repeat bigram ban                            │
│           • Top-p nucleus sampling (p=0.95)                 │
│         ↓                                                   │
│         Draft text (all tokens generated simultaneously)     │
│         ↓                                                   │
│  Qwen3-0.6B Validation (1 forward pass, teacher forcing):   │
│         ↓                                                   │
│         Per-segment log-prob scoring                        │
│         ↓                                                   │
│         Bad segments identified (below threshold)           │
│         ↓                                                   │
│  Qwen3 Segment Regeneration (AR generation for bad spans)   │
│         ↓                                                   │
│         Final coherent text                                 │
└─────────────────────────────────────────────────────────────┘
```

### MDLM-BPE v4 Model (Optimized)

```
182.8M params (-9.2% vs v3)
├── Token embedding: 10K BPE vocab → 1024 dims (weight-tied to output)
├── 10× Transformer blocks
│   ├── RoPE positional encoding (rotate_half pattern)
│   ├── AdaLN timestep conditioning (4×d_model, simplified)
│   ├── RMSNorm (not LayerNorm — faster, same quality)
│   ├── SwiGLU FFN (not GELU — better perplexity, fewer params)
│   └── Flash attention (non-causal, SDPA)
├── RMSNorm final
├── Weight-tied output projection (no bias)
└── Full-parallel unmasking (cosine schedule, all positions simultaneously)
```

**Key design choice**: The model predicts ALL positions simultaneously via iterative diffusion — no left-to-right decoding. Early testing showed that semi-autoregressive (left-to-right block) generation improved coherence slightly, but once adaptive guidance and AR-oracle validation were added, full-parallel mode achieved both higher speed AND better quality (see [Benchmark](#benchmark-measured)).

## Benchmarks

> **Protocol note (2026-09-04):** all numbers below were re-measured with a
> strict protocol (`scripts/benchmark_standard.py`): CUDA synchronize before
> and after each timed region, per-shape warmup, ≥10 repetitions, median ± IQR,
> explicit TPS = batch × seq / dt. Forward TPS use random weights (shape-
> equivalent — weights don't affect forward timing). Generation tok/s count
> **only generated tokens, excluding the prompt**. Hardware: RTX 3090, torch 2.6.0+cu124.

### Forward throughput (MDLM v4, 182.8M params, bf16, seq=128)

| Batch | TPS (median) | vs seq=256 |
|------:|-------------:|-----------:|
| 1     | ~3,700–4,000 | 8,296      |
| 8     | ~32,000–45,000 | 59,677   |
| 32    | ~109,000–130,000 | 122,512 |

Measured 2026-09-04 across three independent runs. Batch=32 is stable (IQR
<0.1 ms); batch=1 is latency-bound and noisier (IQR ~7 ms) — treat it as
~3,500–4,000 TPS, not a point estimate. Note: **fp32 beats bf16-autocast at
batch=1** (5,185 vs 3,654 TPS) — autocast per-op overhead dominates when
latency-bound.

### Generation speed (8 samples × 100 new tokens, guidance on, t=0.7)

| Method | tok/s (generated only) |
|--------|-----------------------:|
| Full-parallel (32 steps)            | ~113 |
| Semi-AR, block=32                   | ~31  |
| Full-parallel + guidance            | ~15  |
| Full-parallel + guidance + Qwen3-0.6B oracle validation | ~19 |

The Qwen3-0.6B validation pass is cheap (~1.3× slower than plain guidance)
and measurably improves oracle log-prob of the output (−11.82 → −9.88 on the
same samples). **Honest trade-off:** semi-AR block decoding is 2× faster than
guided full-parallel; the full-parallel path wins on throughput only without
guidance. There is no single "fastest" mode — pick per requirement.

### What changed vs earlier reports

Earlier numbers in this README (5,952/53,783/123,341 forward TPS; 96.5 tok/s
generation; 2.4× vs semi-AR) were measured with looser methodology: no per-
shape warmup, single runs, prompt tokens counted as generated in one branch
but not another, and fp32-vs-bf16 mixed comparison tables. Keep those numbers
only as historical context — the table above supersedes them.

## Failed Approaches (Documented)

Three HRM approaches were tested and failed. Full details in [`docs/phase10_results.md`](docs/phase10_results.md):

| Approach | Δ log-prob | Why It Failed |
|----------|:----------:|---------------|
| Semantic HRM (MDLM embeddings) | — | 201M embeddings too weak for drift detection |
| AR-Oracle guided regeneration | -0.02 | MDLM too weak to generate good replacements even with oracle bias |
| AR-Oracle token replacement | -0.71 | Cross-vocabulary token mapping breaks coherence |

The successful approach: **segment-level regeneration** by the oracle (+0.22), not token-level operations.

## Project Structure

```
latent-space-language-diffusion-model/
├── src/                        # Core modules
│   ├── mdlm_v4_optimized.py    #   ★ v4 model: RMSNorm, SwiGLU, plain-CE MDLM loss
│   ├── mdlm_bpe_v3.py          #   v3 model (legacy, for comparison)
│   ├── inference_v4.py         #   ★ v4 inference: vectorized guidance, batch gen
│   ├── logit_guidance_v4.py    #   ★ v4 guidance: torch.compile friendly
│   ├── ar_control.py           #   199M param AR control model (causal, KV cache)
│   ├── logit_guidance.py       #   v3 adaptive guidance (rep/freq/n-gram/top-p)
│   ├── hrm_refiner.py          #   RepetitionReviewer (geometric, 0 params)
│   ├── hybrid_speculative.py   #   MDLM draft + Qwen3 segment refinement
│   ├── ar_oracle_hrm.py        #   Qwen3 oracle scoring
│   ├── ar_oracle_hrm_v2.py     #   Direct token replacement (experimental)
│   └── semantic_hrm.py         #   Embedding-based coherence (experimental)
├── scripts/                    # Training & evaluation
│   ├── train_v4.py             #   ★ Train MDLM v4 from scratch
│   ├── benchmark_v4.py         #   ★ v3 vs v4 head-to-head benchmark
│   ├── convert_to_hf_v4.py     #   ★ Convert v4 checkpoint → safetensors
│   ├── train.py                #   Train MDLM v3 from scratch (legacy)
│   ├── train_ar.py             #   Train AR control model (matched params/data)
│   ├── finetune.py             #   SFT fine-tune on UltraChat
│   ├── benchmark.py            #   Full benchmark (all methods, v3)
│   ├── benchmark_parallel.py   #   Full-parallel vs semi-AR comparison
│   ├── benchmark_comparison.py #   Head-to-head: MDLM vs AR control
│   ├── download_data.py        #   Download Ultra-FineWeb
│   ├── prepare_data.py         #   Tokenize + pack into memmap
│   └── train_tokenizer.py      #   Train BPE tokenizer
├── experiments/                # Research archive (Phases 1-8)
│   ├── phase1-4_latent_diffusion/
│   ├── phase5-6_mdlm_hrm/
│   └── phase7-8_topic_agentic/
├── docs/
│   └── phase10_results.md      # Detailed HRM experiment results
└── scripts/legacy/             # Parameter sweep & ablation scripts
```

## Reproduce

### Train from scratch (v4)

```bash
# 1. Train BPE tokenizer (10K vocab)
python scripts/train_tokenizer.py

# 2. Download and prepare data
python scripts/download_data.py          # Ultra-FineWeb, 100K docs
# prepare_data is integrated into train_v4.py

# 3. Train MDLM v4 (~45 minutes on RTX 3090)
python scripts/train_v4.py --epochs 3 --batch-size 32 --seq-len 128 --accum 4

# 4. Benchmark v3 vs v4 (head-to-head)
python scripts/benchmark_v4.py

# 5. Convert to HuggingFace safetensors
python scripts/convert_to_hf_v4.py
```

### Train from scratch (v3, legacy)

```bash
python scripts/train.py --epochs 3 --batch-size 32 --seq-len 128
python scripts/benchmark.py
python scripts/benchmark_parallel.py
```

### Requirements

- Python 3.10+
- PyTorch 2.6+ with CUDA
- NVIDIA GPU (tested on RTX 3090, 24GB VRAM)
- transformers (for Qwen3 oracle)
- tokenizers (HuggingFace BPE)

```bash
pip install -r requirements.txt
```

## Research Lineage

This repo documents a complete research arc. The `experiments/` directory preserves the full history:

| Phase | Question | Outcome |
|-------|----------|---------|
| 1-4 | Can FF/score-matching generate text in SplatsDB latent space? | ❌ Energy collapse in 1024D |
| 5 | Can masked diffusion generate grammatical text? | ✅ 100% grammatical (small vocab) |
| 6 | Can HRM (Generator→Reviewer→Editor) improve quality? | ✅ 84% grammatical, 635 vocab |
| 7 | Can SplatsDB latent space control topic? | ✅ 79% on-topic, 10.3× over baseline |
| 8 | Can the system detect and fill knowledge gaps? | ✅ 100% gap detection accuracy |
| 9-10 | Can MDLM scale to coherent real text at speed? | ✅ 2.4× faster, quality-limited by scale |
| 11 | Is the quality gap architectural or just scale? | AR control wins on coherence (-2.94 vs -3.07 oracle LP), MDLM wins on speed (46 vs 37 tok/s) |

## Citation

If you use this work, please cite:

```bibtex
@misc{schwabauer2026latentspace,
  title={Latent Space Language Diffusion Model: Parallel Text Generation via Masked Diffusion with AR-Oracle Validation},
  author={Schwabauer, Brian Tomas},
  year={2026},
  publisher={GitHub},
  url={https://github.com/schwabauerbriantomas-gif/latent-space-language-diffusion-model}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- 🤗 **[HuggingFace Model Card](https://huggingface.co/brianschwabauer/latent-space-language-diffusion-model)** — open weights + full inference code
- **[SplatsDB](https://github.com/schwabauerbriantomas-gif/splatdb)** — vector memory infrastructure
- **[MDLM Logit Guidance](https://github.com/schwabauerbriantomas-gif/mdlm-logit-guidance)** — adaptive guidance module (frequency/repetition/n-gram penalties, top-p)
- **[Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb)** — training data
- **[Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B)** — AR oracle validation model
- **[MDLM](https://arxiv.org/abs/2406.03709)** — Sahoo et al. 2024, masked diffusion language model
- **[Forward-Forward Algorithm](https://arxiv.org/abs/2212.13345)** — Hinton 2022 (explored in Phase 1, refuted for generation)
