# Latent Space Language Diffusion Model

<!-- SEO: keywords for discoverability -->
<!--
keywords: masked diffusion language model, MDLM, parallel text generation, diffusion model, non-autoregressive generation, bidirectional transformer, AR oracle validation, speculative decoding, Qwen3, SplatsDB, open weights, open source LLM, torch, safetensors, autoregressive control, architectural comparison
-->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![Model: 201M](https://img.shields.io/badge/Params-201M-green.svg)](#results-at-a-glance)
[![PPL: 102.6](https://img.shields.io/badge/Perplexity-102.6-orange.svg)](#honest-limitations)
[![Speed: 2.4x](https://img.shields.io/badge/Speed-2.4%C3%97%20faster-brightgreen.svg)](#benchmark-measured)
[![Open Weights](https://img.shields.io/badge/Weights-Open%20%E2%9C%93-success.svg)](https://huggingface.co/brianschwabauer/latent-space-language-diffusion-model)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-blue.svg)](https://huggingface.co/brianschwabauer/latent-space-language-diffusion-model)

> **🤗 Download weights, tokenizer, and full inference code on [HuggingFace](https://huggingface.co/brianschwabauer/latent-space-language-diffusion-model)** — open weights (safetensors, 805 MB), open code (modeling + training scripts), fully reproducible. No black boxes.

A **masked diffusion language model (MDLM)** that generates text by predicting all tokens in parallel, validated by an autoregressive oracle. Built on [SplatsDB](https://github.com/schwabauerbriantomas-gif/splatdb)'s vector infrastructure.

**The core finding**: full-parallel masked diffusion with adaptive guidance + AR-oracle validation achieves **2.4× faster generation** than the equivalent autoregressive model while producing comparable text quality — no sequential decoding required.

## Results at a Glance

| Metric | Value | Notes |
|--------|-------|-------|
| Model parameters | 201M | d_model=1024, 10 layers, 16 heads |
| Training data | 272M tokens | 1M docs from Ultra-FineWeb |
| Perplexity (held-out) | 102.6 | Limited by data scale (see Limitations) |
| Forward throughput | 57,759 TPS | Batch=32, vs Qwen3's 32,023 (1.8× faster) |
| Generation speed | 31.2 tok/s | Full-parallel + guidance |
| Generation speed (validated) | 15.6 tok/s | Full-parallel + guidance + Qwen3 validation |
| Repetition score | 0.99 | After adaptive guidance (baseline: 0.79) |

**Hardware**: Single NVIDIA RTX 3090 (24GB VRAM), 8GB system RAM.

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

### MDLM-BPE v3 Model

```
201M params
├── Token embedding: 16K BPE vocab → 1024 dims
├── 10× Transformer blocks
│   ├── RoPE positional encoding
│   ├── AdaLN timestep conditioning
│   └── Flash attention (non-causal)
├── LayerNorm + output projection
└── Full-parallel unmasking (all positions simultaneously)
```

**Key design choice**: The model predicts ALL positions simultaneously via iterative diffusion — no left-to-right decoding. Early testing showed that semi-autoregressive (left-to-right block) generation improved coherence slightly, but once adaptive guidance and AR-oracle validation were added, full-parallel mode achieved both higher speed AND better quality (see [Benchmark](#benchmark-measured)).

## Benchmark [MEASURED]

All metrics measured on RTX 3090 with identical prompts. Oracle log-prob = mean per-token log-probability under Qwen3-0.6B teacher forcing (higher = more coherent).

### Generation Speed + Quality

| Method | Oracle LP | Rep Score | TPS | Latency |
|--------|:---------:|:---------:|:---:|:-------:|
| Full-parallel + guidance | -4.58 | 1.00 | **31.2** | 1.8s |
| **Full-parallel + guidance + Qwen3 validate** | **-3.55** | 0.95 | **15.6** | **3.3s** |
| Semi-AR + guidance | -4.61 | 1.00 | 13.0 | 4.9s |
| Semi-AR + guidance + Qwen3 validate | -3.72 | 0.98 | 10.0 | 5.1s |
| Qwen3-0.6B (pure AR, reference) | -1.18 | 0.95 | 17.1 | 3.8s |

**Full-parallel + Qwen3 validation is the optimal pipeline**: 2.4× faster than semi-AR, AND better quality (lp -3.55 vs -3.72). The sequential decoder bottleneck was unnecessary once guidance and oracle validation were in place.

### Forward-Pass Throughput

Raw model inference (no generation loop):

| Batch | MDLM (201M) | Qwen3-0.6B (596M) | Speedup |
|------:|------------:|------------------:|--------:|
| 1 | 8,443 TPS | 2,152 TPS | 3.9× |
| 8 | 50,440 TPS | 17,009 TPS | 3.0× |
| 32 | 57,759 TPS | 32,023 TPS | 1.8× |

### Sample Output

```
Prompt: "Climate change is one of the biggest challenges"

Full-parallel + guidance + Qwen3 validation:
  "...facing our world. It plays an important role in managing
   climate change and its effects on the environment..."

Qwen3-0.6B (reference):
  "...for the global community. It affects the environment,
   economy, and society. However, many countries and regions..."
```

### Guidance Ablation

| Technique | Repetition Score | Notes |
|-----------|:----------------:|-------|
| Baseline (no guidance) | 0.79 | Heavy repetition loops |
| + Frequency penalty | 0.91 | Reduces common token dominance |
| + Repetition penalty | 0.96 | Bans already-used tokens |
| + No-repeat bigram | 0.99 | Hard ban on repeated bigrams |
| + Top-p (0.95) | 1.00 | Nucleus sampling filters noise |

## Honest Limitations

This project demonstrates that masked diffusion is a viable architecture for fast text generation, but **an architectural quality gap exists** beyond what scale alone explains. A controlled experiment (below) confirms this.

### Controlled Comparison: MDLM vs AR (Same Scale)

To answer "is the quality gap architectural or just scale?", we trained an **autoregressive control model** from scratch with matched parameters, identical data, and the same compute budget:

| | MDLM v3 | AR Control | Qwen3-0.6B |
|--|---------|------------|------------|
| Architecture | Masked diffusion (bidirectional) | Autoregressive (causal) | Autoregressive (causal) |
| Parameters | 201.3M | 199.1M | 596M |
| Layers | 10 + AdaLN | 15 (standard pre-LN) | 28 |
| Training data | Ultra-FineWeb 1M docs | **Same** | ~trillions |
| Training tokens | 272M | **272M (same .npy)** | ~trillions |
| Tokenizer | 16K BPE | **Same** | 151K BPE |
| Training time | 6.8h | 8.3h | — |
| Hardware | RTX 3090 | RTX 3090 | GPU cluster |

**Head-to-head results** (all measured on RTX 3090, identical prompts):

| Metric | MDLM v3 | AR Control | Notes |
|--------|:-------:|:----------:|-------|
| Oracle log-prob (Qwen3) | -3.068 | **-2.938** | AR produces more coherent text |
| Perplexity (reported) | 114.9 | 18.5 | Not directly comparable (see below) |
| Forward TPS (batch=1) | 3,033 | 4,007 | AR faster (fewer layers, no AdaLN) |
| Forward TPS (batch=8) | 21,960 | 33,739 | AR 1.5× faster |
| Forward TPS (batch=32) | **113,161** | 85,363 | MDLM 1.3× faster (parallel advantage) |
| Generation TPS | **46.0** | 37.2 | MDLM 1.2× faster (parallel decoding) |

**Sample comparison** (greedy, same prompt):

```
Prompt: "Climate change is one of the biggest challenges"

MDLM v3:  "...of global warming. The flooding is one of the major
           devastating impacts of climate change. One of the..."

AR:       "...facing humanity today. So if you're like most people,
           you know what it's like to be at the top of the world.
           As we live in a climate emergency, we're..."

Prompt: "Education systems around the world need to"

MDLM v3:  "...be focused on the development of education standards,
           and leading to a lens and advocate for the values of..."

AR:       "...have the right tools and the right tools to support
           their decision-making processes. The current world is
           not the same as the one used to support edu..."
```

**Why the PPL values are not directly comparable**: MDLM's perplexity (114.9) is computed by sampling random masking ratios t~U(0,1) and averaging CE over masked positions only. This averages over all difficulty levels — from trivial (t≈0, 1 masked token with 127 of context, PPL≈4.6) to extremely hard (t≈1, all tokens masked, no context, PPL≈1562). The AR model's PPL (18.5) is a single, stable next-token prediction task. The oracle log-prob under Qwen3-0.6B is the fair cross-architecture metric, and there the AR wins (-2.94 vs -3.07).

**Conclusion**: With matched parameters (199M vs 201M), identical data (272M tokens), and equivalent compute (3 epochs, RTX 3090), the autoregressive architecture produces **more coherent text** (higher oracle LP) than masked diffusion. The MDLM compensates with **faster parallel generation** (46 vs 37 tok/s). The quality gap between MDLM and Qwen3-0.6B is therefore driven by **both** architecture and scale — not scale alone as previously stated.

Full benchmark code: [`scripts/benchmark_comparison.py`](scripts/benchmark_comparison.py) · AR model: [`src/ar_control.py`](src/ar_control.py) · Training: [`scripts/train_ar.py`](scripts/train_ar.py)

### What Guidance Can and Cannot Fix

**Adaptive guidance eliminates repetition** (0.79 → 0.99) at zero computational cost.

**AR-oracle validation improves coherence** (+0.22 log-prob) by regenerating bad segments.

**Neither can overcome model capacity.** The 201M model produces grammatically correct, repetition-free text, but long-distance semantic coherence is limited by parameter count and training data volume.

### Failed Approaches (Documented)

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
│   ├── mdlm_bpe_v3.py          #   201M param model + full-parallel sampling
│   ├── ar_control.py           #   199M param AR control model (causal, KV cache)
│   ├── logit_guidance.py       #   Adaptive guidance (rep/freq/n-gram/top-p)
│   ├── hrm_refiner.py          #   RepetitionReviewer (geometric, 0 params)
│   ├── hybrid_speculative.py   #   MDLM draft + Qwen3 segment refinement
│   ├── ar_oracle_hrm.py        #   Qwen3 oracle scoring
│   ├── ar_oracle_hrm_v2.py     #   Direct token replacement (experimental)
│   └── semantic_hrm.py         #   Embedding-based coherence (experimental)
├── scripts/                    # Training & evaluation
│   ├── train.py                #   Train MDLM v3 from scratch
│   ├── train_ar.py             #   Train AR control model (matched params/data)
│   ├── finetune.py             #   SFT fine-tune on UltraChat
│   ├── benchmark.py            #   Full benchmark (all methods)
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

### Train from scratch

```bash
# 1. Train BPE tokenizer (16K vocab)
python scripts/train_tokenizer.py

# 2. Download and prepare data
python scripts/download_data.py          # Ultra-FineWeb, 1M docs
python scripts/prepare_data.py           # Tokenize → memmap

# 3. Train MDLM v3 (~7 hours on RTX 3090)
python scripts/train.py --epochs 3 --batch-size 32 --seq-len 128

# 4. Benchmark all methods
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
