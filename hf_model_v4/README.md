---
language:
  - en
library_name: safetensors
license: mit
tags:
  - masked-diffusion
  - text-generation
  - parallel-decoding
  - mdlm
  - diffusion
  - non-autoregressive
  - bidirectional-transformer
  - ar-oracle
  - qwen3
  - open-weights
  - reproducible
metrics:
  - perplexity
pipeline_tag: text-generation
---

# Latent Space Language Diffusion Model (MDLM v4)

**Open weights. Open code. Fully reproducible.**

A **182,795,264-parameter** masked diffusion language model that generates text by predicting all positions in parallel with iterative cosine-schedule unmasking. Trained on a single RTX 3090. Research artifact, not a production model — quality is far below production LLMs, and we say so with numbers below.

## ⚠️ Release v4.1 — what changed and why

The previous weights file in this repo was an **accidental mid-training checkpoint** (train loss ≈ 14.9, perplexity in the millions) from a diverged data-scaling run. It has been replaced with a verified checkpoint. We keep this note so nobody downloads old revisions by mistake:

- Old `model.safetensors` (sha256 `3b69d43a…d5d`, 2026-08-04): **broken** — generates incoherent text.
- New `model.safetensors` (sha256 `e4b3dafedf5a70403dc2afc99de1f57abf3da3e5e156a4b469107fcb6685c605`, 2026-10-19): verified — see measured metrics below.

The training code that produced the diverged runs weighted the loss by `1/(1-t)` (ELBO importance weighting). With logit-normal `t`, rare but extreme weights (up to ×100,000 after clamping) dominated gradients; two long runs diverged this way (logs preserved in the GitHub repo: `results/v4_train_final.log`, `results/v4_train_500k.log`). A controlled ablation (`scripts/diag_v4_loss.py`, receipt in `results/diag_v4_loss.json`) isolated the weighting as the cause, and this checkpoint was trained with **plain cross-entropy on masked positions** ((loss_type='plain' in scripts/train_v4.py)).

## 📊 Measured results (this exact checkpoint)

| Metric | Value | Notes |
|--------|-------|-------|
| Parameters | 182,795,264 (182.8M) | exact, counted from safetensors header |
| Architecture | d_model=1024, 10 layers, 16 heads, d_ff=2816 | RMSNorm + SwiGLU + RoPE + weight tying |
| Vocab | 10,000 BPE | tokenizer included |
| Training data | 27.2M tokens (100K docs, Ultra-FineWeb) | deliberately small; single-GPU budget |
| Perplexity (plain CE, masked positions) | **62.4** | measured on held-out data at release time |
| Forward throughput, batch 1 / 8 / 32 | 6,024 / 56,225 / 130,782 TPS | RTX 3090, seq=128 |
| Full-parallel generation | ~58–113 tok/s (12 samples, no guidance) | 32 diffusion steps, seq=128 |
| Training time | ~50 min | single RTX 3090 |

## 📏 Honest speed framing

Parallel decoding here is **not** free speed against a tuned autoregressive baseline:

- vs our own semi-AR decoding variant: **2.4× faster** (measured, v3-era pipeline)
- vs an equal-size AR control (199M, KV cache, greedy): **1.15× faster** (42.6 vs 37.2 TPS)
- The real advantage is the *throughput profile* (all positions in one forward) and the architecture's headroom — see limitations.

We cite the AR-control number because it is the comparison a reviewer actually cares about.

## 🚧 Known limitations (measured, not vibes)

- **Quality**: an equal-size AR control trained with 30× more tokens reaches perplexity ~18 vs our ~62.4. This model demonstrates an architecture, not a product.
- **Data**: 27.2M training tokens (vs trillions for production models).
- **Prompt adherence**: unconditional-style generation; there is no instruction tuning.
- Known failed experiments (kept for transparency): embedding-based drift detection, token-level oracle replacement, guided-MDLM regeneration — see GitHub repo docs.

## 🚀 Usage (tested end-to-end)

```python
import sys, torch
sys.path.insert(0, ".")                     # repo root
from safetensors.torch import load_file
from mdlm_v4_optimized import MDLMBPEV4, MDLMConfig, BPETokenizer, sample_full_parallel_v4

cfg = MDLMConfig(vocab_size=10000, d_model=1024, n_heads=16,
                 n_layers=10, max_seq_len=256, d_ff=2816, dropout=0.1)
model = MDLMBPEV4(cfg)
model.load_state_dict(load_file("model.safetensors"), strict=True)   # exact fit verified
model.eval()

tok = BPETokenizer("tokenizer.json")
out = sample_full_parallel_v4(
    model, tok,
    prompt_ids=tok.encode("The future of artificial intelligence", add_special=False),
    seq_len=128, n_samples=1, n_steps=32, temperature=0.7,
    device="cuda",
)
print(out[0])
```

> Note: `modeling_mdlm.py` contains the older **v3** architecture (201M, vocab 16K) kept for the training-lineage benchmarks; it does **not** load the v4 weights above. Use `mdlm_v4_optimized.py` as shown.

## 📦 What's included

| File | Description |
|------|-------------|
| `model.safetensors` | v4 weights (182.8M params, fp32, 731 MB) |
| `config.json` | architecture + training metadata |
| `tokenizer.json` | 10K BPE tokenizer |
| `mdlm_v4_optimized.py` | v4 model + training loss + full-parallel sampler |
| `inference_v4.py` | batched generation with adaptive guidance |
| `logit_guidance.py` | frequency/rep/n-gram/top-p guidance |
| `train_v4.py` (GitHub) | full training script |
| `results/` (GitHub) | every benchmark receipt, including the failures |

## 📚 Citation

```bibtex
@misc{schwabauer2026latentspace,
  title={Latent Space Language Diffusion Model: Parallel Text Generation via Masked Diffusion with AR-Oracle Validation},
  author={Schwabauer, Brian Tomas},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/brianschwabauer/latent-space-language-diffusion-model}
}
```

## 📜 License

**MIT** — use, modify, distribute freely.

## 🔗 Related

- [GitHub repository](https://github.com/schwabauerbriantomas-gif/latent-space-language-diffusion-model) — full source, training logs, receipts
- [Sahoo et al. 2024 (MDLM)](https://arxiv.org/abs/2406.03709) — the masked-diffusion formulation this follows
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — AR oracle used for validation experiments
