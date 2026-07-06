"""
Phase 2: Focused experiments after phase 1 revealed epochs is the dominant factor.

Key phase-1 findings:
- 80 epochs → AUROC ~0.66 (FAIL)
- 500 epochs → AUROC ~0.98-0.99 (PASS)
- hidden=128 surprisingly good (0.80 at only 80 ep)
- Best: hidden=256, layers=3, ep=500, thr=(0.05,0.2), lr=0.5, seq=32 → 0.9994

This script:
1. Epochs dose-response curve (100, 200, 300, 400, 500, 700, 1000)
2. Mean-pooling diagnostic: does per-token energy aggregation help?
3. Seed robustness: run best config with 3 seeds
4. seq_len interaction with epochs
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import argparse

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ff_energy import FFEnergy
from phase1_test import make_sequence_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 1024
RESULTS_PATH = REPO / "results" / "autoresearch_sweep.jsonl"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def measure_auroc(model, real_seq, noise_seq, agg="mean_pool"):
    """Measure T1b AUROC with different aggregation strategies."""
    model.eval()
    # Ensure 3D shape [batch, seq_len, dim]
    real_seq = real_seq.reshape(real_seq.shape[0], -1, DIM)
    noise_seq = noise_seq.reshape(noise_seq.shape[0], -1, DIM)
    with torch.no_grad():
        if agg == "mean_pool":
            # Default: mean-pool sequence then compute energy (as in FFEnergy.forward)
            e_real = model(real_seq).cpu().numpy().ravel()
            e_noise = model(noise_seq).cpu().numpy().ravel()
        elif agg == "sum_token_energies":
            # Compute energy per token, then sum across sequence
            e_real = _sum_token_energy(model, real_seq)
            e_noise = _sum_token_energy(model, noise_seq)
        elif agg == "max_token_energy":
            # Max token energy across sequence
            e_real = _max_token_energy(model, real_seq)
            e_noise = _max_token_energy(model, noise_seq)
        elif agg == "var_token_energy":
            # Variance of per-token energies — coherent should be LOW variance
            # (all from same cluster), incoherent should be HIGH variance
            e_real = _var_token_energy(model, real_seq)
            e_noise = _var_token_energy(model, noise_seq)
        else:
            raise ValueError(f"Unknown agg: {agg}")

    labels = np.concatenate([np.zeros(len(e_real)), np.ones(len(e_noise))])
    scores = np.concatenate([e_real, e_noise])
    auroc = roc_auc_score(labels, scores)
    margin = float(np.mean(e_noise) - np.mean(e_real))
    return float(auroc), margin


def _sum_token_energy(model, seq):
    """seq: [batch, seq_len, dim] → [batch] sum of per-token energies."""
    B, S, D = seq.shape
    energies = []
    for t in range(S):
        e = model(seq[:, t, :])  # [batch]
        energies.append(e)
    return torch.stack(energies, dim=1).sum(dim=1).cpu().numpy().ravel()


def _max_token_energy(model, seq):
    B, S, D = seq.shape
    energies = []
    for t in range(S):
        e = model(seq[:, t, :])
        energies.append(e)
    return torch.stack(energies, dim=1).max(dim=1)[0].cpu().numpy().ravel()


def _var_token_energy(model, seq):
    B, S, D = seq.shape
    energies = []
    for t in range(S):
        e = model(seq[:, t, :])
        energies.append(e)
    return torch.stack(energies, dim=1).var(dim=1).cpu().numpy().ravel()


def run_config(hidden, n_layers, epochs, threshold_pos, threshold_neg, lr, seq_len,
               n_samples=1000, seed=42, agg="mean_pool", tag=""):
    torch.manual_seed(seed)
    t0 = time.time()

    real_seq, noise_seq, _ = make_sequence_data(
        n_samples=n_samples, seq_len=seq_len, dim=DIM)
    n_train = int(0.8 * n_samples)
    real_train, real_test = real_seq[:n_train], real_seq[n_train:]
    noise_train, noise_test = noise_seq[:n_train], noise_seq[n_train:]

    model = FFEnergy(dim=DIM, hidden=hidden, n_layers=n_layers).to(DEVICE)
    for layer in model.layers:
        layer.threshold_pos = threshold_pos
        layer.threshold_neg = threshold_neg
        layer.lr = lr

    model.train_ff(real_train, noise_train, epochs=epochs, verbose=False)

    auroc, margin = measure_auroc(model, real_test, noise_test, agg=agg)
    elapsed = time.time() - t0

    result = {
        "timestamp": now_iso(),
        "hidden": hidden, "n_layers": n_layers, "epochs": epochs,
        "threshold_pos": threshold_pos, "threshold_neg": threshold_neg,
        "lr": lr, "seq_len": seq_len, "seed": seed, "agg": agg,
        "tag": tag,
        "auroc": auroc, "margin": margin,
        "elapsed_s": round(elapsed, 1),
        "pass": auroc > 0.85,
    }
    return result


def append_result(result):
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["epochs_curve", "aggregation", "seed_robustness",
                                 "seq_len_epochs", "all"])
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    phases = ["epochs_curve", "aggregation", "seed_robustness", "seq_len_epochs"] \
             if args.phase == "all" else [args.phase]

    for phase in phases:
        print(f"\n{'='*60}")
        print(f"PHASE: {phase}")
        print(f"{'='*60}")

        if phase == "epochs_curve":
            run_epochs_curve()
        elif phase == "aggregation":
            run_aggregation_comparison()
        elif phase == "seed_robustness":
            run_seed_robustness()
        elif phase == "seq_len_epochs":
            run_seq_len_epochs()


def run_epochs_curve():
    """Epochs dose-response at baseline config."""
    print("\nEpochs dose-response (hidden=256, layers=3, lr=0.5, seq=16)...\n")
    for ep in [50, 100, 150, 200, 300, 400, 500, 700, 1000]:
        r = run_config(hidden=256, n_layers=3, epochs=ep,
                       threshold_pos=0.1, threshold_neg=0.3, lr=0.5,
                       seq_len=16, tag="epochs_curve")
        append_result(r)
        print(f"  ep={ep:4d}: AUROC={r['auroc']:.4f} margin={r['margin']:.4f}")


def run_aggregation_comparison():
    """Test if different sequence aggregation strategies help.

    Key question: does mean-pooling destroy structure?
    If per-token variance discrimination works, that means the FF model
    DOES learn per-token energy but mean-pooling averages it out.
    """
    print("\nAggregation strategy comparison (hidden=256, layers=3, ep=500)...\n")
    aggs = ["mean_pool", "sum_token_energies", "max_token_energy", "var_token_energy"]
    for agg in aggs:
        r = run_config(hidden=256, n_layers=3, epochs=500,
                       threshold_pos=0.1, threshold_neg=0.3, lr=0.5,
                       seq_len=16, agg=agg, tag="aggregation")
        append_result(r)
        print(f"  {agg:25s}: AUROC={r['auroc']:.4f} margin={r['margin']:.4f}")


def run_seed_robustness():
    """Run best config with multiple seeds to check robustness."""
    print("\nSeed robustness for best config...\n")
    for seed in [42, 123, 7, 999, 2024]:
        r = run_config(hidden=256, n_layers=3, epochs=500,
                       threshold_pos=0.05, threshold_neg=0.2, lr=0.5,
                       seq_len=32, seed=seed, tag="seed_robustness")
        append_result(r)
        print(f"  seed={seed:4d}: AUROC={r['auroc']:.4f} margin={r['margin']:.4f}")


def run_seq_len_epochs():
    """How does seq_len interact with epochs?"""
    print("\nseq_len × epochs interaction...\n")
    for sl in [4, 8, 16, 32]:
        for ep in [80, 500]:
            r = run_config(hidden=256, n_layers=3, epochs=ep,
                           threshold_pos=0.1, threshold_neg=0.3, lr=0.5,
                           seq_len=sl, tag="seq_len_epochs")
            append_result(r)
            print(f"  seq={sl:2d} ep={ep:3d}: AUROC={r['auroc']:.4f} margin={r['margin']:.4f}")


if __name__ == "__main__":
    main()
