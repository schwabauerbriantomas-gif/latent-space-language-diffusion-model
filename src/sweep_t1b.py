"""
Hyperparameter sweep for T1b: FF energy discrimination of coherent vs
incoherent token sequences.

Baseline: AUROC 0.663 (FAIL, need >0.85).

Saves one JSON line per config to results/autoresearch_sweep.jsonl.
"""
import argparse
import json
import sys
import time
import itertools
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ff_energy import FFEnergy, FFLayer
from phase1_test import make_sequence_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 1024
RESULTS_PATH = REPO / "results" / "autoresearch_sweep.jsonl"


def measure_auroc(model, real_seq, noise_seq):
    """Measure T1b AUROC on held-out sequences."""
    model.eval()
    with torch.no_grad():
        e_real = model(real_seq).cpu().numpy().ravel()
        e_noise = model(noise_seq).cpu().numpy().ravel()
    labels = np.concatenate([np.zeros(len(e_real)), np.ones(len(e_noise))])
    scores = np.concatenate([e_real, e_noise])
    auroc = roc_auc_score(labels, scores)
    margin = float(e_noise.mean() - e_real.mean())
    return float(auroc), margin, float(e_real.mean()), float(e_noise.mean())


def run_config(hidden, n_layers, epochs, threshold_pos, threshold_neg, lr, seq_len,
               n_samples=1000, seed=42):
    """Run a single config, return result dict."""
    torch.manual_seed(seed)
    t0 = time.time()

    # Generate data
    real_seq, noise_seq, centers = make_sequence_data(
        n_samples=n_samples, seq_len=seq_len, dim=DIM)
    n_train = int(0.8 * n_samples)
    real_train, real_test = real_seq[:n_train], real_seq[n_train:]
    noise_train, noise_test = noise_seq[:n_train], noise_seq[n_train:]

    # Build model with custom thresholds/lr per layer
    model = FFEnergy(dim=DIM, hidden=hidden, n_layers=n_layers).to(DEVICE)
    for layer in model.layers:
        layer.threshold_pos = threshold_pos
        layer.threshold_neg = threshold_neg
        layer.lr = lr

    # Train
    model.train_ff(real_train, noise_train, epochs=epochs, verbose=False)

    # Evaluate
    auroc, margin, e_real, e_noise = measure_auroc(model, real_test, noise_test)
    elapsed = time.time() - t0

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "hidden": hidden,
        "n_layers": n_layers,
        "epochs": epochs,
        "threshold_pos": threshold_pos,
        "threshold_neg": threshold_neg,
        "lr": lr,
        "seq_len": seq_len,
        "seed": seed,
        "auroc": auroc,
        "margin": margin,
        "e_real_mean": e_real,
        "e_noise_mean": e_noise,
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
    parser.add_argument("--mode", default="main",
                        choices=["main", "diagnostic", "full"])
    args = parser.parse_args()

    print(f"Device: {DEVICE}, Dim: {DIM}")
    print(f"Results: {RESULTS_PATH}\n")

    if args.mode == "main":
        # Main sweep: focused grid covering all requested params
        configs = build_main_grid()
    elif args.mode == "diagnostic":
        # Diagnostic: per-token energy + structural variants
        configs = build_diagnostic_grid()
    else:
        configs = build_main_grid() + build_diagnostic_grid()

    print(f"Running {len(configs)} configs...\n")
    best_auroc = 0
    best_config = None

    for i, cfg in enumerate(configs):
        label = (f"hidden={cfg['hidden']} layers={cfg['n_layers']} "
                 f"ep={cfg['epochs']} thr=({cfg['threshold_pos']},{cfg['threshold_neg']}) "
                 f"lr={cfg['lr']} seq={cfg['seq_len']}")
        print(f"[{i+1}/{len(configs)}] {label}")
        sys.stdout.flush()

        try:
            result = run_config(**cfg)
            append_result(result)
            tag = " *** PASS ***" if result["pass"] else ""
            print(f"   → AUROC={result['auroc']:.4f} margin={result['margin']:.4f} "
                  f"({result['elapsed_s']:.1f}s){tag}\n")
            if result["auroc"] > best_auroc:
                best_auroc = result["auroc"]
                best_config = result
        except Exception as e:
            print(f"   → ERROR: {e}\n")
            append_result({"error": str(e), "config": cfg,
                           "timestamp": datetime.utcnow().isoformat()})

    print("=" * 60)
    print(f"BEST: AUROC={best_auroc:.4f}")
    print(f"  hidden={best_config['hidden']} layers={best_config['n_layers']} "
          f"ep={best_config['epochs']} thr=({best_config['threshold_pos']},{best_config['threshold_neg']}) "
          f"lr={best_config['lr']} seq={best_config['seq_len']}")
    print(f"  PASS: {best_auroc > 0.85}")


def build_main_grid():
    """Focused grid: vary one or two params at a time from baseline."""
    baseline = dict(hidden=256, n_layers=3, epochs=80,
                    threshold_pos=0.1, threshold_neg=0.3, lr=0.5, seq_len=16)

    configs = []
    # Start with baseline to confirm reproducibility
    configs.append(dict(baseline))

    # 1. Hidden dim sweep (keep rest at baseline)
    for h in [128, 256, 512, 1024]:
        c = dict(baseline); c["hidden"] = h
        if c not in configs: configs.append(c)

    # 2. n_layers sweep
    for nl in [1, 2, 3, 4]:
        c = dict(baseline); c["n_layers"] = nl
        if c not in configs: configs.append(c)

    # 3. Epochs sweep
    for ep in [80, 200, 500]:
        c = dict(baseline); c["epochs"] = ep
        if c not in configs: configs.append(c)

    # 4. Threshold sweep
    for (tp, tn) in [(0.05, 0.2), (0.1, 0.3), (0.05, 0.5)]:
        c = dict(baseline); c["threshold_pos"] = tp; c["threshold_neg"] = tn
        if c not in configs: configs.append(c)

    # 5. LR sweep
    for lr in [0.1, 0.5, 1.0]:
        c = dict(baseline); c["lr"] = lr
        if c not in configs: configs.append(c)

    # 6. seq_len sweep
    for sl in [8, 16, 32]:
        c = dict(baseline); c["seq_len"] = sl
        if c not in configs: configs.append(c)

    # 7. Promising combos: high epochs + high lr + large hidden
    for h, ep, lr in [(512, 500, 1.0), (1024, 500, 1.0), (512, 200, 0.5)]:
        c = dict(baseline); c["hidden"] = h; c["epochs"] = ep; c["lr"] = lr
        if c not in configs: configs.append(c)

    # 8. Best threshold + high epochs + seq_len=32
    for tp, tn in [(0.05, 0.5), (0.05, 0.2)]:
        c = dict(baseline); c["threshold_pos"] = tp; c["threshold_neg"] = tn
        c["epochs"] = 500; c["seq_len"] = 32
        if c not in configs: configs.append(c)

    return configs


def build_diagnostic_grid():
    """Diagnostic experiments to test the mean-pooling hypothesis."""
    # These will be handled specially in main() via diagnostic mode
    return []


if __name__ == "__main__":
    main()
