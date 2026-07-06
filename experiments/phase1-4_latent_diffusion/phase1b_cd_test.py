"""
Phase 1b: Contrastive Divergence test.

Tests whether CD (negatives from Langevin, not random) fixes the T2/T3
failures that plain FF could not. CD is the theoretically correct method
for converting a discriminative energy into a generative one (Hinton 2002).

Usage:
    python src/phase1b_cd_test.py
"""
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ff_energy import FFEnergy, LatentDiffusionSampler
from phase1_test import make_latent_data, make_sequence_data, DEVICE, DIM, _save

RESULTS_DIR = REPO / "results"


def run_cd_test():
    """Train with CD, measure T1b+T2+T3. Compare to FF baseline."""
    print("=" * 60)
    print("Phase 1b: Contrastive Divergence")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    torch.manual_seed(42)
    real, noise, centers = make_latent_data(n_samples=1000)

    results = {
        "test": "phase1b_cd",
        "timestamp": datetime.now().isoformat(),
        "method": "contrastive_divergence",
    }

    # --- Train with CD ---
    model = FFEnergy(dim=DIM, hidden=256, n_layers=3).to(DEVICE)
    print("\n[CD] Training 200 epochs, k=15 Langevin steps per epoch...")
    model.train_cd(real[:800], epochs=200, k_steps=15,
                   lr_langevin=0.05, cd_lr=0.05, verbose=True)
    model.eval()

    # --- T1: discrimination (sanity check — should still pass) ---
    print("\n--- T1: Discrimination ---")
    with torch.no_grad():
        e_real = model(real[800:]).cpu().numpy().ravel()
        e_noise = model(noise[800:]).cpu().numpy().ravel()
    labels = np.concatenate([np.zeros(len(e_real)), np.ones(len(e_noise))])
    scores = np.concatenate([e_real, e_noise])
    auroc = roc_auc_score(labels, scores)
    margin = float(e_noise.mean() - e_real.mean())
    print(f"  E(real)={e_real.mean():.4f}  E(noise)={e_noise.mean():.4f}  margin={margin:.4f}")
    print(f"  AUROC: {auroc:.4f}  {'✅ PASS' if auroc > 0.85 else '❌ FAIL'}")
    results["t1_auroc"] = float(auroc)
    results["t1_margin"] = margin

    # --- T2: score gradient direction (THE CRITICAL TEST) ---
    print("\n--- T2: Score gradient direction ---")
    n_test = 200
    x = torch.randn(n_test, DIM, device=DEVICE) * 2.0
    real_ref = real[:500]
    lr = 0.01
    distances = []
    for step in range(50):
        score = model.score(x)
        x = x - 0.5 * lr * score + np.sqrt(2 * lr) * torch.randn_like(x)
        if step % 10 == 0 or step == 49:
            with torch.no_grad():
                dist = torch.cdist(x, real_ref).min(dim=1)[0]
                d = {"step": step, "median": float(dist.median()),
                     "mean": float(dist.mean())}
                distances.append(d)
                print(f"  step {step:3d}: median={d['median']:.4f} mean={d['mean']:.4f}")
    medians = [d["median"] for d in distances]
    decrease = (medians[0] - medians[-1]) / medians[0]
    results["t2_decrease"] = decrease
    results["t2_distances"] = distances
    results["t2_verdict"] = "PASS" if decrease > 0.15 else "FAIL"
    print(f"  Distance decrease: {decrease*100:.1f}%  {'✅ PASS' if decrease > 0.15 else '❌ FAIL'} (>15%)")

    # --- T3: sampling near data ---
    print("\n--- T3: Sampling near data ---")
    sampler = LatentDiffusionSampler(model, dim=DIM, n_steps=100, lr=0.01)
    samples = sampler.sample(200, device=DEVICE)
    with torch.no_grad():
        s_cpu = samples.cpu()
        dist_to_real = torch.cdist(s_cpu, real[:500].cpu()).min(dim=1)[0]
        near_ratio = (dist_to_real < 1.0).float().mean().item()
        sample_std = s_cpu.std(dim=0).mean().item()
        pairwise = torch.cdist(s_cpu[:50], s_cpu[:50])
        pairwise_mean = pairwise[pairwise > 0].mean().item()
    results["t3_near_ratio"] = near_ratio
    results["t3_sample_std"] = sample_std
    results["t3_verdict"] = "PASS" if near_ratio > 0.3 else "FAIL"
    print(f"  Near real (<1.0): {near_ratio:.3f}  {'✅ PASS' if near_ratio > 0.3 else '❌ FAIL'} (>30%)")
    print(f"  Sample std: {sample_std:.4f}  (diversity)")

    # --- Verdict ---
    t2_pass = decrease > 0.15
    t3_pass = near_ratio > 0.3
    if t2_pass and t3_pass:
        verdict = "CD FIXES THE GRADIENT PROBLEM — diffusion viable"
    elif t2_pass:
        verdict = "CD fixes gradient direction but samples still miss data"
    else:
        verdict = "CD does NOT fix the gradient problem — fundamental limitation"
    results["final_verdict"] = verdict
    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*60}")

    _save(results, "phase1b_cd")
    return model


if __name__ == "__main__":
    run_cd_test()
