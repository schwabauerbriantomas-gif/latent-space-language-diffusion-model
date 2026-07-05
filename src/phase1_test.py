"""
Phase 1 empirical tests for FF-SplatDiffusion.

Three independent pass/fail tests. Each takes <15 min on RTX 3090.
If all 3 fail → the approach is empirically refuted (with data, not theory).

Usage:
    python src/phase1_test.py --test t1   # FF energy discrimination
    python src/phase1_test.py --test t2   # score gradient direction
    python src/phase1_test.py --test t3   # sampling degeneracy
    python src/phase1_test.py --test all  # run all three
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ff_energy import FFEnergy, LatentDiffusionSampler

RESULTS_DIR = REPO / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 1024  # bge-m3 embedding dimension


# ---------------------------------------------------------------------------
# Data generation — simulate SplatsDB latent space
# ---------------------------------------------------------------------------
def make_latent_data(n_samples: int = 1000, n_clusters: int = 20, dim: int = DIM):
    """Simulate SplatsDB latent space: data lives in clusters (splats).

    Real data: points near cluster centers (like tokens near their splat μ).
    Noise: points sampled from wide Gaussian (like random token sequences).
    """
    torch.manual_seed(42)
    # Cluster centers = "splat means"
    centers = torch.randn(n_clusters, dim) * 0.5

    # Real data: pick a cluster, sample near it
    real = []
    for _ in range(n_samples):
        c = centers[torch.randint(0, n_clusters, (1,))]
        point = c + torch.randn(dim) * 0.3  # tight around center
        real.append(point)
    real = torch.stack(real)

    # Noise: wide Gaussian (uniform-ish over the space)
    noise = torch.randn(n_samples, dim) * 2.0

    return real.to(DEVICE), noise.to(DEVICE), centers.to(DEVICE)


def make_sequence_data(n_samples: int = 1000, seq_len: int = 16, dim: int = DIM):
    """Simulate token SEQUENCES (to test if FF captures sequential structure).

    Real sequences: tokens drawn from the SAME cluster (coherent topic).
    Noise sequences: tokens drawn from DIFFERENT clusters (incoherent).
    """
    torch.manual_seed(42)
    n_clusters = 20
    centers = torch.randn(n_clusters, dim) * 0.5

    real_seqs = []
    for _ in range(n_samples):
        c = centers[torch.randint(0, n_clusters, (1,))]
        # All tokens from same cluster = coherent
        seq = c + torch.randn(seq_len, dim) * 0.2
        real_seqs.append(seq)

    noise_seqs = []
    for _ in range(n_samples):
        # Each token from random cluster = incoherent
        seq = []
        for _ in range(seq_len):
            c = centers[torch.randint(0, n_clusters, (1,))]
            seq.append(c + torch.randn(dim) * 0.2)
        noise_seqs.append(torch.stack(seq))

    return (torch.stack(real_seqs).to(DEVICE),
            torch.stack(noise_seqs).to(DEVICE),
            centers)


# ---------------------------------------------------------------------------
# T1: FF energy discrimination
# ---------------------------------------------------------------------------
def test_t1():
    """Hypothesis: E(clean) < E(noise) with AUROC > 0.85.

    Tests BOTH point-level (single embeddings) and sequence-level (coherent vs
    incoherent token sequences). The sequence test is the critical one — it
    tests whether FF captures structure beyond what individual splats encode.
    """
    print("\n" + "=" * 60)
    print("T1: FF Energy Discrimination")
    print("=" * 60)

    from sklearn.metrics import roc_auc_score

    results = {"test": "t1", "timestamp": datetime.utcnow().isoformat()}

    # --- T1a: Point-level (easy baseline) ---
    print("\n[T1a] Point-level discrimination (single embeddings)...")
    real, noise, _ = make_latent_data(n_samples=1000)
    model = FFEnergy(dim=DIM, hidden=256, n_layers=2).to(DEVICE)
    model.train_ff(real[:800], noise[:800], epochs=80, verbose=False)
    model.eval()
    with torch.no_grad():
        e_real = model(real[800:]).cpu().numpy()
        e_noise = model(noise[800:]).cpu().numpy()
    labels = np.concatenate([np.zeros(len(e_real)), np.ones(len(e_noise))])
    scores = np.concatenate([e_real, e_noise])
    auroc_point = roc_auc_score(labels, scores)
    margin_point = float(e_noise.mean() - e_real.mean())
    print(f"  AUROC: {auroc_point:.3f}  margin: {margin_point:.3f}")
    results["t1a_point_auroc"] = float(auroc_point)
    results["t1a_point_margin"] = margin_point

    # --- T1b: Sequence-level (the critical test) ---
    print("\n[T1b] Sequence-level discrimination (coherent vs incoherent)...")
    real_seq, noise_seq, _ = make_sequence_data(n_samples=1000, seq_len=16)
    model_seq = FFEnergy(dim=DIM, hidden=256, n_layers=3).to(DEVICE)
    model_seq.train_ff(real_seq[:800], noise_seq[:800], epochs=80, verbose=False)
    model_seq.eval()
    with torch.no_grad():
        e_real_s = model_seq(real_seq[800:]).cpu().numpy().ravel()
        e_noise_s = model_seq(noise_seq[800:]).cpu().numpy().ravel()
    labels_s = np.concatenate([np.zeros(len(e_real_s)), np.ones(len(e_noise_s))])
    scores_s = np.concatenate([e_real_s, e_noise_s])
    auroc_seq = roc_auc_score(labels_s, scores_s)
    margin_seq = float(e_noise_s.mean() - e_real_s.mean())
    print(f"  AUROC: {auroc_seq:.3f}  margin: {margin_seq:.3f}")
    results["t1b_seq_auroc"] = float(auroc_seq)
    results["t1b_seq_margin"] = margin_seq

    # --- Verdict ---
    pass_point = auroc_point > 0.85
    pass_seq = auroc_seq > 0.85
    results["verdict_t1a"] = "PASS" if pass_point else "FAIL"
    results["verdict_t1b"] = "PASS" if pass_seq else "FAIL"

    print(f"\n[T1] Point AUROC {auroc_point:.3f}: {'✅ PASS' if pass_point else '❌ FAIL'} (>0.85)")
    print(f"[T1] Seq  AUROC {auroc_seq:.3f}: {'✅ PASS' if pass_seq else '❌ FAIL'} (>0.85)")
    if pass_seq:
        print("  → FF captures sequential structure beyond individual splats.")
    else:
        print("  → FF cannot distinguish coherent from incoherent sequences.")

    _save(results, "t1")
    return model_seq if pass_seq else model_seq  # return trained model for T2


# ---------------------------------------------------------------------------
# T2: Score gradient direction
# ---------------------------------------------------------------------------
def test_t2(model: FFEnergy = None):
    """Hypothesis: stepping along -∇E from noise moves toward real data.

    The critical test for diffusion: does the learned score point toward
    the data manifold?
    """
    print("\n" + "=" * 60)
    print("T2: Score Gradient Direction")
    print("=" * 60)

    if model is None:
        # Train a quick model on point data
        real, noise, centers = make_latent_data(n_samples=1000)
        model = FFEnergy(dim=DIM, hidden=256, n_layers=2).to(DEVICE)
        model.train_ff(real[:800], noise[:800], epochs=80, verbose=False)
    else:
        real, noise, centers = make_latent_data(n_samples=1000)

    model.eval()
    results = {"test": "t2", "timestamp": datetime.utcnow().isoformat()}

    # Start from noise, take Langevin steps, measure distance to nearest real
    n_test = 200
    x = torch.randn(n_test, DIM, device=DEVICE) * 2.0
    distances = []
    real_ref = real[:500]  # reference points

    lr = 0.01
    for step in range(50):
        score = model.score(x)
        x = x - 0.5 * lr * score + np.sqrt(2 * lr) * torch.randn_like(x)
        if step % 5 == 0 or step == 49:
            with torch.no_grad():
                # Distance to nearest real point
                dist = torch.cdist(x, real_ref).min(dim=1)[0]
                distances.append({
                    "step": step,
                    "median_dist": float(dist.median()),
                    "mean_dist": float(dist.mean()),
                })

    print("\nDistance to nearest real data during sampling:")
    for d in distances:
        print(f"  step {d['step']:3d}: median={d['median_dist']:.4f}  mean={d['mean_dist']:.4f}")

    # Does distance decrease monotonically (at least in trend)?
    medians = [d["median_dist"] for d in distances]
    start_median = medians[0]
    end_median = medians[-1]
    decrease = (start_median - end_median) / start_median

    results["distances"] = distances
    results["relative_decrease"] = decrease
    results["verdict"] = "PASS" if decrease > 0.15 else "FAIL"

    print(f"\n[T2] Distance decrease: {decrease*100:.1f}%: "
          f"{'✅ PASS' if decrease > 0.15 else '❌ FAIL'} (>15%)")
    if decrease > 0.15:
        print("  → Score gradient points toward data manifold. Diffusion can work.")
    else:
        print("  → Score does not guide toward data. Sampling will fail.")

    _save(results, "t2")


# ---------------------------------------------------------------------------
# T3: Sampling degeneracy
# ---------------------------------------------------------------------------
def test_t3(model: FFEnergy = None):
    """Hypothesis: diffusion samples are non-degenerate (not mode collapse).

    Generates samples, measures repetition ratio. Also checks diversity.
    """
    print("\n" + "=" * 60)
    print("T3: Sampling Degeneracy Check")
    print("=" * 60)

    if model is None:
        real, noise, centers = make_latent_data(n_samples=1000)
        model = FFEnergy(dim=DIM, hidden=256, n_layers=2).to(DEVICE)
        model.train_ff(real[:800], noise[:800], epochs=80, verbose=False)
    else:
        real, noise, centers = make_latent_data(n_samples=1000)

    results = {"test": "t3", "timestamp": datetime.utcnow().isoformat()}

    sampler = LatentDiffusionSampler(model, dim=DIM, n_steps=100, lr=0.01)
    samples, traj = sampler.sample(200, device=DEVICE, return_trajectory=True)

    # Measure 1: spread of samples (should not all collapse to one point)
    with torch.no_grad():
        samples_cpu = samples.cpu()
        sample_std = samples_cpu.std(dim=0).mean().item()
        sample_mean = samples_cpu.mean(dim=0)
        pairwise = torch.cdist(samples_cpu[:50], samples_cpu[:50])
        pairwise_mean = pairwise[pairwise > 0].mean().item()

    # Measure 2: distance to real data (samples should be near real points)
    with torch.no_grad():
        dist_to_real = torch.cdist(samples_cpu, real[:500].cpu()).min(dim=1)[0]
        near_real_ratio = (dist_to_real < 1.0).float().mean().item()

    results["sample_std"] = sample_std
    results["pairwise_dist_mean"] = pairwise_mean
    results["near_real_ratio"] = near_real_ratio

    print(f"\nSample std (diversity):           {sample_std:.4f}")
    print(f"Pairwise distance (diversity):    {pairwise_mean:.4f}")
    print(f"Ratio near real data (<1.0 dist): {near_real_ratio:.3f}")

    # Mode collapse = all samples identical (std ≈ 0)
    not_collapsed = sample_std > 0.05 and pairwise_mean > 0.1
    near_data = near_real_ratio > 0.3

    results["verdict_diversity"] = "PASS" if not_collapsed else "FAIL"
    results["verdict_near_data"] = "PASS" if near_data else "FAIL"
    results["verdict"] = "PASS" if (not_collapsed and near_data) else "FAIL"

    print(f"\n[T3] Diversity: {'✅ PASS' if not_collapsed else '❌ FAIL'} "
          f"(std>0.05, pairwise>0.1)")
    print(f"[T3] Near data: {'✅ PASS' if near_data else '❌ FAIL'} "
          f"(>30% within 1.0 of real)")

    _save(results, "t3")


def _save(results: dict, name: str):
    path = RESULTS_DIR / f"phase1_{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["t1", "t2", "t3", "all"], default="all")
    args = parser.parse_args()

    print(f"Device: {DEVICE}, Dim: {DIM}")

    if args.test in ("t1", "all"):
        model = test_t1()
    else:
        model = None

    if args.test in ("t2", "all"):
        test_t2(model)

    if args.test in ("t3", "all"):
        test_t3(model)
