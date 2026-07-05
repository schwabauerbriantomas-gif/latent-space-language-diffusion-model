"""
Stress test: scale the pipeline to 10, 20, 50, 100 clusters.

Measures:
  - Quality: near_data, diversity, mode coverage
  - Scaling: how does performance change with cluster count?
  - Timing: train + sample time per config
  - Auto-k: does PCA auto-select the right dimensionality?
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from latent_diffusion_pipeline import LatentDiffusionModel, make_clustered_data
from collections import Counter

RESULTS_DIR = REPO / "results"


def count_modes_reached(samples, centers):
    """How many cluster centers have at least one sample nearby?"""
    dists = torch.cdist(samples, centers)
    assigned = dists.argmin(dim=1)
    return len(torch.unique(assigned))


def run_single(n_clusters, n_samples=2000, cluster_spread=0.005, seed=42):
    """Run full pipeline on one config and measure everything."""
    config = f"c{n_clusters}_s{cluster_spread}"
    print(f"\n{'='*70}")
    print(f"STRESS TEST: {config}")
    print(f"{'='*70}")

    t0 = time.time()

    # Generate data
    real, noise, centers = make_clustered_data(
        n_samples=n_samples, n_clusters=n_clusters,
        cluster_spread=cluster_spread, seed=seed
    )
    print(f"Data: {real.shape[0]} samples, {n_clusters} clusters")

    # Fit model
    model = LatentDiffusionModel(
        variance_threshold=0.90,
        score_hidden=256, score_blocks=6, score_epochs=2000,
        svgd_iters=800, svgd_particles=300,
    )
    model.fit(real[:1600], verbose=True)
    fit_time = time.time() - t0

    # Sample
    t1 = time.time()
    samples = model.sample(verbose=True, real_ref_1024=real[:800])
    sample_time = time.time() - t1

    # Evaluate
    metrics = model.evaluate(samples, real[:800])
    metrics["modes_reached"] = count_modes_reached(samples, centers)
    metrics["total_modes"] = n_clusters
    metrics["k"] = model.k
    metrics["fit_time_s"] = fit_time
    metrics["sample_time_s"] = sample_time
    metrics["total_time_s"] = time.time() - t0
    metrics["config"] = config

    # Verdict
    coverage = metrics["modes_reached"] / n_clusters
    if metrics["near_03"] > 0.7 and coverage > 0.8:
        verdict = "✅ SUCCESS"
    elif metrics["near_03"] > 0.3 and coverage > 0.5:
        verdict = "🔶 GOOD"
    elif metrics["near_03"] > 0.1:
        verdict = "🔶 PARTIAL"
    else:
        verdict = "❌ FAIL"
    metrics["verdict"] = verdict

    print(f"\n--- Results: {config} ---")
    print(f"  k (auto): {metrics['k']}")
    print(f"  Near data (<0.3): {metrics['near_03']:.2%}")
    print(f"  Near data (<0.1): {metrics['near_01']:.2%}")
    print(f"  Modes reached: {metrics['modes_reached']}/{n_clusters}")
    print(f"  Diversity: {metrics['diversity']:.4f}")
    print(f"  Median dist: {metrics['median_dist']:.4f}")
    print(f"  Fit time: {fit_time:.0f}s  Sample time: {sample_time:.0f}s")
    print(f"  VERDICT: {verdict}")

    return metrics


if __name__ == "__main__":
    torch.manual_seed(42)
    all_results = {"timestamp": datetime.now().isoformat(), "tests": []}

    configs = [
        {"n_clusters": 5, "cluster_spread": 0.005},
        {"n_clusters": 10, "cluster_spread": 0.005},
        {"n_clusters": 20, "cluster_spread": 0.005},
        {"n_clusters": 50, "cluster_spread": 0.005},
        {"n_clusters": 100, "cluster_spread": 0.005},
    ]

    for cfg in configs:
        try:
            r = run_single(**cfg)
            all_results["tests"].append(r)
        except Exception as e:
            print(f"\nFAILED: {cfg} → {e}")
            import traceback; traceback.print_exc()
            all_results["tests"].append({"config": str(cfg), "error": str(e)})

    # Summary
    print(f"\n{'='*70}")
    print("STRESS TEST SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':>12} {'k':>4} {'near<0.3':>10} {'modes':>8} {'diversity':>10} {'verdict':>10}")
    print("-" * 70)
    for r in all_results["tests"]:
        if "error" in r:
            print(f"{r['config']:>12} {'ERR':>4}")
            continue
        print(f"{r['config']:>12} {r['k']:>4} {r['near_03']:>9.2%} "
              f"{r['modes_reached']:>3}/{r['total_modes']:<3} "
              f"{r['diversity']:>9.4f} {r['verdict']:>10}")

    out = RESULTS_DIR / "stress_test_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")
