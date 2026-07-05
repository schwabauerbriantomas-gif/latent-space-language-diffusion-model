"""
Mathematical analysis of the sampling problem on S^1023.

THE PROBLEM (measured):
  - Score direction: cos(s, true) = +0.970  ✓ correct
  - Score magnitude: ||s|| ≈ 57 everywhere  ✗ constant
  - Langevin sampling: stalls at dist 1.25    ✗

THE QUESTION: What samplers can work given a correct-direction
but constant-magnitude score field on a high-dimensional sphere?

This file contains PURE MATH ANALYSIS + proofs of concept.
Each approach is derived from first principles, not borrowed.

Run sections independently or all together.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from phase2_score_matching_scaled import ScoreNetworkV2, DIM, DEVICE


# ============================================================================
# GEOMETRIC FOUNDATIONS
# ============================================================================

def geometry_analysis():
    """Measure the actual geometry of S^1023 and our data on it."""
    print("=" * 70)
    print("GEOMETRIC ANALYSIS: Why sampling fails on S^1023")
    print("=" * 70)

    d = 1024

    # 1. Concentration of measure
    # For random x,y uniform on S^(d-1), <x,y> ~ N(0, 1/d)
    # So distances concentrate around π/2 ≈ 1.571
    x = F.normalize(torch.randn(10000, d), dim=-1)
    y = F.normalize(torch.randn(10000, d), dim=-1)
    dots = (x * y).sum(dim=-1)
    dists = torch.acos(dots.clamp(-1, 1))

    print(f"\n1. CONCENTRATION OF MEASURE on S^{d-1}:")
    print(f"   Random <x,y>: mean={dots.mean():.6f}  std={dots.std():.6f}")
    print(f"   Random geodesic dist: mean={dists.mean():.4f}  std={dists.std():.4f}")
    print(f"   → All random pairs are at distance π/2 ± {dists.std():.4f}")
    print(f"   → The sphere is 'all equator' — no meaningful near/far")

    # 2. Volume of spherical caps
    # Vol(cap of angle θ) / Vol(S) = I_sin²(θ; (d-1)/2, 1/2)
    # For small θ: ~ θ^(d-1) / (d-1)!!
    # At d=1024, θ=0.1: 0.1^1023 ≈ 0
    print(f"\n2. SPHERICAL CAP VOLUME (where data lives):")
    for theta in [0.01, 0.05, 0.1, 0.3, 0.5]:
        # Approximate: fraction of sphere within angle θ of a point
        # = (sin(θ)/1)^(d-1) for small θ on unit sphere (very rough)
        log_frac = (d - 1) * math.log(max(math.sin(theta), 1e-300))
        print(f"   Cap angle {theta:.2f}: log(volume fraction) = {log_frac:.1f}"
              f"  → fraction ≈ e^{log_frac:.0f}")

    # 3. Our data geometry
    from separable_experiment import make_separable_data
    real, noise, centers, _ = make_separable_data(
        n_samples=2000, n_clusters=5, cluster_spread=0.005
    )
    real_cpu = real.cpu()

    # Intra-cluster angular spread
    for ci in range(5):
        mask = torch.cdist(real_cpu, centers[ci:ci+1].cpu()).squeeze() < 0.3
        cluster_pts = real_cpu[mask]
        if len(cluster_pts) > 1:
            dots_c = (cluster_pts * centers[ci].cpu()).sum(dim=-1)
            angles = torch.acos(dots_c.clamp(-1, 1))
            print(f"\n   Cluster {ci}: {len(cluster_pts)} points, "
                  f"angular spread = {angles.mean():.4f} ± {angles.std():.4f} rad")

    # Inter-cluster angular distances
    inter = torch.cdist(centers.cpu(), centers.cpu())
    inter = inter[inter > 0]
    print(f"\n   Inter-cluster geodesic dist: {inter.mean():.4f} ± {inter.std():.4f}")

    # 4. Effective dimensionality
    # PCA of the data — how many dims capture 99% of variance?
    real_centered = real_cpu - real_cpu.mean(dim=0)
    U, S, V = torch.svd(real_centered)
    var_explained = (S ** 2) / (S ** 2).sum()
    cumvar = var_explained.cumsum(0)
    k99 = (cumvar < 0.99).sum().item() + 1
    k90 = (cumvar < 0.90).sum().item() + 1
    k50 = (cumvar < 0.50).sum().item() + 1
    print(f"\n3. INTRINSIC DIMENSIONALITY (PCA):")
    print(f"   50% variance: {k50} dimensions")
    print(f"   90% variance: {k90} dimensions")
    print(f"   99% variance: {k99} dimensions")
    print(f"   → Data effectively lives in a {k90}-{k99} dimensional subspace!")
    print(f"   → Score matching in 1024D wastes 1024-{k99} = {1024-k99} dimensions")

    return {"k50": k50, "k90": k90, "k99": k99,
            "eigenvalues": S[:20].tolist(),
            "var_explained": var_explained[:20].tolist()}


# ============================================================================
# APPROACH 1: EXACT KERNEL SCORE (ground truth, no learning)
# ============================================================================

def exact_kde_score(x, data, sigma):
    """Exact score from Gaussian KDE.

    p(x) = (1/n) Σ exp(-||x - x_i||² / (2σ²))
    ∇log p(x) = -(1/σ²) * Σ (x - x_i) w_i / Σ w_i

    where w_i = exp(-||x-x_i||²/(2σ²))

    This gives the EXACT score — no neural network approximation.
    If even this fails, the problem is truly geometric, not learning.
    """
    # x: [batch, d], data: [n, d]
    diff = x.unsqueeze(1) - data.unsqueeze(0)  # [batch, n, d]
    sq_dist = (diff ** 2).sum(dim=-1)  # [batch, n]

    log_w = -sq_dist / (2 * sigma ** 2)
    log_w = log_w - log_w.max(dim=-1, keepdim=True).values  # numerical stability
    w = log_w.exp()  # [batch, n]

    # Weighted average of (x - x_i) = -weighted average of (x_i - x)
    # ∇log p = -(1/σ²) Σ (x - x_i) w_i / Σ w_i = (1/σ²) Σ (x_i - x) w_i / Σ w_i
    weighted_diff = (diff * w.unsqueeze(-1)).sum(dim=1)  # [batch, d] = Σ (x-x_i)w_i
    score = -weighted_diff / (w.sum(dim=-1, keepdim=True) * sigma ** 2)

    return score


def test_exact_kde():
    """Can the EXACT score guide sampling?"""
    print("\n" + "=" * 70)
    print("APPROACH 1: Exact KDE Score (no learning — ground truth)")
    print("=" * 70)

    from separable_experiment import make_separable_data
    real, _, centers, _ = make_separable_data(
        n_samples=2000, n_clusters=5, cluster_spread=0.005
    )
    real_ref = real[:800].cpu()
    data = real[:1600].cpu()

    # Test score at various points
    print("\n  Score profile (exact KDE):")
    for sigma in [0.01, 0.05, 0.1, 0.3]:
        # At data point
        x_at = real[:10].cpu()
        s_at = exact_kde_score(x_at, data, sigma)
        norm_at = s_at.norm(dim=-1).mean().item()

        # At random point on sphere
        x_rand = F.normalize(torch.randn(10, DIM), dim=-1)
        s_rand = exact_kde_score(x_rand, data, sigma)
        norm_rand = s_rand.norm(dim=-1).mean().item()

        # Direction at random point
        dist_to_data = torch.cdist(x_rand, real_ref).min(dim=1)[0]
        nn_idx = torch.cdist(x_rand, real_ref).argmin(dim=1)
        dir_to_nn = F.normalize(real_ref[nn_idx] - x_rand, dim=-1)
        cos_dir = F.cosine_similarity(F.normalize(s_rand, dim=-1), dir_to_nn, dim=-1).mean().item()

        print(f"  σ={sigma:.2f}: ||s|| at data={norm_at:.1f}  "
              f"||s|| at random={norm_rand:.1f}  "
              f"ratio={norm_at/max(norm_rand,1e-10):.1f}x  "
              f"cos(dir)={cos_dir:+.3f}")

    # Try sampling with exact score
    print("\n  Sampling with exact KDE score:")
    x = F.normalize(torch.randn(200, DIM), dim=-1)

    for sigma in [0.3, 0.1, 0.05]:
        for step in range(500):
            s = exact_kde_score(x, data, sigma)
            x = x + 0.001 * sigma ** 2 * s
            x = F.normalize(x, dim=-1)
        dist = torch.cdist(x, real_ref).min(dim=1)[0]
        near = (dist < 0.3).float().mean().item()
        print(f"  σ={sigma:.2f}, 500 steps: near(<0.3)={near:.2%}  "
              f"median_dist={dist.median():.4f}")

    return x


# ============================================================================
# APPROACH 2: TANGENT-PROJECTED SCORE (Riemannian)
# ============================================================================

def tangent_projection(score, x):
    """Project score onto the tangent space T_x S^(d-1).

    T_x S^(d-1) = { v : <v, x> = 0 }
    Projection: v_tan = v - <v, x> x

    The radial component <s, x> x is meaningless on the sphere —
    it tries to change ||x|| which is fixed at 1.
    """
    radial = (score * x).sum(dim=-1, keepdim=True) * x
    return score - radial


def test_tangent_score():
    """Does removing the radial component help?"""
    print("\n" + "=" * 70)
    print("APPROACH 2: Tangent-Projected Score (Riemannian)")
    print("=" * 70)

    # Load trained model
    net = ScoreNetworkV2(dim=DIM, hidden=1024, n_blocks=8).to(DEVICE)
    net.load_state_dict(torch.load(REPO / "checkpoints" / "score_net_v2.pt"))
    net.eval()

    from separable_experiment import make_separable_data
    real, _, centers, _ = make_separable_data(
        n_samples=2000, n_clusters=5, cluster_spread=0.005
    )
    real_ref = real[:800]

    # Measure how much of the score is radial vs tangent
    print("\n  Score decomposition at various points:")
    for desc, x in [
        ("at data", real[:200]),
        ("midway", F.normalize(real[:200] + 0.5 * torch.randn(200, DIM, device=DEVICE), dim=-1)),
        ("random", F.normalize(torch.randn(200, DIM, device=DEVICE), dim=-1)),
    ]:
        with torch.no_grad():
            s = net(x, torch.full((200, 1), 0.5, device=DEVICE))
            s_tan = tangent_projection(s, x)
            radial_frac = (s.norm(dim=-1) - s_tan.norm(dim=-1)) / s.norm(dim=-1).clamp(min=1e-10)
            print(f"  {desc:10s}: ||s||={s.norm(dim=-1).mean():.1f}  "
                  f"||s_tan||={s_tan.norm(dim=-1).mean():.1f}  "
                  f"radial fraction={radial_frac.mean():.3f}")

    # Sample with tangent-projected score
    print("\n  Sampling with tangent-projected score:")
    x = F.normalize(torch.randn(200, DIM, device=DEVICE), dim=-1)
    for sigma in [1.0, 0.5, 0.2, 0.1]:
        alpha = 0.001 * sigma ** 2
        for step in range(300):
            with torch.no_grad():
                s = net(x, torch.full((200, 1), sigma, device=DEVICE))
                s_tan = tangent_projection(s, x)
            # Manifold Langevin: move in tangent space, project back
            noise = torch.randn_like(x)
            noise_tan = tangent_projection(noise, x)
            x = x + 0.5 * alpha * s_tan + math.sqrt(alpha) * noise_tan
            x = F.normalize(x, dim=-1)
        dist = torch.cdist(x, real_ref).min(dim=1)[0]
        near = (dist < 0.3).float().mean().item()
        print(f"  σ={sigma:.1f}: near(<0.3)={near:.2%}  median_dist={dist.median():.4f}")


# ============================================================================
# APPROACH 3: PCA-REDUCED SCORE MATCHING
# ============================================================================

def test_pca_reduced():
    """Score matching in the PCA subspace where data actually lives."""
    print("\n" + "=" * 70)
    print("APPROACH 3: PCA-Reduced Score Matching")
    print("=" * 70)

    from separable_experiment import make_separable_data
    real, _, centers, _ = make_separable_data(
        n_samples=2000, n_clusters=5, cluster_spread=0.005
    )
    real_cpu = real.cpu()

    # PCA
    mean = real_cpu.mean(dim=0)
    real_centered = real_cpu - mean
    U, S, V = torch.svd(real_centered)

    cumvar = (S ** 2).cumsum(0) / (S ** 2).sum()
    k = (cumvar < 0.99).sum().item() + 1
    print(f"  Effective dimensionality: {k} (99% variance)")

    # Project to k-dim subspace
    basis = V[:, :k]  # [d, k]
    real_low = real_centered @ basis  # [n, k]
    print(f"  Projected data: {real_low.shape}")
    print(f"  Low-dim data range: [{real_low.min():.3f}, {real_low.max():.3f}]")

    # Now do 2D-style score matching in this k-dim space
    from control_2d import ScoreNet2D, dsm_loss_2d

    # Adapt network for k dimensions
    class ScoreNetKD(ScoreNet2D):
        def __init__(self, k_dim, hidden=256, n_blocks=6):
            super().__init__(hidden=hidden, n_blocks=n_blocks)
            self.input_proj = torch.nn.Linear(k_dim, hidden)
            self.out = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden), torch.nn.SiLU(),
                torch.nn.Linear(hidden, k_dim)
            )

    net_kd = ScoreNetKD(k_dim=k).cuda()
    optimizer = torch.optim.AdamW(net_kd.parameters(), lr=1e-3)

    real_low_gpu = real_low.cuda()
    sigmas = torch.tensor([2.0, 1.0, 0.5, 0.2, 0.1, 0.05], device=DEVICE)

    print(f"\n  Training score network in {k}D...")
    for epoch in range(2000):
        optimizer.zero_grad()
        loss = dsm_loss_2d(net_kd, real_low_gpu[:1600], sigmas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_kd.parameters(), 1.0)
        optimizer.step()
        if epoch % 500 == 0 or epoch == 1999:
            print(f"    epoch {epoch}: loss={loss.item():.4f}")
    net_kd.eval()

    # Sample in k-dim space
    print(f"\n  Annealed Langevin sampling in {k}D:")
    x = torch.randn(200, k, device=DEVICE) * 3

    for sigma in sigmas:
        alpha = 0.01 * sigma.item() ** 2
        for _ in range(300):
            with torch.no_grad():
                s = net_kd(x, torch.full((200, 1), sigma.item(), device=DEVICE))
            x = x + 0.5 * alpha * s + math.sqrt(alpha) * torch.randn_like(x)

    # Map back to 1024D and measure
    x_full = (x @ basis.T) + mean  # [200, 1024]
    dist = torch.cdist(x_full, real_cpu).min(dim=1)[0]
    near_03 = (dist < 0.3).float().mean().item()
    near_01 = (dist < 0.1).float().mean().item()

    # Cluster coverage
    dists_centers = torch.cdist(x_full, centers.cpu())
    assigned = dists_centers.argmin(dim=1)
    modes = len(torch.unique(assigned))

    print(f"  Results (mapped back to 1024D):")
    print(f"    Near data (<0.3): {near_03:.2%}")
    print(f"    Near data (<0.1): {near_01:.2%}")
    print(f"    Modes reached: {modes}/5")
    print(f"    Median distance: {dist.median():.4f}")

    verdict = "✅ SUCCESS" if near_03 > 0.3 else "🔶 PARTIAL" if near_03 > 0.1 else "❌ FAIL"
    print(f"    Verdict: {verdict}")

    return {"k": k, "near_03": near_03, "near_01": near_01, "modes": modes}


# ============================================================================
# APPROACH 4: STEIN VARIATIONAL GRADIENT DESCENT (SVGD)
# ============================================================================

def test_svgd():
    """SVGD: deterministic particle transport. No Langevin noise needed.

    SVGD update for particle x_i:
      dx_i = (1/n) Σ_j [ k(x_j, x_i) ∇_x_i log p(x_i) + ∇_x_i k(x_j, x_i) ]

    The repulsion term ∇k prevents collapse.
    Uses the KDE score as ∇log p.
    """
    print("\n" + "=" * 70)
    print("APPROACH 4: Stein Variational Gradient Descent (SVGD)")
    print("=" * 70)

    from separable_experiment import make_separable_data
    real, _, centers, _ = make_separable_data(
        n_samples=2000, n_clusters=5, cluster_spread=0.005
    )
    real_ref = real[:800].cpu()
    data = real[:1600].cpu()

    def rbf_kernel(x, y, h="median"):
        """RBF kernel k(x,y) = exp(-||x-y||² / h)"""
        sq_dist = torch.cdist(x, y) ** 2
        if h == "median":
            h_val = sq_detached_median(sq_dist)
        else:
            h_val = h
        return torch.exp(-sq_dist / h_val), sq_dist

    def sq_detached_median(sq_dist):
        """Median heuristic for bandwidth."""
        d = sq_dist[sq_dist > 0]
        if len(d) == 0:
            return 1.0
        return d.median().item()

    n_particles = 200
    x = F.normalize(torch.randn(n_particles, DIM), dim=-1)

    print(f"\n  SVGD with {n_particles} particles...")
    step_size = 0.01

    for iteration in range(500):
        # Score (gradient of log density) at each particle
        score = exact_kde_score(x, data, sigma=0.05)  # [n, d]

        # Kernel matrix
        k_xy, sq_dist = rbf_kernel(x, x, h="median")

        # Repulsion term: ∇_x_i k(x_j, x_i) = -2(x_i - x_j)/h * k(x_j, x_i)
        h = sq_detached_median(sq_dist)
        # ∇_x_i Σ_j k(x_j, x_i) = Σ_j -2(x_i - x_j)/h * k(x_j, x_i)
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # [n, n, d]: x_i - x_j
        grad_k = (-2.0 / h * k_xy.unsqueeze(-1) * diff).sum(dim=1)  # [n, d]

        # SVGD update
        phi = (k_xy @ score) / n_particles + grad_k / n_particles
        x = x + step_size * phi
        x = F.normalize(x, dim=-1)  # project to sphere

        if iteration % 100 == 0 or iteration == 499:
            dist = torch.cdist(x, real_ref).min(dim=1)[0]
            near = (dist < 0.3).float().mean().item()
            print(f"  iter {iteration:3d}: near(<0.3)={near:.2%}  "
                  f"median_dist={dist.median():.4f}")

    return x


# ============================================================================
# APPROACH 5: STEREOGRAPHIC PROJECTION (S^d → R^d)
# ============================================================================

def test_stereographic():
    """Map sphere → R^d via stereographic projection, score match there.

    Stereographic projection from north pole:
      φ: S^d \\ {N} → R^d
      φ(x) = x_{1:d} / (1 - x_{d+1})

    Inverse:
      φ⁻¹(y) = (2y, ||y||² - 1) / (||y||² + 1)

    In R^d, standard diffusion works without manifold constraints.
    """
    print("\n" + "=" * 70)
    print("APPROACH 5: Stereographic Projection (S^1023 → R^1023)")
    print("=" * 70)

    from separable_experiment import make_separable_data
    real, _, centers, _ = make_separable_data(
        n_samples=2000, n_clusters=5, cluster_spread=0.005
    )
    real_cpu = real.cpu()

    # Project to R^1023
    def stereo_forward(x):
        """S^d → R^d. x: [..., d+1] → [..., d]"""
        x_d = x[..., :-1]  # first d coords
        x_last = x[..., -1:]  # last coord
        return x_d / (1 - x_last + 1e-8)

    def stereo_inverse(y):
        """R^d → S^d. y: [..., d] → [..., d+1]"""
        sq = (y ** 2).sum(dim=-1, keepdim=True)
        top = 2 * y
        bottom = torch.cat([y, (sq - 1) / (sq + 1)], dim=-1)
        return torch.cat([2 * y, (sq - 1)], dim=-1) / (sq + 1)

    # Project data
    real_proj = stereo_forward(real_cpu)
    print(f"  Projected data shape: {real_proj.shape}")
    print(f"  Projected data norm: mean={real_proj.norm(dim=-1).mean():.2f}  "
          f"std={real_proj.norm(dim=-1).std():.2f}")
    print(f"  (Large norms = points near north pole on sphere)")

    # Score matching in R^1023 (using exact KDE)
    real_ref_proj = real_proj[:800]
    x_proj = torch.randn(200, DIM - 1) * 5  # start from wide Gaussian

    print(f"\n  Langevin sampling in R^{DIM-1} (stereographic coords)...")
    for sigma in [2.0, 1.0, 0.5, 0.1]:
        for step in range(300):
            s = exact_kde_score(x_proj, real_proj[:1600], sigma)
            alpha = 0.001 * sigma ** 2
            x_proj = x_proj + 0.5 * alpha * s + math.sqrt(alpha) * torch.randn_like(x_proj)

        # Map back to sphere
        x_sphere = stereo_inverse(x_proj)
        x_sphere = F.normalize(x_sphere, dim=-1)

        dist = torch.cdist(x_sphere, real_cpu).min(dim=1)[0]
        near = (dist < 0.3).float().mean().item()
        print(f"  σ={sigma:.1f}: near(<0.3)={near:.2%}  "
              f"median_dist={dist.median():.4f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    # 1. Geometric analysis (fast, no GPU training)
    geom = geometry_analysis()

    # 2. Exact KDE score (ground truth — does the exact score work?)
    kde_samples = test_exact_kde()

    # 3. Tangent projection (uses trained model)
    test_tangent_score()

    # 4. PCA-reduced score matching (train in low-D, map back)
    pca_result = test_pca_reduced()

    # 5. SVGD (deterministic, uses exact score)
    test_svgd()

    # 6. Stereographic projection
    test_stereographic()

    print("\n" + "=" * 70)
    print("ALL APPROACHES TESTED")
    print("=" * 70)
