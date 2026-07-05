"""Just the separability analysis — no training, instant."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from phase2_score_matching_scaled import DIM, DEVICE


def measure_config(n_clusters, spread, dim=DIM):
    torch.manual_seed(42)
    centers = torch.randn(n_clusters, dim)
    centers = F.normalize(centers, dim=-1)

    real = []
    for _ in range(2000):
        c = centers[torch.randint(0, n_clusters, (1,)).item()]
        point = c + torch.randn(dim) * spread
        point = F.normalize(point, dim=-1)
        real.append(point)
    real = torch.stack(real).to(DEVICE)
    centers = centers.to(DEVICE)

    dists = torch.cdist(real, centers)
    assigned = dists.argmin(dim=1)
    dist_to_own = dists.gather(1, assigned.unsqueeze(1)).squeeze()
    intra = dist_to_own.mean().item()

    inter = torch.cdist(centers, centers)
    inter = inter[inter > 0].mean().item()

    ratio = inter / intra if intra > 0 else 999
    status = "✅" if ratio > 5 else "❌"
    return n_clusters, spread, intra, inter, ratio, status


print(f"{'Clusters':>8} {'Spread':>8} {'Intra':>8} {'Inter':>8} {'Ratio':>8} {'Status':>8}")
print("-" * 60)
for n_c, sp in [(20, 0.15), (20, 0.05), (20, 0.01), (10, 0.02), (10, 0.01),
                 (5, 0.02), (5, 0.01), (5, 0.005), (3, 0.01), (3, 0.005)]:
    nc, sp, intra, inter, ratio, status = measure_config(n_c, sp)
    print(f"{nc:>8} {sp:>8.3f} {intra:>8.4f} {inter:>8.4f} {ratio:>7.1f}x {status:>8}")
