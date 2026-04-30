"""Side-by-side uniform vs adaptive kNN visualisation for the poster.

Draws:
  left:  uniform k=16 neighbour edges around the airfoil(s)
  right: UDF-adaptive k neighbour edges, points coloured by per-node k

Picks a sample (default: 3001_10-0 — 3-airfoil geometry for visual
richness) and a zoom window (default: wide enough to cover 2-3 airfoils
if the geometry has multiple).

Output: figures/adaptive_knn_03_comparison.png  (replaces existing)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FIG = ROOT / "figures"

SAMPLE = "3001_10-0"                 # 3-airfoil geometry
K_UNIFORM = 16
K_NEAR = 32
K_FAR = 8
LAMBDA = 0.05
ZOOM_HALF_X = 0.95                    # half-width in x (streamwise)
ZOOM_HALF_Y = 0.55                    # half-width in y (normal)
MAX_EDGES_PER_PANEL = 8000            # subsample to avoid overplotting
RNG = np.random.default_rng(0)


def adaptive_k(udf: np.ndarray, k_near: int, k_far: int, sigma: float) -> np.ndarray:
    k_float = k_far + (k_near - k_far) * np.exp(-udf / sigma)
    return np.maximum(k_far, np.round(k_float).astype(int))


def nearest_surface_udf(pos: np.ndarray, surface: np.ndarray) -> np.ndarray:
    tree = cKDTree(surface)
    d, _ = tree.query(pos, k=1)
    return d


def draw_edges(ax, pos2d, knn, in_view_mask, max_edges, color, alpha):
    """Draw edges from each in-view node, capped at max_edges total."""
    srcs = np.where(in_view_mask)[0]
    # Collect all (src, dst) pairs where src is in view; knn might be padded with -1
    pairs = []
    for s in srcs:
        for d in knn[s]:
            if d == -1:
                continue
            pairs.append((s, d))
    pairs = np.array(pairs)
    if len(pairs) > max_edges:
        pairs = pairs[RNG.choice(len(pairs), size=max_edges, replace=False)]
    segs = np.stack([pos2d[pairs[:, 0]], pos2d[pairs[:, 1]]], axis=1)
    from matplotlib.collections import LineCollection
    lc = LineCollection(segs, colors=color, linewidths=0.35, alpha=alpha,
                        zorder=1)
    ax.add_collection(lc)


def main():
    npz = DATA_DIR / f"{SAMPLE}.npz"
    data = np.load(npz)
    pos = data["pos"].astype(np.float32)           # (N, 3)
    idcs = data["idcs_airfoil"].astype(np.int64)   # (M,)
    surface_3d = pos[idcs]

    # 2D projection for visualisation: drop z, project onto x-y.
    pos2d = pos[:, :2]
    surf2d = surface_3d[:, :2]

    # Centre window on the airfoils' mean position.
    cx = float(surf2d[:, 0].mean())
    cy = float(surf2d[:, 1].mean())
    in_view = (
        (pos2d[:, 0] > cx - ZOOM_HALF_X) & (pos2d[:, 0] < cx + ZOOM_HALF_X) &
        (pos2d[:, 1] > cy - ZOOM_HALF_Y) & (pos2d[:, 1] < cy + ZOOM_HALF_Y)
    )
    local_pos = pos2d[in_view]
    surf_in_view = (
        (surf2d[:, 0] > cx - ZOOM_HALF_X) & (surf2d[:, 0] < cx + ZOOM_HALF_X) &
        (surf2d[:, 1] > cy - ZOOM_HALF_Y) & (surf2d[:, 1] < cy + ZOOM_HALF_Y)
    )
    local_surf = surf2d[surf_in_view]

    # UDF from 3D positions to 3D surface, then pull adaptive k per node.
    print(f"  {SAMPLE}: N={len(pos):,}, view keeps {in_view.sum():,} points")
    udf = nearest_surface_udf(pos, surface_3d)
    k_vals = adaptive_k(udf, K_NEAR, K_FAR, LAMBDA)

    # Build 2D kNN for visualisation (graph in poster is 2D slice).
    tree2d = cKDTree(pos2d)
    _, nbrs = tree2d.query(pos2d, k=K_NEAR + 1)
    nbrs = nbrs[:, 1:]                                 # drop self

    knn_uniform = nbrs[:, :K_UNIFORM]
    knn_adaptive = np.full((len(pos2d), K_NEAR), -1, dtype=np.int64)
    for i, ki in enumerate(k_vals):
        knn_adaptive[i, :ki] = nbrs[i, :ki]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)

    # Left: uniform k
    ax = axes[0]
    draw_edges(ax, pos2d, knn_uniform, in_view, MAX_EDGES_PER_PANEL,
               "#88aacc", 0.40)
    ax.scatter(local_pos[:, 0], local_pos[:, 1], s=0.8, c="#2b4a6b",
               alpha=0.7, zorder=2)
    ax.scatter(local_surf[:, 0], local_surf[:, 1], s=3, c="#c44e52",
               zorder=3, label="airfoil surface")
    ax.set_xlim(cx - ZOOM_HALF_X, cx + ZOOM_HALF_X)
    ax.set_ylim(cy - ZOOM_HALF_Y, cy + ZOOM_HALF_Y)
    ax.set_aspect("equal")
    ax.set_title(rf"Uniform  $k = {K_UNIFORM}$",
                 fontsize=12, fontweight="bold", color="#000061")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Right: adaptive k, coloured
    ax = axes[1]
    draw_edges(ax, pos2d, knn_adaptive, in_view, MAX_EDGES_PER_PANEL,
               "#888", 0.25)
    local_k = k_vals[in_view]
    sc = ax.scatter(local_pos[:, 0], local_pos[:, 1], s=1.0, c=local_k,
                    cmap="plasma", vmin=K_FAR, vmax=K_NEAR, alpha=0.85,
                    zorder=2)
    ax.scatter(local_surf[:, 0], local_surf[:, 1], s=3, c="#c44e52",
               zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(r"$k_i$", fontsize=10)
    ax.set_xlim(cx - ZOOM_HALF_X, cx + ZOOM_HALF_X)
    ax.set_ylim(cy - ZOOM_HALF_Y, cy + ZOOM_HALF_Y)
    ax.set_aspect("equal")
    ax.set_title(
        rf"Adaptive  $k_{{\mathrm{{near}}}}{{=}}{K_NEAR}$, "
        rf"$k_{{\mathrm{{far}}}}{{=}}{K_FAR}$, $\lambda{{=}}{LAMBDA}$",
        fontsize=12, fontweight="bold", color="#000061",
    )
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    out = FIG / "adaptive_knn_03_comparison.png"
    out_pdf = FIG / "adaptive_knn_03_comparison.pdf"
    fig.savefig(out, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote: {out}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
