"""
Visualize adaptive k-NN graph construction.

Shows the proposed rule:
    k(d) = k_far + (k_near - k_far) * exp(-d / sigma)
where d = UDF (distance to nearest airfoil surface point).

Produces 3 figures:
  adaptive_knn_01_curve.png      — k(d) curves for various sigma values
  adaptive_knn_02_assignment.png — per-point k assignment on a real point cloud
  adaptive_knn_03_comparison.png — edge density: uniform k=16 vs adaptive

Optional: save the computed adaptive graph to disk for use in training.

Usage:
    /home/agrov/gram/bin/python scripts/visualize_adaptive_knn.py
    /home/agrov/gram/bin/python scripts/visualize_adaptive_knn.py --sample data/1021_1-0.npz
    /home/agrov/gram/bin/python scripts/visualize_adaptive_knn.py --save-graph

Saved graph format (--save-graph):
    {npz_stem}_adaptive_knn_near{k_near}_far{k_far}_s{sigma}.pt
    Contains: {'knn_graph': (N, k_near) int32 padded with -1, 'k_near', 'k_far', 'sigma'}
    Load with: torch.load(path, weights_only=True)
    Switch to uniform k=16: use 'knn_graph' key from existing _feat.pt cache instead.
"""

import argparse
import glob
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.features import _chunked_min_dist


# ─────────────────────────────────────────────────────────────────
# Algorithm
# ─────────────────────────────────────────────────────────────────

def adaptive_k(udf: np.ndarray, k_near: int, k_far: int, sigma: float) -> np.ndarray:
    """Per-point k: k_i = max(k_far, round(k_far + (k_near-k_far)*exp(-d/sigma)))."""
    k_float = k_far + (k_near - k_far) * np.exp(-udf / sigma)
    return np.maximum(k_far, np.round(k_float).astype(int))


def build_adaptive_knn_graph(
    pos: np.ndarray,
    udf: np.ndarray,
    k_near: int,
    k_far: int,
    sigma: float,
) -> np.ndarray:
    """Build adaptive k-NN graph on 3D positions.

    Queries k_near neighbors for every point (one tree pass), then trims
    each row to k_i neighbors determined by UDF, padding unused slots with -1.

    Args:
        pos:    (N, 3) point positions
        udf:    (N,)  unsigned distance to nearest airfoil surface point
        k_near: maximum neighbors (near-surface points)
        k_far:  minimum neighbors (far-field points)
        sigma:  length-scale of the decay (dataset coordinates)

    Returns:
        (N, k_near) int32 array — neighbor indices, unused slots = -1
    """
    k_vals = adaptive_k(udf, k_near, k_far, sigma)
    tree = cKDTree(pos)
    _, nbrs_all = tree.query(pos, k=k_near + 1)   # (N, k_near+1) incl self
    nbrs_all = nbrs_all[:, 1:]                      # drop self → (N, k_near)

    knn = np.full((len(pos), k_near), -1, dtype=np.int32)
    for i, ki in enumerate(k_vals):
        knn[i, :ki] = nbrs_all[i, :ki]
    return knn


def adaptive_knn_save_path(npz_path: str, k_near: int, k_far: int, sigma: float) -> str:
    """Return the save path for an adaptive knn graph alongside its NPZ file.

    e.g. data/1021_1-0.npz → data/1021_1-0_adaptive_knn_near32_far8_s0.05.pt
    """
    stem = str(os.path.splitext(npz_path)[0])
    return f"{stem}_adaptive_knn_near{k_near}_far{k_far}_s{sigma}.pt"


# ─────────────────────────────────────────────────────────────────
# Figure 1: k(d) curves — pure math, no data
# ─────────────────────────────────────────────────────────────────

def plot_kdecay_curves(
    output_path: str,
    k_near: int = 32,
    k_far: int = 8,
    highlight_sigma: float = 0.05,
    error_threshold: float = 0.1,
):
    sigmas = [0.05, 0.10, 0.20, 0.40]
    colors = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d"]
    d = np.linspace(0, 0.60, 500)

    fig, ax = plt.subplots(figsize=(7, 4))

    for sigma, color in zip(sigmas, colors):
        k = k_far + (k_near - k_far) * np.exp(-d / sigma)
        lw = 2.5 if sigma == highlight_sigma else 1.5
        ax.plot(d, k, color=color, linewidth=lw, label=f"σ = {sigma}")
        # Mark d = sigma (1/e decay point)
        k_at_sigma = k_far + (k_near - k_far) * np.exp(-1.0)
        ax.plot(sigma, k_at_sigma, "o", color=color, markersize=5, zorder=5)

    ax.axhline(k_near, color="gray", linewidth=1.0, linestyle="--", label=f"k_near = {k_near}")
    ax.axhline(k_far,  color="gray", linewidth=1.0, linestyle="-.", label=f"k_far  = {k_far}")

    # Error threshold annotation (from LOG.md empirics)
    ax.axvline(
        error_threshold, color="black", linewidth=1.2, linestyle="--", zorder=4,
    )
    ax.text(
        error_threshold + 0.005,
        k_near - 1,
        f"d = {error_threshold}\nerror threshold\n(LOG.md)",
        fontsize=7.5,
        va="top",
        color="black",
    )

    ax.set_yticks(range(k_far, k_near + 1, 4))
    ax.set_ylim(k_far - 2, k_near + 2)
    ax.set_xlim(0, 0.60)
    ax.set_xlabel("d  (UDF: distance to nearest airfoil surface point)", fontsize=11)
    ax.set_ylabel("k  (number of neighbors)", fontsize=11)
    ax.set_title(
        f"Adaptive k-NN rule:  k(d) = k_far + (k_near − k_far) · exp(−d / σ)\n"
        f"  k_near = {k_near}  ·  k_far = {k_far}  ·  dots = 1/e point  ·  bold = recommended σ",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ─────────────────────────────────────────────────────────────────
# Figure 2: k assignment map on a real point cloud
# ─────────────────────────────────────────────────────────────────

def plot_k_assignment(
    pos: np.ndarray,
    surface_pts: np.ndarray,
    udf: np.ndarray,
    output_path: str,
    sigma: float = 0.05,
    k_near: int = 32,
    k_far: int = 8,
    view_radius: float = 2.0,
):
    k_vals = adaptive_k(udf, k_near, k_far, sigma)

    cx, cy = surface_pts[:, 0].mean(), surface_pts[:, 1].mean()
    mask = (
        (pos[:, 0] > cx - view_radius) & (pos[:, 0] < cx + view_radius) &
        (pos[:, 1] > cy - view_radius) & (pos[:, 1] < cy + view_radius)
    )
    local_pos = pos[mask]
    local_k   = k_vals[mask]

    surf_mask = (
        (surface_pts[:, 0] > cx - view_radius) & (surface_pts[:, 0] < cx + view_radius) &
        (surface_pts[:, 1] > cy - view_radius) & (surface_pts[:, 1] < cy + view_radius)
    )
    local_surf = surface_pts[surf_mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        local_pos[:, 0], local_pos[:, 1],
        c=local_k, s=0.4, cmap="plasma",
        vmin=k_far, vmax=k_near, alpha=0.7, zorder=2,
    )
    ax.scatter(local_surf[:, 0], local_surf[:, 1],
               s=1.2, c="red", alpha=0.9, zorder=3, label="airfoil surface")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f"k_i  (adaptive, σ={sigma})", fontsize=10)
    cbar.set_ticks(range(k_far, k_near + 1, 4))

    ax.set_xlim(cx - view_radius, cx + view_radius)
    ax.set_ylim(cy - view_radius, cy + view_radius)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(
        f"Adaptive k assignment  (k_near={k_near}, k_far={k_far}, σ={sigma})\n"
        "Warm = many neighbors (near surface) · Cool = few neighbors (far field)"
    )
    ax.legend(fontsize=8, markerscale=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ─────────────────────────────────────────────────────────────────
# Figure 3: Edge comparison — uniform k=16 vs adaptive, tight zoom
# ─────────────────────────────────────────────────────────────────

def _draw_edges(ax, pos2d, knn, center, zoom_r, max_edges, color, lw):
    cx, cy = center
    in_view = (
        (pos2d[:, 0] > cx - zoom_r) & (pos2d[:, 0] < cx + zoom_r) &
        (pos2d[:, 1] > cy - zoom_r) & (pos2d[:, 1] < cy + zoom_r)
    )
    src_ids = np.where(in_view)[0]
    edges = [(s, int(d)) for s in src_ids for d in knn[s] if d >= 0]
    random.shuffle(edges)
    for s, d in edges[:max_edges]:
        ax.plot(
            [pos2d[s, 0], pos2d[d, 0]],
            [pos2d[s, 1], pos2d[d, 1]],
            color=color, linewidth=lw, alpha=0.4, zorder=1,
        )


def plot_edge_comparison(
    pos: np.ndarray,
    surface_pts: np.ndarray,
    udf: np.ndarray,
    output_path: str,
    k_uniform: int = 16,
    k_near: int = 32,
    k_far: int = 8,
    sigma: float = 0.05,
    zoom_r: float = 0.20,
    max_edges: int = 8000,
):
    pos2d  = pos[:, :2]
    surf2d = surface_pts[:, :2]

    # Leading edge: leftmost airfoil point
    le_idx = np.argmin(surface_pts[:, 0])
    cx, cy = surface_pts[le_idx, 0], surface_pts[le_idx, 1]

    in_view = (
        (pos2d[:, 0] > cx - zoom_r) & (pos2d[:, 0] < cx + zoom_r) &
        (pos2d[:, 1] > cy - zoom_r) & (pos2d[:, 1] < cy + zoom_r)
    )
    local_ids  = np.where(in_view)[0]
    local_pos  = pos2d[local_ids]
    surf_in_view = (
        (surf2d[:, 0] > cx - zoom_r) & (surf2d[:, 0] < cx + zoom_r) &
        (surf2d[:, 1] > cy - zoom_r) & (surf2d[:, 1] < cy + zoom_r)
    )
    local_surf = surf2d[surf_in_view]

    k_vals = adaptive_k(udf, k_near, k_far, sigma)

    # Build graphs in 2D (visualization only — saved graph uses 3D, see --save-graph)
    print("    Building 2D k-NN tree for visualization...")
    tree = cKDTree(pos2d)
    _, nbrs_all = tree.query(pos2d, k=k_near + 1)
    nbrs_all = nbrs_all[:, 1:]

    knn_uniform   = nbrs_all[:, :k_uniform]
    knn_adaptive  = np.full((len(pos2d), k_near), -1, dtype=np.int32)
    for i, ki in enumerate(k_vals):
        knn_adaptive[i, :ki] = nbrs_all[i, :ki]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    titles = [f"Uniform  k = {k_uniform}", f"Adaptive  k_near={k_near}  k_far={k_far}  σ={sigma}"]
    knns   = [knn_uniform, knn_adaptive]

    for ax, knn, title in zip(axes, knns, titles):
        _draw_edges(ax, pos2d, knn, (cx, cy), zoom_r, max_edges, "#aaaaaa", 0.4)

        if knn is knn_adaptive:
            local_k = k_vals[local_ids]
            sc = ax.scatter(local_pos[:, 0], local_pos[:, 1],
                            c=local_k, s=1.5, cmap="plasma",
                            vmin=k_far, vmax=k_near, alpha=0.8, zorder=2)
            plt.colorbar(sc, ax=ax, label="k_i", shrink=0.8)
        else:
            ax.scatter(local_pos[:, 0], local_pos[:, 1],
                       s=1.5, c="steelblue", alpha=0.7, zorder=2)

        ax.scatter(local_surf[:, 0], local_surf[:, 1], s=4, c="red", alpha=0.9, zorder=3)
        ax.set_xlim(cx - zoom_r, cx + zoom_r)
        ax.set_ylim(cy - zoom_r, cy + zoom_r)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    fig.suptitle(
        "Edge density comparison — tight zoom near leading edge\n"
        "(red = airfoil surface · edges subsampled to avoid overplotting)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize adaptive k-NN construction")
    parser.add_argument("--data-dir",   default="/home/agrov/iclr-2026/data")
    parser.add_argument("--output-dir", default="/home/agrov/iclr-2026/figures")
    parser.add_argument("--sample",     default=None,
                        help="Path to a specific .npz file. Defaults to first in data-dir.")
    parser.add_argument("--k-near",  type=int,   default=32)
    parser.add_argument("--k-far",   type=int,   default=8)
    parser.add_argument("--sigma",   type=float, default=0.05)
    parser.add_argument("--zoom-r",  type=float, default=0.20,
                        help="Half-width of tight-zoom window for figure 3")
    parser.add_argument("--save-graph", action="store_true",
                        help=(
                            "Save the adaptive k-NN graph to disk alongside the .npz file. "
                            "Filename: {stem}_adaptive_knn_near{k_near}_far{k_far}_s{sigma}.pt. "
                            "Graph is built on 3D positions. Padded to k_near cols with -1. "
                            "To switch to uniform k=8 or k=16 at training time, load "
                            "'knn_graph' from the existing _feat.pt cache instead."
                        ))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Figure 1: pure math ────────────────────────────────────────
    print("Figure 1: k(d) decay curves...")
    plot_kdecay_curves(
        os.path.join(args.output_dir, "adaptive_knn_01_curve.png"),
        k_near=args.k_near,
        k_far=args.k_far,
        highlight_sigma=args.sigma,
    )

    # ── Load sample data ───────────────────────────────────────────
    if args.sample:
        npz_path = args.sample
    else:
        files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
        if not files:
            print(f"No .npz files found in {args.data_dir}. Skipping figures 2 & 3.")
            return
        npz_path = files[0]

    print(f"\nLoading sample: {os.path.basename(npz_path)}")
    data        = np.load(npz_path)
    pos         = data["pos"].astype(np.float32)    # (N, 3)
    surface_pts = pos[data["idcs_airfoil"]]          # (M, 3)

    print("  Computing UDF (chunked)...")
    udf = _chunked_min_dist(
        torch.from_numpy(pos),
        torch.from_numpy(surface_pts),
    ).numpy()   # (N,)

    # ── Figure 2: k assignment map ─────────────────────────────────
    print("Figure 2: k assignment map...")
    plot_k_assignment(
        pos, surface_pts, udf,
        output_path=os.path.join(args.output_dir, "adaptive_knn_02_assignment.png"),
        sigma=args.sigma, k_near=args.k_near, k_far=args.k_far,
    )

    # ── Figure 3: edge comparison ──────────────────────────────────
    print("Figure 3: edge comparison (uniform vs adaptive)...")
    plot_edge_comparison(
        pos, surface_pts, udf,
        output_path=os.path.join(args.output_dir, "adaptive_knn_03_comparison.png"),
        k_uniform=16, k_near=args.k_near, k_far=args.k_far,
        sigma=args.sigma, zoom_r=args.zoom_r,
    )

    # ── Save graph (optional) ──────────────────────────────────────
    if args.save_graph:
        save_path = adaptive_knn_save_path(npz_path, args.k_near, args.k_far, args.sigma)
        print(f"\nBuilding 3D adaptive k-NN graph for saving...")
        knn = build_adaptive_knn_graph(pos, udf, args.k_near, args.k_far, args.sigma)
        torch.save(
            {
                "knn_graph": torch.from_numpy(knn),   # (N, k_near) int32, -1 = unused
                "k_near":    args.k_near,
                "k_far":     args.k_far,
                "sigma":     args.sigma,
            },
            save_path,
        )
        print(f"  Saved graph → {save_path}")
        print(f"  Shape: {knn.shape}  dtype: {knn.dtype}")
        print(f"  (To use uniform k=16 instead, load 'knn_graph' from _feat.pt)")

    print("\nDone.")


if __name__ == "__main__":
    main()
