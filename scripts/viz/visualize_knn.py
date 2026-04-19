"""
Visualize k-NN graph connectivity on representative samples.

Picks one sample per geometry class (1xxx/2xxx/3xxx) and generates
2D scatter plots (top-down XY view) with k-NN edges at two zoom levels.

Usage:
    python scripts/visualize_knn.py
    python scripts/visualize_knn.py --k 6 8 16 --radius 1.5
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pick_samples(data_dir, per_class=3):
    """Pick multiple samples per geometry class (1xxx, 2xxx, 3xxx).

    Picks from different geometry_ids within each class for variety.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    # Group by geometry class prefix AND geometry_id
    seen_geom_ids = {}  # prefix -> set of geometry_ids already picked
    picked = {}         # prefix -> list of file paths
    for f in files:
        name = os.path.basename(f)
        prefix = name[0]
        geom_id = name.split('_')[0]  # e.g. '1021'
        if prefix not in seen_geom_ids:
            seen_geom_ids[prefix] = set()
            picked[prefix] = []
        if len(picked[prefix]) >= per_class:
            continue
        # Prefer different geometry_ids for variety
        if geom_id not in seen_geom_ids[prefix]:
            seen_geom_ids[prefix].add(geom_id)
            picked[prefix].append(f)
    return [f for k in sorted(picked) for f in picked[k]]


def plot_knn_graph(pos, idcs_airfoil, k, ax, radius, title, max_edges=20000):
    """Plot 2D (XY top-down) view of point cloud with k-NN edges near the airfoil."""
    airfoil_pts = pos[idcs_airfoil]
    centroid = airfoil_pts.mean(axis=0)

    # Select points within radius of airfoil centroid (XY plane)
    dists_xy = np.linalg.norm(pos[:, :2] - centroid[:2], axis=1)
    mask = dists_xy < radius
    local_pos = pos[mask]
    local_indices = np.where(mask)[0]

    airfoil_set = set(idcs_airfoil.tolist())
    is_airfoil = np.array([idx in airfoil_set for idx in local_indices])

    # Build k-NN on local points
    tree = cKDTree(local_pos)
    _, neighbors = tree.query(local_pos, k=k + 1)

    # Collect edges, subsample if needed
    edges = []
    for i in range(len(local_pos)):
        for j in neighbors[i, 1:]:
            if j < len(local_pos):
                edges.append((i, j))

    if len(edges) > max_edges:
        rng = np.random.default_rng(42)
        edge_indices = rng.choice(len(edges), max_edges, replace=False)
        edges = [edges[ei] for ei in edge_indices]

    # Plot edges
    for i, j in edges:
        ax.plot(
            [local_pos[i, 0], local_pos[j, 0]],
            [local_pos[i, 1], local_pos[j, 1]],
            color='#cccccc', linewidth=0.15, zorder=1,
        )

    # Volume points
    vol_mask = ~is_airfoil
    ax.scatter(
        local_pos[vol_mask, 0], local_pos[vol_mask, 1],
        s=0.3, c='steelblue', alpha=0.5, zorder=2, label='Volume',
    )

    # Airfoil points
    ax.scatter(
        local_pos[is_airfoil, 0], local_pos[is_airfoil, 1],
        s=1.0, c='red', alpha=0.8, zorder=3, label='Airfoil',
    )

    ax.set_xlim(centroid[0] - radius, centroid[0] + radius)
    ax.set_ylim(centroid[1] - radius, centroid[1] + radius)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, markerscale=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return len(local_pos), len(edges)


def visualize_sample(sample_path, k_values, radius, output_dir):
    """Generate wide + tight zoom figures for one sample."""
    data = np.load(sample_path)
    pos = data['pos']
    idcs_airfoil = data['idcs_airfoil']
    sample_name = os.path.basename(sample_path).replace('.npz', '')

    print(f"\n{sample_name}: {pos.shape[0]:,} pts, {len(idcs_airfoil):,} airfoil pts")

    for zoom_label, r in [("wide", radius), ("tight", radius / 3)]:
        fig, axes = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 6))
        if len(k_values) == 1:
            axes = [axes]

        for ax, k in zip(axes, k_values):
            print(f"  k={k}, {zoom_label} (r={r:.2f})...", end=" ")
            n_pts, n_edges = plot_knn_graph(pos, idcs_airfoil, k, ax, r, f'k={k}')
            print(f"{n_pts:,} pts, {n_edges:,} edges")

        fig.suptitle(f'{sample_name} — {zoom_label} zoom (r={r:.2f})', fontsize=14, y=1.02)
        fig.tight_layout()

        out_path = os.path.join(output_dir, f'knn_{sample_name}_{zoom_label}.png')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, nargs='+', default=[6, 8, 16])
    parser.add_argument('--radius', type=float, default=1.5)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    samples = pick_samples(args.data_dir)
    print(f"Selected {len(samples)} samples:")
    for s in samples:
        print(f"  {os.path.basename(s)}")

    for sample_path in samples:
        visualize_sample(sample_path, args.k, args.radius, args.output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
