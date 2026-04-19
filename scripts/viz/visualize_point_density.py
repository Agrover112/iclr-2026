"""
Visualize local point density distribution across the point cloud.

Computes point density as: k-NN distance to k-th neighbor (smaller = denser).
Visualizes on 2D (XY top-down) view with heatmap coloring.

Usage:
    python scripts/visualize_point_density.py
    python scripts/visualize_point_density.py --k 8 --samples 3
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
    """Pick 3 samples per geometry class (1xxx, 2xxx, 3xxx)."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    seen_geom_ids = {}
    picked = {}
    for f in files:
        name = os.path.basename(f)
        prefix = name[0]
        geom_id = name.split('_')[0]
        if prefix not in seen_geom_ids:
            seen_geom_ids[prefix] = set()
            picked[prefix] = []
        if len(picked[prefix]) >= per_class:
            continue
        if geom_id not in seen_geom_ids[prefix]:
            seen_geom_ids[prefix].add(geom_id)
            picked[prefix].append(f)
    return [f for k in sorted(picked) for f in picked[k]]


def compute_density(pos, k=8):
    """Compute local point density as k-NN distance to k-th neighbor.

    Smaller distance = denser region.
    """
    tree = cKDTree(pos)
    dists, _ = tree.query(pos, k=k + 1)
    # Return distance to k-th neighbor (index k, since 0 is self)
    return dists[:, k]


def visualize_sample(sample_path, k, output_dir):
    """Generate density heatmap for one sample."""
    data = np.load(sample_path)
    pos = data['pos']
    idcs_airfoil = data['idcs_airfoil']
    sample_name = os.path.basename(sample_path).replace('.npz', '')

    print(f"  {sample_name}: computing density (k={k})...", end=' ', flush=True)
    density = compute_density(pos, k=k)
    print("rendering...", end=' ', flush=True)

    airfoil_pts = pos[idcs_airfoil]
    centroid = airfoil_pts.mean(axis=0)

    # Create figure with two subplots: full view + zoomed
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, radius, title_suf in zip(axes, [2.0, 0.6], ["full", "zoom"]):
        dists_xy = np.linalg.norm(pos[:, :2] - centroid[:2], axis=1)
        mask = dists_xy < radius
        local_pos = pos[mask]
        local_density = density[mask]

        # Scatter with density as color
        scatter = ax.scatter(
            local_pos[:, 0], local_pos[:, 1],
            c=local_density, s=1, cmap='viridis_r', alpha=0.7
        )

        # Overlay airfoil points
        airfoil_mask = np.array([idx in set(idcs_airfoil) for idx in np.where(mask)[0]])
        ax.scatter(
            local_pos[airfoil_mask, 0], local_pos[airfoil_mask, 1],
            s=2, c='red', alpha=0.5, label='Airfoil'
        )

        ax.set_xlim(centroid[0] - radius, centroid[0] + radius)
        ax.set_ylim(centroid[1] - radius, centroid[1] + radius)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title_suf} (r={radius})')
        ax.legend(fontsize=8)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('k-NN distance (k=8)', fontsize=9)

    fig.suptitle(f'Point Density — {sample_name}\n(darker = denser)', fontsize=12, y=1.02)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f'density_{sample_name}_k{k}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {out_path}")

    # Print statistics
    print(f"      Density stats (k-NN distance to 8-th neighbor):")
    print(f"        Min:    {density.min():.4f} (densest)")
    print(f"        Max:    {density.max():.4f} (sparsest)")
    print(f"        Mean:   {density.mean():.4f}")
    print(f"        Median: {np.median(density):.4f}")

    # Check if density is non-uniform (ratio of max to min)
    ratio = density.max() / (density.min() + 1e-6)
    if ratio > 3:
        print(f"        Ratio (max/min): {ratio:.1f} — NON-UNIFORM (like real CFD) ✓")
    else:
        print(f"        Ratio (max/min): {ratio:.1f} — UNIFORM (warping flattened it)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='figures')
    parser.add_argument('--samples', type=int, default=3,
                        help='Number of samples per geometry class')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    samples = pick_samples(args.data_dir, per_class=args.samples)
    print(f"Point Density Analysis (k={args.k})\n")
    print(f"Selected {len(samples)} samples:")
    for s in samples:
        print(f"  {os.path.basename(s)}")
    print()

    for sample_path in samples:
        visualize_sample(sample_path, args.k, args.output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
