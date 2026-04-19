"""
3D rotating GIF of k-NN graph around the airfoil.

Generates one GIF per sample, 3 samples per geometry class (1xxx/2xxx/3xxx).

Usage:
    python scripts/visualize_knn_3d.py
    python scripts/visualize_knn_3d.py --k 8 --frames 24 --radius 0.8
"""

import argparse
import glob
import io
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pick_samples(data_dir, per_class=3):
    """Pick multiple samples per geometry class (1xxx, 2xxx, 3xxx).

    Picks from different geometry_ids within each class for variety.
    """
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


def render_frame(local_pos, is_airfoil, edges, centroid, radius, elev, azim, k, sample_name):
    """Render one 3D frame and return as PIL Image."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i, j in edges:
        ax.plot(
            [local_pos[i, 0], local_pos[j, 0]],
            [local_pos[i, 1], local_pos[j, 1]],
            [local_pos[i, 2], local_pos[j, 2]],
            color='#cccccc', linewidth=0.2, zorder=1,
        )

    vol = ~is_airfoil
    ax.scatter(
        local_pos[vol, 0], local_pos[vol, 1], local_pos[vol, 2],
        s=0.2, c='steelblue', alpha=0.4, zorder=2,
    )
    ax.scatter(
        local_pos[is_airfoil, 0], local_pos[is_airfoil, 1], local_pos[is_airfoil, 2],
        s=0.8, c='red', alpha=0.7, zorder=3,
    )

    ax.set_xlim(centroid[0] - radius, centroid[0] + radius)
    ax.set_ylim(centroid[1] - radius, centroid[1] + radius)
    ax.set_zlim(centroid[2] - radius, centroid[2] + radius)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f'{sample_name} | k={k}', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_gif(sample_path, k, radius, n_frames, output_dir):
    """Generate a rotating 3D GIF for one sample."""
    data = np.load(sample_path)
    pos = data['pos']
    idcs_airfoil = data['idcs_airfoil']
    sample_name = os.path.basename(sample_path).replace('.npz', '')

    airfoil_pts = pos[idcs_airfoil]
    centroid = airfoil_pts.mean(axis=0)

    dists = np.linalg.norm(pos - centroid, axis=1)
    mask = dists < radius
    local_pos = pos[mask]
    local_indices = np.where(mask)[0]

    airfoil_set = set(idcs_airfoil.tolist())
    is_airfoil = np.array([idx in airfoil_set for idx in local_indices])

    tree = cKDTree(local_pos)
    _, neighbors = tree.query(local_pos, k=k + 1)

    edges = []
    for i in range(len(local_pos)):
        for j in neighbors[i, 1:]:
            if j < len(local_pos):
                edges.append((i, j))

    max_edges = 8000
    if len(edges) > max_edges:
        rng = np.random.default_rng(42)
        edge_indices = rng.choice(len(edges), max_edges, replace=False)
        edges = [edges[ei] for ei in edge_indices]

    print(f"  {sample_name}: {len(local_pos):,} pts, {len(edges):,} edges, {n_frames} frames")

    frames = []
    for i in range(n_frames):
        azim = (360 / n_frames) * i
        print(f"    frame {i+1}/{n_frames} (azim={azim:.0f})", end='\r', flush=True)
        img = render_frame(local_pos, is_airfoil, edges, centroid, radius, elev=25, azim=azim, k=k, sample_name=sample_name)
        frames.append(img)
    print()

    out_path = os.path.join(output_dir, f'knn_3d_{sample_name}_k{k}.gif')
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
    )
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--radius', type=float, default=0.8)
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    samples = pick_samples(args.data_dir)
    print(f"Generating 3D k-NN GIFs (k={args.k}, radius={args.radius}, {args.frames} frames)")
    print(f"Selected {len(samples)} samples:")
    for s in samples:
        print(f"  {os.path.basename(s)}")

    for sample_path in samples:
        make_gif(sample_path, args.k, args.radius, args.frames, args.output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
