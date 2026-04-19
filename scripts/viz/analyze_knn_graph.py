"""
Analyze kNN graph statistics across dataset files.

Computes statistics useful for designing GNN depth, attention radius,
SE(3) equivariant networks, and Transformers.

Statistics computed:
  - Edge length distribution (min/mean/max/std) — informs attention radius
  - Receptive field per layer (# unique nodes reachable after L hops)
  - Spatial reach per layer (3D distance covered after L hops)
  - Estimated diameter (via BFS sampling)
  - Clustering coefficient (how cliquey neighborhoods are)
  - Near-surface vs bulk edge length comparison
  - Expander quality metrics (spectral gap, per-hop expansion ratio, conductance)

Usage:
    # Analyze 10 random files (fast, ~1 min)
    /home/agrov/gram/bin/python scripts/analyze_knn_graph.py

    # Analyze more files for better statistics
    /home/agrov/gram/bin/python scripts/analyze_knn_graph.py --n-files 50

    # Analyze specific geometry type
    /home/agrov/gram/bin/python scripts/analyze_knn_graph.py --prefix 2
"""

import argparse
import glob
import os
import random
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.features import feat_cache_path

# BFS hops to analyze — also = minimum GNN depth to cover that receptive field
MAX_HOPS = 15
# Number of seed nodes to sample for BFS-based stats
N_BFS_SEEDS = 100


# ─────────────────────────────────────────────────────────────────
# Core analysis functions
# ─────────────────────────────────────────────────────────────────

def edge_length_stats(pos: np.ndarray, knn: np.ndarray) -> dict:
    """
    Distribution of edge lengths (Euclidean distance between connected points).
    Informs:
      - Attention radius for local Transformer / SE(3) network
      - Whether graph is local or long-range
    """
    N, k = knn.shape
    src = np.repeat(np.arange(N), k)           # (N*k,)
    dst = knn.reshape(-1)                       # (N*k,)
    diffs = pos[dst] - pos[src]                 # (N*k, 3)
    lengths = np.linalg.norm(diffs, axis=1)     # (N*k,)
    return {
        "min":    float(lengths.min()),
        "p5":     float(np.percentile(lengths, 5)),
        "mean":   float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p95":    float(np.percentile(lengths, 95)),
        "max":    float(lengths.max()),
        "std":    float(lengths.std()),
    }


def _bfs_one_seed(
    knn: np.ndarray,
    seed: int,
    max_hops: int,
    pos: np.ndarray | None = None,
) -> list[dict]:
    """
    Vectorized BFS from a single seed using boolean numpy arrays.

    Replaces Python set-based BFS. Key ops:
        knn[frontier].ravel()   — fetch all neighbor indices in one numpy op
        visited boolean array   — O(1) membership test via fancy indexing
        new_mask boolean array  — deduplication without np.unique (avoid sort)
        np.flatnonzero          — O(N) scan to extract frontier indices

    Returns list of length max_hops, each entry:
        visited_count : int    — total nodes visited including seed
        new_count     : int    — nodes added this hop (frontier size)
        max_dist      : float  — max 3D distance from seed to any visited node
                                 (only meaningful if pos is provided)
    """
    N = knn.shape[0]
    visited  = np.zeros(N, dtype=bool)
    new_mask = np.zeros(N, dtype=bool)   # reusable temp; reset after each hop
    visited[seed] = True
    frontier = np.array([seed], dtype=np.int32)

    seed_pos = pos[seed] if pos is not None else None
    running_max_dist = 0.0
    prev_count = 1
    hops = []

    for _ in range(max_hops):
        if len(frontier) == 0:
            # Graph exhausted — pad remaining hops with last values
            pad = {"visited_count": prev_count, "new_count": 0, "max_dist": running_max_dist}
            hops.extend([pad] * (max_hops - len(hops)))
            break

        neighbors = knn[frontier].ravel()           # (|frontier|*k,)
        new_mask[neighbors] = True                  # mark all candidates
        new_mask &= ~visited                        # keep only unvisited
        visited   |= new_mask                       # mark as visited

        frontier = np.flatnonzero(new_mask).astype(np.int32)
        new_mask[frontier] = False                  # reset for next hop

        visited_count = int(visited.sum())

        if pos is not None and len(frontier) > 0:
            # Incremental max distance: new frontier is the wavefront
            # running max is non-decreasing, so only check new nodes
            new_dists = np.linalg.norm(pos[frontier] - seed_pos, axis=1).max()
            running_max_dist = max(running_max_dist, float(new_dists))

        hops.append({
            "visited_count": visited_count,
            "new_count":     visited_count - prev_count,
            "max_dist":      running_max_dist,
        })
        prev_count = visited_count

    return hops


def bfs_stats_per_hop(
    pos: np.ndarray,
    knn: np.ndarray,
    seeds: list[int],
    max_hops: int,
) -> tuple[dict, dict, dict]:
    """
    Single BFS pass per seed computing receptive field, spatial reach,
    and expansion ratio all at once (was 3 separate BFS loops before).

    Returns:
        receptive_field  : {h: {mean, std}}  — nodes reachable at hop h
        spatial_reach    : {h: {mean, std}}  — max 3D distance at hop h
        expansion_ratio  : {h: float}        — new_count / prev_visited_count at hop h
    """
    hop_visited   = {h: [] for h in range(1, max_hops + 1)}
    hop_reach     = {h: [] for h in range(1, max_hops + 1)}
    hop_expansion = {h: [] for h in range(1, max_hops + 1)}

    for seed in seeds:
        hops = _bfs_one_seed(knn, seed, max_hops, pos=pos)
        prev_visited = 1
        for h, s in enumerate(hops, 1):
            hop_visited[h].append(s["visited_count"])
            hop_reach[h].append(s["max_dist"])
            ratio = s["new_count"] / prev_visited if prev_visited > 0 else 0.0
            hop_expansion[h].append(ratio)
            prev_visited = s["visited_count"]

    receptive_field = {
        h: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for h, v in hop_visited.items()
    }
    spatial_reach = {
        h: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for h, v in hop_reach.items()
    }
    expansion_ratio = {h: float(np.mean(v)) for h, v in hop_expansion.items()}

    return receptive_field, spatial_reach, expansion_ratio


def estimate_diameter(knn: np.ndarray, n_seeds: int = 10) -> int:
    """
    Estimate graph diameter via BFS from random seeds.
    True diameter is expensive (O(N²)); this is a lower bound.
    Informs minimum GNN depth needed to propagate information across the graph.
    """
    N = knn.shape[0]
    max_depth = 0
    seeds = random.sample(range(N), min(n_seeds, N))

    for seed in seeds:
        # Vectorized BFS to full convergence (no hop cap)
        visited  = np.zeros(N, dtype=bool)
        new_mask = np.zeros(N, dtype=bool)
        visited[seed] = True
        frontier = np.array([seed], dtype=np.int32)
        depth = 0
        while len(frontier) > 0:
            neighbors = knn[frontier].ravel()
            new_mask[neighbors] = True
            new_mask &= ~visited
            visited   |= new_mask
            frontier   = np.flatnonzero(new_mask).astype(np.int32)
            new_mask[frontier] = False
            if len(frontier) > 0:
                depth += 1
        max_depth = max(max_depth, depth)

    return max_depth


def clustering_coefficient(knn: np.ndarray, sample_size: int = 1000) -> float:
    """
    Average local clustering coefficient on a random sample of nodes.
    High value → neighbors are also connected to each other (cliquey).
    Low value → neighbors don't know each other (tree-like, info spreads fast).
    Informs whether deeper GNNs get rapidly diminishing returns.
    """
    knn_set = [set(knn[i].tolist()) for i in range(len(knn))]
    nodes = random.sample(range(len(knn)), min(sample_size, len(knn)))
    coeffs = []
    k = knn.shape[1]
    for node in nodes:
        neighbors = knn_set[node]
        edges_among = sum(
            1 for nb in neighbors if neighbors & knn_set[nb]
        )
        possible = k * (k - 1)
        coeffs.append(edges_among / possible if possible > 0 else 0.0)
    return float(np.mean(coeffs))


def expander_metrics(
    knn: np.ndarray,
    expansion_ratio_per_hop: dict,
    subgraph_size: int = 3000,
) -> dict:
    """
    Expander graph quality metrics.

    A good expander = information spreads fast, few hops needed for global context.

    Metrics:
      1. Per-hop expansion ratio: pre-computed by bfs_stats_per_hop (no extra BFS here)

      2. Spectral gap (λ₂ of normalized Laplacian on a subgraph sample):
         → close to 0 = weak expander / disconnected clusters
         → close to 1 = excellent expander (Ramanujan-like)
         Computed on a BFS subgraph of `subgraph_size` nodes (full 100k too large
         for sparse eigensolver in reasonable time).

      3. Approximate conductance (Cheeger constant lower bound):
         φ ≈ λ₂ / 2  (Cheeger inequality: λ₂/2 ≤ φ ≤ √(2λ₂))
         Conductance measures worst-case bottleneck: how hard is it to cut the graph
         into two balanced halves?
         → high φ = no bottlenecks, message passing flows freely
    """
    N = knn.shape[0]

    # ── Spectral gap on BFS subgraph (vectorized) ──────────────────
    # Grow subgraph from a random seed via vectorized BFS
    sub_seed = random.randint(0, N - 1)
    visited  = np.zeros(N, dtype=bool)
    new_mask = np.zeros(N, dtype=bool)
    visited[sub_seed] = True
    frontier = np.array([sub_seed], dtype=np.int32)

    while visited.sum() < subgraph_size and len(frontier) > 0:
        neighbors = knn[frontier].ravel()
        new_mask[neighbors] = True
        new_mask &= ~visited
        visited   |= new_mask
        frontier   = np.flatnonzero(new_mask).astype(np.int32)
        new_mask[frontier] = False

    sub_nodes = np.flatnonzero(visited)[:subgraph_size]
    node_to_idx = {int(n): i for i, n in enumerate(sub_nodes)}
    n = len(sub_nodes)

    # Build symmetric adjacency (keep only edges within subgraph)
    rows, cols = [], []
    for i, node in enumerate(sub_nodes):
        for nb in knn[node].tolist():
            if nb in node_to_idx:
                j = node_to_idx[nb]
                rows.append(i); cols.append(j)
                rows.append(j); cols.append(i)

    A = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n), dtype=np.float32
    )
    A = (A > 0).astype(np.float32)  # binarize (deduplicate)

    # Normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    degrees = np.asarray(A.sum(axis=1)).flatten()
    degrees = np.maximum(degrees, 1e-9)
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
    L = sp.eye(n) - d_inv_sqrt @ A @ d_inv_sqrt

    # Two smallest eigenvalues: λ₀ ≈ 0 (trivial), λ₁ = spectral gap
    try:
        eigenvalues = scipy.sparse.linalg.eigsh(
            L, k=2, which="SM", tol=1e-4, maxiter=1000, return_eigenvectors=False
        )
        eigenvalues = np.sort(np.abs(eigenvalues))
        spectral_gap = float(eigenvalues[1])   # λ₂ (second smallest)
    except Exception:
        spectral_gap = float("nan")

    # Cheeger bound: φ ≈ λ₂ / 2
    conductance_lb = spectral_gap / 2.0

    return {
        "expansion_ratio_per_hop": expansion_ratio_per_hop,
        "spectral_gap":            spectral_gap,
        "conductance_lower_bound": conductance_lb,
        "subgraph_size":           n,
    }


def near_vs_bulk_edge_lengths(pos: np.ndarray, knn: np.ndarray, udf: np.ndarray,
                               near_thresh: float = 0.1) -> dict:
    """
    Compare edge lengths near the airfoil vs in the bulk.
    Near-surface: mesh is refined → shorter edges, denser graph.
    Bulk: coarser → longer edges.
    Informs whether a single k works well everywhere or adaptive k would help.
    """
    N, k = knn.shape
    src = np.repeat(np.arange(N), k)
    dst = knn.reshape(-1)
    lengths = np.linalg.norm(pos[dst] - pos[src], axis=1)

    near_mask = udf[src] < near_thresh
    bulk_mask = ~near_mask

    def stats(arr):
        return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(len(arr))}

    return {
        f"near_surface (udf<{near_thresh})": stats(lengths[near_mask]),
        "bulk": stats(lengths[bulk_mask]),
    }


# ─────────────────────────────────────────────────────────────────
# Per-file analysis
# ─────────────────────────────────────────────────────────────────

def analyze_file(npz_path: str) -> dict | None:
    cache_path = feat_cache_path(npz_path)
    if not os.path.exists(cache_path):
        return None

    data = np.load(npz_path)
    pos = data["pos"].astype(np.float32)           # (N, 3)

    cache = torch.load(cache_path, weights_only=True)
    if "knn_graph" not in cache:
        return None
    knn = cache["knn_graph"].numpy().astype(np.int32)  # (N, 16)

    udf = None
    if "udf_truncated" in cache:
        udf = cache["udf_truncated"].numpy().squeeze()  # (N,)

    seeds = random.sample(range(len(pos)), N_BFS_SEEDS)

    # Single BFS pass per seed for receptive field + spatial reach + expansion ratio
    receptive_field, spatial_reach, expansion_ratio = bfs_stats_per_hop(
        pos, knn, seeds, MAX_HOPS
    )

    result = {}
    result["edge_lengths"]     = edge_length_stats(pos, knn)
    result["receptive_field"]  = receptive_field
    result["spatial_reach"]    = spatial_reach
    result["diameter_est"]     = estimate_diameter(knn, n_seeds=10)
    result["clustering_coeff"] = clustering_coefficient(knn)
    result["expander"]         = expander_metrics(knn, expansion_ratio)
    if udf is not None:
        result["near_vs_bulk"] = near_vs_bulk_edge_lengths(pos, knn, udf)

    return result


# ─────────────────────────────────────────────────────────────────
# Aggregate and print
# ─────────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    def mean_over(key, subkey):
        vals = [r[key][subkey] for r in results if key in r]
        return float(np.mean(vals))

    agg = {}

    # Edge lengths
    for stat in ["min", "p5", "mean", "median", "p95", "max", "std"]:
        agg[f"edge_length_{stat}"] = mean_over("edge_lengths", stat)

    # Receptive field per hop
    agg["receptive_field"] = {}
    for h in range(1, MAX_HOPS + 1):
        vals = [r["receptive_field"][h]["mean"] for r in results if "receptive_field" in r]
        agg["receptive_field"][h] = float(np.mean(vals))

    # Spatial reach per hop
    agg["spatial_reach"] = {}
    for h in range(1, MAX_HOPS + 1):
        vals = [r["spatial_reach"][h]["mean"] for r in results if "spatial_reach" in r]
        agg["spatial_reach"][h] = float(np.mean(vals))

    # Diameter
    agg["diameter_est"] = float(np.mean([r["diameter_est"] for r in results]))

    # Clustering
    agg["clustering_coeff"] = float(np.mean([r["clustering_coeff"] for r in results]))

    # Expander metrics
    agg["spectral_gap"] = float(np.nanmean([r["expander"]["spectral_gap"] for r in results]))
    agg["conductance_lb"] = float(np.nanmean([r["expander"]["conductance_lower_bound"] for r in results]))
    agg["expansion_ratio_per_hop"] = {}
    for h in range(1, MAX_HOPS + 1):
        vals = [r["expander"]["expansion_ratio_per_hop"][h] for r in results if "expander" in r]
        agg["expansion_ratio_per_hop"][h] = float(np.mean(vals))

    return agg


def print_report(agg: dict, n_files: int, k: int):
    print(f"\n{'═'*60}")
    print(f"  kNN Graph Analysis  ({n_files} files, k={k})")
    print(f"{'═'*60}")

    print("\n── Edge Length Distribution ──────────────────────────────")
    print(f"  min    : {agg['edge_length_min']:.4f}")
    print(f"  p5     : {agg['edge_length_p5']:.4f}")
    print(f"  mean   : {agg['edge_length_mean']:.4f}  ← typical interaction distance")
    print(f"  median : {agg['edge_length_median']:.4f}")
    print(f"  p95    : {agg['edge_length_p95']:.4f}  ← safe attention radius")
    print(f"  max    : {agg['edge_length_max']:.4f}")
    print(f"  std    : {agg['edge_length_std']:.4f}")
    print(f"\n  → Transformer local attention radius: ~{agg['edge_length_p95']:.3f}")
    print(f"  → SE(3) cutoff radius: ~{agg['edge_length_p95']:.3f}")

    print("\n── Receptive Field per GNN Layer ─────────────────────────")
    print(f"  {'Hops':>5}  {'Nodes reachable':>16}  {'% of 100k':>10}")
    for h, mean_nodes in agg["receptive_field"].items():
        pct = 100 * mean_nodes / 100_000
        bar = "█" * min(40, int(pct / 2.5))
        print(f"  {h:>5}  {mean_nodes:>16,.0f}  {pct:>9.1f}%  {bar}")
    print(f"\n  → GNN depth for ~global context: "
          f"{next((h for h, v in agg['receptive_field'].items() if v > 50_000), MAX_HOPS)}+ layers")

    print("\n── Spatial Reach per GNN Layer ───────────────────────────")
    print(f"  {'Hops':>5}  {'Max 3D distance':>16}")
    for h, mean_reach in agg["spatial_reach"].items():
        print(f"  {h:>5}  {mean_reach:>16.4f}")

    print(f"\n── Diameter Estimate (BFS lower bound) ───────────────────")
    print(f"  ~{agg['diameter_est']:.1f} hops")
    print(f"  → GNN needs ≥{int(agg['diameter_est'])} layers for full graph coverage")

    print(f"\n── Clustering Coefficient ────────────────────────────────")
    print(f"  {agg['clustering_coeff']:.4f}  (0=tree-like, 1=fully cliquey)")
    if agg['clustering_coeff'] > 0.5:
        print(f"  → High clustering: deeper GNNs may over-smooth quickly")
    else:
        print(f"  → Low clustering: information spreads efficiently per layer")

    print(f"\n── Expander Quality ──────────────────────────────────────")
    sg = agg['spectral_gap']
    phi = agg['conductance_lb']
    print(f"  Spectral gap (λ₂ of norm. Laplacian, subgraph): {sg:.4f}")
    print(f"  Conductance lower bound (λ₂/2):                 {phi:.4f}")
    if sg > 0.5:
        quality = "excellent expander — information mixes fast"
    elif sg > 0.2:
        quality = "moderate expander — decent mixing"
    elif sg > 0.05:
        quality = "weak expander — clustering slows mixing"
    else:
        quality = "poor expander / near-disconnected — bottlenecks present"
    print(f"  → {quality}")

    print(f"\n  Per-hop expansion ratio (mean new nodes / visited so far):")
    print(f"  {'Hop':>5}  {'Expansion ratio':>16}  note")
    for h, ratio in agg["expansion_ratio_per_hop"].items():
        note = ""
        if ratio > 0.5:
            note = "strong growth"
        elif ratio > 0.1:
            note = "moderate growth"
        elif ratio > 0.01:
            note = "slowing"
        else:
            note = "saturated"
        print(f"  {h:>5}  {ratio:>16.3f}  {note}")

    print(f"\n{'═'*60}")
    print(f"\n  GNN Design Recommendations")
    print(f"{'─'*60}")
    depth_for_global = next((h for h, v in agg['receptive_field'].items() if v > 50_000), MAX_HOPS)
    print(f"  Depth          : {depth_for_global}–{depth_for_global+2} layers for semi-global context")
    print(f"  Attention radius: {agg['edge_length_p95']:.3f} (covers 95% of edges)")
    print(f"  Hidden dim     : scale with depth — deeper → wider to avoid bottleneck")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/agrov/iclr-2026/data")
    parser.add_argument("--n-files", type=int, default=10,
                        help="Number of files to analyze (default: 10). Ignored if --one-per-geometry.")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Filter by geometry prefix, e.g. '1' for single-airfoil")
    parser.add_argument("--one-per-geometry", action="store_true",
                        help="Deterministically pick one file per unique geometry_id. "
                             "Most correct estimate: pos is identical within a geometry so "
                             "additional files add zero new information about graph structure.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=16)
    args = parser.parse_args()

    random.seed(args.seed)

    all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.prefix:
        all_files = [f for f in all_files if os.path.basename(f).startswith(args.prefix)]

    if args.one_per_geometry:
        import re
        seen = set()
        files = []
        for f in all_files:
            m = re.match(r"(\d+)_", os.path.basename(f))
            if m:
                gid = m.group(1)
                if gid not in seen:
                    seen.add(gid)
                    files.append(f)
        print(f"One-per-geometry mode: {len(files)} unique geometries found.")
    else:
        files = random.sample(all_files, min(args.n_files, len(all_files)))
    print(f"Analyzing {len(files)} files (k={args.knn_k})...")

    results = []
    for i, path in enumerate(files):
        t0 = time.perf_counter()
        r = analyze_file(path)
        elapsed = time.perf_counter() - t0
        name = os.path.basename(path)
        if r:
            results.append(r)
            print(f"  [{i+1}/{len(files)}] {name}  {elapsed:.1f}s")
        else:
            print(f"  [{i+1}/{len(files)}] {name}  SKIPPED (no knn_graph in cache)")

    if not results:
        print("No results — run precompute_features.py --features knn_graph first.")
        return

    agg = aggregate(results)
    print_report(agg, len(results), args.knn_k)


if __name__ == "__main__":
    main()
