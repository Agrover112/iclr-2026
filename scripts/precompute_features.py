"""
Precompute per-point geometric features for all NPZ files in the dataset.

Run once before training:
    /home/agrov/gram/bin/python scripts/precompute_features.py

By default computes all registered features and saves alongside each NPZ:
    data/1021_1-0.npz      (existing)
    data/1021_1-0_feat.pt  (new — dict mapping feature_name -> Tensor)

To compute only specific features:
    python scripts/precompute_features.py --features udf_truncated udf_gradient

To recompute even if cache already exists:
    python scripts/precompute_features.py --overwrite
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

# Allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch.nn.functional as F
from src.features import FEATURE_REGISTRY, _chunked_nn_search, feat_cache_path


def precompute_file(npz_path: str, feature_names: list[str]) -> dict[str, torch.Tensor]:
    """Compute all requested features for one NPZ file.

    Uses fused NN search when udf_truncated + udf_gradient are both requested
    (one distance matrix pass instead of two).
    """
    data = np.load(npz_path)
    pos = torch.from_numpy(data["pos"]).float()                  # (N, 3)
    idcs_airfoil = torch.from_numpy(data["idcs_airfoil"]).long() # (M,)
    surface_pts = pos[idcs_airfoil]                              # (M, 3)

    features = {}
    fused = {"udf_truncated", "udf_gradient"}
    names_set = set(feature_names)

    # Fused fast path: one NN search for both udf_truncated and udf_gradient
    if fused.issubset(names_set):
        min_dists, nearest_pts = _chunked_nn_search(pos, surface_pts)
        features["udf_truncated"] = min_dists.clamp(max=0.5).unsqueeze(1)
        features["udf_gradient"]  = F.normalize(nearest_pts - pos, dim=1)

    # Compute any remaining features individually
    for name in feature_names:
        if name not in features:
            features[name] = FEATURE_REGISTRY[name](pos, surface_pts)

    return features


def _worker(args):
    """Top-level function required for multiprocessing pickling."""
    torch.set_num_threads(1)  # prevent each worker spawning many threads → oversubscription
    npz_path, feature_names, overwrite = args
    out_path = feat_cache_path(npz_path)

    # Check which features are missing from the existing cache
    missing = list(feature_names)
    existing = {}
    if os.path.exists(out_path):
        if not overwrite:
            existing = torch.load(out_path, weights_only=True)
            missing = [f for f in feature_names if f not in existing]
            if not missing:
                return ("skipped", npz_path, 0.0)

    try:
        t0 = time.perf_counter()
        new_features = precompute_file(npz_path, missing)
        # Merge with existing cache
        existing.update(new_features)
        torch.save(existing, out_path)
        return ("done", npz_path, time.perf_counter() - t0)
    except Exception as e:
        return ("error", npz_path, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/agrov/iclr-2026/data")
    parser.add_argument(
        "--features", nargs="+", default=list(FEATURE_REGISTRY.keys()),
        help=f"Features to compute. Available: {sorted(FEATURE_REGISTRY)}",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if cache file already exists.")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(),
                        help="Parallel worker processes (default: all cores).")
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(npz_files)} NPZ files")
    print(f"Features: {args.features}")
    print(f"Workers:  {args.workers}")
    print(f"Output:   {{stem}}_feat.pt alongside each NPZ")
    print()

    work = [(npz, args.features, args.overwrite) for npz in npz_files]
    skipped = done = errors = 0
    t0 = time.perf_counter()

    with mp.Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, work)):
            status, path, info = result
            name = os.path.basename(path)
            if status == "done":
                done += 1
                print(f"  [{done+skipped:3d}/{len(npz_files)}] {name}  {info:.2f}s")
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"  ERROR {name}: {info}")

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.1f}s — computed: {done}, skipped: {skipped}, errors: {errors}")
    if done > 0:
        print(f"Avg per file: {total * args.workers / done:.2f}s (wall: {total/done:.2f}s)")


if __name__ == "__main__":
    main()
