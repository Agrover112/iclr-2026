"""Dataset loading and splitting with geometry-level stratification."""

import glob
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Dataset = object  # fallback base class for split-only usage
try:
    from omegaconf import DictConfig
except ImportError:
    DictConfig = None  # only needed for load_from_config


class GRAMDataset(Dataset):
    """PyTorch Dataset for GRaM competition data."""

    def __init__(self, file_paths: List[str], features: Optional[List[str]] = None,
                 max_distance: float = 0.0, log_udf: bool = False):
        """
        Initialize dataset with list of .npz file paths.

        Args:
            file_paths:   List of paths to .npz files
            features:     List of feature names to load from precomputed cache
                          (e.g. ["udf_truncated", "udf_gradient"]).
                          If None or empty, no features are loaded — model computes
                          them on the fly (slow but works without precomputation).
            max_distance: If > 0, filter out points whose UDF to the airfoil
                          exceeds this value. Airfoil points are always kept.
                          0 = no filtering (default).
            log_udf:      If True, replace the udf_truncated channel with
                          log(udf + 1e-4), renormalized to [0, 1]. Runtime
                          transform — no cache change. Motivated by law-of-the-wall
                          (u+ ∝ log(y+)): gives the decoder log-scale resolution
                          in the viscous sublayer where magnitude error is largest.
        """
        self.file_paths = file_paths
        self.features = features or []
        self.max_distance = max_distance
        self.log_udf = log_udf

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        """Load single sample and return as torch tensors."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required to load dataset samples")
        import torch
        from src.features import feat_cache_path

        data = np.load(self.file_paths[idx])

        # Extract fields from .npz
        t = torch.from_numpy(data['t']).float()  # (10,)
        pos = torch.from_numpy(data['pos']).float()  # (100k, 3)
        idcs_airfoil = torch.from_numpy(data['idcs_airfoil']).long()  # variable length
        velocity_in = torch.from_numpy(data['velocity_in']).float()  # (5, 100k, 3)
        velocity_out = torch.from_numpy(data['velocity_out']).float()  # (5, 100k, 3)

        # Load precomputed features from cache if requested
        point_features = None
        knn_graph = None
        if self.features:
            from src.features import GRAPH_FEATURES
            cache_path = feat_cache_path(self.file_paths[idx])
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, weights_only=True)
                float_parts = []
                udf_slice = None   # (start, end) in point_features if log_udf active
                col = 0
                for name in self.features:
                    if name not in cached:
                        continue  # feature missing from cache — will be computed on-the-fly
                    if name in GRAPH_FEATURES:
                        knn_graph = cached[name].long()  # (N, k) int64 indices; -1 = unused
                    else:
                        f = cached[name]
                        if f.dim() == 1:
                            f = f.unsqueeze(1)
                        if self.log_udf and name == "udf_truncated":
                            udf_slice = (col, col + f.shape[1])
                        col += f.shape[1]
                        float_parts.append(f)
                if float_parts:
                    point_features = torch.cat(float_parts, dim=1)
                    if udf_slice is not None:
                        import math
                        s, e = udf_slice
                        log_d = torch.log(point_features[:, s:e] + 1e-4)
                        lo, hi = math.log(1e-4), math.log(0.5 + 1e-4)
                        point_features[:, s:e] = (log_d - lo) / (hi - lo)

        # Filter distant points if max_distance is set
        if self.max_distance > 0:
            # Compute UDF: distance from each point to nearest airfoil point
            from src.features import _chunked_min_dist
            surface_pts = pos[idcs_airfoil]
            udf = _chunked_min_dist(pos, surface_pts)  # (N,)

            # Keep points within max_distance OR on the airfoil surface
            airfoil_mask = torch.zeros(pos.shape[0], dtype=torch.bool)
            airfoil_mask[idcs_airfoil] = True
            keep = (udf <= self.max_distance) | airfoil_mask

            # Apply mask to all point-indexed tensors
            pos = pos[keep]
            velocity_in = velocity_in[:, keep]
            velocity_out = velocity_out[:, keep]
            if point_features is not None:
                point_features = point_features[keep]
            if knn_graph is not None:
                # Remap neighbor indices to new point positions after filtering.
                # -1 entries (adaptive_knn_graph padding) map to -1 via the sentinel below.
                old_to_new_knn = torch.full((keep.shape[0],), -1, dtype=torch.long)
                old_to_new_knn[keep] = torch.arange(keep.sum())
                trimmed = knn_graph[keep]                         # (N_keep, k)
                valid = trimmed >= 0
                remapped = torch.where(valid, old_to_new_knn[trimmed.clamp(min=0)], trimmed)
                knn_graph = remapped                              # (N_keep, k), -1 preserved

            # Remap airfoil indices to new positions
            old_to_new = torch.full((keep.shape[0],), -1, dtype=torch.long)
            old_to_new[keep] = torch.arange(keep.sum())
            idcs_airfoil = old_to_new[idcs_airfoil]
            idcs_airfoil = idcs_airfoil[idcs_airfoil >= 0]  # drop any that fell outside

        sample = {
            't': t,
            'pos': pos,
            'idcs_airfoil': idcs_airfoil,
            'velocity_in': velocity_in,
            'velocity_out': velocity_out,
        }
        if point_features is not None:
            sample['point_features'] = point_features
        if knn_graph is not None:
            sample['knn_graph'] = knn_graph

        return sample


def sim_key(filename: str) -> str:
    """Extract simulation key '{geometry_id}_{sim_id}' from a filename.

    e.g. "1021_10-3.npz" -> "1021_10"

    All 5 chunks of a simulation share the same sim_key and must stay in the
    same split to avoid temporal leakage.
    """
    return os.path.basename(filename).split("-")[0]


def get_geometry_to_files(data_dir: str) -> Dict[str, List[str]]:
    """
    Group .npz files by simulation key '{geometry_id}_{sim_id}'.

    Filename pattern: {geometry_id}_{sim_id}-{chunk_id}.npz

    The competition's "geometry" is a unique physical airfoil arrangement
    identified by (geometry_id, sim_id) — NOT just geometry_id, which only has
    22 unique values for the 162 simulations in the training set.

    Args:
        data_dir: Path to data directory containing .npz files

    Returns:
        Dict mapping "{geometry_id}_{sim_id}" -> list of file paths (5 per sim)
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    geom2files = {}

    for f in files:
        key = sim_key(f)  # e.g., "1021_10" from "1021_10-3.npz"
        geom2files.setdefault(key, []).append(f)

    return geom2files


def split_by_geometry(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    data_fraction: float = 1.0,
) -> Dict[str, List[str]]:
    """
    Stratified split by geometry_id to prevent leakage.

    All files from the same airfoil geometry stay together in one split.

    Args:
        data_dir: Path to data directory
        train_ratio: Fraction of geometries for training (default 0.7)
        val_ratio: Fraction of geometries for validation (default 0.15)
        seed: Random seed for reproducibility
        data_fraction: Use only this fraction of total geometries (default 1.0 = all)

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to file lists
    """
    assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be < 1.0"
    assert 0.0 < data_fraction <= 1.0, "data_fraction must be in (0, 1]"

    # Group files by geometry
    geom2files = get_geometry_to_files(data_dir)

    # Get sorted list of unique geometry IDs
    geoms = sorted(geom2files.keys())

    # Sample a fraction of geometries if requested
    if data_fraction < 1.0:
        random.seed(seed)
        num_geoms_to_use = max(1, int(len(geoms) * data_fraction))
        geoms = random.sample(geoms, num_geoms_to_use)
        geoms = sorted(geoms)  # Re-sort after sampling

    # Shuffle geometries with given seed
    random.seed(seed)
    random.shuffle(geoms)

    # Partition geometries - ensure val == test by design.
    # val_test_each is computed first; train absorbs any odd remainder so both
    # val and test always receive exactly the same number of geometries.
    n = len(geoms)
    val_test_each = (n - int(train_ratio * n)) // 2
    train_count = n - 2 * val_test_each

    train_geoms = set(geoms[:train_count])
    val_geoms = set(geoms[train_count:train_count + val_test_each])
    test_geoms = set(geoms[train_count + val_test_each:train_count + 2 * val_test_each])

    # Build file lists per split
    splits = {
        'train': [f for g in train_geoms for f in geom2files[g]],
        'val': [f for g in val_geoms for f in geom2files[g]],
        'test': [f for g in test_geoms for f in geom2files[g]],
    }

    # Validation checks — compare file count against selected geometries only
    # (not len(geom2files) which is geometry count, not file count)
    selected_file_count = sum(len(geom2files[g]) for g in train_geoms | val_geoms | test_geoms)
    assert len(splits['train']) + len(splits['val']) + len(splits['test']) == selected_file_count, \
        "Splits do not cover all selected files"

    assert train_geoms.isdisjoint(val_geoms), "Geometry leakage: train ∩ val is non-empty"
    assert train_geoms.isdisjoint(test_geoms), "Geometry leakage: train ∩ test is non-empty"
    assert val_geoms.isdisjoint(test_geoms), "Geometry leakage: val ∩ test is non-empty"

    return splits


def get_datasets(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    data_fraction: float = 1.0,
) -> Tuple[GRAMDataset, GRAMDataset, GRAMDataset]:
    """
    Create train/val/test datasets with stratified split.

    Args:
        data_dir: Path to data directory
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed
        data_fraction: Use only this fraction of total data (default 1.0 = all)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    splits = split_by_geometry(data_dir, train_ratio, val_ratio, seed, data_fraction)

    train_dataset = GRAMDataset(splits['train'])
    val_dataset = GRAMDataset(splits['val'])
    test_dataset = GRAMDataset(splits['test'])

    return train_dataset, val_dataset, test_dataset


def print_split_stats(data_dir: str, seed: int = 42, data_fraction: float = 1.0):
    """Print split statistics."""
    splits = split_by_geometry(data_dir, seed=seed, data_fraction=data_fraction)
    geom2files = get_geometry_to_files(data_dir)

    all_geoms = len(geom2files)
    all_files = sum(len(f) for f in geom2files.values())

    print(f"\n=== Dataset Split (seed={seed}, data_fraction={data_fraction}) ===")
    print(f"Total geometries available: {all_geoms}")
    print(f"Total files available: {all_files}")

    split_files = sum(len(f) for f in splits.values())
    split_geoms = len(set(sim_key(f) for f in sum(splits.values(), [])))

    if data_fraction < 1.0:
        print(f"Geometries used: {split_geoms} ({data_fraction*100:.1f}%)")
        print(f"Files used: {split_files}\n")
    else:
        print()

    for split_name in ['train', 'val', 'test']:
        files = splits[split_name]
        geoms = set(sim_key(f) for f in files)
        geom_pct = 100 * len(geoms) / split_geoms if split_geoms > 0 else 0
        file_pct = 100 * len(files) / split_files if split_files > 0 else 0
        print(f"{split_name.upper():5s}: {len(geoms):3d} simulations ({geom_pct:5.1f}%), {len(files):4d} files ({file_pct:5.1f}%)")


def load_from_config(cfg: DictConfig) -> Tuple[GRAMDataset, GRAMDataset, GRAMDataset]:
    """
    Load train/val/test datasets from config.

    Args:
        cfg: OmegaConf DictConfig with data and training settings

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_fraction = getattr(cfg.data, 'fraction', 1.0)

    return get_datasets(
        data_dir=cfg.data.path,
        train_ratio=cfg.data.train_split,
        val_ratio=cfg.data.val_split,
        seed=cfg.training.seed,
        data_fraction=data_fraction,
    )


if __name__ == "__main__":
    # Load config with CLI overrides
    from src.config import load_config, print_config

    cfg = load_config(config_path='local')  # Default to local config
    env = cfg.get('environment', 'unknown')
    print_config(cfg, env=env)

    # Print split stats
    data_fraction = getattr(cfg.data, 'fraction', 1.0)
    print_split_stats(cfg.data.path, seed=cfg.training.seed, data_fraction=data_fraction)

    # Load datasets
    train_ds, val_ds, test_ds = load_from_config(cfg)
    print(f"\nDatasets created:")
    print(f"  train: {len(train_ds)} samples")
    print(f"  val: {len(val_ds)} samples")
    print(f"  test: {len(test_ds)} samples")

    # Example: load single sample
    sample = train_ds[0]
    print(f"\nSample shapes:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
