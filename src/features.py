"""
Per-point geometric feature computation for GRaM competition data.

Feature registry: add new features by adding a function and registering it in
FEATURE_REGISTRY. Each function takes (pos, surface_pts) and returns a Tensor.

Used by:
  scripts/precompute_features.py  — precompute and cache to disk (run once before training)
  models/base.py                  — compute on the fly at inference time
"""

from pathlib import Path

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────

def _chunked_nn_search(
    pts: torch.Tensor,
    surface_pts: torch.Tensor,
    chunk: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Core nearest-neighbour primitive. ONE distance matrix pass, two outputs.

    Chunked to avoid OOM: full 100k × 24k matrix = ~9.6 GB. At chunk=2000 it's
    2000 × 24000 × 4 bytes = 192 MB peak.

    Args:
        pts:         (N, 3) query points
        surface_pts: (M, 3) airfoil surface points

    Returns:
        min_dists:    (N,)   minimum distance to any surface point
        nearest_pts:  (N, 3) coordinates of nearest surface point
    """
    min_dists = []
    nearest_pts = []
    for i in range(0, pts.shape[0], chunk):
        d = torch.cdist(pts[i : i + chunk], surface_pts)   # (chunk, M)
        min_d, idx = d.min(dim=1)                           # (chunk,), (chunk,)
        min_dists.append(min_d)
        nearest_pts.append(surface_pts[idx])                # (chunk, 3)
    return torch.cat(min_dists), torch.cat(nearest_pts)


def _chunked_min_dist(pts: torch.Tensor, surface_pts: torch.Tensor) -> torch.Tensor:
    """Min distance only. Calls _chunked_nn_search and discards nearest_pts."""
    min_dists, _ = _chunked_nn_search(pts, surface_pts)
    return min_dists


def _chunked_nearest_surface_pt(pts: torch.Tensor, surface_pts: torch.Tensor) -> torch.Tensor:
    """Nearest surface point only. Calls _chunked_nn_search and discards min_dists."""
    _, nearest_pts = _chunked_nn_search(pts, surface_pts)
    return nearest_pts


# ─────────────────────────────────────────────────────────────────
# Feature functions
# Each takes (pos: Tensor (N,3), surface_pts: Tensor (M,3)) → Tensor
# ─────────────────────────────────────────────────────────────────

def compute_local_density(pos: torch.Tensor, surface_pts: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Local point density: distance to k-th nearest neighbor.

    Smaller value = denser region. surface_pts is unused but accepted
    to match the feature function signature.

    Returns (N, 1) float32.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(pos.numpy())
    dists, _ = tree.query(pos.numpy(), k=k + 1)
    return torch.from_numpy(dists[:, k]).float().unsqueeze(1)


def compute_udf(pos: torch.Tensor, surface_pts: torch.Tensor) -> torch.Tensor:
    """Unsigned distance from each point to nearest airfoil surface point.

    Returns (N, 1) float32.
    """
    return _chunked_min_dist(pos, surface_pts).unsqueeze(1)


def compute_udf_truncated(
    pos: torch.Tensor,
    surface_pts: torch.Tensor,
    d_max: float = 0.5,
) -> torch.Tensor:
    """UDF capped at d_max. All points farther than d_max get value d_max.

    Focuses numerical precision on the near-surface region where error is high.
    Far-field points (uniform freestream) all get the same saturation value.

    Args:
        d_max: truncation distance in same units as pos (default 0.5)

    Returns (N, 1) float32.
    """
    return _chunked_min_dist(pos, surface_pts).clamp(max=d_max).unsqueeze(1)


def compute_udf_gradient(pos: torch.Tensor, surface_pts: torch.Tensor) -> torch.Tensor:
    """Unit vector from each point TOWARD its nearest airfoil surface point.

    This is -∇UDF: points toward the surface ≈ inward surface normal.
    Encodes the direction the no-slip boundary condition acts in.

    Returns (N, 3) float32.
    """
    nearest = _chunked_nearest_surface_pt(pos, surface_pts)  # (N, 3)
    direction = nearest - pos                                 # toward surface
    return F.normalize(direction, dim=1)


# ─────────────────────────────────────────────────────────────────
# Feature registry
# Add new features here. Key = feature name used in configs.
# ─────────────────────────────────────────────────────────────────

FEATURE_REGISTRY: dict[str, callable] = {
    "udf":            compute_udf,            # (N, 1)
    "udf_truncated":  compute_udf_truncated,  # (N, 1)
    "udf_gradient":   compute_udf_gradient,   # (N, 3)
    "local_density":  compute_local_density,   # (N, 1)
}

# Feature output dimensions (number of channels per feature)
FEATURE_DIMS: dict[str, int] = {
    "udf":            1,
    "udf_truncated":  1,
    "udf_gradient":   3,
    "local_density":  1,
}


def compute_point_features(
    pos: torch.Tensor,
    surface_pts: torch.Tensor,
    feature_names: list[str],
) -> torch.Tensor:
    """Compute and concatenate multiple features for a single sample.

    Fast path: when both udf_truncated and udf_gradient are requested together,
    runs one NN search instead of two (2x speedup for the common case).

    Args:
        pos:           (N, 3) point cloud positions
        surface_pts:   (M, 3) airfoil surface points = pos[idcs_airfoil]
        feature_names: list of keys from FEATURE_REGISTRY

    Returns:
        (N, F_total) concatenated features, float32
    """
    for name in feature_names:
        if name not in FEATURE_REGISTRY:
            raise ValueError(
                f"Unknown feature '{name}'. Available: {sorted(FEATURE_REGISTRY)}"
            )

    # Fast path: fuse udf_truncated + udf_gradient into one NN search
    fused = {"udf_truncated", "udf_gradient"}
    if fused.issubset(set(feature_names)):
        min_dists, nearest_pts = _chunked_nn_search(pos, surface_pts)
        cache = {
            "udf_truncated": min_dists.clamp(max=0.5).unsqueeze(1),
            "udf_gradient":  F.normalize(nearest_pts - pos, dim=1),
        }
        remaining = [n for n in feature_names if n not in fused]
        parts = [cache[n] for n in feature_names if n in fused]
        # Preserve requested order
        parts = []
        for name in feature_names:
            if name in cache:
                parts.append(cache[name])
            else:
                parts.append(FEATURE_REGISTRY[name](pos, surface_pts))
        return torch.cat(parts, dim=1)

    # Default: compute each feature independently
    return torch.cat([FEATURE_REGISTRY[n](pos, surface_pts) for n in feature_names], dim=1)


def total_feature_dim(feature_names: list[str]) -> int:
    """Total number of feature channels for a given feature list."""
    return sum(FEATURE_DIMS[n] for n in feature_names)


# ─────────────────────────────────────────────────────────────────
# Cache path helper (used by precompute script + dataloader)
# ─────────────────────────────────────────────────────────────────

def feat_cache_path(npz_path: str) -> str:
    """Return the feature cache .pt path for a given .npz file.

    e.g. data/1021_1-0.npz → data/1021_1-0_feat.pt
    """
    return str(Path(npz_path).with_suffix("")) + "_feat.pt"
