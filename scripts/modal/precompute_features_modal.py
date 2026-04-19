"""
Precompute geometric features for all NPZ files in the Modal `gram-data` volume.

Reuses src/features.py verbatim. Monkey-patches _chunked_nn_search to run
torch.cdist on GPU (L4) — the only GPU-worthy step. cKDTree paths stay CPU.

Output schema matches scripts/data/precompute_features.py exactly:
    {stem}_feat.pt  = dict[str, Tensor]  keyed by feature name.

Run:
    /home/agrov/gram/bin/modal run scripts/modal/precompute_features_modal.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.5.1", "numpy==2.1.3", "scipy==1.14.1", "tqdm==4.67.1")
    .add_local_python_source("src")
)

volume = modal.Volume.from_name("gram-data", create_if_missing=False)
app = modal.App("gram-precompute-features", image=image)

DEFAULT_FEATURES = "udf,udf_truncated,udf_gradient,local_density,knn_graph,adaptive_knn_graph"


@app.function(gpu="L4", cpu=8.0, memory=16384, volumes={"/data": volume}, timeout=7200)
def precompute_all(features: list[str], overwrite: bool = False):
    import os, glob, time, traceback
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    from src import features as feat_mod
    from src.features import (
        FEATURE_REGISTRY, compute_knn_graph, compute_adaptive_knn_graph,
        feat_cache_path,
    )
    from functools import partial

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── GPU-accelerated NN search (monkey-patch) ──
    _orig_nn = feat_mod._chunked_nn_search

    def _gpu_nn_search(pts, surface_pts, chunk=8192):
        pts_g = pts.to(device, non_blocking=True)
        surf_g = surface_pts.to(device, non_blocking=True)
        min_dists, nearest_pts = [], []
        for i in range(0, pts_g.shape[0], chunk):
            d = torch.cdist(pts_g[i:i+chunk], surf_g)
            mn, idx = d.min(dim=1)
            min_dists.append(mn)
            nearest_pts.append(surf_g[idx])
        return torch.cat(min_dists).cpu(), torch.cat(nearest_pts).cpu()

    feat_mod._chunked_nn_search = _gpu_nn_search

    # Rebuild registry with k=16 for knn_graph, defaults for adaptive
    registry = dict(FEATURE_REGISTRY)
    registry["knn_graph"] = partial(compute_knn_graph, k=16)
    registry["adaptive_knn_graph"] = partial(
        compute_adaptive_knn_graph, k_near=32, k_far=8, sigma=0.05
    )

    npz_files = sorted(glob.glob("/data/*.npz"))
    print(f"Found {len(npz_files)} NPZ files")

    done = skipped = 0
    failed: list[tuple[str, str]] = []
    t0 = time.perf_counter()

    for npz in tqdm(npz_files):
        out_path = feat_cache_path(npz)

        existing = {}
        missing = list(features)
        if os.path.exists(out_path) and not overwrite:
            try:
                existing = torch.load(out_path, weights_only=True)
                missing = [f for f in features if f not in existing]
            except Exception:
                existing, missing = {}, list(features)
            if not missing:
                skipped += 1
                continue

        try:
            data = np.load(npz)
            pos = torch.from_numpy(data["pos"]).float()
            idcs = torch.from_numpy(data["idcs_airfoil"]).long()
            surface_pts = pos[idcs]

            new_features = {}
            fused = {"udf_truncated", "udf_gradient"}
            if fused.issubset(set(missing)):
                min_d, near = feat_mod._chunked_nn_search(pos, surface_pts)
                new_features["udf_truncated"] = min_d.clamp(max=0.5).unsqueeze(1)
                new_features["udf_gradient"]  = F.normalize(near - pos, dim=1)

            for name in missing:
                if name not in new_features:
                    new_features[name] = registry[name](pos, surface_pts)

            existing.update(new_features)
            # Atomic write: tmp → rename. Prevents half-written _feat.pt on crash.
            tmp_path = out_path + ".tmp"
            torch.save(existing, tmp_path)
            os.replace(tmp_path, out_path)
            done += 1
        except Exception as e:
            failed.append((os.path.basename(npz), traceback.format_exc()))
            print(f"ERROR {os.path.basename(npz)}: {e}")

    # Note: no volume.commit() — mounted writes auto-persist; closure commit is a no-op
    total = time.perf_counter() - t0
    print(f"\nDone in {total:.1f}s — computed: {done}, skipped: {skipped}, errors: {len(failed)}")
    if failed:
        print("\nFailed files:")
        for name, tb in failed:
            print(f"  {name}")
            print("    " + tb.replace("\n", "\n    "))
    return {"done": done, "skipped": skipped, "failed": [f[0] for f in failed]}


VALID_FEATURES = {"udf", "udf_truncated", "udf_gradient", "local_density", "knn_graph", "adaptive_knn_graph"}


@app.local_entrypoint()
def main(features: str = DEFAULT_FEATURES, overwrite: bool = False):
    feature_list = [f.strip() for f in features.split(",") if f.strip()]
    if not feature_list:
        raise SystemExit("ERROR: --features is empty. Pass a comma-separated list.")
    unknown = [f for f in feature_list if f not in VALID_FEATURES]
    if unknown:
        raise SystemExit(
            f"ERROR: unknown features {unknown}. Valid: {sorted(VALID_FEATURES)}"
        )
    print(f"Requested features: {feature_list}")
    result = precompute_all.remote(features=feature_list, overwrite=overwrite)
    print(f"\nSummary: {result}")
    if result["failed"]:
        print(f"Retry these {len(result['failed'])} files by rerunning (script is resumable).")
