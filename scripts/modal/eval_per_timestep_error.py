"""
Run MLP / Fixed EGNN / gEGNO (submission) on the 70/15/15 test split and save
per-sample per-timestep L2 errors as CSV for the horizon-bar chart figure.

Checkpoints (all on `gram-runs`):
  MLP            : /mlp/local_seed42/best_model.pt
  Fixed EGNN     : /fixed_egnn/fegnn_d6h32_20260409-133815/seed_42/best_model.pt
  gEGNO (submit) : /gated_egno_meanres/egno_meanres_h96_submission_20260419-151939/seed_42/best_model.pt

Test split: split_by_geometry(data_fraction=1.0, train_ratio=0.7, val_ratio=0.15, seed=42)
  ~120 files across ~24 geometries (disjoint from any model's training geoms
  under the SAME split; note submission model trained with df=1.0 95/2.5/2.5 so
  those 120 files are mostly in its train set — caveat noted in the figcaption).

Outputs (to gram-runs volume):
  /runs/eval/per_timestep_errors.csv  columns: file, model, frame, l2

Run (user launches):
    /home/agrov/gram/bin/modal run scripts/modal/eval_per_timestep_error.py \\
        --n-samples 40

Then download:
    modal volume get gram-runs eval/per_timestep_errors.csv results/

And plot locally:
    /home/agrov/gram/bin/python scripts/poster/make_horizon_bars.py
"""
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.modal_image import image

# Ship src/ and models/ as python sources.
# NB: do NOT also mount /submission — that package is also named `models` and
# collides with the main project's `models/` on sys.path (only one wins, and
# the loser imports fail with ModuleNotFoundError).
eval_image = (
    image
    .add_local_python_source("src")
    .add_local_python_source("models")
)

app = modal.App("eval-per-timestep-error", image=eval_image)
data_volume = modal.Volume.from_name("gram-data",  create_if_missing=False)
runs_volume = modal.Volume.from_name("gram-runs",  create_if_missing=True)


# --- checkpoint paths on the volume ---
CKPT_MLP   = "/runs/mlp/local_seed42/best_model.pt"
CKPT_EGNN  = "/runs/fixed_egnn/fegnn_d6h32_20260409-133815/seed_42/best_model.pt"
CKPT_GEGNO = "/runs/gated_egno_meanres/egno_meanres_h96_submission_20260419-151939/seed_42/best_model.pt"


@app.function(
    gpu=os.environ.get("MODAL_GPU", "L40S"),
    timeout=3600,
    cpu=4.0,
    memory=32 * 1024,
    volumes={"/data": data_volume, "/runs": runs_volume},
)
def eval_models(n_samples: int = 0, seed: int = 42) -> str:
    """n_samples = 0 → use the entire test split (~120 files for df=1.0 70/15/15)."""
    import csv
    import random
    import numpy as np
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- data split ---
    # Full 810 files, geometry-stratified 70/15/15 (seed 42).
    # Test set ≈ 120 files; caveat — some may overlap with each checkpoint's
    # training geoms since models were trained under different df/split configs.
    from src.data import split_by_geometry
    splits = split_by_geometry(
        data_dir="/data",
        train_ratio=0.7, val_ratio=0.15,
        seed=seed, data_fraction=1.0,
    )
    test_files = sorted(splits["test"])
    random.Random(seed).shuffle(test_files)
    if n_samples > 0:
        test_files = test_files[:n_samples]
    print(f"Test split: {len(splits['test'])} files · evaluating {len(test_files)}")

    # --- load models ---
    loaders: dict[str, callable] = {}

    def _load_mlp():
        from models.mlp.model import MLP
        m = MLP()
        sd = torch.load(CKPT_MLP, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        m.load_state_dict(sd, strict=False)
        return m.to(device).eval()
    loaders["MLP"] = _load_mlp

    def _load_egnn():
        from models.fixed_egnn.model import FixedEGNNModel
        m = FixedEGNNModel(depth=6, hidden_dim=32)
        sd = torch.load(CKPT_EGNN, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        m.load_state_dict(sd, strict=False)
        return m.to(device).eval()
    loaders["EGNN (fixed)"] = _load_egnn

    def _load_gegno():
        # Submission checkpoint is hidden_dim=96, depth=4 (default depth).
        from models.gated_egno_meanres.model import GatedEGNOMeanResModel
        m = GatedEGNOMeanResModel(hidden_dim=96, depth=4)
        sd = torch.load(CKPT_GEGNO, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        m.load_state_dict(sd, strict=True)
        return m.to(device).eval()
    loaders["gEGNO"] = _load_gegno

    models: dict[str, torch.nn.Module] = {}
    for name, fn in loaders.items():
        try:
            models[name] = fn()
            n = sum(p.numel() for p in models[name].parameters())
            print(f"  [ok]  {name:<14} | {n:,} params")
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")

    if not models:
        raise RuntimeError("no model loaded; check checkpoint paths")

    # --- iterate samples ---
    out_dir = "/runs/eval"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "per_timestep_errors.csv")

    rows = []
    with torch.no_grad():
        for fi, fname in enumerate(test_files, start=1):
            path = os.path.join("/data", fname)
            if not os.path.exists(path):
                print(f"  [skip missing] {fname}")
                continue
            data = np.load(path)
            t = torch.from_numpy(data["t"]).float().unsqueeze(0).to(device)
            pos = torch.from_numpy(data["pos"]).float().unsqueeze(0).to(device)
            idcs_airfoil = [torch.from_numpy(data["idcs_airfoil"]).long().to(device)]
            vin = torch.from_numpy(data["velocity_in"]).float().unsqueeze(0).to(device)
            gt = torch.from_numpy(data["velocity_out"]).float().to(device)   # (5, N, 3)

            print(f"[{fi:02d}/{len(test_files)}] {fname}  N={pos.shape[1]:,}", flush=True)
            for mname, model in models.items():
                try:
                    # For models that compute features on the fly (EGNN / gEGNO),
                    # precompute them and push to device — `_compute_batch_features`
                    # uses scipy (CPU path) and otherwise leaves tensors on CPU,
                    # which breaks scatter_add when the model is on CUDA.
                    feats = getattr(model, "FEATURES", None)
                    if feats and hasattr(model, "_compute_batch_features"):
                        pf, kg = model._compute_batch_features(pos, idcs_airfoil)
                        if pf is not None: pf = pf.to(device)
                        if kg is not None: kg = kg.to(device)
                        pred = model(t, pos, idcs_airfoil, vin, pf, kg)[0]
                    else:
                        pred = model(t, pos, idcs_airfoil, vin)[0]     # (5, N, 3)
                    # per-frame mean-L2-per-point
                    per_frame = (pred - gt).norm(dim=-1).mean(dim=-1).cpu().tolist()
                    for fi_idx, val in enumerate(per_frame):
                        rows.append({
                            "file":  fname,
                            "model": mname,
                            "frame": fi_idx,
                            "l2":    float(val),
                        })
                    mean = sum(per_frame) / len(per_frame)
                    print(f"   {mname:<14}  mean = {mean:.4f}   per-t = "
                          + " ".join(f"{v:.3f}" for v in per_frame))
                except Exception as e:
                    print(f"   {mname:<14}  [FAIL] {type(e).__name__}: {e}")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "model", "frame", "l2"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}")
    return csv_path


@app.local_entrypoint()
def main(n_samples: int = 0, seed: int = 42):
    """n_samples = 0 → all ~120 test files (default). Set >0 to cap for a quick run."""
    csv_path = eval_models.remote(n_samples=n_samples, seed=seed)
    print(f"\nDone. CSV on volume: {csv_path}")
    print("\nDownload:")
    print("    modal volume get gram-runs eval/per_timestep_errors.csv results/")
    print("\nPlot:")
    print("    /home/agrov/gram/bin/python scripts/poster/make_horizon_bars.py")
