"""
Run the submission model (GatedEGNOMeanResModel, h=96) on real NPZ samples
and save predictions in the schema expected by `scripts/viz/export_predictions.py`.

The submission package (`submission/`) is shipped as-is and the model auto-loads
its bundled `state_dict.pt`. Predictions are written to
`/runs/predictions/{sample}_submission.pt` on the `gram-runs` volume.

Run (user launches):
    /home/agrov/gram/bin/modal run scripts/modal/save_submission_predictions.py

Edit the `samples` list in `main()` to pick which files to predict on.

Then download and visualize:
    modal volume get gram-runs predictions/1021_1-0_submission.pt figures/predictions/
    /home/agrov/gram/bin/python scripts/viz/export_predictions.py \\
        --predictions figures/predictions/1021_1-0_submission.pt \\
        --data-path data/1021_1-0.npz \\
        --output-dir figures/predictions/submission_cfd \\
        --no-csv --no-render --cfd --vtp --pressure
"""
import modal


TORCH_VERSION = "2.10.0"
CUDA_VERSION = "cu128"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"torch=={TORCH_VERSION}", "torchvision",
        extra_index_url=f"https://download.pytorch.org/whl/{CUDA_VERSION}",
    )
    .pip_install("torch_geometric")
    .pip_install(
        "pyg_lib", "torch_scatter", "torch_cluster",
        find_links=f"https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html",
    )
    .pip_install("numpy", "scipy", "neuraloperator")
    .add_local_dir("/home/agrov/iclr-2026/submission", "/submission")
)

data_volume = modal.Volume.from_name("gram-data", create_if_missing=False)
runs_volume = modal.Volume.from_name("gram-runs", create_if_missing=True)
app = modal.App("save-submission-predictions", image=image)


@app.function(gpu="L4", timeout=1800, cpu=4.0, memory=16384,
              volumes={"/data": data_volume, "/runs": runs_volume})
def save_predictions(sample_files: list[str], k_knn: int = 16):
    import os
    import sys
    sys.path.insert(0, "/submission")

    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    from models import GatedEGNOMeanResModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nBuilding model (auto-loads bundled state_dict.pt)...")
    model = GatedEGNOMeanResModel().to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  h={model.hidden_dim} d={model.depth} | {n_params:,} params")

    out_dir = "/runs/predictions"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'Sample':<22} {'N':>7} {'M':>7} {'metric':>8}  {'out file':<40}")
    print("-" * 90)
    results = []
    for f in sample_files:
        path = os.path.join("/data", f)
        if not os.path.exists(path):
            print(f"  [missing] {f}")
            continue

        data = np.load(path)
        t = torch.from_numpy(data["t"]).float().unsqueeze(0).to(device)
        pos = torch.from_numpy(data["pos"]).float().unsqueeze(0).to(device)
        idcs_airfoil = [torch.from_numpy(data["idcs_airfoil"]).long().to(device)]
        velocity_in = torch.from_numpy(data["velocity_in"]).float().unsqueeze(0).to(device)
        velocity_out_gt = torch.from_numpy(data["velocity_out"]).float()

        with torch.no_grad():
            pred = model(t, pos, idcs_airfoil, velocity_in).cpu()  # (1, 5, N, 3)
        pred = pred[0]                                               # (5, N, 3)

        metric = (pred - velocity_out_gt).norm(dim=2).mean().item()

        # Compute KNN graph (k=16) for downstream vorticity viz
        pos_np = data["pos"]
        tree = cKDTree(pos_np)
        _, knn = tree.query(pos_np, k=k_knn + 1)
        knn = torch.from_numpy(knn[:, 1:]).long()

        sample_name = os.path.splitext(f)[0]
        out_path = os.path.join(out_dir, f"{sample_name}_submission.pt")
        torch.save({
            "pred":         pred,
            "gt":           velocity_out_gt,
            "pos":          torch.from_numpy(pos_np).float(),
            "idcs_airfoil": torch.from_numpy(data["idcs_airfoil"]).long(),
            "knn_graph":    knn,
        }, out_path)

        N = pos_np.shape[0]
        M = len(data["idcs_airfoil"])
        print(f"{f:<22} {N:>7,} {M:>7,} {metric:>8.4f}  {out_path}")
        results.append({"file": f, "metric": metric, "out": out_path})

    return results


@app.local_entrypoint()
def main():
    # Pick samples — same ones used for the EGNO comparison viz.
    samples = [
        "1021_19-2.npz",
    ]
    print(f"Running submission model on {len(samples)} samples...")
    results = save_predictions.remote(sample_files=samples)
    print(f"\nResults: {results}")
    print("\nTo download:")
    for r in results:
        name = r["file"].replace(".npz", "_submission.pt")
        print(f"  modal volume get gram-runs predictions/{name} figures/predictions/")
    print("\nTo visualize (per sample):")
    print("  /home/agrov/gram/bin/python scripts/viz/export_predictions.py \\")
    print("    --predictions figures/predictions/<sample>_submission.pt \\")
    print("    --data-path data/<sample>.npz \\")
    print("    --output-dir figures/predictions/submission_cfd \\")
    print("    --no-csv --no-render --cfd --vtp --pressure")
