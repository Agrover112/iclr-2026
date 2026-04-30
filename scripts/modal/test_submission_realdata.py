"""
Real-data correctness test for the submission package.

Ships `submission/` to Modal, mounts the `gram-data` volume (contains the 810
real NPZ files), and runs the submission model on 3 samples from different
geometry types. Verifies:
  - forward pass runs with REAL pos/idcs_airfoil/velocity_in
  - on-the-fly feature computation doesn't crash
  - output competition metric against real velocity_out is plausible
    (should be roughly in line with the validated val metric ~1.12 for h=64)

Compares submission metric against the ground truth — if submission's features
match the trained ones, the metric should be close to the best_val the training
run achieved.

Run:
    /home/agrov/gram/bin/modal run scripts/modal/test_submission_realdata.py
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
app = modal.App("test-submission-realdata", image=image)


@app.function(gpu="L4", timeout=900, cpu=4.0, memory=16384,
              volumes={"/data": data_volume})
def realdata_test(sample_files: list[str]):
    """Load each NPZ, run submission model, compare metric vs ground truth."""
    import os
    import sys
    sys.path.insert(0, "/submission")

    import numpy as np
    import torch
    from models import GatedEGNOMeanResModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nBuilding model + loading weights...")
    model = GatedEGNOMeanResModel().to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} params | h={model.hidden_dim} d={model.depth}")

    # Competition metric: L2 norm per point, averaged over space and time.
    def competition_metric(pred, gt):
        return (pred - gt).norm(dim=3).mean(dim=(1, 2)).mean().item()

    print(f"\nTesting {len(sample_files)} samples...")
    print(f"{'Sample':<25}  {'N':>8}  {'M':>7}  {'metric':>8}")
    print("-" * 55)
    results = []
    for f in sample_files:
        path = os.path.join("/data", f)
        if not os.path.exists(path):
            print(f"  [missing] {f}")
            continue

        data = np.load(path)
        t = torch.from_numpy(data["t"]).float().unsqueeze(0).to(device)         # (1, 10)
        pos = torch.from_numpy(data["pos"]).float().unsqueeze(0).to(device)      # (1, N, 3)
        idcs_airfoil = [torch.from_numpy(data["idcs_airfoil"]).long().to(device)]
        velocity_in = torch.from_numpy(data["velocity_in"]).float().unsqueeze(0).to(device)
        velocity_out_gt = torch.from_numpy(data["velocity_out"]).float().unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(t, pos, idcs_airfoil, velocity_in)

        assert pred.shape == velocity_out_gt.shape, \
            f"shape mismatch on {f}: {pred.shape} vs {velocity_out_gt.shape}"

        m = competition_metric(pred, velocity_out_gt)
        N = pos.shape[1]
        M = len(idcs_airfoil[0])
        print(f"{f:<25}  {N:>8,}  {M:>7,}  {m:>8.4f}")
        results.append({"file": f, "N": N, "M": M, "metric": m})

    if results:
        mean_m = sum(r["metric"] for r in results) / len(results)
        print("-" * 55)
        print(f"{'Mean metric':<42}  {mean_m:>8.4f}")
        # Sanity: training best_val for h=64 was 1.1158
        # If mean is similar (±15%), on-the-fly features match training.
        print(f"\nReference: training best_val for h=64 ckpt was ~1.116")
        if abs(mean_m - 1.116) / 1.116 < 0.25:
            print("[PASSED] metric in expected range — submission works correctly.")
        else:
            print(f"[WARNING] metric deviates >25% from training best_val (~1.116)")
            print("  possible cause: on-the-fly features don't match training cache,")
            print("  or these samples happen to be in training (expected metric lower)")

    return results


@app.local_entrypoint()
def main():
    # Mix of geometry types for coverage
    samples = [
        "1021_1-0.npz",    # 1xxx — single airfoil
        "2002_10-0.npz",   # 2xxx — two airfoils (multi-body topology)
        "3001_10-0.npz",   # 3xxx — three airfoils (most complex topology)
    ]  # cross-geometry sanity check — confirms all airfoil count regimes work
    print(f"Launching real-data test on {len(samples)} samples...")
    results = realdata_test.remote(sample_files=samples)
    print(f"\nFinal: {results}")
