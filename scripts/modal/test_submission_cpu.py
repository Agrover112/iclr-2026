"""
CPU test for the submission package — verifies it works without GPU.

Julian's eval environment might be CPU-only. This test confirms the submission
model forward pass completes correctly on CPU with a real NPZ sample.

Run:
    /home/agrov/gram/bin/modal run scripts/modal/test_submission_cpu.py
"""
import modal


TORCH_VERSION = "2.10.0"
CUDA_VERSION = "cu128"

# CPU-only image — still uses pyg wheels (they have CPU variants).
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
app = modal.App("test-submission-cpu", image=image)


# NO gpu= flag — pure CPU container.
@app.function(timeout=1200, cpu=8.0, memory=32768,
              volumes={"/data": data_volume})
def cpu_test():
    import os
    import sys
    import time
    sys.path.insert(0, "/submission")

    import numpy as np
    import torch
    from models import GatedEGNOMeanResModel

    assert not torch.cuda.is_available(), "Expected CPU-only container"
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"torch={torch.__version__}")

    print("\nBuilding model + loading weights on CPU...")
    t0 = time.perf_counter()
    model = GatedEGNOMeanResModel().to(device)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} params, "
          f"loaded in {time.perf_counter()-t0:.1f}s")

    # Single real sample
    path = "/data/1021_1-0.npz"
    data = np.load(path)
    t = torch.from_numpy(data["t"]).float().unsqueeze(0)
    pos = torch.from_numpy(data["pos"]).float().unsqueeze(0)
    idcs_airfoil = [torch.from_numpy(data["idcs_airfoil"]).long()]
    velocity_in = torch.from_numpy(data["velocity_in"]).float().unsqueeze(0)
    velocity_out_gt = torch.from_numpy(data["velocity_out"]).float().unsqueeze(0)

    print(f"\nForward pass on CPU (this will be slow)...")
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(t, pos, idcs_airfoil, velocity_in)
    fwd_time = time.perf_counter() - t0
    print(f"  completed in {fwd_time:.1f}s")

    assert pred.shape == velocity_out_gt.shape
    metric = (pred - velocity_out_gt).norm(dim=3).mean(dim=(1, 2)).mean().item()
    print(f"  output shape: {tuple(pred.shape)}")
    print(f"  metric on 1021_1-0: {metric:.4f}")

    print("\n[PASSED] CPU forward pass works.")
    print(f"  Estimated full-test-set time (95 samples): ~{95 * fwd_time / 60:.0f} min")
    return {"ok": True, "forward_time_s": fwd_time, "metric": metric}


@app.local_entrypoint()
def main():
    print("Launching CPU-only test...")
    result = cpu_test.remote()
    print(f"\nResult: {result}")
