"""
Smoke test the submission package on Modal with placeholder (random) data.

Ships the `submission/` directory to a Modal container and runs the
equivalent of `submission/main.py` with a small batch. Verifies:
  - state_dict.pt loads without shape mismatches
  - forward pass runs end-to-end without crashing
  - output shape is exactly (B, 5, N, 3)

Run:
    /home/agrov/gram/bin/modal run scripts/modal/test_submission_placeholder.py
"""
import modal


TORCH_VERSION = "2.10.0"
CUDA_VERSION = "cu128"

# Inline image — matches `submission/requirements.txt` exactly, no `src/` needed.
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

app = modal.App("test-submission-placeholder", image=image)


@app.function(gpu="L4", timeout=600, cpu=4.0, memory=16384)
def smoke_test(batch_size: int = 1, num_pos: int = 100_000):
    """Build model, load weights, run forward on random data, assert shape."""
    import sys
    sys.path.insert(0, "/submission")

    import torch
    from models import GatedEGNOMeanResModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # === Build model (should auto-load state_dict.pt) ===
    print("\n[1/3] Building model and loading weights...")
    model = GatedEGNOMeanResModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ok — {n_params:,} parameters")
    print(f"  hidden_dim={model.hidden_dim}, depth={model.depth}, heads={model.heads}")

    # === Build placeholder data (mirrors submission/main.py) ===
    print(f"\n[2/3] Creating placeholder data: batch={batch_size}, N={num_pos}...")
    NUM_T_IN, NUM_T_OUT = 5, 5
    t = torch.rand((batch_size, NUM_T_IN + NUM_T_OUT), device=device)
    pos = torch.rand((batch_size, num_pos, 3), device=device)
    idcs_airfoil = [
        torch.randint(num_pos, size=(int(m),), device=device)
        for m in torch.randint(3142, 24198, size=(batch_size,))
    ]
    velocity_in = torch.rand((batch_size, NUM_T_IN, num_pos, 3), device=device)
    ground_truth = torch.rand((batch_size, NUM_T_OUT, num_pos, 3), device=device)
    print(f"  ok — idcs_airfoil lengths: {[len(idc) for idc in idcs_airfoil]}")

    # === Run forward pass ===
    print("\n[3/3] Running forward pass...")
    model.eval()
    with torch.no_grad():
        velocity_out = model(t, pos, idcs_airfoil, velocity_in)

    expected_shape = (batch_size, NUM_T_OUT, num_pos, 3)
    assert velocity_out.shape == expected_shape, \
        f"shape mismatch: got {velocity_out.shape}, expected {expected_shape}"
    print(f"  ok — output shape {tuple(velocity_out.shape)}")

    metric = (velocity_out - ground_truth).norm(dim=3).mean(dim=(1, 2))
    print(f"  metric (vs random GT): {metric.mean().item():.4f} +/- {metric.std().item():.4f}")

    print("\n[PASSED] Smoke test — submission package is functional.")
    return {
        "ok": True,
        "params": n_params,
        "output_shape": tuple(velocity_out.shape),
        "metric_mean": metric.mean().item(),
    }


@app.local_entrypoint()
def main(batch_size: int = 1, num_pos: int = 100_000):
    print(f"Launching smoke test: batch_size={batch_size}, num_pos={num_pos}")
    result = smoke_test.remote(batch_size=batch_size, num_pos=num_pos)
    print(f"\nResult: {result}")
