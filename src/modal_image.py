"""
Modal image definition for GPU training.

Built once, cached forever — subsequent runs reuse the cached image.
Only rebuilds if you change something here.

CUDA 12.4 + PyTorch 2.11.0 + torch_geometric stack.

Usage in any Modal script:
    from src.modal_image import image
"""

import modal

TORCH_VERSION = "2.10.0"
CUDA_VERSION = "cu128"   # officially-supported PyG wheel combo

image = (
    modal.Image.debian_slim(python_version="3.12")

    # --- PyTorch with CUDA ---
    .pip_install(
        f"torch=={TORCH_VERSION}",
        "torchvision",
        extra_index_url=f"https://download.pytorch.org/whl/{CUDA_VERSION}",
    )

    # --- torch_geometric (pure-torch, no extension deps required) ---
    .pip_install("torch_geometric")

    # --- Optional PyG extensions: scatter + cluster (for future GNN/PointNet++) ---
    # torch_sparse dropped (not needed, pure-torch fallback works)
    # torch_spline_conv removed from PyG, folded into pyg_lib
    .pip_install(
        "pyg_lib",
        "torch_scatter",
        "torch_cluster",
        find_links=f"https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html",
    )

    # --- Everything else ---
    .pip_install(
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "omegaconf",
        "wandb",
        "huggingface_hub",
        "datasets",
        "pyarrow",
        "tqdm",
        "pytest",
    )

    # --- Neural operator library (for SpectralConv etc.) ---
    .pip_install("neuraloperator")
)
