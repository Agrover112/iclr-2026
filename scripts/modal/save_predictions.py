"""
Run inference on Modal GPU and save predictions for local visualization.

Usage:
    /home/agrov/gram/bin/modal run scripts/modal/save_predictions.py \
        --model fixed_egnn \
        --checkpoint /runs/fixed_egnn/best_model.pt \
        --data-path /data/1021_1-0.npz

Output is downloaded to figures/predictions/{sample_name}_preds.pt
"""

import modal
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.modal_image import image

training_image = (
    image
    .add_local_python_source("src")
    .add_local_python_source("models")
)

app = modal.App("gram-save-predictions", image=training_image)
data_volume = modal.Volume.from_name("gram-data", create_if_missing=False)
runs_volume = modal.Volume.from_name("gram-runs", create_if_missing=True)


@app.function(
    gpu="T4",
    timeout=600,
    volumes={"/data": data_volume, "/runs": runs_volume},
)
def predict_and_save(
    model_name: str,
    checkpoint_path: str,
    npz_path: str,
    features: str = "",
    gnn_depth: int = 0,
    gnn_hidden: int = 0,
):
    import torch
    import numpy as np
    from src.train import get_model_class
    from src.data import GRAMDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve features
    ModelClass = get_model_class(model_name)
    if features:
        feature_list = [f.strip() for f in features.split(",") if f.strip()]
    else:
        feature_list = getattr(ModelClass, "FEATURES", []) or []
    print(f"Features: {feature_list}")

    # Load model with architecture kwargs
    model_kwargs = {}
    if gnn_depth  > 0: model_kwargs["depth"]      = gnn_depth
    if gnn_hidden > 0: model_kwargs["hidden_dim"]  = gnn_hidden
    try:
        model = ModelClass(features=feature_list, **model_kwargs)
    except TypeError:
        try:
            model = ModelClass(features=feature_list)
        except TypeError:
            model = ModelClass()

    state = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    print(f"Model loaded: {model_name}")

    # Load sample
    ds = GRAMDataset([npz_path], features=feature_list)
    sample = ds[0]

    t = sample["t"].unsqueeze(0).to(device)
    pos = sample["pos"].unsqueeze(0).to(device)
    idcs_airfoil = [sample["idcs_airfoil"].to(device)]
    velocity_in = sample["velocity_in"].unsqueeze(0).to(device)
    velocity_out = sample["velocity_out"]  # (5, N, 3) — keep on CPU
    point_features = sample.get("point_features")
    if point_features is not None:
        point_features = point_features.unsqueeze(0).to(device)
    knn_graph = sample.get("knn_graph")
    if knn_graph is not None:
        knn_graph = knn_graph.unsqueeze(0).to(device)

    # Inference
    print("Running inference...")
    with torch.no_grad():
        pred = model(t, pos, idcs_airfoil, velocity_in, point_features, knn_graph)

    pred = pred[0].cpu()  # (5, N, 3)
    pos_cpu = sample["pos"]  # (N, 3)
    idcs_cpu = sample["idcs_airfoil"]  # (M,)
    knn_cpu = sample.get("knn_graph")  # (N, k) or None

    # Metric
    metric = (pred - velocity_out).norm(dim=2).mean().item()
    print(f"Overall metric: {metric:.6f}")

    # Save
    save_dict = {
        "pred": pred,
        "gt": velocity_out,
        "pos": pos_cpu,
        "idcs_airfoil": idcs_cpu,
    }
    if knn_cpu is not None:
        save_dict["knn_graph"] = knn_cpu

    sample_name = os.path.splitext(os.path.basename(npz_path))[0]
    out_path = f"/runs/predictions/{sample_name}_preds.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(save_dict, out_path)
    print(f"Saved: {out_path}")

    return {"metric": metric, "path": out_path, "sample": sample_name}


@app.local_entrypoint()
def main(
    model: str = "fixed_egnn",
    checkpoint: str = "/runs/fixed_egnn/best_model.pt",
    data_path: str = "/data/1021_1-0.npz",
    features: str = "",
    gnn_depth: int = 0,
    gnn_hidden: int = 0,
):
    result = predict_and_save.remote(model, checkpoint, data_path, features,
                                     gnn_depth, gnn_hidden)
    print(f"\nResult: {result}")
    print(f"\nTo download predictions:")
    print(f"  modal volume get gram-runs predictions/{result['sample']}_preds.pt figures/predictions/")
    print(f"\nThen visualize locally:")
    print(f"  /home/agrov/gram/bin/python scripts/viz/export_predictions.py \\")
    print(f"    --predictions figures/predictions/{result['sample']}_preds.pt \\")
    print(f"    --data-path data/{result['sample']}.npz \\")
    print(f"    --html --no-render")
