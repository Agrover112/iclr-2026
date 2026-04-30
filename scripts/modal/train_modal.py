"""
Modal GPU training script.

Usage:
    # Single run (default GPU=L4, timeout=3h)
    modal run scripts/modal/train_modal.py

    # Multiple seeds in parallel (like SLURM --array)
    modal run scripts/modal/train_modal.py --seeds 42 1 2

    # Custom GPU and timeout (via environment variables)
    MODAL_GPU=A100 MODAL_TIMEOUT_SEC=7200 modal run scripts/modal/train_modal.py

    # Or via CLI flags
    modal run scripts/modal/train_modal.py --gpu L40S --timeout-hours 2 --seeds 42
"""

import modal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.modal_image import image

# Ship src/ and models/ as proper Python packages on the container's sys.path.
# add_local_python_source > add_local_dir for code (handles imports correctly).
training_image = (
    image
    .add_local_python_source("src")
    .add_local_python_source("models")
)

app = modal.App("gram-training", image=training_image)

# Persistent volumes — separate data (read) from runs (write) for safety.
data_volume = modal.Volume.from_name("gram-data", create_if_missing=False)
runs_volume = modal.Volume.from_name("gram-runs", create_if_missing=True)

# GPU and timeout from environment or CLI defaults
DEFAULT_GPU = os.environ.get("MODAL_GPU", "L40S")
DEFAULT_TIMEOUT_SEC = int(os.environ.get("MODAL_TIMEOUT_SEC", "86400"))  # 24 hours


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SEC,
    volumes={"/data": data_volume, "/runs": runs_volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_one_seed(
    seed: int,
    model: str = "residual_mlp",
    epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    data_fraction: float = 1.0,
    run_tag: str = "",
    wandb_project: str = "iclr-2026",
    wandb_entity: str = "agrgnn-kth-royal-institute-of-technology",
    features: str = "",
    no_udf: bool = False,
    max_distance: float = 0.0,
    batch_size: int = 1,
    weight_decay: float = 0.01,
    warmup_epochs: int = 0,
    grad_accum: int = 8,
    gnn_depth: int = 0,
    gnn_hidden: int = 0,
    gnn_heads: int = 0,
    gnn_dropout: float = -1.0,
    update_coords: bool = False,
    # Don't start a Modal bool param with `no_` — Modal's CLI auto-generates
    # `--no-<arg>` for the False case, which silently collides with a literal
    # `no_` prefix and set the wrong value. `enforce_no_slip=True` means
    # "apply the no-slip mask" (airfoil predictions → 0); False disables it.
    enforce_no_slip: bool = True,
    loss_fn: str = "mse",  # "mse" (default) or "l2" — see src/train.py --loss
    log_udf: bool = False,  # runtime log-transform the udf_truncated channel — no precompute
    resume_from: str = "",  # path on /runs volume to a best_model.pt for warm-start finetuning
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    # gEGNO / gated_egno_meanres specific: ablate the per-edge gate by passing
    # --no-use-gate (Modal auto-flips `use_gate: True` to False). No collision
    # since the kwarg is `use_*` not `no_*`.
    use_gate: bool = True,
):
    import subprocess, time as _time
    ts = _time.strftime("%Y%m%d-%H%M%S")
    tag = run_tag or f"df{data_fraction}"
    output_dir = f"/runs/{model}/{tag}_{ts}"
    cmd = [
        "python", "-m", "src.train",
        "--model", model,
        "--seeds", str(seed),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--lr", str(lr),
        "--data-fraction", str(data_fraction),
        "--data-path", "/data",
        "--output-dir", output_dir,
        "--run-tag", tag,
        "--wandb-project", wandb_project,
        "--wandb-entity", wandb_entity,
        "--weight-decay", str(weight_decay),
        "--warmup-epochs", str(warmup_epochs),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--max-distance", str(max_distance),
    ]
    if features:
        cmd.append("--features")
        cmd.extend([f.strip() for f in features.split(",") if f.strip()])
    if no_udf:
        cmd.append("--no-udf")
    if gnn_depth   > 0:   cmd.extend(["--gnn-depth",   str(gnn_depth)])
    if gnn_hidden  > 0:   cmd.extend(["--gnn-hidden",  str(gnn_hidden)])
    if gnn_heads   > 0:   cmd.extend(["--gnn-heads",   str(gnn_heads)])
    if gnn_dropout >= 0:  cmd.extend(["--gnn-dropout", str(gnn_dropout)])
    if update_coords:     cmd.append("--update-coords")
    cmd.append("--use-gate" if use_gate else "--no-use-gate")
    # argparse.BooleanOptionalAction also treats leading `--no-` as negation,
    # so src/train.py uses the affirmative name `--enforce-no-slip` to match.
    cmd.append("--enforce-no-slip" if enforce_no_slip else "--no-enforce-no-slip")
    cmd.extend(["--loss", loss_fn])
    if log_udf:
        cmd.append("--log-udf")
    if resume_from:
        cmd.extend(["--resume-from", resume_from])
    cmd.extend(["--train-ratio", str(train_ratio), "--val-ratio", str(val_ratio)])
    # No cwd needed — src/ is on sys.path via add_local_python_source.
    # Stream stdout/stderr live so Modal logs show training progress in real time.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Training failed for seed {seed} (rc={rc})")


@app.local_entrypoint()
def main(
    seeds: str = "42",
    model: str = "residual_mlp",
    epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    data_fraction: float = 1.0,
    run_tag: str = "",
    wandb_project: str = "iclr-2026",
    wandb_entity: str = "agrgnn-kth-royal-institute-of-technology",
    features: str = "",
    no_udf: bool = False,
    max_distance: float = 0.0,
    batch_size: int = 1,
    weight_decay: float = 0.01,
    warmup_epochs: int = 0,
    grad_accum: int = 8,
    gnn_depth: int = 0,
    gnn_hidden: int = 0,
    gnn_heads: int = 0,
    gnn_dropout: float = -1.0,
    update_coords: bool = False,
    enforce_no_slip: bool = True,
    loss_fn: str = "mse",
    log_udf: bool = False,
    resume_from: str = "",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_gate: bool = True,
    gpu: str = DEFAULT_GPU,
    timeout_hours: float = DEFAULT_TIMEOUT_SEC / 3600,
):
    seed_list = [int(s) for s in seeds.split(",")]
    print(f"GPU: {gpu} | Timeout: {timeout_hours:.1f}h | enforce_no_slip: {enforce_no_slip} | loss: {loss_fn} | log_udf: {log_udf}")
    print(f"Split: train={train_ratio} val={val_ratio} test={1 - train_ratio - val_ratio:.3f}")
    if resume_from:
        print(f"Warm-start from: {resume_from}")
    print(f"Launching {len(seed_list)} parallel GPU job(s): seeds={seed_list} features={features or '(model default)'}")

    # All seeds run in parallel — like SLURM --array
    list(train_one_seed.starmap([
        (seed, model, epochs, patience, lr, data_fraction, run_tag, wandb_project, wandb_entity,
         features, no_udf, max_distance, batch_size, weight_decay, warmup_epochs, grad_accum,
         gnn_depth, gnn_hidden, gnn_heads, gnn_dropout, update_coords, enforce_no_slip, loss_fn,
         log_udf, resume_from, train_ratio, val_ratio, use_gate)
        for seed in seed_list
    ]))

    print("All seeds done.")
