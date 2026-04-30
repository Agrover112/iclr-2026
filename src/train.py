"""
General training script — wraps any model, runs multiple seeds, early stopping.

Usage:
    python -m src.train --model mlp --seeds 42 1 2 --epochs 100 --patience 10

Saves per-seed metrics and a cross-seed summary to results/{model}/
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from src.data import split_by_geometry, GRAMDataset
from src.features import GRAPH_FEATURES

# Registry: name -> import path and class
MODEL_REGISTRY = {
    "mlp":          ("models.mlp.model",          "MLP"),
    "residual_mlp": ("models.residual_mlp.model", "ResidualMLP"),
    "gat":          ("models.gat.model",           "GATModel"),
    "egnn":         ("models.egnn.model",          "EGNNModel"),
    "fixed_egnn":   ("models.fixed_egnn.model",    "FixedEGNNModel"),
    "fixed_egnn_recurrent": ("models.fixed_egnn_recurrent.model", "FixedEGNNRecurrentModel"),
    "fixed_egnn_gated": ("models.fixed_egnn_gated.model", "FixedEGNNGatedModel"),
    "fixed_egnn_attn": ("models.fixed_egnn_attn.model", "FixedEGNNAttnModel"),
    "fixed_egnn_gated_tconv": ("models.fixed_egnn_gated_tconv.model", "FixedEGNNGatedTconvModel"),
    "fixed_egnn_gated_spectral": ("models.fixed_egnn_gated_spectral.model", "FixedEGNNGatedSpectralModel"),
    "gated_egno": ("models.gated_egno.model", "GatedEGNOModel"),
    "gated_egno_meanres": ("models.gated_egno_meanres.model", "GatedEGNOMeanResModel"),
}


_UDF_FEATURES = {"udf", "udf_truncated", "udf_gradient"}
_KNN_GRAPH_MAP = {"fixed": "knn_graph", "adaptive": "adaptive_knn_graph"}


def resolve_features(model_class, features_override, no_udf, knn):
    """Build final feature list from model defaults + CLI flags.

    Priority: --features (explicit) > --no-udf / --knn flags > model default.

    Args:
        model_class:       model class (used to read default FEATURES)
        features_override: explicit list from --features, or None
        no_udf:            if True, remove all UDF float features
        knn:               'none', 'fixed', 'adaptive', or None (keep model default)

    Returns:
        list[str] of feature names
    """
    base = list(features_override if features_override is not None
                else getattr(model_class, 'FEATURES', []) or [])

    if no_udf:
        base = [f for f in base if f not in _UDF_FEATURES]

    if knn is not None:
        base = [f for f in base if f not in GRAPH_FEATURES]
        if knn != 'none':
            base.append(_KNN_GRAPH_MAP[knn])

    return base


def get_model_class(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    module_path, class_name = MODEL_REGISTRY[name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def collate_fn(batch):
    t = torch.stack([b['t'] for b in batch])
    pos = torch.stack([b['pos'] for b in batch])
    idcs_airfoil = [b['idcs_airfoil'] for b in batch]
    velocity_in = torch.stack([b['velocity_in'] for b in batch])
    velocity_out = torch.stack([b['velocity_out'] for b in batch])

    # Float features (UDF etc.) — None triggers on-the-fly computation in base.py
    point_features = (
        torch.stack([b['point_features'] for b in batch])
        if 'point_features' in batch[0] else None
    )

    # knn_graph — separate from float features, used by GNN models
    knn_graph = (
        torch.stack([b['knn_graph'] for b in batch])
        if 'knn_graph' in batch[0] else None
    )

    return t, pos, idcs_airfoil, velocity_in, velocity_out, point_features, knn_graph


@torch.no_grad()
def competition_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    """L2 norm per point, averaged over space and time. Competition evaluation metric."""
    return (pred - target).norm(dim=3).mean(dim=(1, 2)).mean().item()


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.best_epoch = 0
        self.counter = 0

    def step(self, val_metric: float, epoch: int) -> bool:
        """Returns True if training should stop."""
        if val_metric < self.best - self.min_delta:
            self.best = val_metric
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def run_split(model, loader, optimizer=None, accum_steps=1, device=None, loss_fn="mse"):
    """Run one pass over a data split. Returns (loss, metric, elapsed_s, avg_grad_norm).

    loss_fn: "mse" (default) or "l2" — see `compute_loss` for details.
    """
    is_train = optimizer is not None
    model.train(is_train)
    if device is None:
        device = next(model.parameters()).device

    total_loss = 0.0
    total_metric = 0.0
    total_grad_norm = 0.0
    grad_norm_steps = 0
    t0 = time.perf_counter()
    n = len(loader)

    if is_train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(is_train):
        for i, (t, pos, idcs_airfoil, velocity_in, velocity_out, point_features, knn_graph) in enumerate(loader):
            t             = t.to(device)
            pos           = pos.to(device)
            velocity_in   = velocity_in.to(device)
            velocity_out  = velocity_out.to(device)
            idcs_airfoil  = [x.to(device) for x in idcs_airfoil]
            if point_features is not None:
                point_features = point_features.to(device)
            if knn_graph is not None:
                knn_graph = knn_graph.to(device)
            pred = model(t, pos, idcs_airfoil, velocity_in, point_features, knn_graph)
            if loss_fn == "l2":
                # L2-norm loss — matches the competition metric exactly. Slower
                # convergence from random init (linear gradient vs MSE's quadratic),
                # but metric-aligned once near optimum.
                loss = (pred - velocity_out).norm(dim=3).mean()
            else:
                # MSE (default). Quadratic gradient → fast recovery from random-init
                # explosion. See LOG.md 2026-04-14 — beat L2 by 22-25% on test at d=6.
                loss = F.mse_loss(pred, velocity_out)
            metric = competition_metric(pred, velocity_out)
            total_loss += loss.item()
            total_metric += metric

            if is_train:
                (loss / accum_steps).backward()
                if (i + 1) % accum_steps == 0 or (i + 1) == n:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                    total_grad_norm += grad_norm
                    grad_norm_steps += 1
                    optimizer.step()
                    optimizer.zero_grad()

            phase = "train" if is_train else "val  "
            elapsed = time.perf_counter() - t0
            print(f"  [{phase}] {i+1:3d}/{n}  metric={metric:.4f}  elapsed={elapsed:.1f}s",
                  end='\r', flush=True)

    print()
    elapsed = time.perf_counter() - t0
    avg_grad_norm = total_grad_norm / max(grad_norm_steps, 1)
    if n == 0:
        return 0.0, 0.0, elapsed, avg_grad_norm
    return total_loss / n, total_metric / n, elapsed, avg_grad_norm


def train_one_seed(model_name, splits, seed, epochs, lr, patience, accum_steps, output_dir,
                    features=None, max_distance=0.0, wandb_project=None, wandb_entity=None, use_wandb=False,
                    data_fraction=1.0, run_tag=None, weight_decay=0.01, warmup_epochs=0, batch_size=1,
                    gnn_depth=None, gnn_hidden=None, gnn_heads=None, gnn_dropout=None,
                    update_coords=None, no_slip_mask=None, loss_fn="mse", log_udf=False,
                    resume_from=None, use_gate=None):
    # Resolve feature names: CLI override > model default > empty
    ModelClass = get_model_class(model_name)
    if features is not None:
        feature_names = features
    else:
        feature_names = getattr(ModelClass, 'FEATURES', []) or []

    # Build datasets with feature names and optional distance filtering
    train_ds = GRAMDataset(splits['train'], features=feature_names, max_distance=max_distance, log_udf=log_udf)
    val_ds = GRAMDataset(splits['val'], features=feature_names, max_distance=max_distance, log_udf=log_udf)
    test_ds = GRAMDataset(splits['test'], features=feature_names, max_distance=max_distance, log_udf=log_udf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                             num_workers=4, pin_memory=True)

    # Model + optimizer (ModelClass already resolved above for feature names)
    # Pass features override + GNN kwargs; models ignore unknown kwargs via TypeError fallback
    gnn_kwargs = {}
    if gnn_depth   is not None: gnn_kwargs['depth']      = gnn_depth
    if gnn_hidden  is not None: gnn_kwargs['hidden_dim'] = gnn_hidden
    if gnn_heads   is not None: gnn_kwargs['heads']      = gnn_heads
    if gnn_dropout is not None: gnn_kwargs['dropout']    = gnn_dropout
    if update_coords is not None: gnn_kwargs['update_coords'] = update_coords
    if no_slip_mask  is not None: gnn_kwargs['no_slip_mask']  = no_slip_mask
    if use_gate      is not None: gnn_kwargs['use_gate']      = use_gate
    try:
        model = ModelClass(features=feature_names, **gnn_kwargs)
    except TypeError:
        try:
            model = ModelClass(features=feature_names)
        except TypeError:
            model = ModelClass()
    # Re-initialize weights with this seed so each seed is genuinely different
    torch.manual_seed(seed)
    if not getattr(model, "preserves_init", False):
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    # Warm-start from a checkpoint (weights only — optimizer state is fresh,
    # so use lower LR + no warmup to avoid destabilizing a near-converged model).
    if resume_from is not None:
        state = torch.load(resume_from, weights_only=True, map_location='cpu')
        model.load_state_dict(state)
        print(f"  Resumed weights from {resume_from}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if warmup_epochs <= 0:
        warmup_epochs = max(1, epochs // 10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6),
        ],
        milestones=[warmup_epochs],
    )
    early_stopping = EarlyStopping(patience=patience)

    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = seed_dir / "best_model.pt"

    # Resolve effective flags from the constructed model (covers model defaults
    # when the caller left a kwarg as None). These go into wandb config so
    # sweeps over e.g. no_slip_mask can group/filter runs correctly.
    effective_no_slip_mask = getattr(model, 'no_slip_mask', None)
    effective_update_coords = getattr(model, 'update_coords', None)

    # WandB init
    run_name = f"{model_name}_seed{seed}_{int(time.time())}"
    wb_run = None
    if use_wandb and _WANDB_AVAILABLE:
        wb_tags = [t for t in [run_tag, f"noslip={effective_no_slip_mask}"] if t]
        wb_run = wandb.init(
            project=wandb_project or "iclr-2026",
            entity=wandb_entity,
            name=run_name,
            config={
                "model": model_name,
                "seed": seed,
                "epochs": epochs,
                "lr": lr,
                "patience": patience,
                "accum_steps": accum_steps,
                "features": features or [],
                "max_distance": max_distance,
                "data_fraction": data_fraction,
                "run_tag": run_tag or "",
                "output_dir": str(output_dir),
                "weight_decay": weight_decay,
                "warmup_epochs": warmup_epochs,
                "batch_size": batch_size,
                "gnn_depth": gnn_depth,
                "gnn_hidden": gnn_hidden,
                "gnn_heads": gnn_heads,
                "gnn_dropout": gnn_dropout,
                "no_slip_mask": effective_no_slip_mask,
                "update_coords": effective_update_coords,
                "loss_fn": loss_fn,
                "log_udf": log_udf,
            },
            tags=wb_tags or None,
            reinit=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # GPU probe — report device name + total VRAM so we know exactly what
    # Modal provisioned (the --gpu flag picks the type; this confirms it).
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name} | VRAM: {p.total_memory / 1e9:.1f} GB | "
              f"SMs: {p.multi_processor_count} | CC: {p.major}.{p.minor}")

    def log_gpu_memory():
        """Print and log GPU memory stats."""
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        mem_alloc_gb = torch.cuda.memory_allocated() / 1e9
        mem_reserved_gb = torch.cuda.memory_reserved() / 1e9
        mem_max_gb = torch.cuda.max_memory_allocated() / 1e9
        mem_str = f"GPU: {mem_alloc_gb:.2f}G allocated, {mem_reserved_gb:.2f}G reserved, {mem_max_gb:.2f}G peak"
        print(f"  {mem_str}")
        if wb_run is not None:
            wandb.log({
                "gpu/memory_allocated_gb": mem_alloc_gb,
                "gpu/memory_reserved_gb": mem_reserved_gb,
                "gpu/max_memory_allocated_gb": mem_max_gb,
            })

    print(f"\n{'─'*55}")
    print(f"  Seed {seed} | {model_name} | {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
    print(f"  Device: {device}")
    print(f"{'─'*55}")
    print(f"{'Epoch':>5}  {'Train loss':>10}  {'Train err':>9}  {'Val loss':>8}  {'Val err':>7}  {'Time':>6}  {'GPU Mem':>15}")
    print(f"{'─'*55}")

    epoch_records = []
    stopped_early = False

    for epoch in range(1, epochs + 1):
        train_loss, train_metric, train_time, grad_norm = run_split(model, train_loader, optimizer, accum_steps, device, loss_fn=loss_fn)
        val_loss, val_metric, val_time, _ = run_split(model, val_loader, device=device, loss_fn=loss_fn)
        scheduler.step()
        total_time = train_time + val_time
        current_lr = optimizer.param_groups[0]['lr']

        # GPU memory info
        gpu_info = ""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_gb = torch.cuda.memory_allocated() / 1e9
            gpu_info = f"{mem_gb:>7.2f}G"

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_metric:>9.4f}  "
              f"{val_loss:>8.4f}  {val_metric:>7.4f}  {total_time:>5.1f}s  {gpu_info:>15}")

        if wb_run is not None:
            log_dict = {
                "train_loss": train_loss,
                "train_metric": train_metric,
                "val_loss": val_loss,
                "val_metric": val_metric,
                "lr": current_lr,
                "grad_norm": grad_norm,
            }
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                log_dict["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                log_dict["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                log_dict["gpu/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            wandb.log(log_dict, step=epoch)
        else:
            log_gpu_memory()

        # Clear GPU cache to reduce memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        epoch_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_metric": train_metric,
            "val_loss": val_loss,
            "val_metric": val_metric,
            "time_s": total_time,
        })

        stopped = early_stopping.step(val_metric, epoch)
        if early_stopping.counter == 0:
            torch.save(model.state_dict(), best_ckpt_path)
        if stopped:
            stopped_early = True
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best val_metric={early_stopping.best:.4f} at epoch {early_stopping.best_epoch})")
            break

    # Re-evaluate train + val on best checkpoint, then evaluate test.
    print(f"\n  Loading best checkpoint (epoch {early_stopping.best_epoch})...")
    model.load_state_dict(torch.load(best_ckpt_path))
    best_train_loss, best_train_metric, _, _ = run_split(model, train_loader, device=device, loss_fn=loss_fn)
    best_val_loss,   best_val_metric_eval, _, _ = run_split(model, val_loader, device=device, loss_fn=loss_fn)
    test_loss, test_metric, _, _ = run_split(model, test_loader, device=device, loss_fn=loss_fn)

    # Pretty final summary table (stdout)
    print(f"\n  {'─'*55}")
    print(f"  {'Split':<8} {'Loss':>12} {'Metric':>12}")
    print(f"  {'─'*55}")
    print(f"  {'Train':<8} {best_train_loss:>12.4f} {best_train_metric:>12.4f}")
    print(f"  {'Val':<8}   {best_val_loss:>12.4f} {best_val_metric_eval:>12.4f}")
    print(f"  {'Test':<8}  {test_loss:>12.4f} {test_metric:>12.4f}   (naive: 1.68)")
    print(f"  {'─'*55}")

    if wb_run is not None:
        # Log at epoch+1 so it doesn't collide with the last training step
        wandb.log({
            "final/train_loss":   best_train_loss,
            "final/train_metric": best_train_metric,
            "final/val_loss":     best_val_loss,
            "final/val_metric":   best_val_metric_eval,
            "final/test_loss":    test_loss,
            "final/test_metric":  test_metric,
        }, step=epoch + 1)
        wandb.run.summary["best_epoch"]        = early_stopping.best_epoch
        wandb.run.summary["best_train_loss"]   = best_train_loss
        wandb.run.summary["best_train_metric"] = best_train_metric
        wandb.run.summary["best_val_loss"]     = best_val_loss
        wandb.run.summary["best_val_metric"]   = early_stopping.best
        wandb.run.summary["test_loss"]         = test_loss
        wandb.run.summary["test_metric"]       = test_metric
        wandb.finish()

    result = {
        "seed": seed,
        "best_epoch": early_stopping.best_epoch,
        "best_train_loss": best_train_loss,
        "best_train_metric": best_train_metric,
        "best_val_loss": best_val_loss,
        "best_val_metric": early_stopping.best,
        "stopped_early": stopped_early,
        "epochs_run": len(epoch_records),
        "test_loss": test_loss,
        "test_metric": test_metric,
        "epochs": epoch_records,
    }

    with open(seed_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def _positive_int(value):
    v = int(value)
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {v}")
    return v


def _nonneg_int(value):
    v = int(value)
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {v}")
    return v


def _positive_float(value):
    v = float(value)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {v}")
    return v


def _nonneg_float(value):
    v = float(value)
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {v}")
    return v


def _unit_fraction(value):
    v = float(value)
    if v <= 0 or v > 1:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {v}")
    return v


def _prob_float(value):
    v = float(value)
    if v < 0 or v >= 1:
        raise argparse.ArgumentTypeError(f"must be in [0, 1), got {v}")
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--seeds', type=_positive_int, nargs='+', default=[42])
    parser.add_argument('--epochs', type=_positive_int, default=100)
    parser.add_argument('--lr', type=_positive_float, default=1e-3)
    parser.add_argument('--patience', type=_positive_int, default=10)
    parser.add_argument('--grad-accum', type=_positive_int, default=8,
                        help='Accumulate gradients over N batches before stepping (default: 8).')
    parser.add_argument('--data-fraction', type=_unit_fraction, default=0.1)
    parser.add_argument('--data-path', type=str, default='/home/agrov/iclr-2026/data')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--features', type=str, nargs='*', default=None,
                        help='Explicit feature list (overrides model default + --no-udf/--knn flags).')
    parser.add_argument('--no-udf', action='store_true',
                        help='Remove UDF float features (udf, udf_truncated, udf_gradient) from feature set.')
    parser.add_argument('--knn', type=str, default=None, choices=['none', 'fixed', 'adaptive'],
                        help='kNN graph type: none (no graph), fixed (uniform k), adaptive (distance-weighted k).')
    parser.add_argument('--max-distance', type=_nonneg_float, default=0.0,
                        help='Filter out points beyond this UDF distance from airfoil. 0 = no filtering.')
    parser.add_argument('--wandb-project', type=str, default='iclr-2026',
                        help='WandB project name.')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='WandB entity (team/org). Defaults to personal account.')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB logging.')
    parser.add_argument('--run-tag', type=str, default=None,
                        help='Short tag to distinguish this run (e.g. "df0.2_udf"). Added to output path + WandB tags.')
    parser.add_argument('--weight-decay', type=_nonneg_float, default=0.01)
    parser.add_argument('--warmup-epochs', type=_nonneg_int, default=0,
                        help='0 = auto (epochs // 10).')
    parser.add_argument('--batch-size', type=_positive_int, default=1)
    parser.add_argument('--gnn-depth', type=_positive_int, default=None)
    parser.add_argument('--gnn-hidden', type=_positive_int, default=None)
    parser.add_argument('--gnn-heads', type=_positive_int, default=None)
    parser.add_argument('--gnn-dropout', type=_prob_float, default=None)
    parser.add_argument('--use-gate', dest='use_gate',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='gated_egno / gated_egno_meanres only: --use-gate (default) '
                             'keeps the per-edge sigmoid gate; --no-use-gate ablates it.')
    parser.add_argument('--update-coords', action='store_true',
                        help='FixedEGNN only: enable equivariant coordinate updates inside layers.')
    # IMPORTANT: argparse.BooleanOptionalAction silently treats any flag that
    # starts with `--no-` as the negation form, so a literal `--no-slip-mask`
    # flag gets parsed as "set False" regardless of intent. Use an affirmative
    # flag name (`--enforce-no-slip`) here; argparse then generates
    # `--enforce-no-slip` / `--no-enforce-no-slip` cleanly.
    parser.add_argument('--enforce-no-slip', dest='no_slip_mask',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='FixedEGNN only: zero predictions at idcs_airfoil at every rollout step '
                             '(no-slip boundary). Default: model default (on). Disable with --no-enforce-no-slip.')
    parser.add_argument('--loss', dest='loss_fn', choices=['mse', 'l2'], default='mse',
                        help='Training loss: "mse" (default, fast recovery from random init) or '
                             '"l2" (competition-metric-aligned, slower convergence). See LOG.md 2026-04-14.')
    parser.add_argument('--log-udf', action='store_true',
                        help='Runtime-transform the udf_truncated channel to log(udf+1e-4), '
                             'renormalized to [0, 1]. No precompute needed.')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to a state_dict.pt to load AFTER model construction. '
                             'Use with low lr (e.g. 1e-4) + warmup-epochs 0 for warm-start finetuning.')
    parser.add_argument('--train-ratio', type=_unit_fraction, default=0.7,
                        help='Fraction of geometries for training. Default 0.7. For final '
                             'submission use 0.95+ to train on all available data.')
    parser.add_argument('--val-ratio', type=_unit_fraction, default=0.15,
                        help='Fraction of geometries for validation. Default 0.15. Remaining '
                             'goes to test. Must satisfy train_ratio + val_ratio < 1.0.')
    args = parser.parse_args()

    # Resolve final feature list from flags (must run BEFORE tag/output-dir generation)
    ModelClass = get_model_class(args.model)
    args.features = resolve_features(ModelClass, args.features, args.no_udf, args.knn)

    # Auto-generate a descriptive tag if none provided
    if args.run_tag is None:
        feat_slug = "-".join(args.features) if args.features else "none"
        slip_slug = "" if args.no_slip_mask is None else (
            "_noslip" if args.no_slip_mask else "_slip"
        )
        log_slug = "_logudf" if args.log_udf else ""
        args.run_tag = f"df{args.data_fraction}_{feat_slug}{slip_slug}{log_slug}"

    if args.output_dir is None:
        # Include run_tag + timestamp → no collisions on reruns
        import time as _time
        ts = _time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = f"results/{args.model}/{args.run_tag}_{ts}"

    feat_str = ', '.join(args.features) if args.features else '(none)'
    print(f"\nFeatures: {feat_str}")
    if args.max_distance > 0:
        print(f"Max distance filter: {args.max_distance}")

    # Build splits once — all seeds share the same train/val/test files
    print(f"Loading {args.data_fraction*100:.0f}% of data...")
    splits = split_by_geometry(
        args.data_path,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        seed=42,  # fixed split seed — only model init + shuffle vary per seed
        data_fraction=args.data_fraction,
    )
    print(f"Train: {len(splits['train'])} files | "
          f"Val: {len(splits['val'])} files | "
          f"Test: {len(splits['test'])} files")

    # Train across seeds
    all_results = []
    for seed in args.seeds:
        result = train_one_seed(
            model_name=args.model,
            splits=splits,
            seed=seed,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            accum_steps=args.grad_accum,
            output_dir=args.output_dir,
            features=args.features,
            max_distance=args.max_distance,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            use_wandb=not args.no_wandb,
            data_fraction=args.data_fraction,
            run_tag=args.run_tag,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            gnn_depth=args.gnn_depth,
            gnn_hidden=args.gnn_hidden,
            gnn_heads=args.gnn_heads,
            gnn_dropout=args.gnn_dropout,
            update_coords=args.update_coords if args.update_coords else None,
            no_slip_mask=args.no_slip_mask,
            loss_fn=args.loss_fn,
            log_udf=args.log_udf,
            resume_from=args.resume_from,
            use_gate=args.use_gate,
        )
        all_results.append(result)

    # Cross-seed summary
    val_metrics = [r['best_val_metric'] for r in all_results]
    test_metrics = [r['test_metric'] for r in all_results]
    test_losses = [r['test_loss'] for r in all_results]

    def mean_std(vals):
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5
        return mean, std

    val_mean, val_std = mean_std(val_metrics)
    test_mean, test_std = mean_std(test_metrics)
    test_loss_mean, test_loss_std = mean_std(test_losses)

    summary = {
        "model": args.model,
        "seeds": args.seeds,
        "data_fraction": args.data_fraction,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "naive_baseline": 1.68,
        "val_metric":  {"mean": val_mean,  "std": val_std},
        "test_metric": {"mean": test_mean, "std": test_std},
        "test_loss":   {"mean": test_loss_mean, "std": test_loss_std},
        "per_seed": [
            {
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "val_metric": r["best_val_metric"],
                "test_metric": r["test_metric"],
                "test_loss": r["test_loss"],
                "stopped_early": r["stopped_early"],
            }
            for r in all_results
        ],
    }

    out_path = Path(args.output_dir) / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'═'*55}")
    print(f"  Summary ({len(args.seeds)} seed{'s' if len(args.seeds) > 1 else ''})")
    print(f"{'═'*55}")
    print(f"  Val  metric : {val_mean:.4f} ± {val_std:.4f}")
    print(f"  Test metric : {test_mean:.4f} ± {test_std:.4f}  (naive baseline: 1.68)")
    print(f"  Test loss   : {test_loss_mean:.4f} ± {test_loss_std:.4f}")
    print(f"\n  Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
