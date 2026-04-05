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

from src.data import split_by_geometry, GRAMDataset

# Registry: name -> import path and class
MODEL_REGISTRY = {
    "mlp":          ("models.mlp.model",          "MLP"),
    "residual_mlp": ("models.residual_mlp.model", "ResidualMLP"),
}


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

    # Pass precomputed point features if available; None triggers on-the-fly computation
    if 'point_features' in batch[0]:
        point_features = torch.stack([b['point_features'] for b in batch])
    else:
        point_features = None

    return t, pos, idcs_airfoil, velocity_in, velocity_out, point_features


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


def run_split(model, loader, optimizer=None, accum_steps=1):
    """Run one pass over a data split. Returns (loss, metric, elapsed_s)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_metric = 0.0
    t0 = time.perf_counter()
    n = len(loader)

    if is_train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(is_train):
        for i, (t, pos, idcs_airfoil, velocity_in, velocity_out, point_features) in enumerate(loader):
            pred = model(t, pos, idcs_airfoil, velocity_in, point_features)
            loss = F.mse_loss(pred, velocity_out)
            metric = competition_metric(pred, velocity_out)
            total_loss += loss.item()
            total_metric += metric

            if is_train:
                (loss / accum_steps).backward()
                if (i + 1) % accum_steps == 0 or (i + 1) == n:
                    optimizer.step()
                    optimizer.zero_grad()

            phase = "train" if is_train else "val  "
            elapsed = time.perf_counter() - t0
            print(f"  [{phase}] {i+1:3d}/{n}  metric={metric:.4f}  elapsed={elapsed:.1f}s",
                  end='\r', flush=True)

    print()
    elapsed = time.perf_counter() - t0
    return total_loss / n, total_metric / n, elapsed


def train_one_seed(model_name, splits, seed, epochs, lr, patience, accum_steps, output_dir,
                    features=None, max_distance=0.0):
    # Resolve feature names: CLI override > model default > empty
    ModelClass = get_model_class(model_name)
    if features is not None:
        feature_names = features
    else:
        feature_names = getattr(ModelClass, 'FEATURES', []) or []

    # Build datasets with feature names and optional distance filtering
    train_ds = GRAMDataset(splits['train'], features=feature_names, max_distance=max_distance)
    val_ds = GRAMDataset(splits['val'], features=feature_names, max_distance=max_distance)
    test_ds = GRAMDataset(splits['test'], features=feature_names, max_distance=max_distance)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn,
                             num_workers=4, pin_memory=True)

    # Model + optimizer (ModelClass already resolved above for feature names)
    # Pass features override so model's input dim matches the data
    try:
        model = ModelClass(features=feature_names)
    except TypeError:
        # Model doesn't accept features kwarg (e.g. base MLP)
        model = ModelClass()
    # Re-initialize weights with this seed so each seed is genuinely different
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)

    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = seed_dir / "best_model.pt"

    print(f"\n{'─'*55}")
    print(f"  Seed {seed} | {model_name} | {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
    print(f"{'─'*55}")
    print(f"{'Epoch':>5}  {'Train loss':>10}  {'Train err':>9}  {'Val loss':>8}  {'Val err':>7}  {'Time':>6}")
    print(f"{'─'*55}")

    epoch_records = []
    stopped_early = False

    for epoch in range(1, epochs + 1):
        train_loss, train_metric, train_time = run_split(model, train_loader, optimizer, accum_steps)
        val_loss, val_metric, val_time = run_split(model, val_loader)
        total_time = train_time + val_time

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_metric:>9.4f}  "
              f"{val_loss:>8.4f}  {val_metric:>7.4f}  {total_time:>5.1f}s")

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

    # Evaluate on test set using best checkpoint
    print(f"\n  Loading best checkpoint (epoch {early_stopping.best_epoch})...")
    model.load_state_dict(torch.load(best_ckpt_path))
    test_loss, test_metric, _ = run_split(model, test_loader)
    print(f"  Test  loss={test_loss:.4f}  metric={test_metric:.4f}")

    result = {
        "seed": seed,
        "best_epoch": early_stopping.best_epoch,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Accumulate gradients over N batches before stepping (default: 1 = no accumulation)')
    parser.add_argument('--data-fraction', type=float, default=0.1)
    parser.add_argument('--data-path', type=str, default='/home/agrov/iclr-2026/data')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--features', type=str, nargs='*', default=None,
                        help='Per-point features to use (e.g. udf_truncated udf_gradient local_density). '
                             'Omit to use model default. Pass with no args to disable all features.')
    parser.add_argument('--max-distance', type=float, default=0.0,
                        help='Filter out points beyond this UDF distance from airfoil. 0 = no filtering.')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results/{args.model}"

    # Print feature/filter config
    feat_str = ', '.join(args.features) if args.features else '(model default)'
    if args.features is not None and len(args.features) == 0:
        feat_str = '(none)'
    print(f"\nFeatures: {feat_str}")
    if args.max_distance > 0:
        print(f"Max distance filter: {args.max_distance}")

    # Build splits once — all seeds share the same train/val/test files
    print(f"Loading {args.data_fraction*100:.0f}% of data...")
    splits = split_by_geometry(
        args.data_path,
        train_ratio=0.7, val_ratio=0.15,
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
