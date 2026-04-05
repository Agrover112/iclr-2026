"""
Baseline MLP timing run on 10% of data.

Usage:
    python train_mlp.py [--epochs N] [--data-fraction F]

Reports per-epoch wall time, train loss, and val metric so we can estimate
full-dataset training cost before committing to a longer run.
"""

import argparse
import time
import torch
from torch.utils.data import DataLoader

from src.data import split_by_geometry, GRAMDataset
from src.train import collate_fn, competition_metric
from models.mlp.model import MLP


def run_epoch(model, loader, optimizer=None):
    """Run one train or val epoch. Returns (mean_loss, elapsed_seconds)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_metric = 0.0
    t0 = time.perf_counter()
    phase = "train" if is_train else "val  "

    with torch.set_grad_enabled(is_train):
        for i, (t, pos, idcs_airfoil, velocity_in, velocity_out, _point_features) in enumerate(loader):
            pred = model(t, pos, idcs_airfoil, velocity_in)
            loss = torch.nn.functional.mse_loss(pred, velocity_out)
            metric = competition_metric(pred, velocity_out)
            total_metric += metric

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elapsed = time.perf_counter() - t0
            print(f"  [{phase}] {i+1:3d}/{len(loader)}  metric={metric:.4f}  "
                  f"elapsed={elapsed:.1f}s", end='\r', flush=True)

    print()  # newline after \r
    elapsed = time.perf_counter() - t0
    return total_metric / len(loader), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--data-fraction', type=float, default=0.1)
    parser.add_argument('--data-path', type=str, default='/home/agrov/iclr-2026/data')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Data
    print(f"\nLoading {args.data_fraction*100:.0f}% of data (seed={args.seed})...")
    splits = split_by_geometry(
        args.data_path,
        train_ratio=0.7, val_ratio=0.15,
        seed=args.seed, data_fraction=args.data_fraction,
    )
    train_ds = GRAMDataset(splits['train'])
    val_ds = GRAMDataset(splits['val'])
    print(f"Train: {len(train_ds)} files | Val: {len(val_ds)} files")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn,
                            num_workers=4, pin_memory=True)

    # Model
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MLP params: {total_params:,}\n")

    print(f"{'Epoch':>5}  {'Train metric':>12}  {'Val metric':>10}  {'Time (s)':>9}")
    print("-" * 46)

    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        train_metric, train_time = run_epoch(model, train_loader, optimizer)
        val_metric, val_time = run_epoch(model, val_loader)
        total_time = train_time + val_time
        epoch_times.append(total_time)
        print(f"{epoch:>5}  {train_metric:>12.4f}  {val_metric:>10.4f}  {total_time:>9.1f}s")

    avg_epoch = sum(epoch_times) / len(epoch_times)
    print(f"\nAvg time/epoch: {avg_epoch:.1f}s")
    print(f"Estimated full-data (162 sims) time/epoch: {avg_epoch / args.data_fraction:.1f}s "
          f"({avg_epoch / args.data_fraction / 60:.1f} min)")


if __name__ == '__main__':
    main()
