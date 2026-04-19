"""Fetch training runs from W&B for inspection.

Usage:
    /home/agrov/gram/bin/python scripts/analysis/fetch_wandb_runs.py
    /home/agrov/gram/bin/python scripts/analysis/fetch_wandb_runs.py --run-id <id>
    /home/agrov/gram/bin/python scripts/analysis/fetch_wandb_runs.py --state finished --limit 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import wandb

DEFAULT_ENTITY = "agrgnn-kth-royal-institute-of-technology"
DEFAULT_PROJECT = "iclr-2026"
OUT_DIR = Path("results/wandb_dump")


META_COLS = ["id", "name", "state", "tags", "created", "url"]


def _flatten(d: dict, prefix: str) -> dict:
    out = {}
    for k, v in dict(d).items():
        if k.startswith("_"):
            continue
        if isinstance(v, (list, tuple)):
            v = ",".join(map(str, v))
        elif isinstance(v, dict):
            v = json.dumps(v, default=str)
        out[f"{prefix}{k}"] = v
    return out


def summarize(runs) -> pd.DataFrame:
    """Schema-free dump: union of all config.* and summary.* keys across runs.

    No hardcoded keys — if you add a new wandb.config field or wandb.log metric,
    it shows up automatically as a new column. Missing keys become NaN.
    """
    rows = []
    for r in runs:
        row = {
            "id": r.id,
            "name": r.name,
            "state": r.state,
            "tags": ",".join(r.tags or []),
            "created": str(r.created_at),
            "url": r.url,
        }
        row.update(_flatten(r.config, "cfg."))
        row.update(_flatten(r.summary, "sum."))
        rows.append(row)
    df = pd.DataFrame(rows)
    # stable column order: meta first, then cfg.*, then sum.*
    cfg = sorted(c for c in df.columns if c.startswith("cfg."))
    summ = sorted(c for c in df.columns if c.startswith("sum."))
    return df[META_COLS + cfg + summ]


WANDB_FILES = ["config.yaml", "output.log", "wandb-summary.json",
               "wandb-metadata.json", "requirements.txt"]


def dump_files(run, out_dir: Path) -> None:
    rd = out_dir / run.id
    rd.mkdir(parents=True, exist_ok=True)
    for fname in WANDB_FILES:
        try:
            run.file(fname).download(root=str(rd), replace=True)
        except Exception as e:
            print(f"  [warn] {run.id}/{fname}: {e}")


def dump_history(run, out_dir: Path) -> None:
    hist = run.history(pandas=True)
    if hist is None or hist.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    hist.to_csv(out_dir / f"{run.id}_history.csv", index=False)
    with open(out_dir / f"{run.id}_config.json", "w") as f:
        json.dump(dict(run.config), f, indent=2, default=str)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--run-id", default=None, help="Fetch a single run by id (full history).")
    p.add_argument("--state", default=None, help="Filter by state, e.g. finished/running/crashed.")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--with-history", action="store_true", help="Dump per-step metric CSVs.")
    p.add_argument("--with-files", action="store_true",
                   help="Download config.yaml/output.log/wandb-summary.json/etc. per run.")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args()

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"

    if args.run_id:
        run = api.run(f"{path}/{args.run_id}")
        print(f"== {run.name} ({run.id}) — {run.state} ==")
        print("config:", json.dumps(dict(run.config), indent=2, default=str))
        print("summary:", json.dumps(dict(run.summary), indent=2, default=str))
        dump_history(run, args.out)
        if args.with_files:
            dump_files(run, args.out)
        print(f"history -> {args.out}/{run.id}_history.csv")
        return

    filters = {"state": args.state} if args.state else None
    runs = api.runs(path, filters=filters, order="-created_at")
    runs = list(runs)[: args.limit]
    df = summarize(runs)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "runs_summary.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\n[saved] {csv_path}  ({len(df)} runs)")

    if args.with_history:
        for r in runs:
            dump_history(r, args.out)
        print(f"[saved] per-run history CSVs -> {args.out}/")

    if args.with_files:
        for r in runs:
            print(f"[files] {r.id} {r.name}")
            dump_files(r, args.out)
        print(f"[saved] per-run wandb files -> {args.out}/<run_id>/")


if __name__ == "__main__":
    main()
