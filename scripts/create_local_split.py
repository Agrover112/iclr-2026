#!/usr/bin/env python3
"""
Create a local 10% data split for CPU testing.
Selects 10% of simulations (keeping all 5 chunks together) and splits
into train/val/test (70/15/15) by simulation key to prevent leakage.
Saves split metadata to split.json for reproducibility.
"""

import json
from pathlib import Path

from src.data import split_by_geometry, get_geometry_to_files

DATA_DIR = "/home/agrov/iclr-2026/data"
OUTPUT_PATH = Path("/home/agrov/iclr-2026/split.json")

geom2files = get_geometry_to_files(DATA_DIR)
total_sims = len(geom2files)
total_files = sum(len(v) for v in geom2files.values())

splits = split_by_geometry(DATA_DIR, train_ratio=0.7, val_ratio=0.15, seed=42, data_fraction=0.10)

train_files = sorted(splits['train'])
val_files   = sorted(splits['val'])
test_files  = sorted(splits['test'])

# Derive sim lists from file lists
def files_to_sims(files):
    seen = {}
    for f in sorted(files):
        key = Path(f).stem.split("-")[0]
        seen[key] = True
    return list(seen.keys())

train_sims = files_to_sims(train_files)
val_sims   = files_to_sims(val_files)
test_sims  = files_to_sims(test_files)

split_data = {
    "total_sims": total_sims,
    "total_files": total_files,
    "selected_sims": len(train_sims) + len(val_sims) + len(test_sims),
    "selected_files": len(train_files) + len(val_files) + len(test_files),
    "seed": 42,
    "train_sims": train_sims,
    "val_sims": val_sims,
    "test_sims": test_sims,
    "train_files": [Path(f).name for f in train_files],
    "val_files":   [Path(f).name for f in val_files],
    "test_files":  [Path(f).name for f in test_files],
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(split_data, f, indent=2)

total_selected = len(train_files) + len(val_files) + len(test_files)
print(f"Total simulations: {total_sims} ({total_files} files)")
print(f"Selected:          {split_data['selected_sims']} sims ({total_selected} files)")
print(f"Train: {len(train_sims):2d} sims / {len(train_files):3d} files")
print(f"Val:   {len(val_sims):2d} sims / {len(val_files):3d} files")
print(f"Test:  {len(test_sims):2d} sims / {len(test_files):3d} files")
print(f"\nSplit saved to {OUTPUT_PATH}")
