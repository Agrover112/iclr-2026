"""Histogram of UDF (distance-to-airfoil-surface) over a sample of the training set.

Story the figure tells: points are concentrated within the boundary-layer scale
(lambda = 0.05) — which is why adaptive kNN pays off near the wall and why the
UDF feature carries so much signal. The spike at d_max = 0.5 is the truncation
(all free-stream points map there).

Writes figures/poster_udf_hist.{png,pdf}.
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "figures"

NAVY = "#000061"
ACCENT = "#c44e52"
MUTE = "#5a5a65"

LAMBDA = 0.05       # boundary-layer scale (adaptive-kNN decay)
D_MAX = 0.5         # UDF truncation

N_SAMPLES = 80      # number of .npz files to aggregate
BINS = 80           # histogram bins
SEED = 42


def collect_udf(n_samples: int, seed: int = SEED) -> np.ndarray:
    feat_files = sorted(DATA.glob("*_feat.pt"))
    random.Random(seed).shuffle(feat_files)
    feat_files = feat_files[:n_samples]
    chunks = []
    for f in feat_files:
        d = torch.load(f, weights_only=False, map_location="cpu")
        chunks.append(d["udf_truncated"].squeeze(-1).numpy().astype(np.float32))
    return np.concatenate(chunks)


def make_figure(udf: np.ndarray) -> None:
    pct_boundary = float(np.mean(udf <= LAMBDA) * 100)
    pct_near = float(np.mean(udf <= 0.1) * 100)
    median = float(np.median(udf))
    p95 = float(np.percentile(udf, 95))

    fig, (ax_lin, ax_log) = plt.subplots(
        1, 2, figsize=(8.5, 3.0), constrained_layout=True, sharey=False
    )

    # --- left: linear-y zoom into boundary layer (0 - 0.15) ---
    zoom = 0.15
    ax_lin.hist(udf[udf <= zoom], bins=50, range=(0.0, zoom),
                color=NAVY, alpha=0.9, edgecolor="white", linewidth=0.4)
    ax_lin.axvline(LAMBDA, color=ACCENT, linestyle="--", linewidth=1.8)
    ax_lin.text(LAMBDA + 0.002, ax_lin.get_ylim()[1] * 0.94,
                f"λ = {LAMBDA}\nboundary-layer scale",
                color=ACCENT, fontsize=10, va="top", ha="left")
    ax_lin.set_xlabel("UDF  (zoomed: 0–0.15)")
    ax_lin.set_ylabel("number of points")
    ax_lin.set_xlim(0.0, zoom)
    ax_lin.grid(True, axis="y", alpha=0.22)
    ax_lin.spines["top"].set_visible(False)
    ax_lin.spines["right"].set_visible(False)
    ax_lin.set_title(f"{pct_boundary:.0f}% of points within λ",
                     fontsize=11, color=NAVY, pad=6, loc="left")

    # --- right: full range with log-y, showing the long tail ---
    ax_log.hist(udf, bins=BINS, range=(0.0, D_MAX),
                color=NAVY, alpha=0.9, edgecolor="white", linewidth=0.4)
    ax_log.axvline(LAMBDA, color=ACCENT, linestyle="--", linewidth=1.8)
    ax_log.set_yscale("log")
    ax_log.set_xlabel("UDF  (full range: 0–0.5, log-y)")
    ax_log.set_xlim(0.0, D_MAX)
    ax_log.grid(True, axis="y", alpha=0.22, which="major")
    ax_log.minorticks_off()
    ax_log.spines["top"].set_visible(False)
    ax_log.spines["right"].set_visible(False)
    ax_log.set_title(
        f"median d = {median:.3f}  ·  95th pct = {p95:.2f}",
        fontsize=11, color=NAVY, pad=6, loc="left",
    )

    fig.suptitle(
        "Distance-to-airfoil distribution across the point cloud",
        fontsize=12, fontweight="bold", color=NAVY,
    )

    fig.savefig(OUT / "poster_udf_hist.pdf")
    fig.savefig(OUT / "poster_udf_hist.png", dpi=220)
    plt.close(fig)
    print(f"n points: {len(udf):,}  |  ≤λ: {pct_boundary:.1f}%  |  ≤0.1: {pct_near:.1f}%  |  median: {median:.4f}  |  p95: {p95:.3f}")
    print(f"wrote {OUT/'poster_udf_hist.png'}")


if __name__ == "__main__":
    udf = collect_udf(N_SAMPLES)
    make_figure(udf)
