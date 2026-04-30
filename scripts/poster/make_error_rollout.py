"""Per-model error-magnitude rollout for the poster.

Plots |pred - gt|_2 as a z-slice heatmap for each of t=0..4, with one row
per model (EGNN baseline, gEGNO submission). Shared colour scale so the
two rows are directly comparable.

Output: figures/poster_error_rollout.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
SAMPLE = "1021_1-0"

# (label, .pt path) — both saved by scripts/modal/save_predictions.py
MODELS = [
    ("MLP (Baseline)", FIG / "predictions" / f"{SAMPLE}_mlp.pt"),
    ("EGNN",                     FIG / "predictions" / f"{SAMPLE}_preds.pt"),
    ("gEGNO (submission)",       FIG / "predictions" / f"{SAMPLE}_submission.pt"),
]

TIMESTEPS = [0, 1, 2, 3, 4]
SLICE_AXIS = 2
SLICE_THICKNESS = 0.05
GRID = 240
CMAP = "hot"


def load(pt_path):
    d = torch.load(pt_path, weights_only=True, map_location="cpu")
    return d["pos"].numpy(), d["gt"].numpy(), d["pred"].numpy(), d["idcs_airfoil"].numpy()


def slice_mask(pos, is_af):
    slice_val = float(np.median(pos[is_af, SLICE_AXIS]))
    mask = np.abs(pos[:, SLICE_AXIS] - slice_val) <= SLICE_THICKNESS
    return mask, slice_val


def error_field(pos2d, err_mag, grid_x, grid_y):
    return griddata(pos2d, err_mag, (grid_x, grid_y), method="linear", fill_value=np.nan)


def compute_error_fields(pt_path):
    pos, gt, pred, idcs = load(pt_path)
    is_af = np.zeros(pos.shape[0], dtype=bool); is_af[idcs] = True
    mask, slice_val = slice_mask(pos, is_af)
    idx = np.where(mask)[0]
    plot_axes = [a for a in (0, 1, 2) if a != SLICE_AXIS]
    p2d = pos[idx][:, plot_axes]
    xmin, ymin = p2d.min(axis=0); xmax, ymax = p2d.max(axis=0)
    xs = np.linspace(xmin, xmax, GRID); ys = np.linspace(ymin, ymax, GRID)
    gx, gy = np.meshgrid(xs, ys)

    fields = []
    for t in TIMESTEPS:
        err_mag = np.linalg.norm(pred[t, idx] - gt[t, idx], axis=-1)
        fields.append(error_field(p2d, err_mag, gx, gy))

    af = pos[mask & is_af][:, plot_axes]
    return fields, (xmin, xmax, ymin, ymax), af, slice_val


def main():
    rows = []
    extent = None
    af = None
    slice_val = None
    for label, pt in MODELS:
        fields, ext, a, sv = compute_error_fields(pt)
        rows.append((label, fields))
        extent = ext
        af = a
        slice_val = sv

    # Shared colour scale across all panels.
    all_vals = np.concatenate([
        f[~np.isnan(f)].ravel() for _, row_fields in rows for f in row_fields
    ])
    vmax = float(np.percentile(all_vals, 99))

    xmin, xmax, ymin, ymax = extent
    nrows, ncols = len(rows), len(TIMESTEPS)
    fig = plt.figure(figsize=(9.8, 1.5 * len(rows) + 0.6), constrained_layout=True)
    gs = GridSpec(nrows, ncols + 1, figure=fig,
                  width_ratios=[1]*ncols + [0.04], wspace=0.04, hspace=0.07)

    for r, (label, fields) in enumerate(rows):
        for c, field in enumerate(fields):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(field, extent=extent, origin="lower",
                           cmap=CMAP, aspect="equal", vmin=0, vmax=vmax)
            if af.size:
                ax.scatter(af[:, 0], af[:, 1], s=0.5, c="cyan",
                           alpha=0.85, zorder=5)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"$t{{=}}{TIMESTEPS[c]}$", fontsize=10,
                             fontweight="bold", color="#000061")
            if c == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold",
                              color="#000061", labelpad=8)

    cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(im, cax=cax, label=r"$\|\hat v - v\|$")

    fig.suptitle(
        f"Prediction error $|\\hat v - v|$ · sample {SAMPLE} · z-slice at "
        f"{slice_val:.3f} · shared colour scale",
        fontsize=9, color="#4a4a55",
    )

    out_pdf = FIG / "poster_error_rollout.pdf"
    out_png = FIG / "poster_error_rollout.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
