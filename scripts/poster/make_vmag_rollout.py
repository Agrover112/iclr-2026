"""Velocity-magnitude |v| rollout for the poster — far more legible at small
sizes than streamlines or quiver arrows.

Pipeline:
  1. Load `figures/predictions/<sample>_submission.pt` (has pred + gt as
     (5, N, 3) tensors, pos (N, 3), idcs_airfoil (M,)).
  2. Pick a z-slice at the airfoil median.
  3. For each timestep, compute |v| per point and interpolate onto a
     regular 2D grid via scipy.griddata.
  4. Plot a 2×5 grid of coloured heatmaps: GT |v| (top row), predicted |v|
     (bottom row), one column per output timestep, shared colorbar.

Output: figures/poster_vmag_rollout.{pdf,png}
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
SRC_PT = FIG / "predictions" / f"{SAMPLE}_submission.pt"
TIMESTEPS = [0, 1, 2, 3, 4]
SLICE_AXIS = 2             # 0=x, 1=y, 2=z
SLICE_THICKNESS = 0.05
GRID = 240
CMAP = "turbo"


def load_data():
    d = torch.load(SRC_PT, weights_only=True, map_location="cpu")
    pos = d["pos"].numpy()
    gt  = d["gt"].numpy()       # (T, N, 3)
    pred = d["pred"].numpy()    # (T, N, 3)
    idcs = d["idcs_airfoil"].numpy()
    is_af = np.zeros(pos.shape[0], dtype=bool)
    is_af[idcs] = True
    return pos, gt, pred, is_af


def slice_mask(pos, is_af):
    slice_val = float(np.median(pos[is_af, SLICE_AXIS]))
    mask = np.abs(pos[:, SLICE_AXIS] - slice_val) <= SLICE_THICKNESS
    return mask, slice_val


def interp_vmag(p2d, v3d, grid_x, grid_y):
    vmag = np.linalg.norm(v3d, axis=-1)   # (N_slice,)
    return griddata(p2d, vmag, (grid_x, grid_y), method="linear", fill_value=np.nan)


def main():
    pos, gt, pred, is_af = load_data()
    mask, slice_val = slice_mask(pos, is_af)
    idx = np.where(mask)[0]
    plot_axes = [a for a in (0, 1, 2) if a != SLICE_AXIS]
    p2d = pos[idx][:, plot_axes]

    # Build grid spanning the slice extent.
    xmin, ymin = p2d.min(axis=0)
    xmax, ymax = p2d.max(axis=0)
    xs = np.linspace(xmin, xmax, GRID)
    ys = np.linspace(ymin, ymax, GRID)
    grid_x, grid_y = np.meshgrid(xs, ys)

    gt_fields, pred_fields = [], []
    for t in TIMESTEPS:
        gt_fields.append(interp_vmag(p2d, gt[t, idx], grid_x, grid_y))
        pred_fields.append(interp_vmag(p2d, pred[t, idx], grid_x, grid_y))

    # Shared colour scale across ALL panels (use GT for anchor, pred must
    # live in same range for visual fairness).
    all_vals = np.concatenate([f[~np.isnan(f)].ravel() for f in gt_fields + pred_fields])
    vmax = float(np.percentile(all_vals, 99))
    vmin = 0.0

    af_pos = pos[mask & is_af][:, plot_axes]

    fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
    gs = GridSpec(2, len(TIMESTEPS) + 1, figure=fig,
                  width_ratios=[1]*len(TIMESTEPS) + [0.04],
                  wspace=0.04, hspace=0.08)

    def draw(ax, field, title_top):
        im = ax.imshow(field, extent=(xmin, xmax, ymin, ymax),
                       origin="lower", cmap=CMAP, aspect="equal",
                       vmin=vmin, vmax=vmax)
        if af_pos.size:
            ax.scatter(af_pos[:, 0], af_pos[:, 1], s=0.6, c="black",
                       alpha=0.85, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if title_top is not None:
            ax.set_title(title_top, fontsize=11, fontweight="bold",
                         color="#000061")
        return im

    for c, t in enumerate(TIMESTEPS):
        ax_gt  = fig.add_subplot(gs[0, c])
        ax_pr  = fig.add_subplot(gs[1, c])
        im = draw(ax_gt, gt_fields[c], f"$t{{=}}{t}$")
        draw(ax_pr, pred_fields[c], None)
        if c == 0:
            ax_gt.set_ylabel("Ground truth", fontsize=11,
                             fontweight="bold", color="#000061", labelpad=10)
            ax_pr.set_ylabel("gEGNO", fontsize=11,
                             fontweight="bold", color="#000061", labelpad=10)
    cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(im, cax=cax, label="$|v|$")

    fig.suptitle(
        f"Velocity magnitude rollout · sample {SAMPLE} · z-slice at "
        f"{slice_val:.3f} · shared colour scale",
        fontsize=10, color="#4a4a55",
    )

    out_pdf = FIG / "poster_vmag_rollout.pdf"
    out_png = FIG / "poster_vmag_rollout.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
