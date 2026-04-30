"""Build a 2-row rollout comparison for the poster: GT (top) vs gEGNO (bottom)
across the full 5-frame prediction horizon.

Each source PNG has layout [GT | predicted | error] at a single timestep.
We crop the left (GT) and middle (pred) panels for each of t=0..4 and tile
them into a 2x5 grid.

Output: figures/poster_rollout_compare.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
SRC = FIG / "predictions_meanres"
SAMPLE = "1021_1-0"
TIMESTEPS = [0, 1, 2, 3, 4]


def crop_panel(img: Image.Image, which: str) -> Image.Image:
    """Crop a single panel (GT, pred, or err) out of a 3-panel PNG."""
    W, H = img.size
    # Drop the top ~5% sup-title strip so cropped panels are clean.
    top = int(H * 0.05)
    # Each source PNG is a 1x3 grid. Panel boundaries approximately at
    # thirds of width. Slightly inset left/right to avoid colorbar bleed.
    inset = int(W * 0.005)
    if which == "gt":
        left, right = 0, int(W * 1 / 3) - inset
    elif which == "pred":
        left, right = int(W * 1 / 3) + inset, int(W * 2 / 3) - inset
    elif which == "err":
        left, right = int(W * 2 / 3) + inset, W
    else:
        raise ValueError(which)
    return img.crop((left, top, right, H))


def main() -> None:
    rows = []  # (label, list of cropped PIL images per timestep)
    for label, which in [("Ground truth", "gt"), ("gEGNO prediction", "pred")]:
        imgs = []
        for t in TIMESTEPS:
            src = SRC / f"{SAMPLE}_t{t}_streamlines_z.png"
            imgs.append(crop_panel(Image.open(src), which))
        rows.append((label, imgs))

    nrows, ncols = len(rows), len(TIMESTEPS)
    fig = plt.figure(figsize=(13.5, 5.2), constrained_layout=True)
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.03, hspace=0.08)

    for r, (row_label, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"$t{{=}}{TIMESTEPS[c]}$", fontsize=11,
                             color="#000061", fontweight="bold")
            if c == 0:
                ax.set_ylabel(row_label, fontsize=11, color="#000061",
                              fontweight="bold", labelpad=10)

    fig.suptitle(
        f"Prediction rollout: sample {SAMPLE}, z-slice, 5-frame horizon",
        fontsize=10, color="#4a4a55",
    )

    out_pdf = FIG / "poster_rollout_compare.pdf"
    out_png = FIG / "poster_rollout_compare.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
