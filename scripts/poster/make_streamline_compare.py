"""Build the 3-way streamline-error comparison figure for the poster.

Crops the right panel (streamline error magnitude) from the three
per-model rendered PNGs and tiles them horizontally with a model-name
label above each. Source PNGs are produced by
`scripts/viz/export_predictions.py --streamlines`, each a 3-panel
figure (GT | pred | error).

Output: figures/poster_streamline_compare.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"

SAMPLE = "1021_1-0"
TIMESTEP = 4
SLICE = "z"

# (label, dir, colour_accent)
MODELS = [
    ("EGNN (fixed, baseline)",       "predictions_fixed_egnn"),
    ("EGNO (gated)",                  "predictions_egno"),
    ("gEGNO (mean-residual, ours)",  "predictions_meanres"),
]


def load_error_panel(model_dir: str) -> Image.Image:
    """Crop the right third (error-magnitude panel) from a source PNG."""
    p = FIG / model_dir / f"{SAMPLE}_t{TIMESTEP}_streamlines_{SLICE}.png"
    img = Image.open(p)
    W, H = img.size
    # The source is a 1x3 subplot grid + shared suptitle. The right third
    # contains the error heatmap and its colorbar. Strip the top ~5% title
    # strip so we can draw a clean model label.
    left = int(W * 2 / 3)
    top = int(H * 0.04)
    return img.crop((left, top, W, H))


def main() -> None:
    panels = [(lab, load_error_panel(d)) for lab, d in MODELS]

    fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
    gs = GridSpec(1, len(panels), figure=fig, wspace=0.04)
    for ax, (label, img) in zip([fig.add_subplot(gs[0, i]) for i in range(len(panels))], panels):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(label, fontsize=11, fontweight="bold", color="#000061")

    fig.suptitle(
        f"Streamline error magnitude  ·  sample {SAMPLE}  ·  t={TIMESTEP}  ·  {SLICE}-slice",
        fontsize=10, color="#4a4a55",
    )

    out_pdf = FIG / "poster_streamline_compare.pdf"
    out_png = FIG / "poster_streamline_compare.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
