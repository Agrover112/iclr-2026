"""Per-timestep error curve for the poster.

Argues visually for one-shot decoding: error grows sub-exponentially
across the 5-frame horizon for EGNN / EGNO / gEGNO. A hypothetical
autoregressive rollout (drawn with a dashed line) would compound
multiplicatively; our numbers do not.

Output: figures/poster_horizon_curve.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
SRC = FIG / "predictions"

MODELS = [
    ("MLP (Baseline)",               SRC / "3001_10-0_mlp.pt",           "#777777"),
    ("EGNN ($h{=}32$)",              SRC / "3001_10-0_preds.pt",         "#4c72b0"),
    ("gEGNO ($h{=}96$, submission)", SRC / "3001_10-0_submission.pt",    "#c44e52"),
]
SAMPLE = "3001_10-0"


def per_timestep_err(pt):
    d = torch.load(pt, weights_only=True, map_location="cpu")
    return (d["pred"] - d["gt"]).norm(dim=-1).mean(dim=-1).numpy()   # (T,)


def persist_last_frame_err(pt, data_dir):
    """Per-t error of the naive 'copy the most recent input frame' baseline,
    computed on the same sample the .pt was generated from."""
    import numpy as np
    d = torch.load(pt, weights_only=True, map_location="cpu")
    gt = d["gt"].numpy()                               # (T, N, 3)
    # Load the raw .npz to get velocity_in[-1]
    stem = Path(pt).stem.split("_")[0] + "_" + Path(pt).stem.split("_")[1]
    npz_path = Path(data_dir) / f"{stem}.npz".replace("_", "_", 1)
    # Simpler: derive from the sample name used elsewhere
    raw = np.load(Path(data_dir) / (SAMPLE + ".npz"))
    vin = raw["velocity_in"]                            # (5, N, 3)
    last = vin[-1]                                      # (N, 3) — input frame t = -1
    # Broadcast: predict `last` for every output frame and diff against gt
    err = np.linalg.norm(gt - last[None], axis=-1).mean(axis=-1)  # (T,)
    return err


def main():
    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    T = 5
    xs = np.arange(T)

    for label, pt, col in MODELS:
        err = per_timestep_err(pt)
        ax.plot(xs, err, marker="o", linewidth=1.8, color=col, label=label)
        ax.text(xs[-1] + 0.08, err[-1], f"{err[-1]:.2f}",
                color=col, fontsize=9, va="center")

    # Naive persist-last-input-frame reference: predict v_in[-1] for every
    # output frame, then measure per-t error against the ground truth. Real
    # data from the same sample (not a hypothetical compounding curve).
    persist = persist_last_frame_err(MODELS[0][1], data_dir=Path(ROOT) / "data")
    ax.plot(xs, persist, linestyle="--", linewidth=1.2, color="#222",
            marker="x", markersize=6,
            label=r"naive repeat-last-frame ($\hat v_{\mathrm{out}}(t)=v_{\mathrm{in}}(-1)$)")
    ax.text(xs[-1] + 0.08, persist[-1], f"{persist[-1]:.2f}",
            color="#222", fontsize=9, va="center")

    ax.set_xticks(xs, [f"$t{{=}}{t}$" for t in xs])
    ax.set_xlabel("predicted frame index")
    ax.set_ylabel("per-point error $\\|\\hat v - v\\|_2$ (averaged over 100k points)")
    ax.set_title(f"Horizon growth of prediction error · sample {SAMPLE}",
                 fontsize=11, color="#000061")
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.38),
        fontsize=9, frameon=False, ncol=2,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(-0.2, T - 0.3)
    # Leave headroom for end-of-line labels and slack below for the legend.
    ax.set_ylim(bottom=ax.get_ylim()[0])

    out_pdf = FIG / "poster_horizon_curve.pdf"
    out_png = FIG / "poster_horizon_curve.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
