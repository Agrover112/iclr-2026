"""Grouped bar chart of per-timestep L2 error with std whiskers across test samples.

Input : results/per_timestep_errors.csv  (from scripts/modal/eval_per_timestep_error.py)
Output: figures/poster_horizon_bars.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "per_timestep_errors.csv"
OUT = ROOT / "figures"

NAVY = "#000061"
ACCENT = "#c44e52"
MUTE = "#5a5a65"

# colour per model (stable)
MODEL_COLOUR = {
    "MLP":          "#b0b0c0",
    "EGNN (fixed)": "#7a90c4",
    "gEGNO":        ACCENT,
}


def main() -> None:
    df = pd.read_csv(CSV)
    models = [m for m in MODEL_COLOUR if m in df["model"].unique()]
    frames = sorted(df["frame"].unique())

    n_models = len(models)
    x = np.arange(len(frames))
    width = 0.78 / n_models

    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)

    # Collect per-model (xs, means, stds) so we can overlay trend lines after.
    trend = {}
    for i, m in enumerate(models):
        sub = df[df["model"] == m]
        means, stds, ns = [], [], []
        for f in frames:
            v = sub.loc[sub["frame"] == f, "l2"].to_numpy()
            means.append(float(v.mean()))
            stds.append(float(v.std(ddof=1) if len(v) > 1 else 0.0))
            ns.append(int(len(v)))
        offset = (i - (n_models - 1) / 2) * width
        xs = x + offset
        bars = ax.bar(xs, means, width, label=m,
                      color=MODEL_COLOUR[m], edgecolor="white", linewidth=0.6)
        ax.errorbar(xs, means, yerr=stds, fmt="none",
                    ecolor="#333", elinewidth=1.1, capsize=3, capthick=1.1)
        # Place labels above the top whisker so they never overlap the bars/errors.
        for xi, mean, sd in zip(xs, means, stds):
            ax.text(xi, mean + sd + 0.08, f"{mean:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color="#333")
        trend[m] = (xs, means)

    # Dotted trend lines across time, per model — makes the temporal slope legible.
    for m in models:
        xs, means = trend[m]
        ax.plot(xs, means, linestyle=":", linewidth=1.3,
                color=MODEL_COLOUR[m], marker="o", markersize=3.5,
                markerfacecolor=MODEL_COLOUR[m], markeredgecolor="white",
                markeredgewidth=0.6, zorder=4)

    ax.set_xticks(x, [f"t={f}" for f in frames])
    ax.set_xlabel("output frame")
    ax.set_ylabel("mean L2 error  (per point)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend outside the axes on the right so it never overlaps bars.
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=10, handlelength=1.4)

    n_samples = df.groupby("model")["file"].nunique().max()
    ax.set_title(
        f"Per-frame L2 error across architectures  ·  n = {n_samples} test samples  (mean ± std)",
        fontsize=11, color=NAVY, pad=6, loc="left",
    )
    # Leave some headroom so value labels don't clip into the title.
    top = df["l2"].max() + df["l2"].std() * 1.2
    ax.set_ylim(0, top)

    fig.savefig(OUT / "poster_horizon_bars.pdf")
    fig.savefig(OUT / "poster_horizon_bars.png", dpi=220)
    plt.close(fig)

    # Summary table to stdout
    summary = (df.groupby(["model", "frame"])["l2"]
                 .agg(["mean", "std", "count"]).reset_index()
                 .pivot(index="model", columns="frame", values="mean"))
    print("\nPer-model mean L2 by frame:")
    print(summary.round(4).to_string())
    print(f"\nWrote {OUT/'poster_horizon_bars.png'}")


if __name__ == "__main__":
    main()
