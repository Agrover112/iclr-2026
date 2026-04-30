"""Generate poster plots from results/wandb_dump.

Produces (into figures/):
  - poster_loss_curve_marker.pdf : train + val curves + red marker at final test.
  - poster_loss_curve_plain.pdf  : train + val only (final test goes in caption).
  - poster_ablation_grid.pdf     : 2x2 test-only bars; hidden-dim gets a trend line.
  - poster_ablations.csv         : machine-readable copy of the numbers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "results" / "wandb_dump"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# dead-neuron cutoff — all runs below used (post-cutoff) init
CUTOFF = "20260414-144933"

HERO_RUN = "x3ytjdcr"  # egno_meanres_h96_submission

# Ablation panels. "trend": draw a dotted line connecting bar tops.
ABLATIONS = {
    "Graph topology (df=0.15)": {
        # Matched-hparam fixed vs UDF-adaptive kNN on plain EGNN backbone.
        "x_label": "",
        "trend": False,
        "points": [
            ("fixed kNN",    "gaqcwjag"),  # fix_v2_d6              (k=16 Euclidean)
            ("adaptive kNN", "fj8hq9v8"),  # fix_v2_d6_adaptive_fair (UDF-modulated k)
        ],
    },
    "EGNN: learning rate & gating (df=0.15)": {
        "x_label": "",
        "trend": False,
        "points": [
            ("base LR",              "gaqcwjag"),  # fix_v2_d6        (lr 3e-4, no gate)
            ("aggressive LR",        "exsoqtnn"),  # fix_v2_d6_aggro  (lr 2e-3, no gate)
            ("aggressive LR + gate", "7efo82fi"),  # gated_v2_aggro   (lr 2e-3, gated)
        ],
    },
    "Temporal mixing (df=0.15)": {
        "x_label": "",
        "trend": False,
        "points": [
            ("1-D conv", "961a9565"),
            ("spectral", "ilzy2aku"),
        ],
    },
    "Residual target (df=0.15)": {
        # df=1 egno_fd crashed without final metrics; df=0.15 is only complete pair
        "x_label": "",
        "trend": False,
        "points": [
            ("last frame", "t2coe8qz"),
            ("mean flow",  "f5rwtjbn"),
        ],
    },
    "Gating, full stack (df=0.15)": {
        # Matched-hparam (h=32, d=4, lr=2e-3, seed 42) ablation of the per-edge
        # sigmoid gate inside the full EGNO stack (spectral time mixing + mean-residual).
        "x_label": "",
        "trend": False,
        "points": [
            ("no gate", "7qyqxm0r"),  # egno_meanres_nogate_v (2026-04-22)
            ("gate",    "f5rwtjbn"),  # egno_meanres_v1       (2026-04-17)
        ],
    },
    "Hidden dim $h$ (gEGNO, df=1.0)": {
        "x_label": "hidden dim $h$",
        "trend": True,
        "points": [
            ("32", "7v4n7u8h"),
            ("48", "9y0houlx"),
            ("64", "8xjc65gk"),
            ("96", "x3ytjdcr"),
        ],
    },
}


def load_history(run_id: str) -> pd.DataFrame:
    return pd.read_csv(DUMP / f"{run_id}_history.csv")


def final_metric(run_id: str, col: str) -> float | None:
    df = load_history(run_id)
    key = f"final/{col}"
    if key in df.columns and df[key].notna().any():
        return float(df[key].dropna().iloc[-1])
    return None


def _base_curve(ax):
    df = load_history(HERO_RUN)
    train = df.dropna(subset=["train_metric"])
    val = df.dropna(subset=["val_metric"])
    ax.plot(train["_step"], train["train_metric"], linewidth=1.6,
            color="#4c72b0", label="train")
    ax.plot(val["_step"], val["val_metric"], linewidth=1.6,
            color="#dd8452", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("L2 metric")
    ax.set_title("gEGNO $h{=}96$ — training curves")
    ax.grid(True, alpha=0.25)
    return val


def plot_hero_marker() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    val = _base_curve(ax)
    final_test = final_metric(HERO_RUN, "test_metric")
    if final_test is not None:
        x_end = float(val["_step"].iloc[-1])
        ax.scatter([x_end], [final_test], color="#c44e52", s=55, zorder=6,
                   label=f"final test = {final_test:.3f}")
        ax.annotate(f"test = {final_test:.3f}", xy=(x_end, final_test),
                    xytext=(-8, 8), textcoords="offset points",
                    ha="right", va="bottom", fontsize=9, color="#c44e52")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(OUT / "poster_loss_curve_marker.pdf")
    fig.savefig(OUT / "poster_loss_curve_marker.png", dpi=200)
    plt.close(fig)


def plot_hero_plain() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    _base_curve(ax)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(OUT / "poster_loss_curve_plain.pdf")
    fig.savefig(OUT / "poster_loss_curve_plain.png", dpi=200)
    plt.close(fig)


def plot_ablations() -> pd.DataFrame:
    n = len(ABLATIONS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 3.1 * nrows),
                              constrained_layout=True)
    axes = axes.flatten()
    # Hide any unused panels
    for ax in axes[n:]:
        ax.set_visible(False)
    records = []
    for ax, (title, cfg) in zip(axes[:n], ABLATIONS.items()):
        labels, test_vs = [], []
        for lab, rid in cfg["points"]:
            v = final_metric(rid, "val_metric")
            t = final_metric(rid, "test_metric")
            labels.append(lab); test_vs.append(t)
            records.append({"ablation": title, "x": lab, "run_id": rid,
                            "final_val_metric": v, "final_test_metric": t})
        x = np.arange(len(labels))
        bars = ax.bar(x, test_vs, width=0.55, color="#4c72b0")
        valid = [i for i, v in enumerate(test_vs) if v is not None]
        if valid:
            best = min(valid, key=lambda i: test_vs[i])
            bars[best].set_color("#c44e52")
        if cfg.get("trend") and all(v is not None for v in test_vs):
            ax.plot(x, test_vs, linestyle=":", color="#333", linewidth=1.2,
                    marker="o", markersize=4, zorder=3)
        for i, t in enumerate(test_vs):
            if t is not None:
                ax.text(i, t, f"{t:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_xlabel(cfg["x_label"])
        ax.set_ylabel("test L2")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        vs = [v for v in test_vs if v is not None]
        if vs:
            ax.set_ylim(min(vs) * 0.97, max(vs) * 1.05)
    fig.savefig(OUT / "poster_ablation_grid.pdf")
    fig.savefig(OUT / "poster_ablation_grid.png", dpi=200)
    plt.close(fig)
    return pd.DataFrame(records)


if __name__ == "__main__":
    plot_hero_marker()
    plot_hero_plain()
    df = plot_ablations()
    df.to_csv(OUT / "poster_ablations.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nWrote: {OUT/'poster_loss_curve_marker.pdf'}")
    print(f"Wrote: {OUT/'poster_loss_curve_plain.pdf'}")
    print(f"Wrote: {OUT/'poster_ablation_grid.pdf'}")
    print(f"Wrote: {OUT/'poster_ablations.csv'}")
