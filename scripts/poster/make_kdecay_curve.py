"""Render a poster-clean version of the adaptive k(d) decay curve.

Replaces the internal `visualize_adaptive_knn.py` figure which referenced
`LOG.md` in an annotation — stripped for external publication. Also uses
LaTeX-ish mathy axis labels instead of ASCII.

Output: figures/adaptive_knn_01_curve.png  (replaces the existing file)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "figures" / "adaptive_knn_01_curve.png"
OUT_PDF = ROOT / "figures" / "adaptive_knn_01_curve.pdf"

K_NEAR = 32
K_FAR  = 8
HIGHLIGHT_LAMBDA = 0.05


def main():
    lambdas = [0.05, 0.10, 0.20, 0.40]
    colors = ["#c44e52", "#dd8452", "#55a868", "#4c72b0"]
    d = np.linspace(0, 0.60, 500)

    fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)

    for lam, color in zip(lambdas, colors):
        k = K_FAR + (K_NEAR - K_FAR) * np.exp(-d / lam)
        lw = 2.4 if lam == HIGHLIGHT_LAMBDA else 1.3
        ax.plot(d, k, color=color, linewidth=lw,
                label=rf"$\lambda = {lam}$")
        # Mark the 1/e decay point (d = lambda)
        k_at_lam = K_FAR + (K_NEAR - K_FAR) * np.exp(-1.0)
        ax.plot(lam, k_at_lam, "o", color=color, markersize=5, zorder=5)

    ax.axhline(K_NEAR, color="#777", linewidth=0.9, linestyle="--",
               label=rf"$k_{{\mathrm{{near}}}} = {K_NEAR}$")
    ax.axhline(K_FAR,  color="#777", linewidth=0.9, linestyle="-.",
               label=rf"$k_{{\mathrm{{far}}}} = {K_FAR}$")

    ax.set_yticks(range(K_FAR, K_NEAR + 1, 4))
    ax.set_ylim(K_FAR - 2, K_NEAR + 2)
    ax.set_xlim(0, 0.60)
    ax.set_xlabel(r"UDF: distance to nearest airfoil surface point $\tilde d$",
                  fontsize=11)
    ax.set_ylabel(r"$k(\tilde d)$  (number of neighbours)", fontsize=11)
    ax.set_title(
        r"Adaptive $k$-NN neighbour count vs distance to surface",
        fontsize=11, color="#000061",
    )
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(OUT_PDF)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"Wrote: {OUT}")
    print(f"Wrote: {OUT_PDF}")


if __name__ == "__main__":
    main()
