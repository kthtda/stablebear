"""Generate paired dendrogram/barcode figures.

Run from the repository root:
    PYTHONPATH=. python docs/_static/gen_linkage_figs.py
Requires SciPy, Matplotlib, and an installed stablebear backend.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# -- docs snippet start linkage_plot --
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

from stablebear.persistence import linkage_to_barcode
from stablebear.plotting import plot_barcode


def plot_linkage_pair(Z, bc, labels):
    """Plot a monotone linkage matrix alongside its converted barcode."""
    intervals = bc.to_numpy()
    heights = np.unique(Z[:, 2])
    palette = ["#2878b5", "#d87824", "#29956a"]
    colors = {h: palette[i % len(palette)] for i, h in enumerate(heights)}
    foreground = plt.rcParams["text.color"]
    limit = float(heights[-1]) * 1.3

    fig, (tree_ax, bar_ax) = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    dendrogram(
        Z, labels=labels, ax=tree_ax, orientation="right",
        link_color_func=lambda cluster: colors[Z[cluster - len(labels), 2]],
    )
    tree_ax.set_title("Dendrogram", loc="left", weight="bold", pad=14)
    tree_ax.set_ylabel("Observations")
    tree_ax.set_xlabel("Scale (merge height)")
    tree_ax.set_xlim(0, limit)
    tree_ax.set_xticks([0, *heights])
    tree_ax.tick_params(axis="y", labelsize=11)
    for height in heights:
        tree_ax.axvline(height, color=colors[height], alpha=0.25,
                        linewidth=1, linestyle="--", zorder=0)
        bar_ax.axvline(height, color=colors[height], alpha=0.25,
                       linewidth=1, linestyle="--", zorder=0)

    plot_barcode(bc, ax=bar_ax, color=foreground, linewidth=2.5)
    # All births are zero; plot_barcode orders these intervals longest first.
    shown = intervals[np.argsort(intervals[:, 1])[::-1]]
    bar_ax.collections[-1].set_colors([
        foreground if np.isinf(death) else colors[death]
        for _, death in shown
    ])
    interval_labels = [
        r"$[0,\infty)$" if np.isinf(death) else f"[0, {death:g})"
        for _, death in shown
    ]

    bar_ax.set_title("Resulting barcode", loc="left", weight="bold", pad=14)
    bar_ax.set_xlabel("Scale")
    bar_ax.set_xlim(0, limit)
    bar_ax.set_xticks([0, *heights])
    bar_ax.set_yticks(range(len(intervals)), interval_labels)
    bar_ax.set_ylim(-0.5, len(intervals) - 0.5)
    bar_ax.tick_params(axis="y", length=0, pad=10)
    for ax in (tree_ax, bar_ax):
        ax.spines[["top", "right"]].set_visible(False)
    bar_ax.spines["left"].set_visible(False)
    fig.tight_layout(w_pad=4)
    return fig
# -- docs snippet end linkage_plot --


# -- docs snippet start linkage_three_points --
def plot_three_points():
    Z = linkage([[0.0], [1.0], [4.0]], method="single")
    bc = linkage_to_barcode(Z)
    # bc.to_numpy(): [[0, 1], [0, 3], [0, inf]]
    return plot_linkage_pair(Z, bc, labels=["A (0)", "B (1)", "C (4)"])
# -- docs snippet end linkage_three_points --


# -- docs snippet start linkage_tied_merges --
def plot_tied_merges():
    Z = linkage([[0.0], [1.0], [4.0], [5.0]], method="single")
    bc = linkage_to_barcode(Z)
    # bc.to_numpy(): [[0, 1], [0, 1], [0, 3], [0, inf]]
    return plot_linkage_pair(Z, bc, labels=["A (0)", "B (1)", "C (4)", "D (5)"])
# -- docs snippet end linkage_tied_merges --


def main():
    here = Path(__file__).resolve().parent
    for theme, background, foreground in [
        ("light", "white", "#222222"),
        ("dark", "#1a1a2e", "#e0e0e0"),
    ]:
        with plt.rc_context({
            "font.size": 11,
            "axes.facecolor": background,
            "figure.facecolor": background,
            "axes.edgecolor": foreground,
            "axes.labelcolor": foreground,
            "axes.titlecolor": foreground,
            "text.color": foreground,
            "xtick.color": foreground,
            "ytick.color": foreground,
        }):
            for name, plot in [
                ("linkage_three_points", plot_three_points),
                ("linkage_tied_merges", plot_tied_merges),
            ]:
                fig = plot()
                path = here / f"{name}_{theme}.png"
                fig.savefig(path, dpi=180, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved {path}")


if __name__ == "__main__":
    main()
