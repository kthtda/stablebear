"""Generate the figure of the guide on projections and distances."""

import numpy as np
import stablebear as sb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent


# -- docs snippet start projection_inner_product --
def plot_projections(point_color="#2a78d6", projected_color="#eb6834",
                     line_color="0.6"):
    points = np.array([[-1.0, 2.5], [0.5, 3.0], [2.0, 0.5],
                       [3.0, 2.5], [4.0, -1.0]])
    line = sb.LinearSubspace([[2.0, 1.0]])
    # Lengths along the second coordinate count four times as much.
    stretched = sb.InnerProduct([[1.0, 0.0], [0.0, 4.0]])

    projections = [
        ("Euclidean inner product", np.asarray(line.project(points))),
        ("Inner product with Gram matrix diag(1, 4)",
         np.asarray(line.project(stretched(points)).points)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharex=True, sharey=True)
    for ax, (title, projected) in zip(axes, projections):
        ax.axline((0, 0), (2, 1), color=line_color, linewidth=1.0)
        for p, q in zip(points, projected):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=line_color,
                    linewidth=0.8, linestyle="--")
        ax.scatter(points[:, 0], points[:, 1], color=point_color,
                   marker="o", s=45, zorder=3, label="points")
        ax.scatter(projected[:, 0], projected[:, 1], color=projected_color,
                   marker="s", s=40, zorder=3, label="projected points")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 5.0)
        ax.set_ylim(-1.5, 4.0)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig
# -- docs snippet end projection_inner_product --


# -- Generate light and dark variants --
def _save_themed(plot_func, style, bg_color, fg_color, outfile):
    with plt.style.context(style), \
         plt.rc_context({"axes.facecolor": bg_color,
                         "axes.edgecolor": fg_color,
                         "axes.labelcolor": fg_color,
                         "axes.titlecolor": fg_color,
                         "figure.facecolor": bg_color,
                         "text.color": fg_color,
                         "xtick.color": fg_color,
                         "ytick.color": fg_color,
                         "legend.facecolor": bg_color,
                         "legend.edgecolor": fg_color,
                         "legend.labelcolor": fg_color}):
        fig = plot_func()
        fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        print(f"saved {outfile}")


LIGHT = ("default", "white", "black")
DARK = ("dark_background", "#14181e", "#e0e0e0")

if __name__ == "__main__":
    _save_themed(plot_projections, *LIGHT, HERE / "projection_inner_product_light.png")
    _save_themed(lambda: plot_projections("#3987e5", "#d95926", "0.45"),
                 *DARK, HERE / "projection_inner_product_dark.png")
