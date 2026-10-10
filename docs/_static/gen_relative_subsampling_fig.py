"""Generate the relative subsampling example for light and dark themes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# -- docs snippet start relative_subsamples --
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import stablebear as sb
from stablebear.plotting import plot_distance_weight_heatmap


def plot_relative_subsamples(*, uniform=False, dark=False):
    point_color = "#b8c5cf" if dark else "#526474"
    selected_color = "#ff9c70" if dark else "#cf512b"
    query_color = "#16384b"
    outline_color = "#202936" if dark else "#ffffff"
    rng = np.random.default_rng(12)
    points = rng.uniform(-1, 1, size=(100, 2))
    queries = np.array([[-0.6, -0.3], [0.0, 0.55], [0.6, -0.3]])
    if uniform:
        distribution = sb.distributions.Uniform(start=0, end=0.65)
        title = r"Uniform distance weights ($0 \leq d < 0.65$)"
    else:
        distribution = sb.distributions.Gaussian(std=0.4)
        title = "Gaussian distance weights (std = 0.4)"
    peak = distribution.weight(0.0)
    samples = sb.random.subsample_relative(
        sb.PointCloud(points), queries, n_points=10, replace=False,
        distribution=distribution,
        generator=sb.random.Generator(seed=5),
    )

    extent = (-1.15, 1.15, -1.15, 1.15)
    # Higher weights are lighter in both themes, with a wide luminance range.
    field_colors = (["#172331", "#316c90", "#8bd1ed"] if dark else
                    ["#155477", "#78acc9", "#f5f7f9"])
    cmap = (ListedColormap([field_colors[0], field_colors[-1]]) if uniform else
            LinearSegmentedColormap.from_list("distance_weight", field_colors))

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        heatmap = plot_distance_weight_heatmap(
            distribution, query=queries[i], extent=extent, ax=ax,
            cmap=cmap, vmin=0, vmax=peak, zorder=0,
        )
        selected = np.asarray(samples[i, 0])
        ax.scatter(points[:, 0], points[:, 1], s=17, color=point_color,
                   edgecolors=outline_color, linewidths=0.45,
                   label="Reference (100 points)", zorder=2)
        ax.scatter(selected[:, 0], selected[:, 1], s=52, color=selected_color,
                   edgecolors=outline_color, linewidths=0.9,
                   label="Sample (10 points)", zorder=3)
        ax.scatter(*queries[i], s=100, color=query_color, marker="x",
                   linewidths=2,
                   label="Query", zorder=4)
        ax.set(title=f"Query {i + 1}", xlabel="x", aspect="equal",
               xlim=(-1.15, 1.15), ylim=(-1.15, 1.15),
               xticks=[-1, 0, 1], yticks=[-1, 0, 1])
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("#617080" if dark else "#b7c2ca")
            spine.set_linewidth(0.6)
        ax.tick_params(length=3, width=0.6, labelsize=9)
        ax.set_title(f"Query {i + 1}", fontsize=11, pad=10)

    axes[0].set_ylabel("y")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                        bbox_to_anchor=(0.47, 0.01))
    # The legend sits on the page background rather than the heatmap peak.
    legend.legend_handles[-1].set_color("#ffffff" if dark else "#16384b")
    fig.suptitle(title, y=0.98, fontsize=13)
    fig.subplots_adjust(left=0.06, right=0.87, bottom=0.22, top=0.86,
                        wspace=0.12)
    colorbar_ax = fig.add_axes((0.9, 0.22, 0.015, 0.64))
    colorbar = fig.colorbar(heatmap, cax=colorbar_ax, label="Distance weight",
                 ticks=[0, peak] if uniform else None)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=9)
    return fig
# -- docs snippet end relative_subsamples --


if __name__ == "__main__":
    for theme, style, background in [
        ("light", "default", "white"),
        ("dark", "dark_background", "#1a1a2e"),
    ]:
        with plt.style.context(style), plt.rc_context({
            "figure.facecolor": background,
            "axes.facecolor": background,
        }):
            for uniform, suffix in [(False, ""), (True, "_uniform")]:
                fig = plot_relative_subsamples(uniform=uniform, dark=theme == "dark")
                output = Path(__file__).with_name(
                    f"subsampling_relative{suffix}_{theme}.png"
                )
                fig.savefig(output, dpi=180, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved {output}")
