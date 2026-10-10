"""Illustrate mapping points to filter values and then to sampling weights."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# -- docs snippet start filter_function --
import matplotlib.pyplot as plt
import numpy as np
import stablebear as sb


def plot_filter_function(*, dark=False):
    query = np.array([0.0, 0.0])
    points = np.array([
        [0.24, 0.32],   # A: distance 0.4
        [0.64, -0.48],  # B: distance 0.8
        [-0.72, 0.96],  # C: distance 1.2
        [-1.28, -0.96], # D: distance 1.6
    ])
    distances = np.linalg.norm(points - query, axis=1)
    std = 0.9
    distribution = sb.distributions.Gaussian(std=std)
    weights = distribution.weight(distances)

    ink = "#e7edf5" if dark else "#253746"
    muted = "#8996a8" if dark else "#7b8894"
    colors = (["#ffa576", "#73b6f5", "#6bd1b8", "#c8a0ee"] if dark else
              ["#c95223", "#246caa", "#168570", "#8257a6"])
    # Two rows keep text readable at the documentation column's laptop width.
    fig = plt.figure(figsize=(7.2, 5.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.3])
    cloud_ax = fig.add_subplot(grid[0, :])
    line_ax = fig.add_subplot(grid[1, 0])
    weight_ax = fig.add_subplot(grid[1, 1])
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.22, top=0.84,
                        wspace=0.5, hspace=0.4)

    # Keep labels about eight points from their markers, including diagonals.
    # Each point keeps its letter and color throughout the construction.
    for point, label, color in zip(points, "ABCD", colors):
        cloud_ax.plot([query[0], point[0]], [query[1], point[1]],
                      color=color, linewidth=1.1, alpha=0.65)
        cloud_ax.scatter(*point, s=45, color=color, zorder=3)
        cloud_ax.annotate(label, point,
                          xytext=(-8, 0) if label == "D" else (6, 6),
                          textcoords="offset points",
                          ha="right" if label == "D" else "left",
                          va="center" if label == "D" else "bottom",
                          color=color, fontsize=11)
    cloud_ax.scatter(*query, marker="x", s=80, linewidths=2, color=ink, zorder=4)
    cloud_ax.annotate("q", query, xytext=(0, -8), textcoords="offset points",
                      ha="center", va="top", fontsize=11, color=ink)
    cloud_ax.set(xlim=(-1.55, 1.05), ylim=(-1.25, 1.3), aspect="equal")
    cloud_ax.set_axis_off()
    cloud_ax.set_title("1. Measure distance from q", fontsize=11, pad=12, color=ink)

    line_ax.set(xlim=(-0.05, 1.85), ylim=(-0.55, 0.55), yticks=[],
                xticks=[0, 0.4, 0.8, 1.2, 1.6], xlabel=r"Filter value $d = f_q(x)$")
    line_ax.spines[["left", "right", "top"]].set_visible(False)
    line_ax.spines["bottom"].set_position(("data", 0))
    line_ax.spines["bottom"].set_color(muted)
    line_ax.xaxis.set_label_coords(0.5, 0.15)
    line_ax.set_title("2. Place on the filter line", fontsize=11, pad=12, color=ink)
    for distance, label, color in zip(distances, "ABCD", colors):
        line_ax.scatter(distance, 0, s=45, color=color, zorder=3)
        line_ax.annotate(label, (distance, 0), xytext=(0, 8),
                         textcoords="offset points", ha="center", va="bottom",
                         color=color, fontsize=11)

    d = np.linspace(0, 1.85, 400)
    weight_ax.plot(d, distribution.weight(d), color=ink, linewidth=1.8)
    for distance, weight, label, color in zip(distances, weights, "ABCD", colors):
        weight_ax.scatter(distance, 0, s=30, facecolors="none", edgecolors=color, zorder=3)
        weight_ax.annotate("", xy=(distance, weight), xytext=(distance, 0.02),
                           arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
        weight_ax.scatter(distance, weight, s=45, color=color, zorder=4)
        weight_ax.annotate(label, (distance, weight), xytext=(6, 6),
                           textcoords="offset points", ha="left", va="bottom",
                           color=color, fontsize=11)
    weight_ax.set(xlim=(-0.05, 1.85), ylim=(-0.03, 0.5),
                  xticks=[0, 0.4, 0.8, 1.2, 1.6], yticks=[0, 0.25, 0.5],
                  xlabel=r"Filter value $d$", ylabel=r"Weight $W(d)$")
    weight_ax.spines[["top", "right"]].set_visible(False)
    weight_ax.spines["bottom"].set_position(("data", 0))
    for spine in weight_ax.spines.values():
        spine.set_color(muted)
        spine.set_linewidth(0.6)
    weight_ax.set_title("3. Weight by distribution", fontsize=11, pad=12, color=ink)

    for ax in (line_ax, weight_ax):
        ax.tick_params(length=3, width=0.6, labelsize=9, colors=ink)
        ax.grid(False)
    return fig
# -- docs snippet end filter_function --


if __name__ == "__main__":
    for theme, style, background in [
        ("light", "default", "white"),
        ("dark", "dark_background", "#1a1a2e"),
    ]:
        with plt.style.context(style), plt.rc_context({
            "figure.facecolor": background,
            "axes.facecolor": background,
        }):
            fig = plot_filter_function(dark=theme == "dark")
            output = Path(__file__).with_name(f"subsampling_filter_function_{theme}.png")
            fig.savefig(output, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {output}")
