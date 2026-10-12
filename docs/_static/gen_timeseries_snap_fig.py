"""Generate the time series evaluation and snapping figures for the docs."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from stablebear.timeseries import TimeSeries

HERE = Path(__file__).parent


def _plot_steps(ax, series, color):
    times = np.asarray(series.times.values)
    values = np.asarray(series.values)
    ax.step(times, values, where="post", color=color, linewidth=2)
    ax.plot(times, values, "o", color=color, markersize=6, zorder=5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_evaluation():
    series = TimeSeries([1.0, 3.0, 2.0, 4.0], times=[0.0, 1.0, 1.5, 3.5])
    queries = [0.5, 1.5, 2.75]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    _plot_steps(ax, series, plt.rcParams["lines.color"])
    for t in queries:
        ax.axvline(t, color="gray", linestyle="--", linewidth=1)
        ax.plot(t, series(t), "s", color="tab:orange", markersize=7,
                zorder=6)
    ax.set_ylim(0, 5)
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_rounding():
    exact = TimeSeries([1.0, 2.0, 3.0, 4.0], times=[0.0, 0.1, 0.2, 0.3],
                       snap_tolerance=0.0)
    query = 0.3 - 0.1
    color = plt.rcParams["lines.color"]

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(7, 2.6), gridspec_kw={"width_ratios": [1, 1.3]})

    # Whole series: the query looks like it sits exactly on 0.2.
    grid = np.linspace(0.0, 0.3, 1001)
    ax_full.plot(grid, [exact(t) for t in grid], color=color, linewidth=2)
    ax_full.plot([0.0, 0.1, 0.2, 0.3], [1.0, 2.0, 3.0, 4.0], "o",
                 color=color, markersize=6, zorder=5)
    ax_full.axvline(query, color="tab:orange", linestyle="--", linewidth=1)
    ax_full.set_title("query 0.3 - 0.1", fontsize=10)
    ax_full.set_xlabel("time (s)")
    ax_full.set_ylabel("series(t)")
    ax_full.set_ylim(0, 5)

    # Zoom around 0.2, in units of 1e-17 s. The small ticks are the
    # float64 values; 0.3 - 0.1 is the one just below 0.2.
    scale = 1e-17
    floats = [0.2]
    for _ in range(3):
        floats.insert(0, np.nextafter(floats[0], 0.0))
    floats.append(np.nextafter(0.2, 1.0))
    offsets = [(f - 0.2) / scale for f in floats]
    left, right = offsets[0] - 1, offsets[-1] + 1
    ax_zoom.plot([left, 0.0], [2.0, 2.0], color=color, linewidth=2)
    ax_zoom.plot([0.0, 0.0], [2.0, 3.0], color=color, linewidth=2)
    ax_zoom.plot([0.0, right], [3.0, 3.0], color=color, linewidth=2)
    ax_zoom.plot(0.0, 3.0, "o", color=color, markersize=6, zorder=5)
    ax_zoom.plot(offsets, [0.0] * len(offsets), "|", color="gray",
                 markersize=10, clip_on=False)
    q = (query - 0.2) / scale
    ax_zoom.axvline(q, color="tab:orange", linestyle="--", linewidth=1)
    ax_zoom.plot(q, exact(query), "s", color="tab:orange", markersize=7,
                 zorder=6)
    ax_zoom.annotate("0.3 - 0.1", (q, exact(query)), xytext=(q - 2.2, 2.6),
                     fontsize=8, arrowprops={"arrowstyle": "->",
                                             "color": "gray"})
    ax_zoom.annotate("observation\nat 0.2", (0.0, 3.0),
                     xytext=(0.6, 3.4), fontsize=8)
    ax_zoom.set_xlim(left, right)
    ax_zoom.set_ylim(0, 5)
    ax_zoom.set_title("zoomed in at 0.2", fontsize=10)
    ax_zoom.set_xlabel("time - 0.2 (units of 1e-17 s)")
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_snap_distances():
    series = TimeSeries([1.0, 2.0, 3.0, 4.0], times=[0.0, 0.1, 0.2, 0.3])
    by_spacing = series.snap_tolerance * (0.2 - 0.1)
    by_eps = series.snap_eps * np.finfo(np.float64).eps * 0.2
    cutoff = min(by_spacing, by_eps)
    queries = [("0.3 - 0.1", 0.3 - 0.1), ("0.2 - 1e-9", 0.2 - 1e-9),
               ("0.19", 0.19)]
    color = plt.rcParams["lines.color"]

    fig, ax = plt.subplots(figsize=(7, 2.0))
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1e-18)  # time runs left to right, toward 0.2
    ax.set_ylim(0, 1)
    ax.axvspan(1e-18, cutoff, facecolor="tab:orange", alpha=0.3,
               hatch="///", edgecolor="tab:orange", linewidth=0)
    ax.axvline(by_eps, color="tab:orange", linewidth=1.5)
    ax.axvline(by_spacing, color="gray", linestyle="--", linewidth=1)
    ax.text(by_eps, 0.05, "1024 \u00d7 eps \u00d7 0.2 ", ha="right",
            fontsize=8)
    ax.text(by_spacing, 0.05, "1e-6 \u00d7 0.1 ", ha="right", fontsize=8)
    ax.text(1e-16, 0.82, "returns 3.0", ha="center", fontsize=8)
    ax.text(1e-4, 0.82, "returns 2.0", ha="center", fontsize=8)
    for label, t in queries:
        d = 0.2 - t
        ax.plot(d, 0.35, "s", color=color, markersize=7)
        ax.annotate(f"{label}", (d, 0.35), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("distance of the query below 0.2 (s), log scale")
    fig.tight_layout()
    return fig


# -- Generate light and dark variants --
def _save_themed(plot_func, style, bg_color, fg_color, line_color, outfile):
    with plt.style.context(style), \
         plt.rc_context({"axes.facecolor": bg_color,
                         "axes.edgecolor": fg_color,
                         "axes.labelcolor": fg_color,
                         "axes.titlecolor": fg_color,
                         "figure.facecolor": bg_color,
                         "text.color": fg_color,
                         "xtick.color": fg_color,
                         "ytick.color": fg_color,
                         "lines.color": line_color}):
        fig = plot_func()
        fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        print(f"saved {outfile}")


LIGHT = ("default", "white", "black", "steelblue")
DARK = ("dark_background", "#14181e", "#e0e0e0", "#5dade2")

if __name__ == "__main__":
    for name, plot_func in [("timeseries_evaluation", plot_timeseries_evaluation),
                            ("timeseries_rounding", plot_timeseries_rounding),
                            ("timeseries_snap_distances", plot_timeseries_snap_distances)]:
        _save_themed(plot_func, *LIGHT, HERE / f"{name}_light.png")
        _save_themed(plot_func, *DARK, HERE / f"{name}_dark.png")
