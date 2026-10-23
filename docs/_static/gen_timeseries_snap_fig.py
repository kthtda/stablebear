"""Generate the time series figures for the docs."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from stablebear.timeseries import TimeDelayEmbedding, Times, TimeSeries, Window

HERE = Path(__file__).parent


def _plot_steps(ax, series, color):
    times = np.asarray(series.times.to("s"))
    values = np.asarray(series.values)
    # Each step [t_i, t_{i+1}) is closed on the left and open on the right.
    for t0, t1, v in zip(times[:-1], times[1:], values[:-1]):
        ax.plot([t0, t1], [v, v], color=color, linewidth=2)
        ax.plot(t1, v, "o", color=color, markersize=6, zorder=5,
                markerfacecolor=plt.rcParams["axes.facecolor"])
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
def plot_timeseries_channels():
    a = TimeSeries([1.0, 2.0, 3.0, 4.0], times=[0.0, 0.1, 0.2, 0.3])
    b = TimeSeries([10.0, 20.0, 30.0],
                   times=Times([50.0, 90.0, 130.0], unit="ms"))
    series = TimeSeries.from_channels([a, b], unit="s")
    query = 0.1
    values = series(query)

    fig, axes = plt.subplots(2, 1, figsize=(6, 3.5), sharex=True)
    for i, (ax, channel, top) in enumerate(zip(axes, (a, b), (5, 40))):
        ax.axvspan(0.05, 0.13, color="gray", alpha=0.2, linewidth=0)
        _plot_steps(ax, channel, plt.rcParams["lines.color"])
        ax.axvline(query, color="gray", linestyle="--", linewidth=1)
        ax.plot(query, np.asarray(values[i]), "s", color="tab:orange",
                markersize=7, zorder=6)
        ax.set_ylim(0, top)
        ax.set_ylabel("ab"[i])
    axes[0].set_xlabel("")
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_vector_channel():
    a = TimeSeries([1.0, 2.0, 3.0, 4.0], times=[0.0, 0.1, 0.2, 0.3])
    v = TimeSeries([[1.0, 0.0], [0.0, 1.0], [-1.0, 2.0]],
                   times=[0.05, 0.1, 0.15])
    series = TimeSeries.from_channels([a, v])
    query = 0.125
    values = series(query)
    color = plt.rcParams["lines.color"]

    def column(x):
        return rf"$\binom{{{x[0]:g}}}{{{x[1]:g}}}$"

    fig, (ax_a, ax_v) = plt.subplots(2, 1, figsize=(6, 3.5), sharex=True)
    for ax in (ax_a, ax_v):
        ax.axvspan(0.05, 0.15, color="gray", alpha=0.2, linewidth=0)
    ax_a.axvline(query, color="gray", linestyle="--", linewidth=1)

    _plot_steps(ax_a, a, color)
    ax_a.plot(query, float(values[0]), "s", color="tab:orange", markersize=7,
              zorder=6)
    ax_a.set_ylim(0, 5)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("a")

    # Channel v: the time axis as a line, with each vector above its time
    # and the queried vector below.
    ax_v.axhline(0, color=color, linewidth=1)
    for t, x in zip(np.asarray(v.times), np.asarray(v.values)):
        ax_v.plot(t, 0, "o", color=color, markersize=6, zorder=5)
        ax_v.text(t, 0.25, column(x), ha="center", va="bottom", fontsize=14)
    ax_v.plot(query, 0, "s", color="tab:orange", markersize=7, zorder=6)
    ax_v.text(query, -0.25, column(np.asarray(values[1])), ha="center",
              va="top", fontsize=14, color="tab:orange")
    ax_v.set_ylim(-1.3, 1.3)
    ax_v.set_yticks([])
    ax_v.set_ylabel("v")
    ax_v.set_xlabel("time (s)")
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


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_embedding():
    t = np.linspace(0.0, 6.0, 49)
    signal = TimeSeries(np.sin(np.pi * t), times=t)
    embedded = TimeDelayEmbedding(dimension=2, delay=0.5)(signal)
    query = 1.0
    color = plt.rcParams["lines.color"]

    fig, (ax_signal, ax_points) = plt.subplots(
        1, 2, figsize=(7, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})

    ax_signal.step(t, np.asarray(signal.values), where="post", color=color,
                   linewidth=1.5)
    for s in (query - 0.5, query):
        ax_signal.axvline(s, color="gray", linestyle="--", linewidth=1)
        ax_signal.plot(s, signal(s), "s", color="tab:orange", markersize=7,
                       zorder=6)
    ax_signal.set_xlabel("time (s)")
    ax_signal.set_ylabel("x(t)")

    points = np.array([np.asarray(embedded(s)) for s in t[t >= 0.5]])
    ax_points.plot(points[:, 0], points[:, 1], "o", color=color,
                   markersize=4)
    ax_points.plot(*np.asarray(embedded(query)), "s", color="tab:orange",
                   markersize=7, zorder=6)
    ax_points.set_aspect("equal")
    ax_points.set_xlabel("x(t - 0.5)")
    ax_points.set_ylabel("x(t)")
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_windows():
    signal = TimeSeries([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
                        times=np.arange(9.0))
    windows, bounds = Window(size=4.0, step=2.0, anchor="start")(
        signal, return_bounds=True)
    bounds = np.asarray(bounds)
    color = plt.rcParams["lines.color"]

    fig = plt.figure(figsize=(7, 4.4))
    grid = fig.add_gridspec(2, len(bounds), height_ratios=[1.4, 1])
    ax_signal = fig.add_subplot(grid[0, :])
    ax_signal.step(np.asarray(signal.times.to("s")), np.asarray(signal.values),
                   where="post", color=color, linewidth=1.5)
    # One bar per window above the signal, staggered so they do not overlap.
    # Each window is [a, b): closed at the start, open at the end.
    for i, (a, b) in enumerate(bounds):
        y = 3.6 + 0.7 * i
        ax_signal.plot([a, b], [y, y], color="tab:orange", linewidth=2)
        ax_signal.plot(a, y, "o", color="tab:orange", markersize=6, zorder=5)
        ax_signal.plot(b, y, "o", color="tab:orange", markersize=6,
                       zorder=5, markerfacecolor=plt.rcParams["axes.facecolor"])
        ax_signal.text((a + b) / 2, y + 0.08, f"window {i}: [{a:g}, {b:g})",
                       ha="center", va="bottom", fontsize=8)
    ax_signal.set_xlim(-0.3, 8.3)
    ax_signal.set_ylim(-0.3, 5.6)
    ax_signal.set_xticks(range(9))
    ax_signal.set_yticks([0, 1, 2, 3])
    ax_signal.set_xlabel("time (s)")
    ax_signal.set_ylabel("signal")

    # Each window is evaluated on its domain [0, 4).
    times = np.linspace(0.0, 4.0, 400, endpoint=False)
    for i in range(len(bounds)):
        ax = fig.add_subplot(grid[1, i])
        ax.plot(times, np.asarray(windows[i](times)), color=color,
                linewidth=1.5, drawstyle="steps-post")
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(-0.3, 3.3)
        ax.set_xticks(range(5))
        ax.set_yticks([0, 1, 2, 3] if i == 0 else [])
        ax.set_title(f"window {i}", fontsize=9)
    fig.tight_layout()
    return fig


def _plot_window_rows(rows):
    """Plot ``signal`` above one row of window bars per (label, Window)."""
    signal = TimeSeries([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
                        times=np.arange(9.0))
    color = plt.rcParams["lines.color"]
    hollow = plt.rcParams["axes.facecolor"]
    t0, t1 = 0.0, 8.0

    fig, (ax_signal, ax_rows) = plt.subplots(
        2, 1, figsize=(7, 1.9 + 0.7 * len(rows)), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 0.6 * len(rows)]})
    ax_signal.step(np.asarray(signal.times.to("s")), np.asarray(signal.values),
                   where="post", color=color, linewidth=1.5)
    ax_signal.set_ylim(-0.3, 3.4)
    ax_signal.set_yticks([0, 1, 2, 3])
    ax_signal.set_ylabel("signal")

    # Closed ends are filled, open ends hollow. Partial windows are dashed
    # and closed where they are cut at the end of the series. Consecutive
    # windows alternate in height so that shared ends stay visible.
    for row, (label, window) in enumerate(rows):
        _, bounds = window(signal, return_bounds=True)
        for k, (a, b) in enumerate(np.asarray(bounds)):
            y = len(rows) - 1 - row + (0.15 if k % 2 else -0.15)
            partial = b - a < window.size
            closed_a = window.closed in ("left", "both") or (partial and a == t0)
            closed_b = window.closed in ("right", "both") or (partial and b == t1)
            ax_rows.plot([a, b], [y, y], color="tab:orange", linewidth=2,
                         linestyle="--" if partial else "-")
            for x, closed in ((a, closed_a), (b, closed_b)):
                ax_rows.plot(x, y, "o", color="tab:orange", markersize=6,
                             zorder=5, markerfacecolor=None if closed else hollow)
    ax_rows.set_ylim(-0.5, len(rows) - 0.5)
    ax_rows.set_yticks(range(len(rows)))
    ax_rows.set_yticklabels([label for label, _ in reversed(rows)])
    ax_rows.set_xlim(-0.3, 8.3)
    ax_rows.set_xticks(range(9))
    ax_rows.set_xlabel("time (s)")
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_window_anchors():
    return _plot_window_rows([
        ('anchor="start"', Window(size=3.0, anchor="start")),
        ('anchor="end"', Window(size=3.0)),
    ])


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_window_partial():
    return _plot_window_rows([
        ('anchor="start"', Window(size=3.0, anchor="start", partial=True)),
        ('anchor="end"', Window(size=3.0, partial=True)),
    ])


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
                            ("timeseries_channels", plot_timeseries_channels),
                            ("timeseries_vector_channel", plot_timeseries_vector_channel),
                            ("timeseries_rounding", plot_timeseries_rounding),
                            ("timeseries_snap_distances", plot_timeseries_snap_distances),
                            ("timeseries_embedding", plot_timeseries_embedding),
                            ("timeseries_windows", plot_timeseries_windows),
                            ("timeseries_window_anchors", plot_timeseries_window_anchors),
                            ("timeseries_window_partial", plot_timeseries_window_partial)]:
        _save_themed(plot_func, *LIGHT, HERE / f"{name}_light.png")
        _save_themed(plot_func, *DARK, HERE / f"{name}_dark.png")
