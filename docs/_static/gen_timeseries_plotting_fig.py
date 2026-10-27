"""Generate the figures for the time series plotting docs."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from gen_timeseries_snap_fig import DARK, LIGHT, _save_themed
from stablebear.plotting import plot_timeseries
from stablebear.timeseries import Times, TimeSeries, Window

HERE = Path(__file__).parent


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_steps():
    series = TimeSeries([1.0, 3.0, 2.0, 4.0], times=[0.0, 1.0, 1.5, 3.5])

    fig, ax = plt.subplots(figsize=(6, 2.5))
    plot_timeseries(series, ax=ax, markers=True)
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_window_in_place():
    series = TimeSeries([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
                        times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    window = Window(size=3.0, step=3.0, anchor="start")(series)[1]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    plot_timeseries(series, ax=ax, color="gray", linewidth=1)
    plot_timeseries(window, ax=ax, linewidth=3)
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_units():
    seconds = TimeSeries([1.0, 3.0, 2.0], times=[0.0, 1.0, 2.5])
    milliseconds = TimeSeries([0.5, 1.5, 2.5],
                              times=Times([500.0, 1500.0, 3000.0], unit="ms"))

    fig, ax = plt.subplots(figsize=(6, 2.5))
    plot_timeseries(seconds, ax=ax, markers=True, label="in s")
    plot_timeseries(milliseconds, ax=ax, markers=True, label="in ms")
    ax.legend()
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_sharex():
    seconds = TimeSeries([1.0, 3.0, 2.0], times=[0.0, 1.0, 2.5])
    milliseconds = TimeSeries([0.5, 1.5, 2.5],
                              times=Times([500.0, 1500.0, 3000.0], unit="ms"))

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6, 3.5), sharex=True)
    plot_timeseries(seconds, ax=top, markers=True)
    plot_timeseries(milliseconds, ax=bottom, markers=True, color="tab:orange")
    top.set_xlabel("")
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_windows_relative():
    series = TimeSeries([0.5, 2.0, 1.0, 3.5, 2.5, 0.0, 3.0, 1.5],
                        times=[0.0, 0.7, 1.9, 2.4, 3.6, 4.9, 5.5, 7.2])
    windows = Window(size=2.4, anchor="start")(series)

    fig, (ax_in_place, ax_relative) = plt.subplots(1, 2, figsize=(8, 2.5))
    plot_timeseries(windows, ax=ax_in_place, markers=True)
    plot_timeseries(windows, ax=ax_relative, relative=True, markers=True)
    ax_in_place.set_title("in place", fontsize=10)
    ax_relative.set_title("relative=True", fontsize=10)
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_some_windows():
    series = TimeSeries([0.5, 2.0, 1.0, 3.5, 2.5, 0.0, 3.0, 1.5],
                        times=[0.0, 0.7, 1.9, 2.4, 3.6, 4.9, 5.5, 7.2])
    windows = Window(size=2.4, anchor="start")(series)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    plot_timeseries(series, ax=ax, color="gray", linewidth=1)
    plot_timeseries(windows[[0, 2]], ax=ax, linewidth=3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for name, plot_func in [("timeseries_plot_steps", plot_timeseries_steps),
                            ("timeseries_plot_window_in_place", plot_timeseries_window_in_place),
                            ("timeseries_plot_units", plot_timeseries_units),
                            ("timeseries_plot_sharex", plot_timeseries_sharex),
                            ("timeseries_plot_windows_relative", plot_timeseries_windows_relative),
                            ("timeseries_plot_some_windows", plot_timeseries_some_windows)]:
        _save_themed(plot_func, *LIGHT, HERE / f"{name}_light.png")
        _save_themed(plot_func, *DARK, HERE / f"{name}_dark.png")
