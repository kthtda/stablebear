"""Generate the figures for the time series plotting docs."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from gen_timeseries_snap_fig import DARK, LIGHT, _save_themed
from stablebear.plotting import plot_timeseries
from stablebear.timeseries import TimeSeries, Window

HERE = Path(__file__).parent


# -- included in docs via literalinclude :pyobject: --
def plot_timeseries_steps():
    series = TimeSeries([1.0, 3.0, 2.0, 4.0], times=[0.0, 1.0, 1.5, 3.5])

    fig, ax = plt.subplots(figsize=(6, 2.5))
    plot_timeseries(series, ax=ax)
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


if __name__ == "__main__":
    for name, plot_func in [("timeseries_plot_steps", plot_timeseries_steps),
                            ("timeseries_plot_window_in_place", plot_timeseries_window_in_place)]:
        _save_themed(plot_func, *LIGHT, HERE / f"{name}_light.png")
        _save_themed(plot_func, *DARK, HERE / f"{name}_dark.png")
