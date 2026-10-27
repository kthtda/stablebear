import numpy as np

from plot_helpers import FigureGallery, gallery_fixture, ax_fixture

from stablebear.plotting import plot_timeseries
from stablebear.timeseries import Times, TimeSeries, Window

_gallery = FigureGallery()
_show_gallery = gallery_fixture(_gallery)
ax = ax_fixture(_gallery)


def _series():
    return TimeSeries([10.0, 20.0, 30.0], times=Times([0.0, 1.5, 4.0], unit="s"))


def test_a_series_is_drawn_as_steps_through_its_observations(ax):
    (line,) = plot_timeseries(_series(), ax=ax)

    assert line.get_drawstyle() == "steps-post"
    assert np.array_equal(line.get_xdata(), [0.0, 1.5, 4.0])
    assert np.array_equal(line.get_ydata(), [10.0, 20.0, 30.0])


def test_a_window_holds_its_last_value_to_its_end(ax):
    # The window [2, 5) holds 20, observed at 1.5, then 30 at 4; it ends at
    # 5, after its last observation.
    window = Window(size=3.0, step=2.0, closed="left", partial=False, anchor="start")(
        TimeSeries([10.0, 20.0, 30.0, 40.0], times=[0.0, 1.5, 4.0, 6.0]))[1]

    (line,) = plot_timeseries(window, ax=ax)

    assert np.array_equal(line.get_xdata(), [2.0, 4.0, 5.0])
    assert np.array_equal(line.get_ydata(), [20.0, 30.0, 30.0])


def test_the_time_axis_is_labeled_with_its_unit(ax):
    plot_timeseries(_series(), ax=ax)

    assert ax.xaxis.get_units() == "s"
    assert ax.get_xlabel() == "time (s)"


def test_a_label_set_before_drawing_is_kept(ax):
    ax.set_xlabel("elapsed")

    plot_timeseries(_series(), ax=ax)

    assert ax.get_xlabel() == "elapsed"


def test_no_axes_draws_on_the_current_axes(ax):
    import matplotlib.pyplot as plt
    plt.sca(ax)

    plot_timeseries(_series())

    assert len(ax.get_lines()) == 1


def test_markers_mark_each_observation(ax):
    window = Window(size=3.0, step=2.0, closed="left", partial=False, anchor="start")(
        TimeSeries([10.0, 20.0, 30.0, 40.0], times=[0.0, 1.5, 4.0, 6.0]))[1]

    (line,) = plot_timeseries(window, ax=ax, markers=True)

    # The observations at 2 and 4, not the end of the window at 5.
    assert line.get_marker() == "o"
    assert line.get_markevery() == [0, 1]


def test_keyword_arguments_reach_the_line(ax):
    (line,) = plot_timeseries(_series(), ax=ax, color="red", label="signal")

    assert line.get_color() == "red"
    assert line.get_label() == "signal"


def test_unit_sets_the_unit_of_the_time_axis(ax):
    (line,) = plot_timeseries(_series(), ax=ax, unit="ms")

    assert np.array_equal(line.get_xdata(), [0.0, 1500.0, 4000.0])
    assert ax.get_xlabel() == "time (ms)"


def test_a_series_in_another_unit_is_drawn_in_the_unit_of_the_axis(ax):
    plot_timeseries(_series(), ax=ax)
    milliseconds = TimeSeries([1.0, 2.0], times=Times([500.0, 2500.0], unit="ms"))

    (line,) = plot_timeseries(milliseconds, ax=ax)

    assert np.array_equal(line.get_xdata(), [0.5, 2.5])
    assert ax.get_xlabel() == "time (s)"
