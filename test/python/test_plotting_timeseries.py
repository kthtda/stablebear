import numpy as np
import pytest

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


def test_axes_that_share_x_share_the_time_unit():
    import matplotlib.pyplot as plt
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
    plot_timeseries(_series(), ax=top)

    (line,) = plot_timeseries(TimeSeries([1.0, 2.0], times=Times([500.0, 2500.0], unit="ms")), ax=bottom)

    assert np.array_equal(line.get_xdata(), [0.5, 2.5])
    plt.close(fig)


def test_a_unit_other_than_that_of_the_axis_raises(ax):
    plot_timeseries(_series(), ax=ax)

    with pytest.raises(ValueError, match="'s', not 'ms'"):
        plot_timeseries(_series(), ax=ax, unit="ms")


def test_an_axis_of_dates_raises(ax):
    ax.plot(np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[D]"), [0.0, 1.0])

    with pytest.raises(TypeError):
        plot_timeseries(_series(), ax=ax)


def _windows():
    # Windows of 2 s at 0, 2, and 4 s.
    series = TimeSeries([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return Window(size=2.0, step=2.0, closed="left", partial=False, anchor="start")(series)


def test_a_tensor_of_windows_is_drawn_one_line_each_in_place(ax):
    lines = plot_timeseries(_windows(), ax=ax)

    assert [list(line.get_xdata()) for line in lines] == [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]]
    assert len({line.get_color() for line in lines}) == 3


def test_relative_draws_each_window_from_zero(ax):
    lines = plot_timeseries(_windows(), ax=ax, relative=True)

    assert [list(line.get_xdata()) for line in lines] == [[0.0, 1.0, 2.0]] * 3


def test_an_empty_tensor_draws_nothing(ax):
    assert plot_timeseries(_windows()[0:0], ax=ax) == []
    assert ax.get_lines() == []


def test_a_tensor_with_two_dimensions_raises(ax):
    series = TimeSeries([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], times=np.arange(9.0))
    windows = Window(size=2.0, step=2.0, closed="left", partial=False, anchor="start")(series).reshape((2, 2))

    with pytest.raises(ValueError, match="1-D"):
        plot_timeseries(windows, ax=ax)


def test_vector_observations_raise(ax):
    with pytest.raises(ValueError, match="scalar"):
        plot_timeseries(TimeSeries([[1.0, 2.0], [3.0, 4.0]], times=[0.0, 1.0]), ax=ax)


def _channels():
    a = TimeSeries([1.0, 2.0], times=Times([0.0, 2.0], unit="s"))
    b = TimeSeries([10.0, 20.0, 30.0], times=Times([0.0, 1000.0, 2000.0], unit="ms"))
    c = TimeSeries([-1.0, -2.0], times=Times([0.0, 2.0], unit="s"))
    return TimeSeries.from_channels([a, b, c], unit="s")


def test_several_channels_are_drawn_on_new_rows_sharing_time():
    import matplotlib.pyplot as plt
    lines = plot_timeseries(_channels())

    rows = [line.axes for line in lines]
    assert len(set(rows)) == 3
    assert [list(line.get_ydata()) for line in lines] == [[1.0, 2.0], [10.0, 20.0, 30.0], [-1.0, -2.0]]
    assert rows[0].get_shared_x_axes().joined(rows[0], rows[2])
    plt.close(rows[0].figure)


def test_channels_in_other_units_are_drawn_in_the_unit_of_the_series():
    import matplotlib.pyplot as plt
    lines = plot_timeseries(_channels())

    assert list(lines[1].get_xdata()) == [0.0, 1.0, 2.0]
    plt.close(lines[0].axes.figure)


def test_only_the_last_row_is_labeled():
    import matplotlib.pyplot as plt
    lines = plot_timeseries(_channels())

    assert [line.axes.get_xlabel() for line in lines] == ["", "", "time (s)"]
    plt.close(lines[0].axes.figure)


def test_several_channels_are_drawn_on_given_axes_in_order():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3)

    lines = plot_timeseries(_channels(), ax=axes)

    assert [line.axes for line in lines] == list(axes)
    plt.close(fig)


@pytest.mark.parametrize("n_axes", [1, 2])
def test_several_channels_need_one_axes_each(n_axes):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n_axes, squeeze=False)
    ax = axes[0, 0] if n_axes == 1 else list(axes[0])

    with pytest.raises(ValueError, match=r"series\[i\]"):
        plot_timeseries(_channels(), ax=ax)
    plt.close(fig)


def _dated():
    return TimeSeries([10.0, 20.0, 30.0], times=Times([0.0, 1.5, 4.0], unit="h", epoch="2023-01-01"))


def test_a_series_with_an_epoch_is_drawn_at_its_dates(ax):
    (line,) = plot_timeseries(_dated(), ax=ax)

    expected = np.array(["2023-01-01T00:00", "2023-01-01T01:30", "2023-01-01T04:00"], dtype="datetime64[us]")
    assert np.array_equal(line.get_xdata(), expected)
    assert ax.get_xlabel() == ""


def test_windows_with_an_epoch_are_drawn_at_their_dates(ax):
    windows = Window(size=2.0, step=2.0, closed="left", partial=False, anchor="start")(_dated())

    lines = plot_timeseries(windows, ax=ax)

    # The window at 2 h holds 20 from 02:00 and ends at 04:00.
    expected = np.array(["2023-01-01T02:00", "2023-01-01T04:00"], dtype="datetime64[us]")
    assert np.array_equal(lines[1].get_xdata(), expected)


def test_dates_are_rounded_to_microseconds(ax):
    series = TimeSeries([1.0, 2.0], times=Times([0.0, 1500.4], unit="ns", epoch="2023-01-01"))

    (line,) = plot_timeseries(series, ax=ax)

    expected = np.array(["2023-01-01T00:00:00.000000", "2023-01-01T00:00:00.000002"], dtype="datetime64[us]")
    assert np.array_equal(line.get_xdata(), expected)
