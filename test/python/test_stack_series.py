import numpy as np
import pytest

import stablebear as sb
from stablebear.timeseries import DateTime, TimeDelta, TimeDeltaTensor, TimeSeries, cut_windows


def _two_scalar_channels():
    a = TimeSeries([0.0, 10.0, 25.0], times=TimeDeltaTensor([0.0, 1.0, 2.5], "s"))
    b = TimeSeries([-5.0, -15.0, -30.0], times=TimeDeltaTensor([0.5, 1.5, 3.0], "s"))
    return TimeSeries.from_channels([a, b], unit="s")


def test_stacked_scalar_channels_give_vectors_at_every_time_in_the_domain():
    series = _two_scalar_channels()
    # Each value holds until the next. The domain is [0.5, 2.5], so a's time
    # 0 and b's time 3 fall outside it, and a point (*) comes at its start
    # and at every observation inside it.
    #
    # time     0    0.5  1    1.5  2    2.5  3
    # a        0         10             25
    # b             -5        -15            -30
    # domain        [-------------------]
    # stacked       *    *    *         *

    stacked = sb.stack(series, axis=0)

    assert stacked.n_channels == 1
    assert stacked.domain == series.domain
    assert np.array_equal(stacked.times.count, [0.5, 1.0, 1.5, 2.5])
    assert stacked.values.array_equal(sb.FloatTensor(np.array([
        [0.0, -5.0],    # 0.5: a(0), b(0.5)
        [10.0, -5.0],   # 1.0: a(1), b(0.5)
        [10.0, -15.0],  # 1.5: a(1), b(1.5)
        [25.0, -15.0],  # 2.5: a(2.5), b(1.5)
    ])))


def _two_vector_channels():
    a = TimeSeries([[10.0, 20.0], [20.0, 40.0]], times=TimeDeltaTensor([1.0, 2.0], "s"))
    b = TimeSeries([[-10.0, -20.0], [-20.0, -40.0]], times=TimeDeltaTensor([1.0, 2.0], "s"))
    return TimeSeries.from_channels([a, b], unit="s")


def _pcf(value):
    return sb.Pcf(np.array([[0.0, value], [1.0, 0.0]]))


def _two_pcf_vector_channels():
    # Pairs of PCFs of values 10 * t and 20 * t for a, and their negatives
    # for b, at time t.
    a = TimeSeries(sb.tensor([[_pcf(10.0), _pcf(20.0)],
                              [_pcf(20.0), _pcf(40.0)]]), times=TimeDeltaTensor([1.0, 2.0], "s"))
    b = TimeSeries(sb.tensor([[_pcf(-10.0), _pcf(-20.0)],
                              [_pcf(-20.0), _pcf(-40.0)]]), times=TimeDeltaTensor([1.0, 2.0], "s"))
    return TimeSeries.from_channels([a, b], unit="s")


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
@pytest.mark.parametrize("t", [TimeDelta(1.0, "s"), TimeDelta(2.0, "s")])
@pytest.mark.parametrize("make_series", [
    pytest.param(_two_vector_channels, id="floats"),
    pytest.param(_two_pcf_vector_channels, id="pcfs"),
])
def test_stacked_channels_stack_as_tensors_stack(make_series, t, axis):
    series = make_series()

    stacked = sb.stack(series, axis=axis)

    assert stacked.dtype is series.dtype
    assert stacked(t).array_equal(sb.stack([series[0](t), series[1](t)], axis=axis))


def test_stacked_window_keeps_its_open_start_and_origin():
    a = TimeSeries([0.0, 10.0, 20.0, 40.0], times=TimeDeltaTensor([0.0, 1.0, 2.0, 4.0], "s"))
    b = TimeSeries([0.0, -10.0, -20.0, -40.0], times=TimeDeltaTensor([0.0, 1.0, 2.0, 4.0], "s"))
    source = TimeSeries.from_channels([a, b], unit="s")
    # The window (1, 3] of the source, in its own time from 1. It holds the
    # value at 1 from its open start, so its first point is at 0.
    #
    # source   0    1    2    3    4
    # a        0    10   20        40
    # b        0    -10  -20       -40
    # window        (---------]
    # own time      0    1    2
    # stacked       *    *
    window = cut_windows(source, size=TimeDelta(2.0, "s"), step=TimeDelta(1.0, "s"), closed="right",
                         anchor="start")[1]

    stacked = sb.stack(window, axis=0)

    assert stacked.domain == window.domain
    assert stacked.origin == TimeDelta(1.0, "s")
    assert np.array_equal(stacked.times.count, [0.0, 1.0])
    assert np.array_equal(np.asarray(stacked(TimeDelta(0.5, "s"))), [10.0, -10.0])


def test_stacked_series_keeps_its_unit_and_epoch():
    a = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([1.0, 2.0], "h"), epoch="2023-01-01")
    b = TimeSeries([-10.0, -20.0], times=TimeDeltaTensor([1.0, 2.0], "h"), epoch="2023-01-01")

    stacked = sb.stack(TimeSeries.from_channels([a, b], unit="h"), axis=0)

    assert stacked.unit == "h"
    assert stacked.epoch == DateTime("2023-01-01")


@pytest.mark.parametrize("join", [sb.stack, sb.concatenate])
def test_list_of_series_raises_naming_from_channels_and_tensor(join):
    a = TimeSeries([[10.0, 20.0], [20.0, 40.0]], times=TimeDeltaTensor([1.0, 2.0], "s"))
    b = TimeSeries([[-10.0, -20.0], [-20.0, -40.0]], times=TimeDeltaTensor([1.5, 2.0], "s"))

    with pytest.raises(TypeError, match=r"from_channels.*sb\.tensor"):
        join([a, b], axis=0)


def _channels_of_different_shapes():
    a = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([1.0, 2.0], "s"))
    b = TimeSeries([[-10.0, -20.0], [-20.0, -40.0]], times=TimeDeltaTensor([1.0, 2.0], "s"))
    return TimeSeries.from_channels([a, b], unit="s")


def _channels_of_different_origins():
    source = TimeSeries([0.0, 10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.0, 3.0], "s"))
    # The window [1, 3), whose origin is 1.
    window = cut_windows(source, size=TimeDelta(2.0, "s"), step=TimeDelta(1.0, "s"), closed="left", partial=False,
                         anchor="start")[1]
    return TimeSeries.from_channels([window, source], unit="s")


@pytest.mark.parametrize("make_series, axis, message", [
    pytest.param(_channels_of_different_shapes, 0, "same shape", id="shapes"),
    pytest.param(_channels_of_different_origins, 0, "different origins", id="origins"),
    pytest.param(_two_scalar_channels, 2, "out of range", id="axis-2"),
    pytest.param(_two_scalar_channels, -3, "out of range", id="axis-minus-3"),
])
def test_stack_of_series_rejects_invalid_input(make_series, axis, message):
    series = make_series()

    with pytest.raises(ValueError, match=message):
        sb.stack(series, axis=axis)


def _vector_channels_of_different_lengths():
    a = TimeSeries([[10.0], [20.0]], times=TimeDeltaTensor([1.0, 2.0], "s"))
    b = TimeSeries([[-10.0, -20.0], [-20.0, -40.0]], times=TimeDeltaTensor([1.0, 2.0], "s"))
    return TimeSeries.from_channels([a, b], unit="s")


@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("t", [TimeDelta(1.0, "s"), TimeDelta(2.0, "s")])
@pytest.mark.parametrize("make_series", [
    pytest.param(_two_vector_channels, id="floats"),
    pytest.param(_two_pcf_vector_channels, id="pcfs"),
    pytest.param(_vector_channels_of_different_lengths, id="lengths"),
])
def test_concatenated_channels_join_as_tensors_concatenate(make_series, t, axis):
    series = make_series()

    concatenated = sb.concatenate(series, axis=axis)

    assert concatenated.dtype is series.dtype
    assert concatenated(t).array_equal(sb.concatenate([series[0](t), series[1](t)], axis=axis))


def _two_matrix_channels_of_different_widths():
    # Columns of 10 * t for a, and of -10 * t and -20 * t for b.
    a = TimeSeries([[[10.0], [10.0]],
                    [[20.0], [20.0]]], times=TimeDeltaTensor([1.0, 2.0], "s"))
    b = TimeSeries([[[-10.0, -20.0], [-10.0, -20.0]],
                    [[-20.0, -40.0], [-20.0, -40.0]]], times=TimeDeltaTensor([1.0, 2.0], "s"))
    return TimeSeries.from_channels([a, b], unit="s")


@pytest.mark.parametrize("axis", [1, -1])
@pytest.mark.parametrize("t", [TimeDelta(1.0, "s"), TimeDelta(2.0, "s")])
def test_concatenated_matrix_channels_join_along_their_columns(t, axis):
    series = _two_matrix_channels_of_different_widths()

    concatenated = sb.concatenate(series, axis=axis)

    assert concatenated(t).array_equal(sb.concatenate([series[0](t), series[1](t)], axis=axis))


def _channels_of_different_ranks():
    a = TimeSeries([[10.0, 20.0], [20.0, 40.0]], times=TimeDeltaTensor([1.0, 2.0], "s"))
    b = TimeSeries([[[-10.0], [-20.0]],
                    [[-20.0], [-40.0]]], times=TimeDeltaTensor([1.0, 2.0], "s"))
    return TimeSeries.from_channels([a, b], unit="s")


def _vector_channels_of_different_origins():
    source = TimeSeries([[0.0], [10.0], [20.0], [30.0]], times=TimeDeltaTensor([0.0, 1.0, 2.0, 3.0], "s"))
    # The window [1, 3), whose origin is 1.
    window = cut_windows(source, size=TimeDelta(2.0, "s"), step=TimeDelta(1.0, "s"), closed="left", partial=False,
                         anchor="start")[1]
    return TimeSeries.from_channels([window, source], unit="s")


@pytest.mark.parametrize("make_series, axis, message", [
    pytest.param(_two_scalar_channels, 0, "sb.stack", id="scalars"),
    pytest.param(_two_matrix_channels_of_different_widths, 0, "differ other than along axis", id="shapes"),
    pytest.param(_channels_of_different_ranks, 0, "differ other than along axis", id="ranks"),
    pytest.param(_vector_channels_of_different_origins, 0, "different origins", id="origins"),
    pytest.param(_two_vector_channels, 1, "out of range", id="axis-1"),
    pytest.param(_two_vector_channels, -2, "out of range", id="axis-minus-2"),
])
def test_concatenate_of_series_rejects_invalid_input(make_series, axis, message):
    series = make_series()

    with pytest.raises(ValueError, match=message):
        sb.concatenate(series, axis=axis)
