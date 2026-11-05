import numpy as np
import pytest

from stablebear.timeseries import Domain, TimeDelta, TimeDeltaTensor, TimeSeries


def _series():
    return TimeSeries([0.0, 10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.0, 3.0], "s"))


@pytest.mark.parametrize("start_closed, end_closed", [(True, True), (True, False), (False, True), (False, False)])
def test_restricted_series_has_the_given_domain(start_closed, end_closed):
    domain = Domain(TimeDelta(1.0, "s"), TimeDelta(2.0, "s"), start_closed=start_closed, end_closed=end_closed)

    restricted = _series().restrict(domain)

    assert restricted.domain == domain


def test_restricted_series_holds_the_value_in_effect_at_its_start():
    domain = Domain(TimeDelta(0.5, "s"), TimeDelta(2.5, "s"), start_closed=True, end_closed=True)

    restricted = _series().restrict(domain)

    assert np.array_equal(restricted.times.count, [0.5, 1.0, 2.0])
    assert np.array_equal(np.asarray(restricted.values), [0.0, 10.0, 20.0])
    assert restricted(TimeDelta(2.5, "s")) == 20.0


def test_restricted_series_keeps_its_unit_and_origin():
    domain = Domain(TimeDelta(1.0, "s"), TimeDelta(2.0, "s"), start_closed=False, end_closed=True)

    restricted = _series().restrict(domain)

    assert restricted.unit == "s"
    assert restricted.origin == TimeDelta(0.0, "s")


def test_each_channel_is_restricted_as_a_series_of_its_own():
    a = TimeSeries([0.0, 10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0, 2.0], "s"))
    b = TimeSeries([5.0, 15.0], times=TimeDeltaTensor([0.5, 1.5], "s"))
    domain = Domain(TimeDelta(1.0, "s"), TimeDelta(1.5, "s"), start_closed=True, end_closed=True)

    restricted = TimeSeries.from_channels([a, b]).restrict(domain)

    assert restricted.array_equal(TimeSeries.from_channels([a.restrict(domain), b.restrict(domain)]))


def test_restricted_series_without_a_unit_takes_numbers():
    series = TimeSeries([0.0, 10.0, 20.0], times=[0.0, 1.0, 2.0])

    domain = Domain(0.5, 1.5, start_closed=True, end_closed=False)

    restricted = series.restrict(domain)

    assert restricted.domain == domain


@pytest.mark.parametrize("domain, error, message", [
    pytest.param(Domain(TimeDelta(-1.0, "s"), TimeDelta(2.0, "s"), True, True), ValueError, "reaches outside",
                 id="before-start"),
    pytest.param(Domain(TimeDelta(1.0, "s"), TimeDelta(4.0, "s"), True, True), ValueError, "reaches outside",
                 id="after-end"),
    pytest.param(Domain(1.0, 2.0, True, True), TypeError, "duration", id="numbers"),
])
def test_restrict_rejects_invalid_domains(domain, error, message):
    with pytest.raises(error, match=message):
        _series().restrict(domain)


def test_restrict_cannot_include_an_end_the_series_leaves_out():
    open_end = _series().restrict(Domain(TimeDelta(0.0, "s"), TimeDelta(2.0, "s"), start_closed=True,
                                         end_closed=False))

    with pytest.raises(ValueError, match="reaches outside"):
        open_end.restrict(Domain(TimeDelta(1.0, "s"), TimeDelta(2.0, "s"), start_closed=True, end_closed=True))


def test_restrict_rejects_a_series_with_an_epoch():
    series = TimeSeries([0.0, 10.0], times=TimeDeltaTensor([0.0, 1.0], "h"), epoch="2023-01-01")

    with pytest.raises(TypeError, match="with_epoch"):
        series.restrict(Domain(TimeDelta(0.0, "h"), TimeDelta(1.0, "h"), start_closed=True, end_closed=True))
