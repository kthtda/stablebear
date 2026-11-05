import pytest

from stablebear.timeseries import Domain, TimeDelta


def test_domain_of_one_time_includes_both_ends():
    domain = Domain(TimeDelta(1.0, "s"), TimeDelta(1.0, "s"), start_closed=True, end_closed=True)

    assert domain.start == domain.end == TimeDelta(1.0, "s")


@pytest.mark.parametrize("start, end, start_closed, end_closed", [
    pytest.param(TimeDelta(2.0, "s"), TimeDelta(1.0, "s"), True, True, id="reversed"),
    pytest.param(TimeDelta(1.0, "s"), TimeDelta(1.0, "s"), True, False, id="one-time-open-end"),
    pytest.param(TimeDelta(1000.0, "ms"), TimeDelta(1.0, "s"), False, True, id="one-time-in-two-units"),
])
def test_domain_holding_no_time_raises(start, end, start_closed, end_closed):
    with pytest.raises(ValueError, match="holds no time"):
        Domain(start, end, start_closed=start_closed, end_closed=end_closed)
