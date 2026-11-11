import pytest

from stablebear.timeseries import DateTime, Domain, TimeDelta


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


# The domain [1 s, 3 s), with a closed start and an open end.
@pytest.mark.parametrize("t, inside", [
    pytest.param(TimeDelta(2.0, "s"), True, id="inside"),
    pytest.param(TimeDelta(1.0, "s"), True, id="closed-start"),
    pytest.param(TimeDelta(3.0, "s"), False, id="open-end"),
    pytest.param(TimeDelta(0.5, "s"), False, id="before"),
    pytest.param(TimeDelta(4.0, "s"), False, id="after"),
])
def test_time_in_a_domain(t, inside):
    domain = Domain(TimeDelta(1.0, "s"), TimeDelta(3.0, "s"), start_closed=True, end_closed=False)

    assert (t in domain) is inside


def test_time_in_a_domain_with_an_open_start_and_a_closed_end():
    domain = Domain(TimeDelta(1.0, "s"), TimeDelta(3.0, "s"), start_closed=False, end_closed=True)

    assert TimeDelta(1.0, "s") not in domain
    assert TimeDelta(3.0, "s") in domain


def test_number_in_a_domain_of_numbers_is_compared_with_its_ends():
    # 1.0 equals start_closed, True, but is outside the domain.
    domain = Domain(2.0, 4.0, start_closed=True, end_closed=True)

    assert 1.0 not in domain
    assert 3.0 in domain


@pytest.mark.parametrize("start, end", [
    pytest.param(TimeDelta(1.0, "s"), 2.0, id="duration-and-number"),
    pytest.param(DateTime("2023-01-01"), TimeDelta(1.0, "s"), id="date-and-duration"),
])
def test_domain_with_ends_of_two_kinds_raises(start, end):
    with pytest.raises(TypeError, match="both durations"):
        Domain(start, end, start_closed=True, end_closed=True)
