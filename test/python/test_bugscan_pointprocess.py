import numpy as np
import pytest

import stablebear as sb
from stablebear.point_process import poisson


def _gen():
    return sb.random.Generator(1)


# ---------------------------------------------------------------------------
# Bugs #32 & #33: sample_poisson segfaulted on a non-finite mean -- lambda =
# rate * volume reached std::poisson_distribution, whose behavior is undefined
# for a non-finite or non-positive mean. A non-finite or non-positive rate, or
# a non-finite or degenerate lo/hi, must raise a clean ValueError instead of
# crashing the interpreter.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [float("inf"), float("-inf"), float("nan"), -5.0, 0.0])
def test_sample_poisson_bad_rate_raises(rate):
    with pytest.raises(ValueError):
        poisson.sample_poisson((1,), dim=2, rate=rate, generator=_gen())


@pytest.mark.parametrize(
    "lo, hi",
    [
        ([0.0, 0.0], [float("inf"), 1.0]),
        ([float("-inf"), 0.0], [1.0, 1.0]),
        ([0.0, 0.0], [float("nan"), 1.0]),
        ([float("nan"), 0.0], [1.0, 1.0]),
    ],
)
def test_sample_poisson_nonfinite_bounds_raise(lo, hi):
    with pytest.raises(ValueError):
        poisson.sample_poisson((1,), dim=2, rate=1.0, lo=lo, hi=hi, generator=_gen())


@pytest.mark.parametrize(
    "lo, hi",
    [
        ([0.0, 2.0], [1.0, 2.0]),
        ([0.0, 3.0], [1.0, 2.0]),
    ],
)
def test_sample_poisson_degenerate_box_raises(lo, hi):
    with pytest.raises(ValueError):
        poisson.sample_poisson((1,), dim=2, rate=1.0, lo=lo, hi=hi, generator=_gen())


def test_sample_poisson_underflowing_mean_raises():
    with pytest.raises(ValueError):
        poisson.sample_poisson(
            (1,), dim=2, rate=1.0, lo=[0.0, 0.0], hi=[1e-200, 1e-200], generator=_gen()
        )


# ---------------------------------------------------------------------------
# Bug #31: noisy_sin / noisy_cos segfaulted when n_points == 0, because
# noisy_function unconditionally writes randomTs.front()/pts.back() on the
# empty vectors produced for nPoints == 0. n_points must be >= 1.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [sb.random.noisy_sin, sb.random.noisy_cos])
def test_noisy_trig_zero_points_raises(fn):
    with pytest.raises(ValueError):
        fn((1,), n_points=0)


@pytest.mark.parametrize("fn", [sb.random.noisy_sin, sb.random.noisy_cos])
def test_noisy_trig_one_point_ok(fn):
    out = fn((3,), n_points=1)
    assert out.shape == (3,)


# ---------------------------------------------------------------------------
# Bug #35: generators returned an unfilled element for a 0-d shape (), which
# holds one element.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [sb.random.noisy_sin, sb.random.noisy_cos])
def test_noisy_trig_fills_the_element_of_a_0d_shape(fn):
    out = fn((), n_points=6, generator=_gen())

    assert not out.array_equal(sb.zeros((), dtype=sb.pcf32))


def test_sample_poisson_fills_the_element_of_a_0d_shape():
    X = poisson.sample_poisson((), dim=2, rate=5.0, generator=_gen())

    assert not X.array_equal(sb.zeros((), dtype=sb.pcloud64))
