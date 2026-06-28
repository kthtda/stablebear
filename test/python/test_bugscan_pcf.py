import numpy as np
import pytest

import stablebear as sb

Pcf = sb.Pcf


# ---------------------------------------------------------------------------
# Bug #11: Pcf construction validated t0 on the *unsorted* input row 0, then
# std::sort could move a negative time to the front -- yielding a PCF with
# t0 < 0 that happily evaluated at negative times. Construction now rejects
# unsorted input outright (instead of silently sorting) and requires the first
# time to be 0; evaluation guards against 0 directly.
# ---------------------------------------------------------------------------


def test_unsorted_input_rejected():
    # Breakpoints out of time order are no longer silently sorted; they are
    # rejected so a misplaced/negative time can never slip in unnoticed.
    with pytest.raises(ValueError, match="non-decreasing time order"):
        Pcf(np.array([[2.0, 2.0], [0.0, 1.0]], dtype=np.float64))


def test_unsorted_negative_time_rejected():
    # A negative time on a later row makes the input unsorted -> rejected.
    with pytest.raises(ValueError, match="non-decreasing time order"):
        Pcf(np.array([[0.0, 1.0], [2.0, 2.0], [-1.0, 99.0]], dtype=np.float64))


def test_negative_first_time_rejected():
    # Sorted input whose (genuine) minimum time is negative -> t=0 check.
    with pytest.raises(ValueError, match="t=0"):
        Pcf(np.array([[-1.0, 5.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float64))


def test_min_time_above_zero_rejected():
    with pytest.raises(ValueError, match="t=0"):
        Pcf(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))


def test_intpcf_negative_first_time_rejected():
    with pytest.raises(ValueError, match="t=0"):
        Pcf(np.array([[-1, 9], [0, 1], [2, 2]], dtype=np.int64))


def test_eval_negative_time_scalar_raises():
    f = Pcf(np.array([[0.0, 1.0], [2.0, 2.0]], dtype=np.float64))
    with pytest.raises(Exception, match="time 0"):
        f(-0.5)


def test_eval_negative_time_array_raises():
    f = Pcf(np.array([[0.0, 1.0], [2.0, 2.0]], dtype=np.float64))
    with pytest.raises(Exception, match="time 0"):
        f(np.array([-0.5, 0.5]))


# ---------------------------------------------------------------------------
# Bug #56: Pcf construction silently accepted duplicate breakpoint times
# (contradicting the strictly-increasing contract) and turned an empty (0, 2)
# input array into a fabricated 1-point PCF. Both are now rejected.
# ---------------------------------------------------------------------------


def test_pcf_rejects_duplicate_times():
    with pytest.raises(ValueError, match="strictly increasing"):
        Pcf(np.array([[0.0, 5.0], [1.0, 6.0], [1.0, 7.0]]))


def test_pcf_rejects_duplicate_first_times():
    with pytest.raises(ValueError, match="strictly increasing"):
        Pcf(np.array([[0.0, 5.0], [0.0, 6.0]]))


def test_pcf_rejects_empty_array():
    with pytest.raises(ValueError, match="empty"):
        Pcf(np.zeros((0, 2)))


def test_pcf_strictly_increasing_still_constructs():
    f = Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.0]]))
    assert f.size == 3
