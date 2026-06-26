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
# Bug #12: Pcf.__pow__ and tensor __pow__ called PyErr_WarnEx and ignored its
# return code. Under `-W error` (warnings escalated to exceptions) the warning
# becomes a pending Python exception, but the binding still returned a value, so
# pybind11 raised a confusing SystemError. The bindings must now propagate the
# RuntimeWarning-as-error instead.
# ---------------------------------------------------------------------------


def test_pcf_pow_propagates_warning_as_error():
    import warnings
    f = Pcf(np.array([[0.0, -2.0], [1.0, 4.0]], dtype=np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning):
            f ** 0.5


def test_tensor_pow_propagates_warning_as_error():
    import warnings
    t = sb.FloatTensor(np.array([-2.0, 4.0], dtype=np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning):
            t ** 0.5


def test_pcf_pow_warning_still_emitted_by_default():
    import warnings
    f = Pcf(np.array([[0.0, -2.0], [1.0, 4.0]], dtype=np.float64))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = f ** 0.5
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_pcf_pow_no_warning_for_finite_result():
    import warnings
    f = Pcf(np.array([[0.0, 4.0], [1.0, 9.0]], dtype=np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = f ** 0.5  # all-positive base -> finite, no warning
    assert g(0.5) == 2.0
