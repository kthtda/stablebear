import numpy as np
import pytest

import stablebear as sb


# ---------------------------------------------------------------------------
# Bug #19: split(tensor, 0) performed an integer modulo-by-zero
# (axis_size % n_sections) and aborted the interpreter with SIGFPE.
# array_split already guarded n_sections == 0; split now does too, raising a
# clean, catchable ValueError.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "t",
    [
        pytest.param(sb.FloatTensor(np.arange(6, dtype=np.float64)), id="float"),
        pytest.param(sb.IntTensor(np.arange(6, dtype=np.int64)), id="int"),
        pytest.param(sb.zeros((6,)), id="pcf"),
        pytest.param(sb.FloatTensor(np.zeros((0,))), id="empty"),
    ],
)
def test_split_zero_sections_raises(t):
    with pytest.raises(ValueError):
        sb.split(t, 0)


def test_split_zero_sections_explicit_axis_raises():
    t = sb.FloatTensor(np.arange(12, dtype=np.float64).reshape(3, 4))
    with pytest.raises(ValueError):
        sb.split(t, 0, axis=1)


def test_array_split_zero_sections_still_raises():
    with pytest.raises(ValueError):
        sb.array_split(sb.FloatTensor(np.arange(6.0)), 0)
