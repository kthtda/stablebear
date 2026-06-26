import numpy as np
import pytest

import stablebear as sb


# ---------------------------------------------------------------------------
# Bugs #16 & #9: broadcast_to returned a writable stride-0 view, so assigning
# through it (#16) or doing in-place arithmetic on it (#9) silently corrupted
# the shared source (a single write scattered across the broadcast axis).
# NumPy marks broadcast views read-only; mirror that by raising ValueError.
# ---------------------------------------------------------------------------


def _broadcast_view():
    src = sb.FloatTensor(np.array([[1.0, 2.0, 3.0]]))  # (1, 3)
    return src, src.broadcast_to((4, 3))               # strides (0, 1)


def test_setitem_element_through_broadcast_raises():
    src, b = _broadcast_view()
    with pytest.raises(ValueError):
        b[0, 0] = 999.0
    assert np.asarray(src).tolist() == [[1.0, 2.0, 3.0]]  # source untouched


def test_setitem_slice_through_broadcast_raises():
    src, b = _broadcast_view()
    with pytest.raises(ValueError):
        b[:, 0] = 7.0
    with pytest.raises(ValueError):
        b[:] = 99.0
    assert np.asarray(src).tolist() == [[1.0, 2.0, 3.0]]


def test_inplace_arithmetic_through_broadcast_raises():
    src = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))
    for op in ("__iadd__", "__isub__", "__imul__", "__itruediv__", "__ipow__"):
        b = src.broadcast_to((2, 3))     # axis-0 stride 0
        with pytest.raises(ValueError):
            getattr(b, op)(2.0)
    assert np.asarray(src).tolist() == [1.0, 2.0, 3.0]  # source untouched


def test_inplace_with_tensor_rhs_through_broadcast_raises():
    src = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))
    b = src.broadcast_to((2, 3))
    rhs = sb.FloatTensor(np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]))
    with pytest.raises(ValueError):
        b += rhs
    assert np.asarray(src).tolist() == [1.0, 2.0, 3.0]


def test_prepended_broadcast_axis_is_read_only():
    src = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))  # (3,)
    b = src.broadcast_to((2, 3))                      # prepended axis, stride 0
    with pytest.raises(ValueError):
        b[0, 0] = 5.0


def test_int_tensor_broadcast_is_read_only():
    src = sb.IntTensor(np.array([[1, 2, 3]]))
    b = src.broadcast_to((4, 3))
    with pytest.raises(ValueError):
        b[1, 2] = 777
    assert np.asarray(src).tolist() == [[1, 2, 3]]


# --- the fix must not over-reach: these remain writeable ---------------------


def test_broadcast_view_is_still_readable():
    src = sb.FloatTensor(np.array([[1.0, 2.0, 3.0]]))
    b = src.broadcast_to((4, 3))
    assert np.asarray(b).tolist() == [[1.0, 2.0, 3.0]] * 4


def test_copy_of_broadcast_is_writeable():
    src = sb.FloatTensor(np.array([[1.0, 2.0, 3.0]]))
    c = src.broadcast_to((4, 3)).copy()
    c[0, 0] = 999.0                                  # independent storage
    assert np.asarray(c)[0].tolist() == [999.0, 2.0, 3.0]
    assert np.asarray(c)[1].tolist() == [1.0, 2.0, 3.0]
    assert np.asarray(src).tolist() == [[1.0, 2.0, 3.0]]


def test_newaxis_size1_stride0_is_writeable():
    """A size-1 axis (from newaxis) may have stride 0 but has no aliasing, so it
    stays writeable -- only size>1 stride-0 axes are read-only."""
    t = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))[None]   # (1, 3)
    t[0, 0] = 5.0
    assert np.asarray(t).tolist() == [[5.0, 2.0, 3.0]]
