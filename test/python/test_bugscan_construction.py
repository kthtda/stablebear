import numpy as np
import pytest

import stablebear as sb


# ---------------------------------------------------------------------------
# Bug #3: constructing a numeric/bool tensor from a 0-d (scalar) ndarray
# silently dropped the value and returned the dtype's zero, because walk_impl
# skipped rank-0 tensors (shape ()), conflating them with zero-size extents.
# A rank-0 tensor has exactly one element and must preserve it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "TensorType, np_dtype, value",
    [
        (sb.FloatTensor, np.float64, 3.5),
        (sb.FloatTensor, np.float32, -2.25),
        (sb.IntTensor, np.int64, 11),
        (sb.IntTensor, np.int32, -7),
        (sb.BoolTensor, np.bool_, True),
        (sb.BoolTensor, np.bool_, False),
    ],
)
def test_scalar_ndarray_preserves_value(TensorType, np_dtype, value):
    """A 0-d ndarray holds one element; the rank-0 tensor must keep its value."""
    t = TensorType(np.array(value, dtype=np_dtype))
    assert tuple(t.shape) == ()
    assert t.size == 1
    assert np.asarray(t).item() == value


def test_one_element_1d_still_preserved():
    """Regression guard: the 1-element 1-D case always worked; keep it working."""
    t = sb.FloatTensor(np.array([3.5]))
    assert tuple(t.shape) == (1,)
    assert np.asarray(t).tolist() == [3.5]


def test_reshape_to_rank0_preserves_value():
    """reshape([]) of a populated 1-element tensor yields a rank-0 tensor that
    still holds the value (the shape-() storage was always populated; only the
    ndarray->tensor walk dropped it)."""
    t = sb.FloatTensor(np.array([3.5])).reshape([])
    assert tuple(t.shape) == ()
    assert np.asarray(t).item() == 3.5


def test_zero_size_extent_stays_empty():
    """Contrast: a genuine zero-size extent has no elements and is unaffected."""
    for shape in [(0,), (3, 0), (0, 5)]:
        t = sb.FloatTensor(np.zeros(shape))
        assert t.size == 0
        assert tuple(t.shape) == shape
