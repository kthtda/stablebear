import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb
from stablebear.base_tensor import FloatTensor, IntTensor


# ---------------------------------------------------------------------------
# Bug #191: in-place tensor arithmetic ignored LHS/RHS aliasing. When the RHS
# is a view of the same buffer (a[1:] += a[:-1], a += a[::-1]), the in-place
# walk read elements it had already updated, producing traversal-order-
# dependent results. NumPy materializes the RHS first; so do we now.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("np_dtype,TensorT", [(np.float32, FloatTensor),
                                              (np.float64, FloatTensor),
                                              (np.int32, IntTensor),
                                              (np.int64, IntTensor)])
class TestInplaceAliasing:
    def test_shifted_overlap_add(self, np_dtype, TensorT):
        a = TensorT(np.array([1, 2, 3, 4], dtype=np_dtype))
        a[1:] += a[:-1]
        expected = np.array([1, 2, 3, 4], dtype=np_dtype)
        expected[1:] = expected[1:] + expected[:-1].copy()
        npt.assert_array_equal(np.asarray(a), expected)  # [1, 3, 5, 7]

    def test_reversed_self_add(self, np_dtype, TensorT):
        a = TensorT(np.array([1, 2, 3, 4], dtype=np_dtype))
        a += a[::-1]
        npt.assert_array_equal(np.asarray(a), [5, 5, 5, 5])

    def test_shifted_overlap_mul(self, np_dtype, TensorT):
        a = TensorT(np.array([1, 2, 3, 4], dtype=np_dtype))
        a[1:] *= a[:-1]
        npt.assert_array_equal(np.asarray(a), [1, 2, 6, 12])

    def test_2d_row_overlap(self, np_dtype, TensorT):
        arr = np.arange(1, 7, dtype=np_dtype).reshape(3, 2)
        a = TensorT(arr.copy())
        a[1:] += a[:-1]
        expected = arr.copy()
        expected[1:] = expected[1:] + expected[:-1].copy()
        npt.assert_array_equal(np.asarray(a), expected)

    def test_non_aliasing_unaffected(self, np_dtype, TensorT):
        a = TensorT(np.array([1, 2, 3, 4], dtype=np_dtype))
        b = TensorT(np.array([10, 20, 30, 40], dtype=np_dtype))
        a += b
        npt.assert_array_equal(np.asarray(a), [11, 22, 33, 44])


def test_pcf_tensor_shifted_overlap_add():
    """Aliasing handling must also hold for PCF elements."""
    X = sb.zeros((3,))
    X[0] = sb.Pcf(np.array([[0.0, 1.0]], dtype=np.float32))
    X[1] = sb.Pcf(np.array([[0.0, 2.0]], dtype=np.float32))
    X[2] = sb.Pcf(np.array([[0.0, 4.0]], dtype=np.float32))

    X[1:] += X[:-1]

    npt.assert_array_equal(np.asarray(X(0.0)), [1.0, 3.0, 6.0])
