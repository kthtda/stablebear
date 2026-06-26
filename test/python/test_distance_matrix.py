import numpy as np
import pytest

from stablebear.distance_matrix import DistanceMatrix
from stablebear.typing import float32, float64


@pytest.fixture(params=[float32, float64], ids=["f32", "f64"])
def dtype(request):
    return request.param


class TestConstruction:
    def test_empty(self, dtype):
        dm = DistanceMatrix(0, dtype=dtype)
        assert dm.size == 0
        assert dm.storage_count == 0

    def test_size_1(self, dtype):
        dm = DistanceMatrix(1, dtype=dtype)
        assert dm.size == 1
        assert dm.storage_count == 0

    def test_size_n(self, dtype):
        dm = DistanceMatrix(5, dtype=dtype)
        assert dm.size == 5
        assert dm.storage_count == 10  # 5*4/2

    def test_defaults_to_float64(self):
        dm = DistanceMatrix(3)
        dm[0, 1] = 1.0
        dense = dm.to_dense()
        assert dense.dtype == np.float64

    def test_rejects_bad_dtype(self):
        with pytest.raises(TypeError, match="Unsupported dtype"):
            DistanceMatrix(3, dtype=int)

    def test_copy_wraps_same_data(self, dtype):
        dm1 = DistanceMatrix(3, dtype=dtype)
        dm1[0, 1] = 1.0
        dm2 = DistanceMatrix(dm1)
        assert dm2[0, 1] == 1.0


class TestAccess:
    def test_diagonal_is_zero(self, dtype):
        dm = DistanceMatrix(3, dtype=dtype)
        for i in range(3):
            assert dm[i, i] == 0.0

    def test_set_and_get(self, dtype):
        dm = DistanceMatrix(4, dtype=dtype)
        dm[0, 1] = 2.5
        assert dm[0, 1] == pytest.approx(2.5)

    def test_symmetric(self, dtype):
        dm = DistanceMatrix(4, dtype=dtype)
        dm[0, 3] = 7.0
        assert dm[3, 0] == pytest.approx(7.0)

    def test_reject_negative(self, dtype):
        dm = DistanceMatrix(3, dtype=dtype)
        with pytest.raises(ValueError):
            dm[0, 1] = -1.0

    def test_reject_nonzero_diagonal(self, dtype):
        dm = DistanceMatrix(3, dtype=dtype)
        with pytest.raises(ValueError):
            dm[1, 1] = 5.0

    def test_set_diagonal_zero_ok(self, dtype):
        dm = DistanceMatrix(3, dtype=dtype)
        dm[1, 1] = 0.0  # should not raise

    def test_out_of_bounds(self, dtype):
        dm = DistanceMatrix(3, dtype=dtype)
        with pytest.raises(IndexError):
            _ = dm[3, 0]


class TestToDense:
    def test_roundtrip(self, dtype):
        dm = DistanceMatrix(3, dtype=dtype)
        dm[0, 1] = 1.0
        dm[0, 2] = 2.0
        dm[1, 2] = 3.0
        dense = dm.to_dense()
        expected = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
        ], dtype=np.float32 if dtype is float32 else np.float64)
        np.testing.assert_array_almost_equal(dense, expected)

    def test_empty_to_dense(self, dtype):
        dm = DistanceMatrix(0, dtype=dtype)
        dense = dm.to_dense()
        assert dense.shape == (0, 0)


class TestFromDense:
    def test_roundtrip_f32(self):
        arr = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float32)
        dm = DistanceMatrix.from_dense(arr)
        np.testing.assert_array_almost_equal(dm.to_dense(), arr)

    def test_roundtrip_f64(self):
        arr = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float64)
        dm = DistanceMatrix.from_dense(arr)
        np.testing.assert_array_almost_equal(dm.to_dense(), arr)

    def test_rejects_nonzero_diagonal(self):
        arr = np.array([[1, 0], [0, 0]], dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError)):
            DistanceMatrix.from_dense(arr)

    def test_rejects_asymmetric(self):
        arr = np.array([[0, 1], [2, 0]], dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError)):
            DistanceMatrix.from_dense(arr)

    def test_rejects_negative(self):
        arr = np.array([[0, -1], [-1, 0]], dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError)):
            DistanceMatrix.from_dense(arr)

    def test_rejects_unsupported_dtype(self):
        arr = np.array([[0, 1], [1, 0]], dtype=np.int32)
        with pytest.raises(TypeError):
            DistanceMatrix.from_dense(arr)


class TestShapeLenDiagonal:
    def test_shape(self, dtype):
        dm = DistanceMatrix(4, dtype=dtype)
        assert dm.shape == (4, 4)

    def test_len(self, dtype):
        dm = DistanceMatrix(4, dtype=dtype)
        assert len(dm) == 4

    def test_diagonal_all_zero(self, dtype):
        np_float = np.float32 if dtype is float32 else np.float64
        arr = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
        ], dtype=np_float)
        dm = DistanceMatrix.from_dense(arr)
        diag = dm.diagonal()
        np.testing.assert_array_equal(diag, np.zeros(3))
        assert diag.dtype == np_float

    def test_empty_shape_len_diagonal(self, dtype):
        dm = DistanceMatrix(0, dtype=dtype)
        assert dm.shape == (0, 0)
        assert len(dm) == 0
        assert dm.diagonal().shape == (0,)


class TestRowSliceIndexing:
    def _matrix(self, dtype):
        np_float = np.float32 if dtype is float32 else np.float64
        arr = np.array([
            [0, 1, 2, 4],
            [1, 0, 3, 5],
            [2, 3, 0, 6],
            [4, 5, 6, 0],
        ], dtype=np_float)
        return DistanceMatrix.from_dense(arr), arr

    def test_row_bare_int(self, dtype):
        dm, arr = self._matrix(dtype)
        np.testing.assert_array_equal(dm[1], arr[1])

    def test_row_int_slice(self, dtype):
        dm, arr = self._matrix(dtype)
        np.testing.assert_array_equal(dm[1, :], arr[1, :])

    def test_column(self, dtype):
        dm, arr = self._matrix(dtype)
        np.testing.assert_array_equal(dm[:, 2], arr[:, 2])

    def test_subarray(self, dtype):
        dm, arr = self._matrix(dtype)
        np.testing.assert_array_equal(dm[0:2, 1:3], arr[0:2, 1:3])

    def test_negative_row(self, dtype):
        dm, arr = self._matrix(dtype)
        np.testing.assert_array_equal(dm[-1], arr[-1])

    def test_int_pair_still_scalar(self, dtype):
        dm, arr = self._matrix(dtype)
        assert dm[1, 2] == arr[1, 2]
        assert np.isscalar(dm[1, 2])
