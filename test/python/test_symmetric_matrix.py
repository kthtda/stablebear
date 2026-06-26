import numpy as np
import pytest

import stablebear as sb
from stablebear.typing import float32, float64


DTYPES = [float32, float64]
NP_DTYPES = {float32: np.float32, float64: np.float64}


@pytest.fixture(params=DTYPES, ids=["float32", "float64"])
def dtype(request):
    return request.param


class TestConstruction:
    def test_construct(self, dtype):
        m = sb.SymmetricMatrix(5, dtype=dtype)
        assert m.size == 5
        assert m.storage_count == 5 + 4 + 3 + 2 + 1

    def test_construct_zero_size(self, dtype):
        m = sb.SymmetricMatrix(0, dtype=dtype)
        assert m.size == 0
        assert m.storage_count == 0

    def test_construct_size_one(self, dtype):
        m = sb.SymmetricMatrix(1, dtype=dtype)
        assert m.size == 1
        assert m.storage_count == 1

    def test_construct_defaults_to_float64(self):
        m = sb.SymmetricMatrix(5)
        assert m.size == 5
        assert m.dtype == sb.float64

    def test_construct_rejects_bad_dtype(self):
        with pytest.raises(TypeError, match="Unsupported dtype"):
            sb.SymmetricMatrix(5, dtype=int)

    def test_construct_from_symmetric_matrix(self, dtype):
        m = sb.SymmetricMatrix(3, dtype=dtype)
        m[0, 1] = 42.0
        m2 = sb.SymmetricMatrix(m)
        assert m2[0, 1] == 42.0

    def test_construct_from_cpp_object(self):
        from stablebear import _sb_cpp as cpp
        raw = cpp.SymmetricMatrix_f64(4)
        m = sb.SymmetricMatrix(raw)
        assert m.size == 4

    def test_construct_rejects_bad_type(self):
        with pytest.raises(TypeError, match="Expected int"):
            sb.SymmetricMatrix([1, 2, 3])


class TestAccess:
    def test_zero_initialized(self, dtype):
        m = sb.SymmetricMatrix(3, dtype=dtype)
        for i in range(3):
            for j in range(3):
                assert m[i, j] == 0.0

    def test_set_and_get(self, dtype):
        m = sb.SymmetricMatrix(4, dtype=dtype)
        m[1, 2] = 5.5
        assert m[1, 2] == 5.5

    def test_symmetry(self, dtype):
        m = sb.SymmetricMatrix(4, dtype=dtype)
        m[0, 3] = 9.0
        assert m[3, 0] == 9.0

    def test_diagonal(self, dtype):
        m = sb.SymmetricMatrix(3, dtype=dtype)
        m[1, 1] = 2.0
        assert m[1, 1] == 2.0

    def test_set_both_directions_same_element(self, dtype):
        m = sb.SymmetricMatrix(3, dtype=dtype)
        m[0, 2] = 1.0
        assert m[2, 0] == 1.0
        m[2, 0] = 7.0
        assert m[0, 2] == 7.0

    def test_out_of_bounds_get(self, dtype):
        m = sb.SymmetricMatrix(3, dtype=dtype)
        with pytest.raises(IndexError):
            _ = m[3, 0]

    def test_out_of_bounds_set(self, dtype):
        m = sb.SymmetricMatrix(3, dtype=dtype)
        with pytest.raises(IndexError):
            m[0, 3] = 1.0


class TestToDense:
    def test_shape(self, dtype):
        m = sb.SymmetricMatrix(4, dtype=dtype)
        assert m.to_dense().shape == (4, 4)

    def test_dtype(self, dtype):
        m = sb.SymmetricMatrix(3, dtype=dtype)
        assert m.to_dense().dtype == NP_DTYPES[dtype]

    def test_values(self, dtype):
        m = sb.SymmetricMatrix(3, dtype=dtype)
        m[0, 1] = 1.0
        m[0, 2] = 2.0
        m[1, 2] = 3.0
        m[0, 0] = 10.0

        expected = np.array([
            [10.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ])
        np.testing.assert_array_equal(m.to_dense(), expected)

    def test_empty(self, dtype):
        m = sb.SymmetricMatrix(0, dtype=dtype)
        assert m.to_dense().shape == (0, 0)


class TestShapeLenDiagonal:
    def test_shape(self, dtype):
        m = sb.SymmetricMatrix(4, dtype=dtype)
        assert m.shape == (4, 4)

    def test_len(self, dtype):
        m = sb.SymmetricMatrix(4, dtype=dtype)
        assert len(m) == 4

    def test_diagonal(self, dtype):
        arr = np.array([
            [10.0, 1.0, 2.0],
            [1.0, 20.0, 3.0],
            [2.0, 3.0, 30.0],
        ], dtype=NP_DTYPES[dtype])
        m = sb.SymmetricMatrix.from_dense(arr)
        np.testing.assert_array_equal(m.diagonal(), [10.0, 20.0, 30.0])

    def test_empty_shape_len_diagonal(self, dtype):
        m = sb.SymmetricMatrix(0, dtype=dtype)
        assert m.shape == (0, 0)
        assert len(m) == 0
        assert m.diagonal().shape == (0,)


class TestRowSliceIndexing:
    def _matrix(self, dtype):
        arr = np.array([
            [10.0, 1.0, 2.0, 4.0],
            [1.0, 20.0, 3.0, 5.0],
            [2.0, 3.0, 30.0, 6.0],
            [4.0, 5.0, 6.0, 40.0],
        ], dtype=NP_DTYPES[dtype])
        return sb.SymmetricMatrix.from_dense(arr), arr

    def test_row_bare_int(self, dtype):
        m, arr = self._matrix(dtype)
        np.testing.assert_array_equal(m[1], arr[1])

    def test_row_int_slice(self, dtype):
        m, arr = self._matrix(dtype)
        np.testing.assert_array_equal(m[1, :], arr[1, :])

    def test_column(self, dtype):
        m, arr = self._matrix(dtype)
        np.testing.assert_array_equal(m[:, 2], arr[:, 2])

    def test_subarray(self, dtype):
        m, arr = self._matrix(dtype)
        np.testing.assert_array_equal(m[0:2, 1:3], arr[0:2, 1:3])

    def test_negative_row(self, dtype):
        m, arr = self._matrix(dtype)
        np.testing.assert_array_equal(m[-1], arr[-1])

    def test_int_pair_still_scalar(self, dtype):
        m, arr = self._matrix(dtype)
        assert m[1, 2] == arr[1, 2]
        assert np.isscalar(m[1, 2])


class TestRepr:
    def test_contains_size(self, dtype):
        m = sb.SymmetricMatrix(5, dtype=dtype)
        assert "5" in repr(m)
