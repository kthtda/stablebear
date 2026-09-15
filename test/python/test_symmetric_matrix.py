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


class TestArrayConstruction:
    @pytest.mark.parametrize(
        ("np_dtype", "sb_dtype"),
        [(np.float32, float32), (np.float64, float64)],
    )
    def test_squareform_preserves_dtype_and_values(self, np_dtype, sb_dtype):
        array = np.array(
            [[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=np_dtype)
        matrix = sb.SymmetricMatrix(array)

        assert matrix.dtype is sb_dtype
        np.testing.assert_array_equal(matrix.to_dense(), array)

    def test_compact_uses_upper_triangle_row_major_order(self):
        compact = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        matrix = sb.SymmetricMatrix(compact)

        expected = np.array([
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6],
        ], dtype=np.float64)
        np.testing.assert_array_equal(matrix.to_dense(), expected)

    def test_rejects_integer_input_without_explicit_dtype(self):
        with pytest.raises(TypeError, match="use float32 or float64"):
            sb.SymmetricMatrix(np.array([1, 2, 3], dtype=np.int32))

    @pytest.mark.parametrize("np_dtype", [np.int32, np.uint64, np.float16])
    def test_explicit_dtype_converts_input(self, np_dtype):
        matrix = sb.SymmetricMatrix(
            np.array([1, 2, 3], dtype=np_dtype), dtype=float32)
        assert matrix.dtype is float32
        np.testing.assert_array_equal(
            matrix.to_dense(), np.array([[1, 2], [2, 3]], dtype=np.float32))

    def test_noncontiguous_inputs(self):
        square = np.array([
            [1, 2, 3], [2, 4, 5], [3, 5, 6],
        ], dtype=np.float64)[::-1, ::-1]
        np.testing.assert_array_equal(
            sb.SymmetricMatrix(square).to_dense(), square)

        compact = np.arange(1, 13, dtype=np.float32)[::2]
        assert not compact.flags.c_contiguous
        np.testing.assert_array_equal(
            sb.SymmetricMatrix(compact).to_dense(),
            np.array([[1, 3, 5], [3, 7, 9], [5, 9, 11]], dtype=np.float32),
        )

    def test_input_is_copied(self):
        compact = np.array([1, 2, 3], dtype=np.float64)
        matrix = sb.SymmetricMatrix(compact)
        compact[:] = 99
        assert matrix[0, 0] == 1

    @pytest.mark.parametrize("length", [2, 4, 5])
    def test_rejects_invalid_compact_length(self, length):
        with pytest.raises(ValueError, match=r"n\*\(n\+1\)/2"):
            sb.SymmetricMatrix(np.ones(length))

    @pytest.mark.parametrize(
        "array",
        [np.array(1.0), np.zeros((2, 2, 2)), np.zeros((2, 3))],
    )
    def test_rejects_invalid_shape(self, array):
        with pytest.raises(ValueError):
            sb.SymmetricMatrix(array)

    def test_rejects_asymmetric_squareform(self):
        with pytest.raises(ValueError, match="symmetric"):
            sb.SymmetricMatrix(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            sb.SymmetricMatrix(np.array([np.nan]))

    def test_empty_compact_and_square_are_zero_by_zero(self):
        compact = sb.SymmetricMatrix(np.array([], dtype=np.float64))
        square = sb.SymmetricMatrix(np.empty((0, 0), dtype=np.float64))
        assert compact.size == square.size == 0

    @pytest.mark.parametrize("np_dtype", [np.float16, np.complex128, np.bool_])
    def test_rejects_other_unsupported_array_dtypes(self, np_dtype):
        with pytest.raises(TypeError, match="use float32 or float64"):
            sb.SymmetricMatrix(np.array([1, 2, 3], dtype=np_dtype))

    def test_from_dense_is_deprecated(self):
        array = np.array([[1, 2], [2, 3]], dtype=np.float64)
        with pytest.warns(DeprecationWarning, match=r"SymmetricMatrix\(array\)"):
            matrix = sb.SymmetricMatrix.from_dense(array)
        np.testing.assert_array_equal(matrix.to_dense(), array)


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


class TestRepr:
    def test_contains_size(self, dtype):
        m = sb.SymmetricMatrix(5, dtype=dtype)
        assert "5" in repr(m)
