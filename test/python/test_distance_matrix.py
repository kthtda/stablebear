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


class TestArrayConstruction:
    @pytest.mark.parametrize(
        ("np_dtype", "sb_dtype"),
        [(np.float32, float32), (np.float64, float64)],
    )
    def test_squareform_preserves_dtype_and_values(self, np_dtype, sb_dtype):
        array = np.array(
            [[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np_dtype)
        matrix = DistanceMatrix(array)

        assert matrix.dtype is sb_dtype
        np.testing.assert_array_equal(matrix.to_dense(), array)

    def test_compact_uses_scipy_condensed_order(self):
        compact = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        matrix = DistanceMatrix(compact)

        expected = np.array([
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0],
        ], dtype=np.float64)
        np.testing.assert_array_equal(matrix.to_dense(), expected)

    def test_rejects_integer_input_without_explicit_dtype(self):
        with pytest.raises(TypeError, match="use float32 or float64"):
            DistanceMatrix(np.array([1, 2, 3], dtype=np.int32))

    @pytest.mark.parametrize("np_dtype", [np.int32, np.uint64, np.float16])
    def test_explicit_dtype_converts_input(self, np_dtype):
        compact = np.array([1, 2, 3], dtype=np_dtype)
        matrix = DistanceMatrix(compact, dtype=float32)

        expected = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
        ], dtype=np.float32)
        assert matrix.dtype is float32
        np.testing.assert_array_equal(matrix.to_dense(), expected)

    def test_noncontiguous_inputs(self):
        square = np.array([
            [0, 1, 2], [1, 0, 3], [2, 3, 0],
        ], dtype=np.float64)[::-1, ::-1]
        np.testing.assert_array_equal(
            DistanceMatrix(square).to_dense(), square)

        compact = np.arange(1, 13, dtype=np.float32)[::2]
        assert not compact.flags.c_contiguous
        matrix = DistanceMatrix(compact)
        np.testing.assert_array_equal(
            matrix.to_dense(),
            np.array([
                [0, 1, 3, 5],
                [1, 0, 7, 9],
                [3, 7, 0, 11],
                [5, 9, 11, 0],
            ], dtype=np.float32),
        )

    def test_input_is_copied(self):
        compact = np.array([1, 2, 3], dtype=np.float64)
        matrix = DistanceMatrix(compact)
        compact[:] = 99
        assert matrix[0, 1] == 1

    @pytest.mark.parametrize("length", [2, 4, 5])
    def test_rejects_invalid_compact_length(self, length):
        with pytest.raises(ValueError, match=r"n\*\(n-1\)/2"):
            DistanceMatrix(np.ones(length))

    @pytest.mark.parametrize(
        "array",
        [np.array(1.0), np.zeros((2, 2, 2)), np.zeros((2, 3))],
    )
    def test_rejects_invalid_shape(self, array):
        with pytest.raises(ValueError):
            DistanceMatrix(array)

    @pytest.mark.parametrize(
        "array",
        [
            np.array([-1.0]),
            np.array([np.nan]),
            np.array([[0.0, -1.0], [-1.0, 0.0]]),
            np.array([[0.0, 1.0], [2.0, 0.0]]),
            np.array([[1.0]]),
        ],
    )
    def test_rejects_invalid_distance_values(self, array):
        with pytest.raises(ValueError):
            DistanceMatrix(array)

    def test_empty_compact_is_one_by_one(self):
        matrix = DistanceMatrix(np.array([], dtype=np.float64))
        assert matrix.size == 1
        np.testing.assert_array_equal(matrix.to_dense(), np.zeros((1, 1)))

    def test_empty_square_is_zero_by_zero(self):
        matrix = DistanceMatrix(np.empty((0, 0), dtype=np.float64))
        assert matrix.size == 0

    @pytest.mark.parametrize("np_dtype", [np.float16, np.complex128, np.bool_])
    def test_rejects_other_unsupported_array_dtypes(self, np_dtype):
        with pytest.raises(TypeError, match="use float32 or float64"):
            DistanceMatrix(np.array([1, 2, 3], dtype=np_dtype))

    def test_from_dense_is_deprecated(self):
        array = np.array([[0, 1], [1, 0]], dtype=np.float64)
        with pytest.warns(DeprecationWarning, match=r"DistanceMatrix\(array\)"):
            matrix = DistanceMatrix.from_dense(array)
        np.testing.assert_array_equal(matrix.to_dense(), array)


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



@pytest.mark.filterwarnings(
    r"ignore:DistanceMatrix\.from_dense\(\) is deprecated since 0\.4\.7:DeprecationWarning")
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
