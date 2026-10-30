import numpy as np
import pytest
import stablebear as sb


def _assert_matches_numpy(a, b, **tolerances):
    """Assert that sb.allclose(a, b) agrees with np.allclose on the same inputs."""
    expected = np.allclose(np.asarray(a), np.asarray(b), **tolerances)
    assert sb.allclose(a, b, **tolerances) is expected


class TestAllcloseFloatTensor:
    def test_identical(self):
        a = sb.FloatTensor([1.0, 2.0, 3.0])
        _assert_matches_numpy(a, a)

    def test_equal_copies(self):
        a = sb.FloatTensor([1.0, 2.0, 3.0])
        b = a.copy()
        _assert_matches_numpy(a, b)

    def test_within_atol(self):
        a = sb.FloatTensor([1.0, 2.0, 3.0])
        b = sb.FloatTensor([1.0 + 1e-9, 2.0, 3.0])
        _assert_matches_numpy(a, b)

    def test_outside_atol(self):
        a = sb.FloatTensor([1.0, 2.0, 3.0])
        b = sb.FloatTensor([1.0, 2.1, 3.0])
        _assert_matches_numpy(a, b)

    def test_custom_atol(self):
        a = sb.FloatTensor([1.0, 2.0, 3.0])
        b = sb.FloatTensor([1.0, 2.05, 3.0])
        _assert_matches_numpy(a, b, atol=0.01)
        _assert_matches_numpy(a, b, atol=0.1)

    def test_multidimensional(self):
        a = sb.FloatTensor(np.ones((3, 4), dtype=np.float32))
        b = sb.FloatTensor(np.ones((3, 4), dtype=np.float32))
        _assert_matches_numpy(a, b)

    def test_equal_infinities_are_close(self):
        a = sb.FloatTensor([1.0, np.inf, -np.inf])
        _assert_matches_numpy(a, a.copy())

    def test_opposite_infinities_are_not_close(self):
        _assert_matches_numpy(sb.FloatTensor([np.inf]), sb.FloatTensor([-np.inf]))

    def test_infinity_is_not_close_to_finite(self):
        _assert_matches_numpy(sb.FloatTensor([np.inf]), sb.FloatTensor([1e300]))

    def test_nan_is_not_close_to_itself(self):
        a = sb.FloatTensor([1.0, np.nan])
        _assert_matches_numpy(a, a.copy())


class TestAllcloseBroadcasting:
    def test_row_broadcasts_against_matrix(self):
        a = sb.FloatTensor([[1.0, 2.0, 3.0],
                            [1.0, 2.0, 3.0]])
        _assert_matches_numpy(a, sb.FloatTensor([1.0, 2.0, 3.0]))
        _assert_matches_numpy(a, sb.FloatTensor([1.0, 2.0, 4.0]))

    def test_column_broadcasts_against_matrix(self):
        a = sb.FloatTensor([[1.0, 1.0],
                            [2.0, 2.0]])
        _assert_matches_numpy(a, sb.FloatTensor([[1.0], [2.0]]))

    def test_scalar_broadcasts_against_tensor(self):
        a = sb.FloatTensor([2.0, 2.0])
        _assert_matches_numpy(a, 2.0)
        _assert_matches_numpy(a, 3.0)

    def test_incompatible_shapes_raise(self):
        a = sb.FloatTensor([1.0, 2.0, 3.0])
        b = sb.FloatTensor([1.0, 2.0])
        with pytest.raises(ValueError):
            np.allclose(np.asarray(a), np.asarray(b))
        with pytest.raises(ValueError, match="broadcast"):
            sb.allclose(a, b)


class TestAllcloseInputTypes:
    @pytest.mark.parametrize("np_dtype", [np.int32, np.int64])
    def test_int_tensors(self, np_dtype):
        a = sb.IntTensor(np.array([1, 2, 3], dtype=np_dtype))
        b = sb.IntTensor(np.array([1, 2, 4], dtype=np_dtype))
        _assert_matches_numpy(a, a.copy())
        _assert_matches_numpy(a, b)

    def test_bool_tensors(self):
        a = sb.BoolTensor(np.array([True, False]))
        b = sb.BoolTensor(np.array([False, False]))
        _assert_matches_numpy(a, a.copy())
        _assert_matches_numpy(a, b)

    def test_int_tensor_and_float_tensor(self):
        a = sb.IntTensor(np.array([1, 2], dtype=np.int32))
        _assert_matches_numpy(a, sb.FloatTensor([1.0, 2.0]))

    def test_float32_and_float64_tensors(self):
        a = sb.FloatTensor(np.array([1.0, 2.0], dtype=np.float32))
        b = sb.FloatTensor(np.array([1.0, 2.0], dtype=np.float64))
        _assert_matches_numpy(a, b)

    def test_tensor_and_ndarray(self):
        a = sb.FloatTensor([1.0, 2.0])
        _assert_matches_numpy(a, np.array([1.0, 2.0]))
        _assert_matches_numpy(np.array([1.0, 2.0]), a)

    def test_tensor_and_list(self):
        _assert_matches_numpy(sb.FloatTensor([1.0, 2.0]), [1.0, 2.0])


class TestAllcloseDistanceMatrix:
    def test_indexed_matches_dense(self):
        matrices = sb.DistanceMatrixTensor(
            np.array([
                [0, 1, 2, 3],
                [1, 0, 1, 2],
                [2, 1, 0, 1],
                [3, 2, 1, 0],
            ], dtype=np.float64))
        indexed = matrices[sb.NestedTensor(sb.indices([3, 0, 3]))][()]
        dense = sb.DistanceMatrix(np.asarray(indexed))
        assert sb.allclose(indexed, dense)
        dense[0, 1] = 9
        assert not sb.allclose(indexed, dense)

    def test_identical(self):
        m = sb.DistanceMatrix(5, dtype=sb.float64)
        m[0, 1] = 3.14
        assert sb.allclose(m, m)

    def test_within_tolerance(self):
        a = sb.DistanceMatrix(3, dtype=sb.float64)
        b = sb.DistanceMatrix(3, dtype=sb.float64)
        a[0, 1] = 1.0
        b[0, 1] = 1.0 + 1e-9
        assert sb.allclose(a, b)

    def test_outside_tolerance(self):
        a = sb.DistanceMatrix(3, dtype=sb.float64)
        b = sb.DistanceMatrix(3, dtype=sb.float64)
        a[0, 1] = 1.0
        b[0, 1] = 2.0
        assert not sb.allclose(a, b)

    def test_equal_infinities_are_close(self):
        a = sb.DistanceMatrix(3, dtype=sb.float64)
        b = sb.DistanceMatrix(3, dtype=sb.float64)
        a[0, 1] = np.inf
        b[0, 1] = np.inf
        assert sb.allclose(a, b)

    def test_infinity_is_not_close_to_finite(self):
        a = sb.DistanceMatrix(3, dtype=sb.float64)
        b = sb.DistanceMatrix(3, dtype=sb.float64)
        a[0, 1] = np.inf
        b[0, 1] = 1e300
        assert not sb.allclose(a, b)


class TestAllcloseSymmetricMatrix:
    def test_identical(self):
        m = sb.SymmetricMatrix(5, dtype=sb.float64)
        m[0, 1] = 3.14
        assert sb.allclose(m, m)

    @pytest.mark.parametrize("entry", [(0, 0), (0, 1)])
    def test_within_tolerance(self, entry):
        a = sb.SymmetricMatrix(3, dtype=sb.float64)
        b = sb.SymmetricMatrix(3, dtype=sb.float64)
        a[entry] = 1.0
        b[entry] = 1.0 + 1e-9
        assert sb.allclose(a, b)
        b[entry] = 2.0
        assert not sb.allclose(a, b)

    def test_equal_infinities_are_close(self):
        a = sb.SymmetricMatrix(3, dtype=sb.float64)
        b = sb.SymmetricMatrix(3, dtype=sb.float64)
        a[0, 1] = np.inf
        b[0, 1] = np.inf
        assert sb.allclose(a, b)

    def test_opposite_infinities_are_not_close(self):
        a = sb.SymmetricMatrix(3, dtype=sb.float64)
        b = sb.SymmetricMatrix(3, dtype=sb.float64)
        a[0, 1] = np.inf
        b[0, 1] = -np.inf
        assert not sb.allclose(a, b)

    def test_infinity_is_not_close_to_finite(self):
        a = sb.SymmetricMatrix(3, dtype=sb.float64)
        b = sb.SymmetricMatrix(3, dtype=sb.float64)
        a[0, 1] = np.inf
        b[0, 1] = 1e300
        assert not sb.allclose(a, b)

    def test_nan_is_not_close_to_itself(self):
        a = sb.SymmetricMatrix(3, dtype=sb.float64)
        a[0, 1] = np.nan
        assert not sb.allclose(a, a)


@pytest.mark.parametrize("matrix_type", [sb.DistanceMatrix, sb.SymmetricMatrix])
@pytest.mark.parametrize("dtype", [sb.float32, sb.float64])
def test_allclose_empty_and_singleton_matrices_differ(matrix_type, dtype):
    assert not sb.allclose(matrix_type(0, dtype=dtype), matrix_type(1, dtype=dtype))


class TestAllcloseMatrixPrecision:
    @pytest.mark.parametrize("matrix_type", [sb.DistanceMatrix, sb.SymmetricMatrix])
    def test_matrices_of_different_precision(self, matrix_type):
        a = matrix_type(3, dtype=sb.float32)
        b = matrix_type(3, dtype=sb.float64)
        a[0, 1] = 2.0
        b[0, 1] = 2.0
        assert sb.allclose(a, b)
        b[0, 1] = 3.0
        assert not sb.allclose(a, b)


class TestAllcloseErrors:
    def test_matrix_and_tensor_raise(self):
        a = sb.FloatTensor([1.0])
        b = sb.DistanceMatrix(1, dtype=sb.float64)
        with pytest.raises(TypeError, match="only with another DistanceMatrix"):
            sb.allclose(a, b)

    def test_distance_matrix_and_symmetric_matrix_raise(self):
        a = sb.DistanceMatrix(2, dtype=sb.float64)
        b = sb.SymmetricMatrix(2, dtype=sb.float64)
        with pytest.raises(TypeError, match="only with another DistanceMatrix"):
            sb.allclose(a, b)

    def test_no_tensor_raises(self):
        with pytest.raises(TypeError, match="at least one tensor or matrix"):
            sb.allclose(42, 42)
