import numpy as np
import pytest
import stablebear as sb


class TestAllcloseFloatTensor:
    def test_identical(self):
        a = sb.FloatTensor([1.0, 2.0, 3.0])
        assert sb.allclose(a, a)

    def test_equal_copies(self):
        a = sb.FloatTensor([1.0, 2.0, 3.0])
        b = a.copy()
        assert sb.allclose(a, b)

    def test_within_atol(self):
        a = sb.FloatTensor([1.0, 2.0, 3.0])
        b = sb.FloatTensor([1.0 + 1e-9, 2.0, 3.0])
        assert sb.allclose(a, b)

    def test_outside_atol(self):
        a = sb.FloatTensor([1.0, 2.0, 3.0])
        b = sb.FloatTensor([1.0, 2.1, 3.0])
        assert not sb.allclose(a, b)

    def test_custom_atol(self):
        a = sb.FloatTensor([1.0, 2.0, 3.0])
        b = sb.FloatTensor([1.0, 2.05, 3.0])
        assert not sb.allclose(a, b, atol=0.01)
        assert sb.allclose(a, b, atol=0.1)

    def test_multidimensional(self):
        a = sb.FloatTensor(np.ones((3, 4), dtype=np.float32))
        b = sb.FloatTensor(np.ones((3, 4), dtype=np.float32))
        assert sb.allclose(a, b)


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


@pytest.mark.parametrize("matrix_type", [sb.DistanceMatrix, sb.SymmetricMatrix])
@pytest.mark.parametrize("dtype", [sb.float32, sb.float64])
def test_allclose_empty_and_singleton_matrices_differ(matrix_type, dtype):
    assert not sb.allclose(matrix_type(0, dtype=dtype), matrix_type(1, dtype=dtype))


class TestAllcloseErrors:
    def test_mismatched_types(self):
        a = sb.FloatTensor([1.0])
        b = sb.DistanceMatrix(1, dtype=sb.float64)
        with pytest.raises(TypeError, match="same supported type"):
            sb.allclose(a, b)

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="same supported type"):
            sb.allclose(42, 42)
