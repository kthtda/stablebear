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

    def test_within_tolerance(self):
        a = sb.SymmetricMatrix(3, dtype=sb.float64)
        b = sb.SymmetricMatrix(3, dtype=sb.float64)
        a[0, 1] = 1.0
        b[0, 1] = 1.0 + 1e-9
        assert sb.allclose(a, b)


class TestAllcloseErrors:
    def test_mismatched_types(self):
        a = sb.FloatTensor([1.0])
        b = sb.DistanceMatrix(1, dtype=sb.float64)
        with pytest.raises(TypeError, match="matrix type"):
            sb.allclose(a, b)

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="stablebear tensor or matrix"):
            sb.allclose(42, 42)


class TestAllcloseInfinity:
    # Bug #51: |inf - inf| is NaN, so the bare formula reported +inf != +inf and
    # gave false positives for opposite-sign infinities.
    def test_same_sign_infinity_is_close(self):
        a = sb.FloatTensor([1.0, np.inf, 3.0])
        assert sb.allclose(a, a)

    def test_opposite_sign_infinity_not_close(self):
        assert not sb.allclose(sb.FloatTensor([np.inf]), sb.FloatTensor([-np.inf]))

    def test_finite_vs_infinity_not_close(self):
        assert not sb.allclose(sb.FloatTensor([np.inf]), sb.FloatTensor([1.0]))

    def test_nan_never_close(self):
        assert not sb.allclose(sb.FloatTensor([np.nan]), sb.FloatTensor([np.nan]))

    def test_matches_numpy_isclose(self):
        a = np.array([1.0, np.inf, -np.inf, np.nan])
        b = np.array([1.0, np.inf, np.inf, np.nan])
        assert sb.allclose(sb.FloatTensor(a), sb.FloatTensor(b)) == bool(
            np.allclose(a, b))  # both False (the nan/opposite-inf entries)


class TestAllcloseShape:
    # Bug #52: a shape mismatch silently returned False instead of broadcasting
    # or raising, hiding real shape bugs.
    def test_incompatible_shapes_raise(self):
        a = sb.FloatTensor(np.ones((3,)))
        b = sb.FloatTensor(np.ones((2, 3)))   # not broadcast-compatible vs (3,)? -> compatible
        c = sb.FloatTensor(np.ones((2,)))     # (3,) vs (2,) is incompatible
        with pytest.raises(ValueError):
            sb.allclose(a, c)

    def test_broadcast_compatible_shapes(self):
        a = sb.FloatTensor(np.ones((2, 3)))
        row = sb.FloatTensor(np.ones((3,)))
        assert sb.allclose(a, row)
        col = sb.FloatTensor(np.full((1, 3), 2.0))
        assert not sb.allclose(a, col)


class TestAllcloseInterop:
    # Bug #100: allclose only accepted FloatTensor/DistanceMatrix/SymmetricMatrix.
    def test_int_tensors(self):
        assert sb.allclose(sb.IntTensor([1, 2, 3]), sb.IntTensor([1, 2, 3]))
        assert not sb.allclose(sb.IntTensor([1, 2, 3]), sb.IntTensor([1, 2, 4]))

    def test_bool_tensors(self):
        assert sb.allclose(sb.BoolTensor([True, False]), sb.BoolTensor([True, False]))

    def test_int_vs_float(self):
        assert sb.allclose(sb.IntTensor([1, 2, 3]), sb.FloatTensor([1.0, 2.0, 3.0]))

    def test_float32_vs_float64(self):
        a = sb.FloatTensor(np.ones(3, dtype=np.float32))
        b = sb.FloatTensor(np.ones(3, dtype=np.float64))
        assert sb.allclose(a, b)

    def test_tensor_vs_ndarray(self):
        assert sb.allclose(sb.FloatTensor([1.0, 2.0]), np.array([1.0, 2.0]))


class TestAllcloseMatrixInterop:
    # Bug #100: differing-precision matrices have no shared C++ overload, so
    # allclose falls back to comparing their dense forms via numpy.
    def test_distance_matrix_float32_vs_float64_close(self):
        a = sb.DistanceMatrix(3, dtype=sb.float32)
        b = sb.DistanceMatrix(3, dtype=sb.float64)
        a[0, 1] = 1.0
        b[0, 1] = 1.0
        assert sb.allclose(a, b)

    def test_distance_matrix_float32_vs_float64_not_close(self):
        a = sb.DistanceMatrix(3, dtype=sb.float32)
        b = sb.DistanceMatrix(3, dtype=sb.float64)
        a[0, 1] = 1.0
        b[0, 1] = 2.0
        assert not sb.allclose(a, b)

    def test_symmetric_matrix_float32_vs_float64_close(self):
        a = sb.SymmetricMatrix(3, dtype=sb.float32)
        b = sb.SymmetricMatrix(3, dtype=sb.float64)
        a[0, 1] = 1.0
        b[0, 1] = 1.0
        assert sb.allclose(a, b)

    def test_symmetric_matrix_float32_vs_float64_not_close(self):
        a = sb.SymmetricMatrix(3, dtype=sb.float32)
        b = sb.SymmetricMatrix(3, dtype=sb.float64)
        a[0, 1] = 1.0
        b[0, 1] = 2.0
        assert not sb.allclose(a, b)

    def test_distance_matrix_vs_symmetric_matrix_raises(self):
        a = sb.DistanceMatrix(3, dtype=sb.float64)
        b = sb.SymmetricMatrix(3, dtype=sb.float64)
        with pytest.raises(TypeError, match="matrix type"):
            sb.allclose(a, b)
