import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb
import stablebear.persistence as pers


def test_assign_tensor_to_slice_1d():
    X0 = sb.random.noisy_sin((10,))
    X1 = sb.random.noisy_sin((10,))

    A = sb.zeros((2, 10))

    A[0, :] = X0
    A[1, :] = X1

    assert A[0, :].array_equal(X0)
    assert A[1, :].array_equal(X1)


def test_assign_tensor_to_slice_nd():
    X0 = sb.random.noisy_sin((10, 10))
    X1 = sb.random.noisy_sin((10, 20))

    A = sb.zeros((10, 30))

    A[:, 0:10] = X0
    A[:, 10:30] = X1

    assert A[:, 0:10].array_equal(X0)
    assert A[:, 10:30].array_equal(X1)


def test_assign_tensor_to_slice_incommensurate_dims():
    X0 = sb.random.noisy_sin((10, 10))

    A = sb.zeros((10, 30))

    with pytest.raises(ValueError):
        A[:, 0:5] = X0


def test_assign_wrong_dtype_raises_error():
    X = sb.random.noisy_sin((10, 10))

    with pytest.raises(TypeError):
        X[2, 3] = 0.5


# --- Bare-slice assignment ---


def _check_broadcast_assign(np_target, np_rhs, slices):
    """Perform the same broadcast assignment on both numpy and stablebear, assert results match."""
    np_a = np_target.copy()
    np_a[slices] = np_rhs

    sb_a = sb.FloatTensor(np_target.copy())
    sb_rhs = sb.FloatTensor(np_rhs)
    sb_a[slices] = sb_rhs

    npt.assert_array_equal(np.asarray(sb_a), np_a)


def test_assign_bare_slice():
    _check_broadcast_assign(
        np.zeros((4, 3)),
        np.array([[1.0, 2.0, 3.0]]),
        slice(1, 3),
    )


# --- Broadcast assignment ---


def test_broadcast_assign_row_to_2d_float():
    """Assign (4,) row to every row of a (3, 4) slice."""
    _check_broadcast_assign(
        np.zeros((3, 4)),
        np.array([1.0, 2.0, 3.0, 4.0]),
        (slice(None), slice(None)),
    )


def test_broadcast_assign_col_to_2d_float():
    """Assign (3, 1) column to fill a (3, 4) slice."""
    _check_broadcast_assign(
        np.zeros((3, 4)),
        np.array([[10.0], [20.0], [30.0]]),
        (slice(None), slice(None)),
    )


def test_broadcast_assign_scalar_tensor_float():
    """Assign (1,) tensor to fill a (5,) slice."""
    _check_broadcast_assign(
        np.zeros((5,)),
        np.array([42.0]),
        slice(None),
    )


def test_broadcast_assign_to_slice_float():
    """Broadcast into a slice, not the whole tensor."""
    _check_broadcast_assign(
        np.zeros((4, 3)),
        np.array([7.0, 8.0, 9.0]),
        (slice(1, 3), slice(None)),
    )


def test_broadcast_assign_incompatible_raises():
    """Assigning (3,) to (4, 2) should fail — not broadcast-compatible."""
    A = sb.FloatTensor(np.zeros((4, 2)))
    rhs = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        A[:, :] = rhs


def test_broadcast_assign_would_expand_target_raises():
    """RHS bigger than target — broadcast shape would differ from target."""
    A = sb.FloatTensor(np.zeros((3,)))
    rhs = sb.FloatTensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    with pytest.raises(ValueError):
        A[:] = rhs


def test_broadcast_assign_pcf():
    """Broadcast a single PCF across a row of a 2D PCF tensor."""
    pcf = sb.random.noisy_sin((1,))

    A = sb.zeros((3,), dtype=sb.pcf32)
    A[:] = pcf

    for i in range(3):
        assert A[i] == pcf[0]


def test_broadcast_assign_barcode():
    """Broadcast a single barcode row across a barcode tensor."""
    bc = pers.Barcode(np.array([[0.0, 1.0], [0.5, 2.0]]))

    row = sb.zeros((1,), dtype=sb.barcode64)
    row[0] = bc

    A = sb.zeros((3,), dtype=sb.barcode64)
    A[:] = row

    for i in range(3):
        assert A[i].is_isomorphic_to(bc)
