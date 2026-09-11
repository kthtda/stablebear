"""Synthetic linkage conversion tests; no SciPy dependency."""

import numpy as np
import pytest

from stablebear.persistence import Barcode, linkage_to_barcode


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("rows, deaths", [
    ([[0, 1, 2.5, 2]], [2.5]),
    ([[0, 1, 1.5, 2], [2, 3, 4.5, 3]], [1.5, 4.5]),
    ([[0, 1, 2, 2], [2, 3, 2, 2], [4, 5, 5, 4]], [2, 2, 5]),
    ([[0, 1, 0, 2], [2, 3, 3, 3]], [3]),
    ([[0, 1, 0, 2]], []),
    ([], []),
])
def test_synthetic_barcodes(rows, deaths, dtype, reduced):
    Z = np.array(rows, dtype=dtype).reshape(-1, 4)
    original = Z.copy()
    Z.flags.writeable = False
    bc = linkage_to_barcode(Z, reduced=reduced)
    expected = [[0, d] for d in deaths]
    if not reduced:
        expected.append([0, np.inf])
    assert isinstance(bc, Barcode)
    assert bc.to_numpy().dtype == dtype
    np.testing.assert_array_equal(bc.to_numpy(), np.array(expected).reshape(-1, 2))
    np.testing.assert_array_equal(Z, original)


def test_noncontiguous_input_and_output_ownership():
    storage = np.zeros((2, 8))
    Z = storage[:, ::2]
    Z[:] = [[0, 1, 1.5, 2], [2, 3, 4.5, 3]]
    bc = linkage_to_barcode(Z)
    Z[:, 2] = 10
    np.testing.assert_array_equal(bc.to_numpy(), [[0, 1.5], [0, 4.5], [0, np.inf]])


@pytest.mark.parametrize("value", [None, [[0, 1, 1, 2]],
    np.ones((1, 4), dtype=np.int64), np.ones((1, 4), dtype=np.float16),
    np.ones((1, 4), dtype=complex), np.ones((1, 4), dtype=object)])
def test_invalid_types(value):
    with pytest.raises(TypeError, match="float32.*float64"):
        linkage_to_barcode(value)


@pytest.mark.parametrize("shape", [(), (4,), (1, 3), (1, 5), (1, 1, 4), (0, 3)])
def test_invalid_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        linkage_to_barcode(np.zeros(shape))


@pytest.mark.parametrize("column, value", [
    (0, -1), (0, 0.5), (0, 3), (0, 1e100), (1, 0),
    (2, -1), (3, 1), (3, 2.5),
    *[(c, v) for c in range(4) for v in [np.nan, np.inf, -np.inf]],
])
def test_invalid_entries(column, value):
    Z = np.array([[0., 1., 1., 2.]])
    Z[0, column] = value
    with pytest.raises(ValueError):
        linkage_to_barcode(Z)


@pytest.mark.parametrize("rows, message", [
    ([[0, 1, 1, 2], [0, 2, 2, 2]], "active"),
    ([[0, 1, 1, 2], [2, 3, 2, 2]], "count"),
    ([[0, 1, 2, 2], [2, 3, 1, 3]], "nondecreasing"),
    ([[0, 4, 1, 2], [2, 3, 2, 3]], "existing"),
])
def test_invalid_hierarchies(rows, message):
    with pytest.raises(ValueError, match=message):
        linkage_to_barcode(np.array(rows, dtype=float))


def test_negative_strides():
    Z = np.array([[0., 1., 1.5, 2.], [2., 3., 4.5, 3.]])
    reversed_storage = Z[::-1, ::-1].copy()
    view = reversed_storage[::-1, ::-1]
    np.testing.assert_array_equal(
        linkage_to_barcode(view).to_numpy(), [[0, 1.5], [0, 4.5], [0, np.inf]])
