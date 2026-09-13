"""Tests for converting linkage matrices to barcodes."""

import numpy as np
import pytest

from stablebear.persistence import Barcode, linkage_to_barcode


def expected_barcode(finite_bars, *, reduced, dtype):
    bars = list(finite_bars)
    if not reduced:
        bars.append([0, np.inf])
    return np.array(bars, dtype=dtype).reshape(-1, 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("reduced", [False, True])
class TestLinkageToBarcode:
    def test_single_observation(self, dtype, reduced):
        # SciPy rejects a single observation; we extend its n - 1 row convention
        # by accepting an empty linkage matrix to represent one observation.
        linkage = np.empty((0, 4), dtype=dtype)

        barcode = linkage_to_barcode(linkage, reduced=reduced)

        assert isinstance(barcode, Barcode)
        expected = expected_barcode([], reduced=reduced, dtype=dtype)
        assert len(barcode) == len(expected)
        np.testing.assert_array_equal(barcode.to_numpy(), expected)


    def test_single_row(self, dtype, reduced):
        linkage = np.array([[0, 1, 1.0, 2]], dtype=dtype)

        barcode = linkage_to_barcode(linkage, reduced=reduced)

        assert isinstance(barcode, Barcode)
        expected = expected_barcode([[0, 1.0]], reduced=reduced, dtype=dtype)
        assert len(barcode) == len(expected)
        np.testing.assert_array_equal(barcode.to_numpy(), expected)
