"""Tests for converting linkage matrices to barcodes."""

import numpy as np
import pytest

from stablebear.persistence import Barcode, linkage_to_barcode


def assert_linkage_barcode(linkage, expected_reduced, *, reduced):
    bars = list(expected_reduced)
    if not reduced:
        bars.append([0, np.inf])
    expected = Barcode(np.array(bars, dtype=linkage.dtype).reshape(-1, 2))

    barcode = linkage_to_barcode(linkage, reduced=reduced)
    assert isinstance(barcode, Barcode)
    assert barcode.is_isomorphic_to(expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("reduced", [False, True])
class TestLinkageToBarcode:
    def test_single_observation(self, dtype, reduced):
        # SciPy rejects a single observation; we extend its n - 1 row convention
        # by accepting an empty linkage matrix to represent one observation.
        linkage = np.empty((0, 4), dtype=dtype)
        expected = []

        assert_linkage_barcode(linkage, expected, reduced=reduced)

    def test_single_row(self, dtype, reduced):
        linkage = np.array([[0, 1, 1.0, 2]], dtype=dtype)
        expected = [[0, 1.0]]

        assert_linkage_barcode(linkage, expected, reduced=reduced)
