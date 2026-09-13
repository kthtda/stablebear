"""Tests for converting linkage matrices to barcodes."""

import numpy as np
import pytest

from stablebear.persistence import Barcode, linkage_to_barcode


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("reduced", [False, True])
def test_single_observation(dtype, reduced):
    # SciPy rejects a single observation; we extend its n - 1 row convention
    # by accepting an empty linkage matrix to represent one observation.
    linkage = np.empty((0, 4), dtype=dtype)

    barcode = linkage_to_barcode(linkage, reduced=reduced)

    assert isinstance(barcode, Barcode)
    expected = np.empty((0, 2), dtype=dtype) if reduced else np.array([[0, np.inf]], dtype=dtype)
    assert len(barcode) == len(expected)
    np.testing.assert_array_equal(barcode.to_numpy(), expected)
