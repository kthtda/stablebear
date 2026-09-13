"""Tests for converting linkage matrices to barcodes."""

import numpy as np
import pytest

from stablebear.persistence import Barcode, linkage_to_barcode


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("reduced", [False, True])
def test_empty_linkage_matrix(dtype, reduced):
    linkage = np.empty((0, 4), dtype=dtype)

    barcode = linkage_to_barcode(linkage, reduced=reduced)

    assert isinstance(barcode, Barcode)
    assert len(barcode) == 0
    assert barcode.to_numpy().shape == (0, 2)
