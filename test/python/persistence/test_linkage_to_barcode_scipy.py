"""Integration coverage for actual SciPy clustering results."""

import numpy as np
import pytest
from scipy.cluster import hierarchy

from stablebear.persistence import linkage_to_barcode


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
@pytest.mark.parametrize("points", [
    [[0.], [1.], [4.]],
    [[0.], [0.], [1.], [1.]],
    [[0.], [2.]],
    np.random.default_rng(221).normal(size=(20, 3)),
])
@pytest.mark.parametrize("reduced", [False, True])
def test_scipy_linkage(method, points, reduced):
    Z = hierarchy.linkage(points, method=method)
    original = Z.copy()
    bc = linkage_to_barcode(Z, reduced=reduced)
    deaths = Z[Z[:, 2] > 0, 2]
    if not reduced:
        deaths = np.append(deaths, np.inf)
    np.testing.assert_array_equal(bc.to_numpy(), np.column_stack((np.zeros(len(deaths)), deaths)))
    np.testing.assert_array_equal(Z, original)
