"""Conversion of hierarchical clustering results to barcodes."""

import numpy as np

from .. import _sb_cpp as cpp
from ..typing import _validate_dtype
from .barcode import Barcode


_BACKEND_MAP = {
    np.float32: cpp.persistence.Linkage32,
    np.float64: cpp.persistence.Linkage64,
}


def linkage_to_barcode(Z: np.ndarray, *, reduced: bool = False) -> Barcode:
    """Convert a SciPy-format linkage matrix to an H0 barcode.

    Parameters
    ----------
    Z : numpy.ndarray
        An ``(n - 1, 4)`` float32 or float64 linkage matrix. Each row contains
        two active cluster indices, a finite nonnegative merge height, and
        the number of observations in the new cluster. Heights must be
        nondecreasing. The output preserves the input precision.
        An empty ``(0, 4)`` matrix represents one observation.
    reduced : bool, optional
        Omit the essential ``[0, inf)`` interval when True (default False).

    Returns
    -------
    Barcode
        Intervals born at zero and dying at the positive merge heights,
        plus one essential interval unless reduced. Zero-length bars are
        omitted. Empty input returns an empty barcode if reduced, otherwise
        one ``[0, inf)`` interval. The input is not modified.

    Raises
    ------
    TypeError
        If Z is not a float32/float64 NumPy array.
    ValueError
        If the shape, cluster references, counts, or heights are invalid.
        Non-monotone linkage matrices, including inversions from centroid
        or median linkage, are rejected rather than repaired.

    Notes
    -----
    SciPy is not required for conversion. Merge heights and tie-breaking
    results are used as supplied; clustering is not recomputed.

    Examples
    --------
    >>> Z = np.array([[0., 1., 1.5, 2.], [2., 3., 4.5, 3.]])
    >>> linkage_to_barcode(Z).to_numpy()
    array([[0. , 1.5],
           [0. , 4.5],
           [0. , inf]])
    """
    if not isinstance(Z, np.ndarray):
        raise TypeError("Z must be a float32 or float64 NumPy array")
    dtype = _validate_dtype(Z.dtype.type, _BACKEND_MAP)
    backend = _BACKEND_MAP[dtype]
    return Barcode(backend.linkage_to_barcode(Z, reduced))
