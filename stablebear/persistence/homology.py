from enum import Enum

import numpy as np

from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..distance_matrix import (
    DistanceMatrix,
    DistanceMatrixTensor,
)
from ..base_tensor import FloatTensor
from ..point_cloud import PointCloud, PointCloudTensor
from ..timeseries.series import TimeSeries
from ..typing import barcode32, barcode64, distmat32, distmat64, float32, float64, pcloud32, pcloud64
from .ph_tensor import BarcodeTensor

cpp_p = cpp.persistence

_DISTMAT_TO_BARCODE_DTYPE = {distmat32: barcode32, distmat64: barcode64}
_PCLOUD_TO_BARCODE_DTYPE = {pcloud32: barcode32, pcloud64: barcode64}
_FLOAT_TO_PCLOUD_DTYPE = {float32: pcloud32, float64: pcloud64}
_FLOAT_TO_BARCODE_DTYPE = {float32: barcode32, float64: barcode64}


class DistanceType(Enum):
    Euclidean = 1


class ComplexType(Enum):
    VietorisRips = 1


def compute_persistent_homology(
    X: PointCloud
    | PointCloudTensor
    | DistanceMatrix
    | DistanceMatrixTensor
    | FloatTensor
    | np.ndarray
    | TimeSeries,
    max_dim: int = 1,
    distance_type: DistanceType = DistanceType.Euclidean,
    complex_type: ComplexType = ComplexType.VietorisRips,
    reduced: bool = False,
    verbose: bool = False,
) -> BarcodeTensor | TimeSeries:
    r"""Compute persistent homology of a point cloud or distance matrix.

    Returns barcodes for homology dimensions 0 through ``max_dim``. When
    ``complex_type`` is ``ComplexType.VietorisRips`` (currently the only
    available option), the computation uses Ripser :footcite:`Bauer2021`. When the input
    contains multiple point clouds or distance matrices, the computations are
    parallelized across them.

    Each point cloud or distance matrix gives ``max_dim + 1`` barcodes, one
    per degree, so the result adds a last dimension of length
    ``max_dim + 1``. Float coordinates are read as point clouds from their
    last two dimensions, ``(n, d)`` for ``n`` points in ``d`` dimensions, one
    point per row; the new dimension replaces those two.

    Parameters
    ----------
    X : PointCloud, PointCloudTensor, DistanceMatrix, DistanceMatrixTensor, FloatTensor, numpy.ndarray, or TimeSeries
        Input data. A ``FloatTensor`` or NumPy array holds point clouds in
        its last two axes, one point per row; a 2-D one is a single cloud.
        A ``DistanceMatrix`` or ``DistanceMatrixTensor`` provides
        precomputed pairwise distances directly; ``distance_type`` is
        ignored in that case.
    max_dim : int, optional
        Maximum homology dimension to compute, by default 1.
    distance_type : DistanceType, optional
        Distance metric for point cloud input, by default
        ``DistanceType.Euclidean``. Ignored when the input is a distance
        matrix.
    complex_type : ComplexType, optional
        Simplicial complex type, by default ``ComplexType.VietorisRips``.
    reduced : bool, optional
        If ``True``, compute reduced homology (no essential H0 class).
        If ``False`` (default), an infinite bar ``[0, inf)`` is added to
        H0 representing the single connected component.
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    BarcodeTensor or TimeSeries
        The barcodes, in the same kind of container as ``X``.

    """

    from .ripser import (
        _compute_barcodes_distmat_ripser,
        _compute_barcodes_euclidean_coordinates_ripser,
        _compute_barcodes_euclidean_pcloud_ripser,
    )

    if complex_type != ComplexType.VietorisRips:
        raise ValueError(f"Unsupported complex type {complex_type}")

    if isinstance(X, TimeSeries):
        return _series_persistent_homology(X, max_dim, distance_type, reduced, verbose)

    if isinstance(X, (DistanceMatrix, DistanceMatrixTensor)):
        task = _compute_barcodes_distmat_ripser(X, max_dim, reduced)
    else:
        if distance_type != DistanceType.Euclidean:
            raise ValueError(
                f"Distance type {distance_type} not supported for complex type {complex_type}."
            )
        if isinstance(X, (np.ndarray, FloatTensor)):
            # The trailing (n, d) axes are point clouds, one point per row.
            task = _compute_barcodes_euclidean_coordinates_ripser(
                FloatTensor(X), max_dim, reduced
            )
        else:
            task = _compute_barcodes_euclidean_pcloud_ripser(X, max_dim, reduced)

    _run_task(lambda: task, verbose=verbose)
    out = BarcodeTensor(task.result())

    # A single cloud or matrix gives shape (max_dim + 1,) directly. A tensor
    # with one element still loses its leading axis, as before.
    if len(out.shape) == 2 and out.shape[0] == 1:
        out = out[0, :]

    return out


def _series_persistent_homology(series, max_dim, distance_type, reduced, verbose):
    from .ripser import _compute_barcodes_euclidean_coordinates_ripser

    if series.dtype in (float32, float64):
        # As for a FloatTensor, the trailing (n, d) axes of each observation
        # are point clouds, one point per row.
        shapes = [series.shape] if series.n_channels == 1 else list(series.shape)
        if any(len(shape) < 2 for shape in shapes):
            raise TypeError(
                "Persistent homology of a numeric time series needs observations "
                f"of shape (..., n, d), one point per row, got shape {series.shape}"
            )
        if distance_type != DistanceType.Euclidean:
            raise ValueError(f"Distance type {distance_type} not supported for time series")
        task = _compute_barcodes_euclidean_coordinates_ripser(series, max_dim, reduced)
    elif series.dtype in (distmat32, distmat64):
        # Distances are given, so distance_type is ignored, as for tensors.
        backend = cpp_p.PersistenceRipser32 if series.dtype is distmat32 else cpp_p.PersistenceRipser64
        task = backend.spawn_ripser_distmat_task(series._data, max_dim, reduced)
    elif series.dtype in (pcloud32, pcloud64):
        if distance_type != DistanceType.Euclidean:
            raise ValueError(f"Distance type {distance_type} not supported for time series")
        backend = cpp_p.PersistenceRipser32 if series.dtype is pcloud32 else cpp_p.PersistenceRipser64
        task = backend.spawn_ripser_pcloud_euclidean_task(series._data, max_dim, reduced)
    else:
        raise TypeError(
            "Persistent homology of a time series needs point clouds, distance "
            f"matrices, or float matrices, got {series.dtype}"
        )
    _run_task(lambda: task, verbose=verbose)
    return TimeSeries._from_cpp(task.result())
