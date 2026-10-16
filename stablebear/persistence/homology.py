from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..distance_matrix import (
    DistanceMatrix,
    DistanceMatrixTensor,
)
from ..base_tensor import FloatTensor
from ..point_cloud import PointCloud, PointCloudTensor
from ..typing import barcode32, barcode64, distmat32, distmat64, float32, float64, pcloud32, pcloud64
from .ph_tensor import BarcodeTensor

if TYPE_CHECKING:
    from ..timeseries import TimeSeries

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
) -> BarcodeTensor:
    r"""Compute persistent homology of a point cloud or distance matrix.

    Returns barcodes for homology dimensions 0 through ``max_dim``. When
    ``complex_type`` is ``ComplexType.VietorisRips`` (currently the only
    available option), the computation uses Ripser :footcite:`Bauer2021`. When the input
    contains multiple point clouds or distance matrices, the computations are
    parallelized across them.

    For an input tensor of shape :math:`(d_1, \ldots, d_n)`, the output has
    shape :math:`(d_1, \ldots, d_n, \texttt{max_dim} + 1)`.

    For a :class:`~stablebear.timeseries.TimeSeries` of point clouds or
    distance matrices, the result is a series of barcodes with the same channels, times, and unit.
    A channel of observation shape ``shape`` gives barcodes of shape
    ``(*shape, max_dim + 1)``. A series of one observation keeps its time
    axis.

    Parameters
    ----------
    X : PointCloud, PointCloudTensor, DistanceMatrix, DistanceMatrixTensor, FloatTensor, numpy.ndarray, or TimeSeries
        Input data. A ``FloatTensor`` or NumPy array is
        interpreted as a single point cloud (one row per point).
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
        A tensor of barcodes, or a series of barcodes for a series input.

    """

    from ..timeseries.series import TimeSeries
    from .ripser import (
        _compute_barcodes_distmat_ripser,
        _compute_barcodes_euclidean_pcloud_ripser,
    )

    if complex_type != ComplexType.VietorisRips:
        raise ValueError(f"Unsupported complex type {complex_type}")

    if isinstance(X, TimeSeries):
        return _series_persistent_homology(X, max_dim, distance_type, reduced, verbose)

    if isinstance(X, (DistanceMatrix, DistanceMatrixTensor)):
        task = _compute_barcodes_distmat_ripser(X, max_dim, reduced)
    else:
        if isinstance(X, (np.ndarray, FloatTensor)):
            X = PointCloud(X)
        match distance_type:
            case DistanceType.Euclidean:
                task = _compute_barcodes_euclidean_pcloud_ripser(
                    X, max_dim, reduced
                )
            case _:
                raise ValueError(
                    f"Distance type {distance_type} not supported for complex type {complex_type}."
                )

    _run_task(lambda: task, verbose=verbose)
    out = BarcodeTensor(task.result())

    # A single cloud or matrix gives shape (max_dim + 1,) directly. A tensor
    # with one element still loses its leading axis, as before.
    if len(out.shape) == 2 and out.shape[0] == 1:
        out = out[0, :]

    return out


def _series_persistent_homology(series, max_dim, distance_type, reduced, verbose):
    from ..timeseries.series import TimeSeries

    if series.dtype in (distmat32, distmat64):
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
            "Persistent homology of a time series needs point clouds or distance "
            f"matrices, got {series.dtype}"
        )
    _run_task(lambda: task, verbose=verbose)
    return TimeSeries._from_cpp(task.result())
