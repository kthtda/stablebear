import warnings
from enum import Enum
from typing import Any, overload

import numpy as np

from .. import _autodiff
from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..distance_matrix import (
    DistanceMatrix,
    DistanceMatrixTensor,
)
from ..base_tensor import FloatTensor
from ..linear import DistanceSpace, _distance_matrices, _points_array
from ..point_cloud import PointCloud, PointCloudTensor
from ..timeseries.series import TimeSeries, TimeSeriesTensor
from ..typing import barcode32, barcode64, distmat32, distmat64, float32, float64, pcloud32, pcloud64
from .ph_tensor import BarcodeTensor, _TrackedBarcodeTensor

cpp_p = cpp.persistence

_DISTMAT_TO_BARCODE_DTYPE = {distmat32: barcode32, distmat64: barcode64}
_PCLOUD_TO_BARCODE_DTYPE = {pcloud32: barcode32, pcloud64: barcode64}
_FLOAT_TO_PCLOUD_DTYPE = {float32: pcloud32, float64: pcloud64}
_FLOAT_TO_BARCODE_DTYPE = {float32: barcode32, float64: barcode64}


class DistanceType(Enum):
    Euclidean = 1


class ComplexType(Enum):
    VietorisRips = 1


@overload
def compute_persistent_homology(
    X: TimeSeries,
    max_dim: int = 1,
    distance_type: DistanceType | None = None,
    complex_type: ComplexType = ComplexType.VietorisRips,
    reduced: bool = False,
    verbose: bool = False,
) -> TimeSeries: ...


@overload
def compute_persistent_homology(
    X: PointCloud | PointCloudTensor | DistanceMatrix | DistanceMatrixTensor | FloatTensor | np.ndarray | DistanceSpace,
    max_dim: int = 1,
    distance_type: DistanceType | None = None,
    complex_type: ComplexType = ComplexType.VietorisRips,
    reduced: bool = False,
    verbose: bool = False,
) -> BarcodeTensor: ...


def _tracked_persistent_homology(X: Any, max_dim: int, reduced: bool) -> BarcodeTensor:
    """The barcodes of one (n, d) point cloud given as a framework array,
    with bars that carry gradients with respect to its points.

    In a Vietoris-Rips complex, a simplex appears at the length of its
    longest edge, so each endpoint of a bar is the distance between two
    points: the critical edge of the simplex that creates or destroys the
    class. Ripser reports these edges with the bars, and the bars are their
    lengths again as framework operations on the points, which the
    framework differentiates. An endpoint without an edge, a birth at 0 in
    dimension 0 or an infinite death, is a constant, and so is an endpoint
    at 0, where a distance has no derivative.
    """
    X = _autodiff._unwrap(X)
    adapter = _autodiff._framework(X)
    if len(X.shape) != 2:
        raise NotImplementedError(
            "compute_persistent_homology with gradients supports a single point cloud of shape (n, d) only.")
    points = adapter.concrete(X)
    if points is None:
        raise NotImplementedError(
            "compute_persistent_homology of points that JAX or tf.function traces abstractly is not supported: "
            "the number of bars depends on their values.")

    dtype = np.float32 if points.dtype == np.float32 else np.float64
    backend = cpp_p.PersistenceRipser32 if dtype == np.float32 else cpp_p.PersistenceRipser64
    content, enumeration, critical_edges = backend.persistence_critical_edges(
        PointCloud(points.astype(dtype))._cpp_point_cloud(), max_dim, reduced)

    # The endpoints in the order birth, death of each bar, with the two
    # points of the edge of each.
    endpoints = content.reshape(-1)
    first, second = critical_edges.reshape(-1, 2).T
    has_edge = (first >= 0) & (endpoints > 0)
    edges = adapter.gather(X, first[has_edge]) - adapter.gather(X, second[has_edge])
    # The endpoints as constants, followed by the lengths of the critical
    # edges, from which each endpoint takes its own.
    values = adapter.concat([adapter.as_tensor(endpoints, like=X), adapter.row_norms(edges)])
    source = np.arange(len(endpoints))
    source[has_edge] = len(endpoints) + np.arange(int(has_edge.sum()))
    bars = adapter.stack_columns(adapter.gather(values, source[0::2]), adapter.gather(values, source[1::2]))
    return _TrackedBarcodeTensor([bars], enumeration)


def _persistent_homology_in_space(space: DistanceSpace, max_dim: int, reduced: bool, verbose: bool) -> Any:
    """The barcodes of the points of a DistanceSpace in its distance."""
    transformed = space.distance._euclidean_points(_points_array(space))
    if transformed is None:
        return compute_persistent_homology(
            _distance_matrices(space), max_dim=max_dim, reduced=reduced, verbose=verbose)
    # The Euclidean distances of the transformed points are the distances
    # of the space.
    return compute_persistent_homology(transformed, max_dim=max_dim, reduced=reduced, verbose=verbose)


def compute_persistent_homology(
    X: PointCloud
    | PointCloudTensor
    | DistanceMatrix
    | DistanceMatrixTensor
    | FloatTensor
    | np.ndarray
    | TimeSeries
    | DistanceSpace,
    max_dim: int = 1,
    distance_type: DistanceType | None = None,
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
    X : PointCloud, PointCloudTensor, DistanceMatrix, DistanceMatrixTensor, FloatTensor, numpy.ndarray, TimeSeries, or DistanceSpace
        Input data. A ``FloatTensor`` or NumPy array holds point clouds in
        its last two axes, one point per row; a 2-D one is a single cloud.
        Distances between the points of a cloud are Euclidean. A
        :class:`~stablebear.linear.DistanceSpace` holds a point cloud with
        another distance, which is then used. A ``DistanceMatrix`` or
        ``DistanceMatrixTensor`` provides precomputed pairwise distances
        directly.
    max_dim : int, optional
        Maximum homology dimension to compute, by default 1.
    distance_type : DistanceType, optional
        Deprecated: put the points in a
        :class:`~stablebear.linear.DistanceSpace` to choose their distance.
        The only value, ``DistanceType.Euclidean``, is the distance of
        plain point clouds. Passing it gives a ``DeprecationWarning``.
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

    Raises
    ------
    TypeError
        If ``distance_type`` is passed with a ``DistanceSpace``.
    NotImplementedError
        If ``X`` is a PyTorch, JAX, or TensorFlow array that is not a single
        point cloud of shape ``(n, d)``, or that JAX or ``tf.function``
        traces: the number of bars depends on the points.

    Notes
    -----
    If ``X`` is a PyTorch, JAX, or TensorFlow array of shape ``(n, d)``, a
    single point cloud, the bars of the result carry gradients with respect
    to the points. Each finite, positive endpoint of a bar is the distance
    between two points, the longest edge of the simplex that creates or
    destroys the class, and its gradient is that of this distance. An
    endpoint at 0 or at infinity has gradient 0. Where a simplex has several
    longest edges, the gradient is that of one of them.

    """

    from .ripser import (
        _compute_barcodes_distmat_ripser,
        _compute_barcodes_euclidean_coordinates_ripser,
        _compute_barcodes_euclidean_pcloud_ripser,
    )

    if complex_type != ComplexType.VietorisRips:
        raise ValueError(f"Unsupported complex type {complex_type}")

    if distance_type is None:
        distance_type = DistanceType.Euclidean
    elif isinstance(X, DistanceSpace):
        raise TypeError("distance_type cannot be passed with a DistanceSpace, which has its own distance.")
    else:
        warnings.warn(
            "distance_type is deprecated: put the points in a DistanceSpace to choose their distance.",
            DeprecationWarning, stacklevel=2)

    if isinstance(X, DistanceSpace):
        return _persistent_homology_in_space(X, max_dim, reduced, verbose)

    if _autodiff._tracked(X) is not None:
        if distance_type != DistanceType.Euclidean:
            raise ValueError(
                f"Distance type {distance_type} not supported for complex type {complex_type}."
            )
        return _tracked_persistent_homology(X, max_dim, reduced)

    if isinstance(X, TimeSeries):
        return _series_persistent_homology(X, max_dim, distance_type, reduced, verbose)
    if isinstance(X, TimeSeriesTensor):
        raise TypeError("A tensor of time series, such as windows, is not a set of point clouds; "
                        "make it one with stablebear.timeseries.to_point_clouds")

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
