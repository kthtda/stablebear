import operator
import warnings

import numpy as np

from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..base_tensor import FloatTensor, IntTensor, PointCloudTensor, _get_backend
from ..distance_matrix import DistanceMatrix, DistanceMatrixTensor
from ..point_cloud import PointCloud
from ..random import _unwrap
from ..typing import (
    distmat32,
    distmat64,
    float32,
    float64,
    pcloud32,
    pcloud64,
    uint64,
)
from .distributions import Gaussian, Uniform
from ..tensor_create import zeros

cpp_samp = cpp.sampling

# The only accepted distributions. Each owns a fully fused C++ fast path
# (distances + weighting + draw) for both point-cloud and distance-matrix
# references; arbitrary callables are intentionally not supported.
_BUILTIN_DISTRIBUTIONS = (Gaussian, Uniform)

_SUBSAMPLE_BACKEND_MAP = {
    float32: cpp_samp.Subsample32,
    float64: cpp_samp.Subsample64,
}


def _as_float_tensor(data, dtype=None):
    """Resolve a (n_points, dim) array-like or FloatTensor to a FloatTensor."""
    if isinstance(data, FloatTensor):
        if dtype is not None and data.dtype != dtype:
            return FloatTensor(np.asarray(data), dtype=dtype)
        return data
    return FloatTensor(data, dtype=dtype)


def _resolve_reference_cloud(reference):
    """Resolve a point-cloud reference to a validated 2-D :class:`FloatTensor`.

    A :class:`PointCloud` (including a subsample's indexed view) or a
    single-cloud :class:`PointCloudTensor` materializes to its coordinates.
    These, like an explicit ``FloatTensor``, are unambiguously points, so only a
    raw square ``(n, n)`` array is screened by the square-array ambiguity guard.
    """
    explicit_cloud = isinstance(reference, (FloatTensor, PointCloud, PointCloudTensor))
    if isinstance(reference, PointCloudTensor):
        if reference.size != 1:
            raise ValueError(
                "point-cloud subsampling needs a single reference cloud, but got a "
                f"PointCloudTensor with {reference.size} clouds."
            )
        # Reduce to the lone cloud: a PointCloud (rank >= 1) or, for a 0-d
        # tensor, already a FloatTensor of its coordinates.
        reference = reference[(0,) * reference.ndim]
    if isinstance(reference, PointCloud):
        reference = reference.materialize()
    cloud = _as_float_tensor(reference)
    if cloud.ndim != 2:
        raise ValueError(
            "reference must be a 2-D (n_reference, dim) point cloud, but got "
            f"{cloud.ndim} dimension(s)."
        )
    if not explicit_cloud:
        _reject_ambiguous_square_reference(cloud)
    return cloud


def _as_distance_matrix(reference):
    """Resolve a distance-matrix reference to a single :class:`DistanceMatrix`."""
    if isinstance(reference, DistanceMatrix):
        return reference
    # DistanceMatrixTensor: must hold exactly one reference matrix.
    if reference.size != 1:
        raise ValueError(
            "distance-matrix subsampling needs a single reference matrix, but got a "
            f"DistanceMatrixTensor with {reference.size} matrices."
        )
    return reference[(0,) * reference.ndim]


def _reject_ambiguous_square_reference(reference_cloud):
    """Raise if an implicit point-cloud reference is a square (n, n) array.

    A square array is ambiguous: n points in n dimensions, or a precomputed
    distance matrix over n points. Reading a distance matrix as a point cloud
    silently computes something different (Euclidean distances *between the
    matrix rows*) at O(n^3) cost, so square references must declare their
    meaning via one of the two wrappers named in the error.
    """
    n_points, dim = reference_cloud.shape
    if n_points != dim:
        return
    raise ValueError(
        f"reference is a square ({n_points}, {n_points}) array, which is "
        "ambiguous: it could be a point cloud or a precomputed distance "
        "matrix. Wrap it in stablebear.DistanceMatrix.from_dense(...) to "
        "subsample by the stored distances, or in stablebear.FloatTensor(...) "
        "to subsample its rows as points."
    )


def _query_is_indices(query):
    """Whether ``query`` is a 1-D integer array of reference indices (vs 2-D
    coordinates). ``None`` (all reference points) is not indices."""
    if query is None:
        return False
    arr = np.asarray(query)
    return arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer)


def _reference_indices(query, n):
    """Validate a 1-D ``query`` as reference row indices (``None`` means all
    ``n`` rows; negative indices count from the end, as in NumPy) and return
    a uint64 ndarray."""
    if query is None:
        return np.arange(n, dtype=np.uint64)
    arr = np.asarray(query).ravel()
    # astype copies, so the in-place shift below cannot touch the caller's array.
    q = arr.astype(np.int64)
    if q.size == 0:
        raise ValueError("query must contain at least one index.")
    # Only signed input gets the negative-index treatment. A huge unsigned
    # value comes out negative from the cast (e.g. uint64 underflow); it must
    # fail the range check below, as in NumPy — not wrap to a valid row.
    if np.issubdtype(arr.dtype, np.signedinteger):
        q[q < 0] += n
    if np.any((q < 0) | (q >= n)):
        raise ValueError(
            f"query indices must be valid reference rows (-{n} <= index < {n}, "
            "negative indices count from the end)."
        )
    return np.ascontiguousarray(q, dtype=np.uint64)


def _element_is_empty(element):
    """Whether a drawn subsample (point cloud or distance matrix) holds zero points."""
    if isinstance(element, DistanceMatrix):
        return element.size == 0
    return element.shape[0] == 0


def _warn_empty_queries(result):
    """Warn once if any query's region held no eligible reference points.

    Callers gate this on ``verbose`` — quiet runs proceed silently with the
    empty subsamples. Emptiness is a per-query property (an all-zero weight
    row empties every instance of that query), so checking one instance per
    query suffices.
    """
    empty = [q for q in range(result.shape[0])
             if _element_is_empty(result[q, 0])]
    if empty:
        shown = ", ".join(map(str, empty[:10])) + (", ..." if len(empty) > 10 else "")
        warnings.warn(
            f"{len(empty)} query point(s) had no reference points with positive "
            f"sampling weight; their subsamples are empty (query indices: {shown}).",
            stacklevel=4,  # helper -> path helper -> subsample_relative -> user code
        )
    return result


def _subsample_distmat(reference, query, *, sample_size, n_instances, distribution,
                       replace, generator, verbose):
    """Distance-matrix path of :func:`subsample_relative` (see its docstring)."""
    source = _as_distance_matrix(reference)
    if query is not None and not _query_is_indices(query):
        raise ValueError(
            "a distance matrix has no coordinates; query must be a 1-D integer "
            "array of reference row indices (or None for all rows)."
        )
    query_indices = IntTensor(_reference_indices(query, source.size), dtype=uint64)
    backend, _ = _get_backend(source, _SUBSAMPLE_BACKEND_MAP)

    # The task fills `out` in place (allocated to (n_query, n_instances) when it
    # runs); pass a placeholder and read it back, as the Ripser pipeline does.
    out = zeros((1,), dtype=distmat32 if source.dtype == float32 else distmat64)

    # Fully fused C++ draw: precomputed distances + built-in distribution.
    task = backend.spawn_subsample_distmat_task(
        source._data, query_indices._data, out._data,
        distribution._native(backend), sample_size, n_instances, replace,
        _unwrap(generator)
    )
    _run_task(lambda: task, verbose=verbose)

    if verbose:
        return _warn_empty_queries(out)
    return out


def _subsample_pointcloud(reference, query, *, sample_size, n_instances, distribution,
                          replace, generator, verbose):
    """Point-cloud path of :func:`subsample_relative` (see its docstring)."""
    # A PointCloud or single-cloud PointCloudTensor (e.g. a subsample fed back
    # in) resolves to a 2-D FloatTensor of coordinates.
    reference_cloud = _resolve_reference_cloud(reference)

    # None or a 1-D integer array selects reference rows (the C++ task reads
    # their coordinates from the reference); a 2-D array gives query coordinates.
    query_is_indices = query is None or _query_is_indices(query)
    if query_is_indices:
        query_tensor = IntTensor(_reference_indices(query, reference_cloud.shape[0]), dtype=uint64)
    else:
        query_tensor = _as_float_tensor(query, dtype=reference_cloud.dtype)
        if query_tensor.ndim != 2:
            raise ValueError(
                "query must be a 2-D (n_query, dim) array of coordinates or a 1-D "
                "integer array of reference indices."
            )
        if reference_cloud.shape[1] != query_tensor.shape[1]:
            raise ValueError("reference and query must have the same dimension.")

    backend, _ = _get_backend(reference_cloud, _SUBSAMPLE_BACKEND_MAP)

    # The task fills `out` in place (allocated to (n_query, n_instances) when it
    # runs); pass a placeholder and read it back, as the Ripser pipeline does.
    out = zeros((1,), dtype=pcloud32 if reference_cloud.dtype == float32 else pcloud64)

    # Fully fused C++ draw: distances + built-in distribution.
    spawn = (backend.spawn_subsample_pcloud_index_query_task if query_is_indices
             else backend.spawn_subsample_pcloud_task)
    task = spawn(
        reference_cloud._data, query_tensor._data, out._data, backend.Euclidean(),
        distribution._native(backend), sample_size, n_instances, replace,
        _unwrap(generator)
    )
    _run_task(lambda: task, verbose=verbose)
    if verbose:
        return _warn_empty_queries(out)
    return out


def subsample_relative(
    reference,
    query=None,
    *,
    sample_size,
    n_instances,
    distribution=None,
    replace=True,
    generator=None,
    verbose=False,
):
    r"""Subsample a reference point cloud relative to each query point.

    Front end of the relative-approach pipeline of Agerberg, Chacholski &
    Ramanujam (2023). For each query point :math:`p`, each reference point
    :math:`r` is weighted by :math:`D(\lVert p - r\rVert)` for the chosen
    distribution :math:`D`, and ``n_instances`` subsamples of up to
    ``sample_size`` points are drawn with those weights. See :doc:`the
    subsampling guide </sampling>` for background, examples, and figures.

    The reference may also be a precomputed distance matrix
    (:class:`~stablebear.DistanceMatrix` or a one-element
    :class:`~stablebear.DistanceMatrixTensor`): distances are then read from
    the matrix, the query points are row indices into it, and each subsample
    is the principal sub-distance-matrix over the drawn points.

    Parameters
    ----------
    reference : array_like, FloatTensor, PointCloud, PointCloudTensor, DistanceMatrix, or DistanceMatrixTensor
        The reference point cloud, shape ``(n_reference, dim)``, or a
        precomputed distance matrix over the reference points. A
        :class:`~stablebear.PointCloud` (or a one-element
        :class:`~stablebear.PointCloudTensor`) is taken as a point cloud —
        so a subsample can be fed straight back in as the reference. A raw
        square ``(n, n)`` array is ambiguous between a point cloud and a
        distance matrix and is rejected: wrap it in
        :class:`~stablebear.DistanceMatrix` (or a one-element
        :class:`~stablebear.DistanceMatrixTensor`) to subsample by stored
        distances, or in :class:`~stablebear.FloatTensor` to subsample its
        rows as points.
    query : array_like, optional
        A 1-D integer array of reference indices (negative indices count from
        the end, as in NumPy), or a 2-D ``(n_query, dim)`` array of
        coordinates; coordinates require a point-cloud reference. If ``None``
        (the default), every reference point is used as a query point.
    sample_size : int
        Maximum number of points per subsample. A subsample is smaller only
        when it runs out of eligible points: drawing without replacement with
        fewer than ``sample_size`` positively weighted reference points, or a
        query point with no positively weighted reference points at all
        (empty subsamples; reported per query when ``verbose=True``).
    n_instances : int
        Number of subsamples drawn per query point.
    distribution : Gaussian or Uniform, optional
        Maps distances to non-negative sampling weights. By default
        :class:`Gaussian` with ``mean=0``, ``sigma=1``.
    replace : bool, optional
        Whether to draw with replacement, by default True (as in the paper).
    generator : Generator, optional
        Random number generator. If ``None``, the global generator is used.
    verbose : bool, optional
        If True, display a progress bar, allow cooperative cancellation
        (e.g. via ``KeyboardInterrupt``), and warn (:class:`UserWarning`)
        about query points whose subsamples came out empty. By default False.

    Returns
    -------
    PointCloudTensor or DistanceMatrixTensor
        Tensor of shape ``(n_query, n_instances)``; element ``[i, j]`` is the
        ``j``-th subsample for query point ``i`` — a point cloud of at most
        ``sample_size`` points, or the sub-distance-matrix over the drawn
        points for distance-matrix input.
    """
    sample_size = operator.index(sample_size)
    n_instances = operator.index(n_instances)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if n_instances <= 0:
        raise ValueError("n_instances must be positive.")
    if distribution is None:
        distribution = Gaussian(mean=0.0, sigma=1.0)
    if not isinstance(distribution, _BUILTIN_DISTRIBUTIONS):
        raise ValueError(
            "distribution must be a built-in distribution "
            f"({' or '.join(d.__name__ for d in _BUILTIN_DISTRIBUTIONS)}) or None."
        )

    # Distance-matrix input: weight by precomputed distances, return sub-matrices.
    if isinstance(reference, (DistanceMatrix, DistanceMatrixTensor)):
        return _subsample_distmat(
            reference, query, sample_size=sample_size, n_instances=n_instances,
            distribution=distribution, replace=replace, generator=generator, verbose=verbose,
        )
    # Point-cloud input: weight by Euclidean distances computed from coordinates.
    return _subsample_pointcloud(
        reference, query, sample_size=sample_size, n_instances=n_instances,
        distribution=distribution, replace=replace, generator=generator, verbose=verbose,
    )
