import numpy as np

from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..base_tensor import FloatTensor, IntTensor, PointCloudTensor
from ..distance_matrix import DistanceMatrix, DistanceMatrixTensor
from ..typing import float32, uint64
from .distributions import Gaussian, Uniform

cpp_samp = cpp.sampling

# The only accepted distributions. Each owns a fully fused C++ fast path
# (distances + weighting + draw) for both point-cloud and distance-matrix
# references; arbitrary callables are intentionally not supported.
_BUILTIN_DISTRIBUTIONS = (Gaussian, Uniform)


def _backend(dtype):
    """The precision-specific C++ subsampling backend for a FloatTensor dtype."""
    return cpp_samp.Subsample32 if dtype == float32 else cpp_samp.Subsample64


def _as_float_tensor(data, dtype=None):
    """Resolve a (n_points, dim) array-like or FloatTensor to a FloatTensor."""
    if isinstance(data, FloatTensor):
        if dtype is not None and data.dtype != dtype:
            return FloatTensor(np.asarray(data), dtype=dtype)
        return data
    return FloatTensor(data, dtype=dtype)


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
    return DistanceMatrix(reference._data._get_element([0] * reference.ndim))


def _query_is_indices(query):
    """Whether ``query`` is a 1-D integer array of reference indices (vs 2-D
    coordinates). ``None`` (all reference points) is not indices."""
    if query is None:
        return False
    arr = np.asarray(query)
    return arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer)


def _reference_indices(query, n):
    """Validate a 1-D ``query`` as reference row indices (``None`` means all
    ``n`` rows) and return a uint64 ndarray."""
    if query is None:
        return np.arange(n, dtype=np.uint64)
    q = np.ascontiguousarray(np.asarray(query).ravel(), dtype=np.uint64)
    if q.size == 0:
        raise ValueError("query must contain at least one index.")
    if np.any(q >= n):
        raise ValueError(
            f"query indices must be valid reference rows (0 <= index < {n})."
        )
    return q


def _subsample_distmat(reference, query, *, sample_size, n_instances, distribution,
                       replace, generator, verbose):
    """Distance-matrix path of :func:`subsample_relative` (see its docstring)."""
    source = _as_distance_matrix(reference)
    if query is not None and not _query_is_indices(query):
        raise ValueError(
            "a distance matrix has no coordinates; query must be a 1-D integer "
            "array of reference row indices (or None for all rows)."
        )
    q = IntTensor(_reference_indices(query, source.size), dtype=uint64)
    backend = _backend(source.dtype)
    gen = generator._gen if generator is not None else None

    # Fully fused C++ draw: precomputed distances + built-in distribution.
    task, result = distribution._sample_subsets_distmat(
        backend, source._data, q._data, sample_size, n_instances, replace, gen
    )
    _run_task(lambda: task, verbose=verbose)
    return DistanceMatrixTensor(result)


def _subsample_pointcloud(reference, query, *, sample_size, n_instances, distribution,
                          replace, generator, verbose):
    """Point-cloud path of :func:`subsample_relative` (see its docstring)."""
    R = _as_float_tensor(reference)
    if query is None:
        X = R
    elif _query_is_indices(query):
        # Query points selected by their order in the reference cloud.
        idx = _reference_indices(query, R.shape[0])
        X = _as_float_tensor(np.asarray(R)[idx], dtype=R.dtype)
    else:
        X = _as_float_tensor(query, dtype=R.dtype)
        if X.ndim != 2:
            raise ValueError(
                "query must be a 2-D (n_query, dim) array of coordinates or a 1-D "
                "integer array of reference indices."
            )

    if R.shape[1] != X.shape[1]:
        raise ValueError("reference and query must have the same dimension.")

    backend = _backend(R.dtype)
    gen = generator._gen if generator is not None else None

    # Fully fused C++ draw: distances + built-in distribution.
    task, result = distribution._sample_subsets(
        backend, R._data, X._data, sample_size, n_instances, replace, gen
    )
    _run_task(lambda: task, verbose=verbose)
    return PointCloudTensor(result)


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

    This is the front end of the relative-approach pipeline of Agerberg,
    Chacholski & Ramanujam (2023). For each query point :math:`p` in *query*, a
    probability over the *reference* point cloud :math:`R` is formed from the
    Euclidean distance :math:`\lVert p - r\rVert` to each reference point,
    passed through a *distribution* :math:`D`,

    .. math::
        \mathrm{prob}(r) \propto D\big(\lVert p - r\rVert\big),\quad r \in R,

    and ``n_instances`` subsamples of ``sample_size`` points are then drawn from
    :math:`R` according to that probability.

    The result feeds directly into the rest of the package::

        subs = subsample_relative(reference, query, sample_size=30, n_instances=2000)
        bcs  = compute_persistent_homology(subs, max_dim=1)
        srs  = barcode_to_stable_rank(bcs)
        rel  = mean(srs, dim=1)   # one relative stable rank per query point

    The reference may also be a precomputed distance matrix
    (:class:`~stablebear.DistanceMatrix` / :class:`~stablebear.DistanceMatrixTensor`)
    instead of a point cloud. The distances are then read straight from the
    matrix (symmetric, zero-diagonal) rather than computed from coordinates, the
    query points are *row indices* into it, and each subsample is the principal
    sub-distance-matrix over the drawn points — returned as a
    :class:`~stablebear.DistanceMatrixTensor`, which feeds the same persistence
    pipeline. At scale, pass a native ``DistanceMatrix`` to avoid materializing a
    dense ``(n, n)`` array.

    Parameters
    ----------
    reference : array_like, FloatTensor, DistanceMatrix, or DistanceMatrixTensor
        The reference point cloud :math:`R`, shape ``(n_reference, dim)``, or a
        precomputed distance matrix over the reference points.
    query : array_like, optional
        The points to view the reference from, in one of two forms:

        * a **1-D integer array of reference indices** — view from those
          reference points, selected by their order (e.g. ``[1, 3, 10]``). Works
          for both point-cloud and distance-matrix references.
        * a **2-D ``(n_query, dim)`` array of coordinates** — view from arbitrary
          vantage points (e.g. a grid, centroids, landmarks). Point-cloud
          references only; a distance matrix has no coordinates.

        If ``None`` (the default), every reference point is used as a query
        point. When the distribution weights every reference point equally (e.g.
        a fully-open :class:`Uniform`), the query does not affect the sampling
        probability, so a single query suffices.
    sample_size : int
        Number of points per subsample (``s`` in the paper).
    n_instances : int
        Number of subsamples drawn per query point (``n`` in the paper).
    distribution : Gaussian or Uniform, optional
        A built-in distribution spec (:class:`Gaussian` or :class:`Uniform`)
        mapping distances to non-negative sampling weights, applied on the fully
        fused C++ fast path. By default :class:`Gaussian` with ``mean=0``,
        ``sigma=1``.
    replace : bool, optional
        Whether each subsample is drawn with replacement, by default True
        (as in the paper).
    generator : Generator, optional
        Random number generator. If ``None``, the global generator is used.
    verbose : bool, optional
        If True, display a progress bar while the subsamples are drawn and allow
        cooperative cancellation (e.g. via ``KeyboardInterrupt``). By default
        False.

    Returns
    -------
    PointCloudTensor or DistanceMatrixTensor
        Tensor of shape ``(n_query, n_instances)``; element ``[i, j]`` is the
        ``j``-th subsample for query point ``i`` — a point cloud of shape
        ``(sample_size, dim)`` for point-cloud input, or a
        ``(sample_size, sample_size)`` sub-distance-matrix for distance-matrix
        input.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if n_instances <= 0:
        raise ValueError("n_instances must be positive.")
    if distribution is None:
        distribution = Gaussian(0.0, 1.0)
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
    # Pointcloud input: 
    return _subsample_pointcloud(
        reference, query, sample_size=sample_size, n_instances=n_instances,
        distribution=distribution, replace=replace, generator=generator, verbose=verbose,
    )
