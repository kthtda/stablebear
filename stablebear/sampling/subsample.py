import numpy as np

from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..base_tensor import FloatTensor, IntTensor, PointCloudTensor
from ..distance_matrix import DistanceMatrix, DistanceMatrixTensor
from ..typing import float32, uint64
from .distributions import Gaussian, Uniform

cpp_samp = cpp.sampling

# Built-in distributions own a fully fused C++ "distance" fast path. For any
# other (custom callable) distribution the distance still runs in C++; only the
# distribution itself is applied in Python before the draw.
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


def _weights_from_distribution(distribution, values, np_float):
    """Apply a custom ``distribution`` element-wise to the ``(n_query,
    n_reference)`` value matrix and validate the resulting sampling weights."""
    weights = np.asarray(distribution(values), dtype=np_float)
    if weights.shape != values.shape:
        raise ValueError(
            "distribution must produce one weight per reference point "
            f"(expected shape {values.shape}, got {weights.shape})."
        )
    if np.any(weights < 0):
        raise ValueError("distribution produced negative sampling weights.")
    if np.any(weights.sum(axis=1) <= 0):
        raise ValueError(
            "distribution produced all-zero weights for at least one query point."
        )
    return weights


def _distance_weights(backend, R, X, distribution, np_float):
    """Weight matrix for the custom-distribution path (point-cloud reference).

    The Euclidean distances are computed once, in parallel C++ — the single
    source of truth, shared with the fused built-in path — then the custom
    ``distribution`` is applied to the whole ``(n_query, n_reference)`` matrix.
    """
    values = np.asarray(FloatTensor(backend.distance_values(R._data, X._data)))
    return _weights_from_distribution(distribution, values, np_float)


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


def _distance_matrix_weights(backend, source, query, distribution, np_float):
    """Weight matrix for the custom-distribution path (distance-matrix reference).

    The per-query distance rows ``source(query[qi], j)`` are gathered once in
    parallel C++, then the custom ``distribution`` is applied to the whole
    ``(n_query, n_reference)`` matrix.
    """
    values = np.asarray(FloatTensor(backend.distance_matrix_values(source._data, query._data)))
    return _weights_from_distribution(distribution, values, np_float)


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

    if isinstance(distribution, _BUILTIN_DISTRIBUTIONS):
        task, result = distribution._sample_subsets_distmat(
            backend, source._data, q._data, sample_size, n_instances, replace, gen
        )
    else:
        np_float = np.float32 if source.dtype == float32 else np.float64
        weights = _distance_matrix_weights(backend, source, q, distribution, np_float)
        weights_tensor = FloatTensor(weights, dtype=source.dtype)
        task, result = backend.sample_subsets_from_probabilities_distmat(
            source._data, weights_tensor._data, sample_size, n_instances, replace, gen
        )

    _run_task(lambda: task, verbose=verbose)
    return DistanceMatrixTensor(result)


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
    distribution : distribution spec or callable, optional
        A built-in spec (:class:`Gaussian`, :class:`Uniform`) or a callable
        mapping distances to non-negative weights. The callable is applied
        element-wise to the whole ``(n_query, n_reference)`` matrix of distances
        and must return an array of the same shape (so any NumPy element-wise
        expression works). By default :class:`Gaussian` with ``mean=0``,
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

    # Distance-matrix input: weight by precomputed distances, return sub-matrices.
    if isinstance(reference, (DistanceMatrix, DistanceMatrixTensor)):
        return _subsample_distmat(
            reference, query, sample_size=sample_size, n_instances=n_instances,
            distribution=distribution, replace=replace, generator=generator, verbose=verbose,
        )

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

    if isinstance(distribution, _BUILTIN_DISTRIBUTIONS):
        # Fully fused C++ draw: distances + built-in distribution.
        task, result = distribution._sample_subsets(
            backend, R._data, X._data, sample_size, n_instances, replace, gen
        )
    else:
        # Custom distribution: the distances are computed in C++, the
        # distribution is applied in Python, then the draw runs in C++.
        np_float = np.float32 if R.dtype == float32 else np.float64
        weights = _distance_weights(backend, R, X, distribution, np_float)
        weights_tensor = FloatTensor(weights, dtype=R.dtype)
        task, result = backend.sample_subsets_from_probabilities(
            R._data, weights_tensor._data, sample_size, n_instances, replace, gen
        )

    _run_task(lambda: task, verbose=verbose)
    return PointCloudTensor(result)
