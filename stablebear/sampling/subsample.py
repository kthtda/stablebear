import numpy as np

from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..base_tensor import FloatTensor, PointCloudTensor
from ..typing import float32
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


def _distance_weights(backend, R, X, distribution, np_float):
    """Weight matrix for the custom-distribution path.

    The Euclidean distances are computed once, in parallel C++ — the single
    source of truth, shared with the fused built-in path — into an
    ``(n_query, n_reference)`` matrix. The custom ``distribution`` is then
    applied to that whole matrix at once (it must act element-wise on the
    distances) to give the sampling weights.
    """
    values = np.asarray(FloatTensor(backend.distance_values(R._data, X._data)))
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

    Parameters
    ----------
    reference : array_like or FloatTensor
        The reference point cloud :math:`R`, shape ``(n_reference, dim)``.
    query : array_like or FloatTensor, optional
        The query points, shape ``(n_query, dim)``. If ``None`` (the default),
        the reference cloud is used as its own query points. When the
        distribution weights every reference point equally (e.g. a fully-open
        :class:`Uniform`), the query points do not affect the sampling
        probability, so a single query point suffices.
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
    PointCloudTensor
        Tensor of shape ``(n_query, n_instances)``; element ``[i, j]`` is the
        ``j``-th subsample (shape ``(sample_size, dim)``) for query point ``i``.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if n_instances <= 0:
        raise ValueError("n_instances must be positive.")

    if distribution is None:
        distribution = Gaussian(0.0, 1.0)

    R = _as_float_tensor(reference)
    X = R if query is None else _as_float_tensor(query, dtype=R.dtype)

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
