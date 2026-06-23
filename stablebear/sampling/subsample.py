#    Copyright 2024-2026 Bjorn Wehlin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np

from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..tensor import FloatTensor, PointCloudTensor
from ..typing import float32, float64, pcloud32, pcloud64
from .distributions import Gaussian, _BuiltinDistribution

cpp_samp = cpp.sampling

_FLOAT_TO_PCLOUD = {float32: pcloud32, float64: pcloud64}


def _get_backend(pcloud_dtype):
    if pcloud_dtype == pcloud32:
        return cpp_samp.Subsample32
    return cpp_samp.Subsample64


def _as_float_tensor(data, dtype=None):
    """Resolve a (n_points, dim) array-like or FloatTensor to a FloatTensor."""
    if isinstance(data, FloatTensor):
        if dtype is not None and data.dtype != dtype:
            return FloatTensor(np.asarray(data), dtype=dtype)
        return data
    return FloatTensor(data, dtype=dtype)


def _filter_values(filter_fn, p, reference_np):
    """Filter values of a single query point ``p`` against all reference points."""
    if isinstance(filter_fn, str):
        if filter_fn == "distance":
            return np.linalg.norm(reference_np - p, axis=1)
        raise ValueError(f"Unknown built-in filter {filter_fn!r}; expected 'distance'.")
    return np.asarray(filter_fn(p, reference_np))


def _compute_probabilities(reference_np, query_np, filter_fn, distribution, np_float):
    n_query = query_np.shape[0]
    n_ref = reference_np.shape[0]
    prob = np.empty((n_query, n_ref), dtype=np_float)

    for i in range(n_query):
        values = _filter_values(filter_fn, query_np[i], reference_np)
        weights = np.asarray(distribution(values), dtype=np_float)
        if weights.shape != (n_ref,):
            raise ValueError(
                "filter/distribution must produce one weight per reference point "
                f"(expected shape ({n_ref},), got {weights.shape})."
            )
        prob[i] = weights

    if np.any(prob < 0):
        raise ValueError("filter/distribution produced negative sampling weights.")
    if np.any(prob.sum(axis=1) <= 0):
        raise ValueError(
            "filter/distribution produced all-zero weights for at least one query point."
        )

    return prob

## TODO: Evaluate if "identity" distribution should be removed
def subsample(
    reference,
    query=None,
    *,
    sample_size,
    n_instances,
    filter_fn="distance",
    distribution=None,
    replace=True,
    generator=None,
    verbose=False,
):
    r"""Subsample a reference point cloud relative to each query point.

    This is the front end of the relative-approach pipeline of Agerberg,
    Chacholski & Ramanujam (2023). For each query point :math:`p` in *query*, a
    probability over the *reference* point cloud :math:`R` is formed by applying
    a *filter* to each pair :math:`(p, r)` and passing the result through a
    *distribution* :math:`D`,

    .. math::
        \mathrm{prob}(r) \propto D\big(\mathrm{filter}(p, r)\big),\quad r \in R,

    and ``n_instances`` subsamples of ``sample_size`` points are then drawn from
    :math:`R` according to that probability.

    The result feeds directly into the rest of the package::

        subs = subsample(reference, query, sample_size=30, n_instances=2000)
        bcs  = compute_persistent_homology(subs, max_dim=1)
        srs  = barcode_to_stable_rank(bcs)
        rel  = mean(srs, dim=1)   # one relative stable rank per query point

    Parameters
    ----------
    reference : array_like or FloatTensor
        The reference point cloud :math:`R`, shape ``(n_reference, dim)``.
    query : array_like or FloatTensor
        The query points, shape ``(n_query, dim)``.
    sample_size : int
        Number of points per subsample (``s`` in the paper).
    n_instances : int
        Number of subsamples drawn per query point (``n`` in the paper).
    filter_fn : str or callable, optional
        Either the built-in ``"distance"`` (Euclidean :math:`\lVert p - r\rVert`,
        evaluated in parallel C++), or a callable ``filter_fn(p, reference) -> (n_reference,)``
        mapping a single query point and the reference array to per-reference
        values. By default ``"distance"``.
    distribution : distribution spec or callable, optional
        A built-in spec (:class:`Gaussian`, :class:`Identity`, :class:`Uniform`)
        or a callable
        ``distribution(values) -> (n_reference,)`` returning non-negative
        weights. By default :class:`Gaussian` with ``mean=0``, ``sigma=1``.
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

    pcloud_dtype = _FLOAT_TO_PCLOUD[R.dtype]
    backend = _get_backend(pcloud_dtype)
    gen = generator._gen if generator is not None else None

    if filter_fn == "distance" and isinstance(distribution, _BuiltinDistribution):
        task, result = distribution._fused_call(
            backend, R._data, X._data, sample_size, n_instances, replace, gen
        )
    else:
        np_float = np.float32 if R.dtype == float32 else np.float64
        prob = _compute_probabilities(
            np.asarray(R), np.asarray(X), filter_fn, distribution, np_float
        )
        prob_tensor = FloatTensor(prob, dtype=R.dtype)
        task, result = backend.sample_subsets_from_probabilities(
            R._data, prob_tensor._data, sample_size, n_instances, replace, gen
        )

    _run_task(lambda: task, verbose=verbose)
    return PointCloudTensor(result)
