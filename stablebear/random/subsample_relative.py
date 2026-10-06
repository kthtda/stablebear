"""Relative subsampling of one reference dataset."""

import warnings

import numpy as np

from .. import _sb_cpp as cpp
from .._validation import _boolean, _positive_integer
from ..async_task import _run_task
from ..base_tensor import FloatTensor, IntTensor
from ..distance_matrix import DistanceMatrix, DistanceMatrixTensor
from ..distributions import Distribution, Gaussian
from ..point_cloud import PointCloud, PointCloudTensor
from ..typing import float32, uint64
from .generator import _unwrap
from .subsample import _partial_policy


def _query_input(query, reference):
    if query is None:
        return None
    if isinstance(query, (PointCloudTensor, DistanceMatrixTensor, DistanceMatrix)):
        raise TypeError("query must be one point cloud, a coordinate array, or an index vector")
    if isinstance(query, IntTensor):
        return query._data
    if not isinstance(query, (PointCloud, FloatTensor)):
        array = np.asarray(query)
        if array.ndim == 1:
            if array.size == 0:
                indices = IntTensor([], dtype=uint64)
            elif np.issubdtype(array.dtype, np.integer):
                indices = IntTensor(array)
            else:
                raise TypeError("query indices must be integers")
            return indices._data
        if array.ndim != 2:
            raise ValueError("query must be a 2-D coordinate array or 1-D index vector")
        query = array
    if isinstance(reference, DistanceMatrix):
        raise ValueError("a distance-matrix reference requires query indices, not coordinates")
    if not isinstance(query, PointCloud) or query.dtype != reference.dtype:
        query = PointCloud(np.asarray(query), dtype=reference.dtype)
    return query._cpp_point_cloud()


def subsample_relative(
    reference,
    query=None,
    *,
    n_points,
    n_samples=1,
    distribution=None,
    replace=True,
    allow_partial="keep",
    discard_duplicates=False,
    generator=None,
    verbose=False,
):
    """Draw distance-weighted subsamples relative to each query point.

    Implements the relative construction of Agerberg, Chachólski, and
    Ramanujam :footcite:`Agerberg2023`, with additional sampling options.

    A reference point is eligible for a query and distribution when its
    sampling weight is positive. Eligibility is counted by reference index,
    before drawing and duplicate removal.

    Parameters
    ----------
    reference : PointCloud or DistanceMatrix
        One source dataset. Coordinates must be finite. Infinite matrix
        distances give zero sampling weight for every distribution.
    query : PointCloud, array_like, FloatTensor, IntTensor, or None, optional
        A coordinate array of shape ``(n_query, dimension)``, or a 1-D integer
        vector of reference indices, including uint32/uint64 tensors.
        Coordinates must be finite. Signed negative indices count from the
        end. ``None`` uses every reference point. Distance-matrix references
        require index queries.
    n_points : int
        Positive number of draws requested per sample.
    n_samples : int, optional
        Positive number of samples per query and distribution, by default 1.
    distribution : Distribution or list of Distribution, optional
        Distribution applied to distances, or a nonempty list of distributions.
        Defaults to ``Gaussian(0, 1)``. A list adds a distribution axis in
        list order.
    replace : bool, optional
        Allow repeated indices within a sample, by default True.
    allow_partial : {"no", "keep", "drop"}, optional
        Policy when fewer than ``n_points`` reference points are eligible:

        * ``"keep"`` (default): without replacement, return all eligible
          points in sampled order; with replacement, fill the requested size
          if any point is eligible. Return an empty sample if none are eligible.
        * ``"no"``: raise ``ValueError`` if the requested size cannot be filled.
          With replacement, one eligible point suffices.
        * ``"drop"``: return an empty sample if fewer than ``n_points`` points
          are eligible, even with replacement.

        For compatibility, False means ``"no"`` and True means ``"keep"``.
    discard_duplicates : bool, optional
        Remove duplicate coordinates (point clouds) or source indices (matrices)
        after drawing, keeping the first occurrence without redraws.
        Defaults to False. This can shorten samples even when
        ``allow_partial="no"``.
    generator : Generator, optional
        Use the global generator when omitted. Each output cell has its own
        random stream, independent of worker scheduling.
    verbose : bool, optional
        Show progress and warn about empty samples by query and distribution,
        by default False. Does not affect sampling results.

    Returns
    -------
    PointCloudTensor or DistanceMatrixTensor
        Indexed tensor of shape ``(n_query, n_samples)`` for one distribution,
        or ``(n_query, n_distributions, n_samples)`` for a list, including a
        one-element list. Matrix samples are principal submatrices in drawn
        index order. Empty queries preserve these axes with ``n_query=0``.
        Samples preserve the reference precision. Later changes to the
        reference do not affect them.
    """
    if not isinstance(reference, (PointCloud, DistanceMatrix)):
        raise TypeError("reference must be one PointCloud or DistanceMatrix")
    n_points = _positive_integer(n_points, "n_points")
    n_samples = _positive_integer(n_samples, "n_samples")
    replace = _boolean(replace, "replace")
    partial_policy = _partial_policy(allow_partial)
    discard_duplicates = _boolean(discard_duplicates, "discard_duplicates")
    verbose = _boolean(verbose, "verbose")
    if distribution is None:
        distribution = Gaussian()
    distribution_axis = isinstance(distribution, list)
    distributions = distribution if distribution_axis else [distribution]
    if not distributions:
        raise ValueError("distribution list must not be empty")
    if not all(isinstance(d, Distribution) for d in distributions):
        raise TypeError("distribution must be a Distribution or a list of them")
    native_query = _query_input(query, reference)
    cloud = isinstance(reference, PointCloud)
    kind = "pcloud" if cloud else "distmat"
    suffix = "32" if reference.dtype == float32 else "64"
    spawn = getattr(cpp.point_process, f"subsample_relative_{kind}{suffix}")
    native_reference = reference._cpp_point_cloud() if cloud else reference._current_matrix()
    task = spawn(
        native_reference, native_query, [d._native for d in distributions],
        n_points, n_samples, distribution_axis, replace, partial_policy,
        discard_duplicates, _unwrap(generator),
    )
    _run_task(lambda: task, verbose=verbose)
    result_type = PointCloudTensor if cloud else DistanceMatrixTensor
    result = result_type(task.result())
    if verbose:
        empty = task.empty_regions()
        if empty:
            warnings.warn(
                f"Empty samples at (query, distribution) indices: {empty}",
                UserWarning, stacklevel=2,
            )
    return result
