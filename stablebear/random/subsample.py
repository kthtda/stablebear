from .. import _sb_cpp as cpp
from .._validation import _boolean, _positive_integer
from ..base_tensor import _get_backend
from ..distance_matrix import DistanceMatrix, DistanceMatrixTensor
from ..point_cloud import PointCloud, PointCloudTensor
from ..tensor_create import zeros
from ..typing import distmat32, distmat64, float32, pcloud32, pcloud64
from .generator import _unwrap


def _partial_policy(value):
    """Return the native policy for ``allow_partial``; bools map to "no"/"keep"."""
    if type(value) is bool:
        value = "keep" if value else "no"
    if not isinstance(value, str):
        raise TypeError("allow_partial must be a string")
    choices = ("no", "keep", "drop")
    if value not in choices:
        options = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"allow_partial must be one of {options}")
    return getattr(cpp.point_process._PartialPolicy, value)


def subsample(
    data,
    n_points,
    n_samples=1,
    *,
    replace=True,
    allow_partial="keep",
    discard_duplicates=False,
    generator=None,
):
    """Draw uniform subsamples from point clouds, distance matrices, or their tensors.

    One sample axis is appended to the input tensor shape. Each input value is
    copied once and shared by its samples, which store only their selected
    point indices. A distance-matrix sample is the compressed symmetric
    principal submatrix induced by those indices.

    Parameters
    ----------
    data : PointCloud, DistanceMatrix, PointCloudTensor, or DistanceMatrixTensor
        Input value or tensor. A single point cloud or distance matrix is
        treated as a zero-dimensional tensor.
    n_points : int
        Number of draws requested for each output point cloud.
    n_samples : int, optional
        Number of output point clouds per input point cloud, by default 1.
    replace : bool, optional
        Whether a logical input row may be drawn repeatedly, by default True. The
        ``allow_partial="drop"`` threshold still applies when True.
    allow_partial : {"no", "keep", "drop"}, optional
        Policy when fewer than ``n_points`` input points are available.
        ``"no"`` raises when the requested draw cannot be filled.
        ``"keep"`` (default) returns all available points without replacement, or an
        empty sample when no points are eligible. With replacement, both policies
        fill the requested size whenever at least one point is eligible.
        ``"drop"`` returns an empty sample when fewer than ``n_points`` points
        are eligible, even with replacement. The check precedes drawing and duplicate removal.
        For compatibility, False means ``"no"`` and True means ``"keep"``.
    discard_duplicates : bool, optional
        For point clouds, keep only the first coordinate-identical drawn point.
        Equality is numeric: signed zeros and same-sign infinities compare
        equal; opposite-sign infinities are distinct, and NaNs do not compare equal.
        Points containing NaN are always retained, including repeated draws
        of the same source point.
        For distance matrices, keep only the first occurrence of each drawn
        source index. Distinct zero-distance indices remain distinct.
    generator : stablebear.random.Generator, optional
        Generator to use. The process-wide generator is used when omitted.

    Returns
    -------
    PointCloudTensor or DistanceMatrixTensor
        A tensor of the same element family and precision, with shape equal to
        the input tensor shape plus (n_samples,). For a single point cloud or
        distance matrix, the result has shape (n_samples,).
    """
    if isinstance(data, PointCloud):
        wrapped = zeros((), dtype=pcloud32 if data.dtype == float32 else pcloud64)
        wrapped[()] = data
        data = wrapped
    elif isinstance(data, DistanceMatrix):
        wrapped = zeros((), dtype=distmat32 if data.dtype == float32 else distmat64)
        wrapped[()] = data
        data = wrapped
    if not isinstance(data, (PointCloudTensor, DistanceMatrixTensor)):
        raise TypeError(
            "data must be a PointCloud, DistanceMatrix, PointCloudTensor, or DistanceMatrixTensor")
    n_points = _positive_integer(n_points, "n_points")
    n_samples = _positive_integer(n_samples, "n_samples")
    replace = _boolean(replace, "replace")
    partial_policy = _partial_policy(allow_partial)
    discard_duplicates = _boolean(discard_duplicates, "discard_duplicates")

    backend, data = _get_backend(
        data,
        {
            pcloud32: cpp.point_process.subsample32,
            pcloud64: cpp.point_process.subsample64,
            distmat32: cpp.point_process.subsample32,
            distmat64: cpp.point_process.subsample64,
        },
    )
    result = backend(
        data._data,
        n_points,
        n_samples,
        replace,
        partial_policy,
        discard_duplicates,
        _unwrap(generator),
    )
    result_type = (
        PointCloudTensor if isinstance(data, PointCloudTensor)
        else DistanceMatrixTensor
    )
    return result_type(result)
