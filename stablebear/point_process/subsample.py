import operator

from .. import _sb_cpp as cpp
from ..base_tensor import PointCloudTensor
from ..random import Generator, _unwrap
from ..typing import pcloud32


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def subsample(
    points,
    n_points,
    n_samples=1,
    *,
    replace=False,
    allow_partial=False,
    discard_duplicates=False,
    generator=None,
):
    """Draw uniform subsamples from every cloud in a point-cloud tensor.

    One sample axis is appended to the input tensor shape. Coordinates are
    copied once per input cloud and shared by that cloud's samples; samples
    store only their selected row indices.

    Parameters
    ----------
    points : PointCloudTensor
        Input tensor. A zero-dimensional tensor represents one point cloud.
    n_points : int
        Number of draws requested for each output cloud.
    n_samples : int, optional
        Number of output clouds per input cloud, by default 1.
    replace : bool, optional
        Whether a logical input row may be drawn repeatedly.
    allow_partial : bool, optional
        Permit fewer draws when an input is too small without replacement, or
        an empty output cloud when sampling an empty input with replacement.
    discard_duplicates : bool, optional
        Keep only the first drawn occurrence of each coordinate-identical
        point, without redrawing.
    generator : stablebear.random.Generator, optional
        Generator to use. The process-wide generator is used when omitted.

    Returns
    -------
    PointCloudTensor
        A tensor with shape equal to points.shape plus (n_samples,).
    """
    if not isinstance(points, PointCloudTensor):
        raise TypeError("points must be a PointCloudTensor")
    n_points = _positive_integer(n_points, "n_points")
    n_samples = _positive_integer(n_samples, "n_samples")
    replace = _boolean(replace, "replace")
    allow_partial = _boolean(allow_partial, "allow_partial")
    discard_duplicates = _boolean(discard_duplicates, "discard_duplicates")
    if generator is not None and not isinstance(generator, Generator):
        raise TypeError("generator must be a stablebear.random.Generator or None")

    backend = cpp.point_process.subsample32 if points.dtype == pcloud32 else cpp.point_process.subsample64
    return PointCloudTensor(
        backend(
            points._data,
            n_points,
            n_samples,
            replace,
            allow_partial,
            discard_duplicates,
            _unwrap(generator),
        )
    )
