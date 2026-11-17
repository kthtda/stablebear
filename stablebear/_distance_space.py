"""DistanceSpace, the base class of what holds points with a distance.

It is part of stablebear.geometry, and is defined here because the point
cloud classes, which that module uses, are distance spaces themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .typing import Dtype, pcloud32, pcloud64

if TYPE_CHECKING:
    from .geometry import Distance


class DistanceSpace:
    """Points together with the distance of their space.

    An operation that uses a distance takes a ``DistanceSpace`` in place of
    the plain points and uses its distance. It is usually made by calling a
    :class:`Distance` on the points.

    A ``PointCloud`` and a ``PointCloudTensor`` are themselves distance
    spaces, of their own points with the Euclidean distance, ``LpDistance(2)``.

    Parameters
    ----------
    points : PointCloud, FloatTensor, or numpy.ndarray
        A point cloud of shape ``(n, d)``, one point per row, which may be a
        PyTorch tensor, or a batch of point clouds as a
        ``PointCloudTensor`` or a time series of point clouds.
    distance : Distance
        The distance between the points.

    Attributes
    ----------
    dtype : Dtype
        The dtype of what holds the points, such as ``pcloud64``.

    Raises
    ------
    ValueError
        If the distance does not apply to points of dimension ``d``.
    """

    dtype: Dtype

    def __init__(self, points: Any, distance: Distance):
        from .geometry import Distance, _as_point_cloud, _dim, _is_float32
        from .point_cloud import PointCloudTensor
        from .timeseries.series import TimeSeries

        if not isinstance(distance, Distance):
            raise TypeError(f"Expected a Distance, got {type(distance).__name__}.")
        cloud = _as_point_cloud(points)
        dim = _dim(cloud)
        if dim is not None:
            distance._check_dim(dim)
        self._points = cloud
        self._distance = distance
        # An attribute, as that of a tensor, which the point cloud classes set.
        if isinstance(cloud, (PointCloudTensor, TimeSeries)):
            self.dtype = cloud.dtype
        else:
            self.dtype = pcloud32 if _is_float32(cloud) else pcloud64

    @property
    def points(self) -> Any:
        """The points, as a ``PointCloud``, as the PyTorch tensor that they were given as, or as the
        ``PointCloudTensor`` or time series of a batch."""
        return self._points

    @property
    def distance(self) -> Distance:
        """The distance of the space."""
        return self._distance

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._points!r}, {self._distance!r})"


# The class belongs to stablebear.geometry, which documents it.
DistanceSpace.__module__ = "stablebear.geometry"

class EuclideanPoints(DistanceSpace):
    """What makes a class of points the DistanceSpace of its own points
    with the Euclidean distance: PointCloud and PointCloudTensor."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Each class of points defines its own constructor."""

    @property
    def points(self) -> Any:
        """The points, which are the object itself."""
        return self

    @property
    def distance(self) -> Distance:
        """The Euclidean distance, ``LpDistance(2)``, which is the distance
        between the points."""
        from .geometry import LpDistance

        return LpDistance(2)
