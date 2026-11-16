"""Bars that carry gradients, from the critical edges of their endpoints.

Each endpoint of a bar of persistent homology or of the homological kernel
is the distance between two points: its critical edge. C++ computes the bars
and their critical edges for every element of a batch, in parallel, and the
functions here compute the bars again as framework operations on the points
of those edges, which the framework differentiates. The points of all clouds
of a batch are one framework array, so a whole batch takes one gather per
column of endpoints. A single point cloud is a batch of one.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .. import _autodiff
from .._tracked import is_tracked
from ..point_cloud import PointCloudTensor
from .ph_tensor import BarcodeTensor, _TrackedBarcodeTensor


def as_batch(points: Any) -> PointCloudTensor:
    """Point clouds as a PointCloudTensor: a PointCloudTensor as it is, an
    (n, d) NumPy or framework array as a batch of one cloud, and a framework
    array of more dimensions as the batch of clouds it holds."""
    if isinstance(points, PointCloudTensor):
        return points
    if _autodiff._framework(points) is not None and len(points.shape) > 2:
        return PointCloudTensor(points)
    return PointCloudTensor([points])


def batch_parts(clouds: PointCloudTensor) -> tuple[Any, np.ndarray, PointCloudTensor]:
    """The points of all clouds of a batch as one (m, d) array, a framework
    array if the batch carries gradients; the first row of each cloud; and
    the batch without gradients, for C++."""
    if is_tracked(clouds):
        tracked: Any = clouds
        return tracked._tvalues, tracked._enumeration.reshape(-1, 2)[:, 0], tracked.detach()
    content, enumeration = clouds._serial()
    return content, enumeration.reshape(-1, 2)[:, 0], clouds


def _endpoints(
    adapter: Any, points: Any, endpoints: np.ndarray, first: np.ndarray, second: np.ndarray, lengths: Callable
) -> Any:
    """The endpoints of bars as a framework array that carries gradients
    with respect to the points, a framework array of shape (m, d).

    Endpoint k is the distance between the points first[k] and second[k],
    its critical edge, and is computed again as lengths(differences), a
    framework operation on the rows of differences of those points. An
    endpoint without an edge (first[k] is negative), such as a birth at 0
    in dimension 0 or an infinite death, is a constant, and so is an
    endpoint at 0, where a distance has no derivative.
    """
    has_edge = (first >= 0) & (endpoints > 0)
    edges = adapter.gather(points, first[has_edge]) - adapter.gather(points, second[has_edge])
    # The endpoints as constants, followed by the lengths of the critical
    # edges, from which each endpoint takes its own.
    values = adapter.concat([adapter.as_tensor(endpoints, like=points), lengths(edges)])
    source = np.arange(len(endpoints))
    source[has_edge] = len(endpoints) + np.arange(int(has_edge.sum()))
    return adapter.gather(values, source)


def tracked_bars(
    adapter: Any,
    births: tuple[Any, np.ndarray],
    deaths: tuple[Any, np.ndarray],
    bars: np.ndarray,
    enumeration: np.ndarray,
    edges: np.ndarray,
    lengths: Callable,
) -> BarcodeTensor:
    """The BarcodeTensor that carries gradients for the bars and critical
    edges that C++ reports for a batch.

    ``bars`` is the (m, 2) array of the bars of all barcodes, ``enumeration``
    their rows per barcode, and ``edges`` the (m, 4) array of the points
    {a, b, c, d} of each bar: its birth is the distance between a and b, and
    its death that between c and d, with -1 where an endpoint has no edge.
    ``births`` and ``deaths`` give, for each, the (points, starts) of the
    batch whose points those are, from batch_parts. The edges count the
    points of each cloud from 0, and the barcodes of a cloud follow each
    other, the same number for every cloud.
    """
    rows = enumeration.reshape(-1, 2)
    n_clouds = len(births[1])
    per_cloud = rows.reshape(n_clouds, -1, 2) if n_clouds else rows.reshape(0, 1, 2)
    bars_of_cloud = per_cloud[:, -1, 1] - per_cloud[:, 0, 0]

    columns = []
    for (points, starts), column, (first, second) in ((births, 0, (0, 1)), (deaths, 1, (2, 3))):
        # Shift the points of each cloud to its rows among all points.
        shift = np.repeat(starts, bars_of_cloud)
        columns.append(_endpoints(
            adapter, points, bars[:, column],
            np.where(edges[:, first] >= 0, edges[:, first] + shift, -1),
            np.where(edges[:, second] >= 0, edges[:, second] + shift, -1), lengths))
    return _TrackedBarcodeTensor([adapter.stack_columns(*columns)], enumeration)
