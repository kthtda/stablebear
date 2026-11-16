"""Bars that carry gradients, from the critical edges of their endpoints.

Each endpoint of a bar of persistent homology or of the homological kernel
is the distance between two points: its critical edge. C++ computes the bars
and their critical edges for every element of a batch, in parallel, and the
functions here compute the bars again as framework operations on the points
of those edges, or on the entries of a distance matrix, which the framework
differentiates. The points of all clouds of a batch are one framework array,
so a whole batch takes one gather per column of endpoints. A single point
cloud is a batch of one.
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


def point_distances(lengths: Callable) -> Callable:
    """How tracked_bars measures an edge between points: ``lengths`` of the
    difference of its two points, rows of the (m, d) array of all points."""
    def measure(adapter, points, first, second, shift):
        return lengths(adapter.gather(points, first + shift) - adapter.gather(points, second + shift))

    return measure


def matrix_entries(adapter: Any, entries: Any, first: np.ndarray, second: np.ndarray, shift: np.ndarray) -> Any:
    """How tracked_bars measures an edge of a distance matrix: its entry,
    among the entries below the diagonal of all matrices by rows, as
    DistanceMatrixTensor.to_serial_content gives them."""
    high, low = np.maximum(first, second), np.minimum(first, second)
    return adapter.gather(entries, shift + high * (high - 1) // 2 + low)


def _endpoints(
    adapter: Any, source: Any, endpoints: np.ndarray, first: np.ndarray, second: np.ndarray, shift: np.ndarray,
    measure: Callable,
) -> Any:
    """The endpoints of bars as a framework array that carries gradients
    with respect to ``source``, a framework array of the points of all
    clouds or of the entries of all matrices.

    Endpoint k is the distance between the points first[k] and second[k] of
    its element, its critical edge, and is computed again as
    ``measure(adapter, source, first, second, shift)``, a framework
    operation, where shift[k] is the first row of the element in the
    source. An endpoint without an edge (first[k] is negative), such as a
    birth at 0 in dimension 0 or an infinite death, is a constant, and so is
    an endpoint at 0, where a distance has no derivative.
    """
    has_edge = (first >= 0) & (endpoints > 0)
    measured = measure(adapter, source, first[has_edge], second[has_edge], shift[has_edge])
    # The endpoints as constants, followed by the measured edges, from which
    # each endpoint takes its own.
    values = adapter.concat([adapter.as_tensor(endpoints, like=source), measured])
    index = np.arange(len(endpoints))
    index[has_edge] = len(endpoints) + np.arange(int(has_edge.sum()))
    return adapter.gather(values, index)


def tracked_bars(
    adapter: Any,
    births: tuple[Any, np.ndarray],
    deaths: tuple[Any, np.ndarray],
    bars: np.ndarray,
    enumeration: np.ndarray,
    edges: np.ndarray,
    measure: Callable,
) -> BarcodeTensor:
    """The BarcodeTensor that carries gradients for the bars and critical
    edges that C++ reports for a batch.

    ``bars`` is the (m, 2) array of the bars of all barcodes, ``enumeration``
    their rows per barcode, and ``edges`` the (m, 4) array of the points
    {a, b, c, d} of each bar: its birth is the distance between a and b, and
    its death that between c and d, with -1 where an endpoint has no edge.
    ``births`` and ``deaths`` give, for each, the (source, starts) of the
    batch whose distances those are: the points of all its clouds with the
    first row of each, from batch_parts, or the entries of all its distance
    matrices with the first entry of each. ``measure`` computes the
    distances from the source: point_distances(lengths) or matrix_entries.
    The edges count the points of each element from 0, and the barcodes of
    an element follow each other, the same number for every element.
    """
    rows = enumeration.reshape(-1, 2)
    n_elements = len(births[1])
    per_element = rows.reshape(n_elements, -1, 2) if n_elements else rows.reshape(0, 1, 2)
    bars_of_element = per_element[:, -1, 1] - per_element[:, 0, 0]

    columns = []
    for (source, starts), column, (first, second) in ((births, 0, (0, 1)), (deaths, 1, (2, 3))):
        columns.append(_endpoints(
            adapter, source, bars[:, column], edges[:, first], edges[:, second],
            np.repeat(starts, bars_of_element), measure))
    return _TrackedBarcodeTensor([adapter.stack_columns(*columns)], enumeration)
