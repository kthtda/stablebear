from typing import Any

import numpy as np

from .. import _autodiff
from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..distance_matrix import (
    DistanceMatrix,
    DistanceMatrixTensor,
)
from ..base_tensor import FloatTensor, _get_backend
from ..linear import DistanceSpace, _distance_matrices, _in_one_framework, _points_array
from ..point_cloud import PointCloud, PointCloudTensor
from ..typing import distmat32, distmat64, float32, float64, pcloud32, pcloud64
from .homology import (
    _DISTMAT_TO_BARCODE_DTYPE,
    _FLOAT_TO_BARCODE_DTYPE,
    _FLOAT_TO_PCLOUD_DTYPE,
    _PCLOUD_TO_BARCODE_DTYPE,
)
from ._critical_edges import as_batch, batch_parts, tracked_bars
from .ph_tensor import BarcodeTensor

cpp_p = cpp.persistence

_FLOAT_TO_DISTMAT_DTYPE = {float32: distmat32, float64: distmat64}


def _spawn_homological_kernel_pcloud_task(
    X: PointCloud | PointCloudTensor,
    Y: PointCloud | PointCloudTensor,
    out: BarcodeTensor,
):
    backend, X = _get_backend(
        X,
        {
            float32: cpp_p.HomologicalKernel32,
            float64: cpp_p.HomologicalKernel64,
            pcloud32: cpp_p.HomologicalKernel32,
            pcloud64: cpp_p.HomologicalKernel64,
        },
    )
    x_data = X._cpp_point_cloud() if isinstance(X, PointCloud) else X._data
    y_data = Y._cpp_point_cloud() if isinstance(Y, PointCloud) else Y._data

    return backend.spawn_homological_kernel_pcloud_task(
        x_data, y_data, out._data
    )


def _spawn_homological_kernel_distmat_task(
    X: DistanceMatrixTensor,
    Y: DistanceMatrixTensor,
    out: BarcodeTensor,
):
    backend, X = _get_backend(
        X, {distmat32: cpp_p.HomologicalKernel32, distmat64: cpp_p.HomologicalKernel64}
    )

    return backend.spawn_homological_kernel_distmat_task(X._data, Y._data, out._data)


def _normalize_kernel_input(X, name: str):
    """Normalize one kernel input to a tensor, wrapping single instances
    into a 1-element tensor."""
    from ..tensor_create import zeros

    if isinstance(X, np.ndarray):
        X = FloatTensor(X)

    if isinstance(X, FloatTensor):
        pcX = zeros((1,), dtype=_FLOAT_TO_PCLOUD_DTYPE[X.dtype])
        pcX[0] = X
        return pcX

    if isinstance(X, DistanceMatrix):
        dmX = zeros((1,), dtype=_FLOAT_TO_DISTMAT_DTYPE[X.dtype])
        dmX[0] = X
        return dmX

    if isinstance(X, (PointCloud, PointCloudTensor, DistanceMatrixTensor)):
        return X

    raise TypeError(f"compute_homological_kernel does not support {name} of type {type(X)}")


def _critical_edges(X: np.ndarray, Y: np.ndarray):
    """(bars, critical edges) of the kernel of one pair of (n, d) point clouds."""
    X, Y = np.asarray(X), np.asarray(Y)
    dtype = np.result_type(X.dtype, Y.dtype, np.float32)
    backend = cpp_p.HomologicalKernel32 if dtype == np.float32 else cpp_p.HomologicalKernel64
    return backend.homological_kernel_critical_edges(
        PointCloud(X.astype(dtype))._cpp_point_cloud(), PointCloud(Y.astype(dtype))._cpp_point_cloud())


def _kernel_bars_vjp(grad_out, X, Y, residuals):
    """Gradients of sum(grad_out * bars) with respect to the points of X and Y.

    Each birth is the distance between two points of Y, and each death the
    distance between two points of X; the gradient of |p_i - p_j| is the unit
    vector from p_j to p_i at p_i and its negative at p_j.
    """
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    (edges,) = residuals
    grads = []
    for points, (i, j), weights in ((X, edges[:, 2:].T, grad_out[:, 1]), (Y, edges[:, :2].T, grad_out[:, 0])):
        diff = points[i] - points[j]
        length = np.linalg.norm(diff, axis=1, keepdims=True)
        step = weights[:, None] * np.divide(diff, length, out=np.zeros_like(diff), where=length > 0)
        grad = np.zeros_like(points)
        np.add.at(grad, i, step)
        np.add.at(grad, j, -step)
        grads.append(grad)
    return tuple(grads)


def _kernel_bars_out(X, Y):
    n = X.shape[0]
    # The bars are float32 only if both point clouds are.
    both_float32 = all(str(P.dtype).endswith("float32") for P in (X, Y))
    return np.array([[0, max(n - 1, 0)]]), np.float32 if both_float32 else np.float64


def _kernel_bars_residuals(X, Y):
    return [((max(X.shape[0] - 1, 0), 4), np.int64)]


@_autodiff.differentiable(
    "X", "Y", vjp=_kernel_bars_vjp, out=_kernel_bars_out, residuals=_kernel_bars_residuals,
    returns=BarcodeTensor)
def _tracked_kernel(X: Any, Y: Any) -> Any:
    """The kernel of one pair of (n, d) point clouds given as framework
    arrays, as a BarcodeTensor of shape (1,) that carries its n - 1 bars as a
    framework array, and as residuals the critical edges, which the vjp
    reuses."""
    bars, edges = _critical_edges(X, Y)
    return BarcodeTensor._from_serial(bars, np.array([[0, len(bars)]])), (edges,)


def _validate_kernel_pair(X, Y):
    """Check that a normalized (X, Y) pair matches in dtype and shape."""
    if X.dtype != Y.dtype:
        raise TypeError(f"X and Y must have the same dtype (got {X.dtype} and {Y.dtype})")
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape (got {X.shape} and {Y.shape})")


_ABSTRACT_BATCH = (
    "compute_homological_kernel of batches, or in a distance that is computed from pairs of points, is not "
    "supported for points that JAX or tf.function traces abstractly: the gradients follow pairs of points "
    "that depend on their values.")


def _tracked_kernel_of_batches(X: Any, Y: Any, distance: Any = None) -> BarcodeTensor:
    """The kernels of aligned point clouds of which some carry gradients,
    as a BarcodeTensor whose bars carry gradients. X and Y are both
    PointCloudTensors, of the same shape, or both single clouds as (n, d)
    arrays, which give a result of shape (1,). The distance is Euclidean,
    or the given Distance, which is computed from pairs of points.

    Each birth is the distance between two points of Y and each death the
    distance between two points of X. C++ reports these pairs with the
    bars, and the bars are their distances again as framework operations
    on the points (see _critical_edges). This needs concrete points.
    """
    if isinstance(X, PointCloudTensor) != isinstance(Y, PointCloudTensor):
        raise TypeError(
            "X and Y must both be batches of point clouds, or both single point clouds "
            f"(got {type(X).__name__} and {type(Y).__name__}).")
    X, Y = as_batch(X), as_batch(Y)
    _validate_kernel_pair(X, Y)
    (x_points, x_starts, x_plain), (y_points, y_starts, y_plain) = batch_parts(X), batch_parts(Y)
    adapter, (x_points, y_points) = _in_one_framework(x_points, y_points)
    if adapter.concrete(x_points) is None or adapter.concrete(y_points) is None:
        raise NotImplementedError(_ABSTRACT_BATCH)

    if distance is None:
        elements, lengths = [x_plain, y_plain], adapter.row_norms
    else:
        elements = [distance._distance_matrices(clouds) for clouds in (x_plain, y_plain)]

        def lengths(differences):
            return distance._lengths(adapter, differences)

    backend = cpp_p.HomologicalKernel32 if X.dtype is pcloud32 else cpp_p.HomologicalKernel64
    # The operation over a pair of tensors takes dense tensors.
    bars, enumeration, edges = backend.homological_kernel_critical_edges(
        *(tensor._data.materialize() if tensor._has_indexed_storage() else tensor._data for tensor in elements))
    return tracked_bars(adapter, (y_points, y_starts), (x_points, x_starts), bars, enumeration, edges, lengths)


def _kernel_in_spaces(X: Any, Y: Any, dim: int, verbose: bool) -> BarcodeTensor:
    """The kernel of the points of two DistanceSpaces in their distance."""
    if not (isinstance(X, DistanceSpace) and isinstance(Y, DistanceSpace)):
        raise TypeError(
            "X and Y must both be DistanceSpaces, with equal distances, or both plain: a plain point cloud "
            "has the Euclidean distance.")
    if X.distance != Y.distance:
        raise TypeError(f"X and Y must have equal distances (got {X.distance!r} and {Y.distance!r}).")
    transformed = [space.distance._euclidean_points(_points_array(space)) for space in (X, Y)]
    if transformed[0] is None and any(_autodiff._tracked(space.points) is not None for space in (X, Y)):
        return _tracked_kernel_of_batches(_points_array(X), _points_array(Y), X.distance)
    if transformed[0] is None:
        return compute_homological_kernel(_distance_matrices(X), _distance_matrices(Y), dim=dim, verbose=verbose)
    # The Euclidean distances of the transformed points are the distances
    # of the spaces.
    return compute_homological_kernel(transformed[0], transformed[1], dim=dim, verbose=verbose)


def compute_homological_kernel(
    X: PointCloud
    | PointCloudTensor
    | DistanceMatrix
    | DistanceMatrixTensor
    | FloatTensor
    | np.ndarray
    | DistanceSpace,
    Y: PointCloud
    | PointCloudTensor
    | DistanceMatrix
    | DistanceMatrixTensor
    | FloatTensor
    | np.ndarray
    | DistanceSpace,
    *,
    dim: int = 0,
    verbose: bool = False,
) -> BarcodeTensor:
    r"""Compute the homological kernel of two point clouds or distance matrices.

    ``X`` carries the larger distances :math:`d` and ``Y`` the smaller
    distances :math:`d'` (:math:`d' \le d` pointwise). Each bar of the
    output barcode is born when two clusters merge under :math:`d'` and
    dies when the same clusters merge under :math:`d`. Both inputs must
    be the same kind and the same shape; element ``i`` of ``X`` is
    paired with element ``i`` of ``Y``. Plain point clouds use the
    Euclidean distance; for another distance, pass two
    :class:`~stablebear.linear.DistanceSpace` objects with equal
    distances, or precomputed distance matrices. When the input contains multiple point clouds or distance
    matrices, the computations are parallelized across them. The method
    and the algorithm used to compute it are described in full in
    :footcite:`KampNorthman2026`.

    Parameters
    ----------
    X : PointCloud, PointCloudTensor, DistanceMatrix, DistanceMatrixTensor, FloatTensor, numpy.ndarray, or DistanceSpace
        Input data for :math:`d`. A ``FloatTensor`` or NumPy array is
        interpreted as a single point cloud (one row per point), and so is
        a PyTorch, JAX, or TensorFlow array (see Notes). A
        :class:`~stablebear.linear.DistanceSpace` holds a point cloud, or a
        batch of them, with the distance to use between its points.
    Y : PointCloud, PointCloudTensor, DistanceMatrix, DistanceMatrixTensor, FloatTensor, numpy.ndarray, or DistanceSpace
        Input data for :math:`d'`, of the same kind as ``X``, with the same number of points per
        element as ``X``. Point clouds stay in ambient dimension: express
        a lower-dimensional projection in the original coordinates.
    dim : int, optional
        Homology dimension of the kernel, by default 0. Only ``dim=0``
        is currently supported.
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    BarcodeTensor
        Kernel barcodes with the same shape as the input, one per input
        element, each with :math:`n - 1` finite bars. Single-instance
        inputs (a NumPy array, ``FloatTensor``, or ``DistanceMatrix``)
        yield a tensor of shape ``(1,)``.

    Raises
    ------
    TypeError
        If ``X`` and ``Y`` are not the same kind or not the same dtype, or
        only one of them is a ``DistanceSpace``, or their distances differ.
    ValueError
        If the (outer) tensor shapes differ, or ``dim`` is negative.
    NotImplementedError
        If ``dim`` is positive.
    RuntimeError
        If :math:`d` does not dominate :math:`d'` (a death would land
        below its birth by more than floating-point roundoff), or if
        paired elements mismatch (for example, two point clouds with
        different point counts, or a point cloud that is not an
        ``(n, dim)`` array).

    Notes
    -----
    If ``X``, ``Y``, or both are PyTorch, JAX, or TensorFlow arrays of shape
    ``(n, d)``, each a single point cloud, the bars of the result carry
    gradients with respect to the points. The other input may then be a
    NumPy array. Each birth is the distance between two points of ``Y`` and
    each death the distance between two points of ``X``, and the gradient
    of a distance of 0 is 0.

    The same holds for the points of two ``DistanceSpace`` objects, in
    their distance, and with the distance of an inner product the bars
    also carry gradients with respect to its Gram matrix when that is such
    an array. With an :class:`~stablebear.linear.LpDistance` other than
    ``LpDistance(2)``, points that JAX or ``tf.function`` traces raise
    ``NotImplementedError``.

    Batches carry gradients as well: ``PointCloudTensor`` objects of which
    at least one carries gradients, such as one made from a list of
    PyTorch, JAX, or TensorFlow arrays or by a projection, or such arrays
    of shape ``(..., n, d)``. Batches that JAX or ``tf.function`` traces
    raise ``NotImplementedError``.

    """
    from ..tensor_create import zeros

    if dim < 0:
        raise ValueError(f"dim must be non-negative (got {dim})")
    if dim > 0:
        raise NotImplementedError(f"only dim=0 is currently supported (got dim={dim})")

    if isinstance(X, DistanceSpace) or isinstance(Y, DistanceSpace):
        return _kernel_in_spaces(X, Y, dim, verbose)

    if _autodiff._tracked(X) is not None or _autodiff._tracked(Y) is not None:
        # A framework array of more than two dimensions is a batch of clouds.
        X, Y = (PointCloudTensor(points) if _autodiff._framework(_autodiff._unwrap(points)) is not None
                and len(_autodiff._unwrap(points).shape) > 2 else points for points in (X, Y))
        if isinstance(X, PointCloudTensor) or isinstance(Y, PointCloudTensor):
            return _tracked_kernel_of_batches(X, Y)
        return _tracked_kernel(X, Y)

    X = _normalize_kernel_input(X, "X")
    Y = _normalize_kernel_input(Y, "Y")

    # --- Point cloud input path ---
    if isinstance(X, PointCloud) and isinstance(Y, PointCloud):
        _validate_kernel_pair(X, Y)
        out = zeros((1,), dtype=_FLOAT_TO_BARCODE_DTYPE[X.dtype])
        task = _spawn_homological_kernel_pcloud_task(X, Y, out)

    elif isinstance(X, PointCloudTensor) and isinstance(Y, PointCloudTensor):
        _validate_kernel_pair(X, Y)
        out = zeros((1,), dtype=_PCLOUD_TO_BARCODE_DTYPE[X.dtype])
        task = _spawn_homological_kernel_pcloud_task(X, Y, out)

    # --- Distance matrix input path ---
    elif isinstance(X, DistanceMatrixTensor) and isinstance(Y, DistanceMatrixTensor):
        _validate_kernel_pair(X, Y)
        out = zeros((1,), dtype=_DISTMAT_TO_BARCODE_DTYPE[X.dtype])
        task = _spawn_homological_kernel_distmat_task(X, Y, out)

    else:
        raise TypeError(
            "X and Y must both be point clouds or both be distance matrices "
            f"(got {type(X).__name__} and {type(Y).__name__})"
        )

    _run_task(lambda: task, verbose=verbose)

    return out
