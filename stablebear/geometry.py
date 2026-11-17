"""Inner products, distances, and linear subspaces of point clouds."""

from __future__ import annotations

from typing import Any, overload

import numpy as np

from . import _autodiff
from .base_tensor import FloatTensor
from . import _sb_cpp as cpp
from .distance_matrix import DistanceMatrix, DistanceMatrixTensor
from ._distance_space import DistanceSpace, EuclideanPoints
from ._tracked import is_tracked, tracked_class
from .point_cloud import PointCloud, PointCloudTensor
from .timeseries.series import TimeSeries
from .typing import Dtype, pcloud32, pcloud64


def _float_array(values: Any, name: str) -> np.ndarray:
    """A float32 array for float32 input and a float64 array for any other
    real input."""
    array = np.asarray(values)
    if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
        raise TypeError(f"{name} must hold real numbers, not {array.dtype}.")
    return array.astype(np.float32 if array.dtype == np.float32 else np.float64)


class _NumPy:
    """The array operations of the framework adapters, for NumPy arrays."""

    matmul = staticmethod(np.matmul)
    transpose = staticmethod(np.transpose)
    solve = staticmethod(np.linalg.solve)
    cholesky = staticmethod(np.linalg.cholesky)


def _in_one_framework(*arrays: Any) -> tuple[Any, list]:
    """The array operations to use for the arrays, and the arrays as that
    takes them: those of a framework if an array is a PyTorch tensor, with the NumPy arrays converted to it, and those of
    NumPy otherwise."""
    like = next((array for array in arrays if _autodiff._framework(array) is not None), None)
    if like is None:
        return _NumPy, list(arrays)
    xp = _autodiff._framework(like)
    return xp, [array if _autodiff._framework(array) is not None else xp.as_tensor(array, like=like)
                for array in arrays]


def _matrix(values: Any, name: str) -> Any:
    """A matrix as a float NumPy array, or as it is if it is a PyTorch tensor."""
    return values if _autodiff._framework(values) is not None else _float_array(values, name)


def _values(matrix: Any) -> np.ndarray:
    """The values of a matrix, a NumPy or a framework array."""
    adapter = _autodiff._framework(matrix)
    return matrix if adapter is None else adapter.to_numpy(matrix)


def _public(matrix: Any) -> Any:
    """A matrix as a property returns it: a FloatTensor, or the framework
    array."""
    return matrix if _autodiff._framework(matrix) is not None else FloatTensor(matrix.copy())


def _is_float32(array: Any) -> bool:
    return "float32" in str(array.dtype)


def _as_point_cloud(points: Any) -> Any:
    """A single point cloud from an (n, d) array, FloatTensor, or
    PointCloud, as a PointCloud. A batch of clouds as a PointCloudTensor or a
    time series, and a framework array of shape (..., n, d), a single cloud
    or a batch of clouds of the same size, as they are."""
    if isinstance(points, PointCloudTensor):
        return points
    if isinstance(points, TimeSeries):
        if points.dtype not in (pcloud32, pcloud64):
            raise TypeError(f"Expected a time series of point clouds, got one of dtype {points.dtype}.")
        return points
    if _autodiff._framework(points) is not None:
        if len(points.shape) < 2:
            raise ValueError(
                f"A point cloud must have shape (n, d), one point per row (got shape {tuple(points.shape)}).")
        # More dimensions are a batch of clouds of the same size.
        return points
    if isinstance(points, PointCloud):
        return points
    if isinstance(points, FloatTensor):
        points = np.asarray(points)
    if isinstance(points, (np.ndarray, list, tuple)):
        array = _float_array(points, "The points")
        if array.ndim != 2:
            raise ValueError(f"A point cloud must have shape (n, d), one point per row (got shape {array.shape}).")
        return PointCloud(array)
    raise TypeError(f"Expected a point cloud, got {type(points).__name__}.")


class InnerProduct:
    r"""An inner product on vectors of dimension :math:`d`, given by its
    Gram matrix.

    The inner product of :math:`x` and :math:`y` is :math:`x^T G y` for the
    Gram matrix :math:`G`. The identity matrix gives the Euclidean inner
    product, which point clouds have without an ``InnerProduct``.

    Calling the inner product on a point cloud gives an
    :class:`InnerProductSpace`, which operations that use an inner product
    or a distance take in place of the point cloud.

    Parameters
    ----------
    gram : array_like
        The Gram matrix, of shape ``(d, d)``, symmetric and positive
        definite. A ``float32`` array gives a ``float32`` inner product,
        and any other real array a ``float64`` one. A PyTorch tensor makes the results of operations that use the inner
        product carry gradients with respect to it.

    Raises
    ------
    ValueError
        If ``gram`` is not a square matrix, not symmetric, or not positive
        definite.
    TypeError
        If ``gram`` does not hold real numbers.

    Examples
    --------
    >>> import stablebear as sb
    >>> ip = sb.InnerProduct([[2.0, 0.0], [0.0, 1.0]])
    >>> ip.dim
    2
    >>> space = ip([[0.0, 0.0], [3.0, 4.0]])
    >>> space.inner_product == ip
    True
    """

    def __init__(self, gram: Any):
        gram = _matrix(gram, "The Gram matrix")
        if len(gram.shape) != 2 or gram.shape[0] != gram.shape[1]:
            raise ValueError(f"The Gram matrix must be square (got shape {tuple(gram.shape)}).")
        values = _values(gram)
        if not np.allclose(values, values.T):
            raise ValueError("The Gram matrix must be symmetric.")
        try:
            np.linalg.cholesky(values)
        except np.linalg.LinAlgError:
            raise ValueError("The Gram matrix must be positive definite.") from None
        self._gram = gram

    @property
    def gram(self) -> Any:
        """The Gram matrix, of shape ``(d, d)``: a ``FloatTensor``, or the
        PyTorch tensor that the inner product was made
        from."""
        return _public(self._gram)

    @property
    def dim(self) -> int:
        """The dimension :math:`d` of the vectors."""
        return int(self._gram.shape[0])

    @property
    def distance(self) -> InnerProductDistance:
        r"""The distance that the inner product induces:
        :math:`\sqrt{(x - y)^T G (x - y)}` between :math:`x` and
        :math:`y`."""
        return InnerProductDistance(self)

    def __call__(self, points: Any) -> InnerProductSpace:
        """The points with this inner product.

        Parameters
        ----------
        points : PointCloud, FloatTensor, or numpy.ndarray
            A point cloud of shape ``(n, d)``, one point per row, which may
            be a PyTorch tensor, or a batch of point
            clouds as a ``PointCloudTensor`` or a time series of point
            clouds.

        Returns
        -------
        InnerProductSpace
            The point cloud with this inner product.

        Raises
        ------
        ValueError
            If the points do not have the dimension of the inner product.
        """
        return InnerProductSpace(points, self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InnerProduct):
            return NotImplemented
        return self._gram is other._gram or np.array_equal(_values(self._gram), _values(other._gram))

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"InnerProduct(dim={self.dim})"


class Distance:
    """A distance between points. The base class of all distances.

    Calling a distance on a point cloud gives a :class:`DistanceSpace`,
    which operations that use a distance, such as
    :func:`~stablebear.persistence.homology.compute_persistent_homology`,
    take in place of the point cloud. A point cloud that is not in a
    ``DistanceSpace`` has the Euclidean distance.

    The distances are :class:`LpDistance` and the distance that an
    :class:`InnerProduct` induces, :attr:`InnerProduct.distance`.
    """

    def __call__(self, points: Any) -> DistanceSpace:
        """The points with this distance.

        Parameters
        ----------
        points : PointCloud, FloatTensor, or numpy.ndarray
            A point cloud of shape ``(n, d)``, one point per row, which may
            be a PyTorch tensor, or a batch of point
            clouds as a ``PointCloudTensor`` or a time series of point
            clouds.

        Returns
        -------
        DistanceSpace
            The point cloud with this distance.
        """
        return DistanceSpace(points, self)

    def _check_dim(self, dim: int) -> None:
        """Raise ValueError if the distance does not apply to points of the
        given dimension."""

    def _euclidean_points(self, X: Any) -> Any:
        """Points whose Euclidean distances are the distances of the points
        X in this distance, of the kind of X (see _multiply); None if the
        distance has none, and is computed from pairs of points."""
        return None

    def _distance_matrices(self, clouds: Any) -> Any:
        """The distances between the points of each cloud of a
        PointCloudTensor or of a time series of point clouds, as a
        DistanceMatrixTensor or a time series of distance matrices, for a
        distance without _euclidean_points."""
        raise NotImplementedError

    def _lengths(self, adapter: Any, differences: Any) -> Any:
        """The distance from x to y for each row x - y of a framework
        array, as an operation of the framework of the given adapter, for a
        distance without _euclidean_points."""
        raise NotImplementedError

    __hash__ = None  # type: ignore[assignment]


class InnerProductDistance(Distance):
    r"""The distance that an inner product induces,
    :math:`\sqrt{(x - y)^T G (x - y)}` between :math:`x` and :math:`y`,
    for the Gram matrix :math:`G` of the inner product. It is what
    :attr:`InnerProduct.distance` gives.

    Parameters
    ----------
    inner_product : InnerProduct
        The inner product.
    """

    def __init__(self, inner_product: InnerProduct):
        if not isinstance(inner_product, InnerProduct):
            raise TypeError(f"Expected an InnerProduct, got {type(inner_product).__name__}.")
        self._inner_product = inner_product

    @property
    def inner_product(self) -> InnerProduct:
        """The inner product that induces the distance."""
        return self._inner_product

    def _check_dim(self, dim: int) -> None:
        if dim != self._inner_product.dim:
            raise ValueError(
                f"The points have dimension {dim}, but the inner product is on vectors of "
                f"dimension {self._inner_product.dim}.")

    def _euclidean_points(self, X: Any) -> Any:
        # With the Cholesky factorization G = C C^T, (x - y)^T G (x - y) is
        # the squared Euclidean length of (x - y)^T C.
        xp, (G,) = _in_one_framework(self._inner_product._gram)
        return _multiply(X, xp.cholesky(G))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distance):
            return NotImplemented
        return isinstance(other, InnerProductDistance) and self._inner_product == other._inner_product

    def __repr__(self) -> str:
        return f"InnerProductDistance({self._inner_product!r})"


class LpDistance(Distance):
    r"""The distance of the :math:`L_p` norm,

    .. math::
        d(x, y) = \Big(\sum_i |x_i - y_i|^p\Big)^{1/p},

    and the largest :math:`|x_i - y_i|` for :math:`p = \infty`.
    ``LpDistance(2)`` is the Euclidean distance, which point clouds have
    without a distance, ``LpDistance(1)`` the taxicab distance, and
    ``LpDistance(float("inf"))`` the Chebyshev distance.

    Parameters
    ----------
    p : float
        The exponent, at least 1, or infinity.

    Raises
    ------
    ValueError
        If ``p`` is less than 1 or not a number.

    Examples
    --------
    >>> import stablebear as sb
    >>> taxicab = sb.LpDistance(1)
    >>> space = taxicab([[0.0, 0.0], [3.0, 4.0]])
    >>> sb.persistence.compute_persistent_homology(space, max_dim=0, reduced=True)[0].to_numpy()
    array([[0., 7.]])
    """

    def __init__(self, p: float):
        _autodiff.require_number(p, "p")
        p = float(p)
        if not p >= 1:
            raise ValueError(f"p must be >= 1 (got {p}).")
        self._p = p

    @property
    def p(self) -> float:
        """The exponent of the norm."""
        return self._p

    def _euclidean_points(self, X: Any) -> Any:
        return X if self._p == 2 else None

    def _distance_matrices(self, clouds: Any) -> Any:
        distances: Any = cpp.pairwise_lp_distances  # overloaded on the container
        if isinstance(clouds, TimeSeries):
            return TimeSeries._from_cpp(distances(clouds._data, self._p))
        return DistanceMatrixTensor(distances(clouds._data, self._p))

    def _lengths(self, adapter: Any, differences: Any) -> Any:
        return adapter.row_lp_norms(differences, self._p)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distance):
            return NotImplemented
        return isinstance(other, LpDistance) and self._p == other._p

    def __repr__(self) -> str:
        return f"LpDistance({self._p:g})"


class InnerProductSpace(DistanceSpace):
    """Points together with the inner product of their space.

    An operation that uses an inner product, such as
    :meth:`LinearSubspace.project`, takes an ``InnerProductSpace`` in place
    of the plain points and uses its inner product. It is also a
    :class:`DistanceSpace`, with the distance that the inner product
    induces. It is usually made by calling an :class:`InnerProduct` on the
    points.

    Parameters
    ----------
    points : PointCloud, FloatTensor, or numpy.ndarray
        A point cloud of shape ``(n, d)``, one point per row, which may be a
        PyTorch tensor, or a batch of point clouds as a
        ``PointCloudTensor`` or a time series of point clouds.
    inner_product : InnerProduct
        The inner product, on vectors of dimension ``d``.

    Raises
    ------
    ValueError
        If the points do not have the dimension of the inner product.
    """

    def __init__(self, points: Any, inner_product: InnerProduct):
        if not isinstance(inner_product, InnerProduct):
            raise TypeError(f"Expected an InnerProduct, got {type(inner_product).__name__}.")
        super().__init__(points, inner_product.distance)
        self._inner_product = inner_product

    @property
    def inner_product(self) -> InnerProduct:
        """The inner product of the space."""
        return self._inner_product

    def __repr__(self) -> str:
        return f"InnerProductSpace({self._points!r}, {self._inner_product!r})"


class LinearSubspace:
    r"""The linear subspace spanned by a set of vectors.

    Parameters
    ----------
    vectors : array_like
        The vectors, as the rows of an array of shape ``(k, d)``: ``k``
        vectors of dimension ``d``. They must be linearly independent, but
        need not be orthogonal or of unit length. A ``float32`` array gives
        a ``float32`` subspace, and any other real array a ``float64`` one.
        A PyTorch tensor makes projections onto the
        subspace carry gradients with respect to the vectors.

    Raises
    ------
    ValueError
        If ``vectors`` is not a matrix, has no rows, or its rows are
        linearly dependent.
    TypeError
        If ``vectors`` does not hold real numbers.

    Examples
    --------
    >>> import numpy as np
    >>> import stablebear as sb
    >>> plane = sb.LinearSubspace([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> plane.dim, plane.ambient_dim
    (2, 3)
    >>> np.asarray(plane.project([[1.0, 2.0, 3.0]]))
    array([[1., 2., 0.]])
    """

    def __init__(self, vectors: Any):
        vectors = _matrix(vectors, "The vectors")
        if len(vectors.shape) != 2 or vectors.shape[0] == 0:
            raise ValueError(
                "The vectors must be the rows of an array of shape (k, d) with k >= 1 "
                f"(got shape {tuple(vectors.shape)}).")
        if np.linalg.matrix_rank(_values(vectors)) < vectors.shape[0]:
            raise ValueError("The vectors must be linearly independent.")
        self._vectors = vectors

    @property
    def vectors(self) -> Any:
        """The vectors that span the subspace, as the rows of a ``(k, d)``
        ``FloatTensor``, or the PyTorch tensor that the
        subspace was made from."""
        return _public(self._vectors)

    @property
    def dim(self) -> int:
        """The dimension :math:`k` of the subspace."""
        return int(self._vectors.shape[0])

    @property
    def ambient_dim(self) -> int:
        """The dimension :math:`d` of the vectors."""
        return int(self._vectors.shape[1])

    @overload
    def project(self, points: InnerProductSpace) -> InnerProductSpace: ...
    @overload
    def project(self, points: PointCloudTensor) -> PointCloudTensor: ...
    @overload
    def project(self, points: TimeSeries) -> TimeSeries: ...
    @overload
    def project(self, points: PointCloud | FloatTensor | np.ndarray | list) -> PointCloud: ...
    @overload
    def project(self, points: Any) -> Any: ...
    def project(self, points: Any) -> Any:
        r"""Project the points of a point cloud orthogonally onto the
        subspace.

        Each point is replaced by the point of the subspace closest to it,
        given in the coordinates of the points: the result has the shape of
        the input. Distances between the projected points are at most
        those between the points.

        What "orthogonal" and "closest" mean depends on the inner product.
        A plain point cloud has the Euclidean one, and an
        :class:`InnerProductSpace` its own. With the vectors of the subspace
        as the rows of :math:`V`, the points as the rows of :math:`X`, and
        the Gram matrix :math:`G` of the inner product, the projected points
        are the rows of

        .. math::
            X G V^T (V G V^T)^{-1} V.

        Parameters
        ----------
        points : PointCloud, PointCloudTensor, TimeSeries, FloatTensor, numpy.ndarray, or InnerProductSpace
            A point cloud of shape ``(n, d)``, one point per row, which may
            be a PyTorch tensor; a batch of point clouds
            as a ``PointCloudTensor`` or a time series of point clouds,
            each of which is projected; or any of these with its inner
            product.

        Returns
        -------
        PointCloud, PointCloudTensor, TimeSeries, or InnerProductSpace
            The projected points, of the shape of the input: a
            ``PointCloud`` for a plain point cloud, a ``PointCloudTensor``
            for a batch, a time series with the times of the input for a
            time series, and an ``InnerProductSpace`` with the same inner
            product for an ``InnerProductSpace``. If the points, the
            vectors of the subspace, or the Gram matrix of the inner
            product are PyTorch tensors, the projected points carry
            gradients with respect to each of them that is one. They are
            then a PyTorch tensor, of the shape of the points, for a single
            point cloud and for points given as a PyTorch tensor of shape
            ``(..., n, d)``, and a ``PointCloudTensor`` that carries
            gradients for a ``PointCloudTensor``.

        Raises
        ------
        ValueError
            If the points do not have the dimension of the vectors of the
            subspace.
        TypeError
            If ``points`` is a :class:`DistanceSpace` that is not an
            ``InnerProductSpace``.
        """
        if chooses_distance(points) and not isinstance(points, InnerProductSpace):
            raise TypeError(
                "Only a DistanceSpace that is an InnerProductSpace can be projected: an orthogonal projection "
                "needs an inner product.")
        space = points if isinstance(points, InnerProductSpace) else None
        cloud = space.points if space is not None else _as_point_cloud(points)
        dim = _dim(cloud)
        if dim is not None and dim != self.ambient_dim:
            raise ValueError(
                f"The points have dimension {dim}, but the subspace is of vectors of dimension "
                f"{self.ambient_dim}.")
        xp, operands = _in_one_framework(self._vectors, *([space.inner_product._gram] if space is not None else []))
        projected = _multiply(_holder(cloud), _projection_matrix(xp, *operands))
        if isinstance(projected, np.ndarray):
            projected = PointCloud(projected)
        return projected if space is None else space.inner_product(projected)

    def __repr__(self) -> str:
        return f"LinearSubspace(dim={self.dim}, ambient_dim={self.ambient_dim})"


def _projection_matrix(xp: Any, V: Any, G: Any = None) -> Any:
    """G V^T (V G V^T)^{-1} V, with the operations of xp: the matrix that
    rows are multiplied by to project them onto the span of the rows of V,
    orthogonally in the inner product with Gram matrix G, the identity if
    None."""
    GVt = xp.transpose(V) if G is None else xp.matmul(G, xp.transpose(V))
    return xp.matmul(GVt, xp.solve(xp.matmul(V, GVt), V))


def _holder(cloud: Any) -> Any:
    """What holds the points of a cloud, as _multiply takes it: an (n, d)
    NumPy array for a PointCloud, and a framework array or a
    PointCloudTensor as it is."""
    return np.asarray(cloud) if isinstance(cloud, PointCloud) else cloud


def _dim(cloud: Any) -> int | None:
    """The dimension of the points of a cloud or a batch, or None for an
    untracked batch, whose clouds would each have to be read to tell."""
    if isinstance(cloud, PointCloudTensor):
        return int(cloud._tvalues.shape[1]) if is_tracked(cloud) else None
    if isinstance(cloud, TimeSeries):
        return None
    return int(cloud.shape[-1])


def _multiply(points: Any, matrix: Any) -> Any:
    """The points, as rows, times a (d, e) matrix, which is a NumPy or a
    framework array.

    The points are an (n, d) NumPy or framework array, or a batch of clouds
    as a PointCloudTensor. If the points or the matrix carry gradients, so
    does the result: a framework array, or a PointCloudTensor that carries
    gradients for a batch, whose points are one framework array and are
    multiplied in one product. An untracked batch times a NumPy matrix is
    multiplied in C++, in parallel over its clouds.
    """
    if isinstance(points, TimeSeries):
        if _autodiff._framework(matrix) is not None:
            raise NotImplementedError("A time series of point clouds does not carry gradients.")
        np_dtype = np.float32 if points.dtype is pcloud32 else np.float64
        multiply_series: Any = cpp.multiply_points  # overloaded on the container
        return TimeSeries._from_cpp(multiply_series(points._data, np.ascontiguousarray(matrix, dtype=np_dtype)))

    if not isinstance(points, PointCloudTensor):
        xp, (points, matrix) = _in_one_framework(points, matrix)
        product = xp.matmul(points, matrix)
        return product.astype(np.result_type(points.dtype, matrix.dtype)) if xp is _NumPy else product

    if not is_tracked(points) and _autodiff._framework(matrix) is None:
        np_dtype = np.float32 if points.dtype is pcloud32 else np.float64
        multiply: Any = cpp.multiply_points  # overloaded on the precision
        return PointCloudTensor(multiply(points._data, np.ascontiguousarray(matrix, dtype=np_dtype)))

    if is_tracked(points):
        tracked_points: Any = points
        content, enumeration = tracked_points._tvalues, tracked_points._enumeration
    else:
        content, enumeration = points._serial()
    xp, (content, matrix) = _in_one_framework(content, matrix)
    if content.shape[1] != matrix.shape[0]:
        raise ValueError(
            f"The points have dimension {content.shape[1]}, but the matrix has {matrix.shape[0]} rows.")
    return tracked_class(PointCloudTensor)([xp.matmul(content, matrix)], enumeration)


def _distance_matrices(space: DistanceSpace) -> Any:
    """The distances between the points of each cloud of an untracked
    space, for a distance without _euclidean_points: one matrix for a single
    cloud."""
    points = space.points
    if not isinstance(points, (PointCloudTensor, TimeSeries)):
        points = PointCloudTensor([np.asarray(points)])
    return space.distance._distance_matrices(points)


def pairwise_distances(space: DistanceSpace) -> Any:
    """The distances between the points of each cloud of a space, in its
    distance; what pdist gives for a DistanceSpace.

    Without gradients, C++ computes the matrices, in parallel over the
    clouds. If the points or the distance carry gradients, the distances
    of all pairs of points of all clouds are one framework operation on
    the rows of the array of all points: a framework array of shape
    (..., n, n) for points that are a framework array, and a
    DistanceMatrixTensor that carries gradients for a PointCloudTensor.
    """
    # The persistence package imports this module.
    from .persistence._critical_edges import as_batch

    distance = space.distance
    transformed = distance._euclidean_points(_points_array(space))
    # The Euclidean distances of the transformed points are the distances
    # of the space.
    holder = _points_array(space) if transformed is None else transformed
    if _autodiff._tracked(holder) is None:
        if transformed is None:
            matrices = _distance_matrices(space)
        else:
            if not isinstance(holder, (PointCloudTensor, TimeSeries)):
                holder = PointCloudTensor([np.asarray(holder)])
            matrices = LpDistance(2)._distance_matrices(holder)
        return matrices if isinstance(space.points, (PointCloudTensor, TimeSeries)) else matrices[0]

    clouds: Any = as_batch(holder)
    points, enumeration = clouds._tvalues, clouds._enumeration
    adapter = _autodiff._framework(points)
    flat = enumeration.reshape(-1, 2)
    sizes = flat[:, 1] - flat[:, 0]
    counts = sizes * (sizes - 1) // 2
    ends = np.cumsum(counts)

    # Entry k of a matrix, of those below its diagonal by rows, is in row
    # i > j and column j, with k = i (i - 1) / 2 + j.
    k = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(ends - counts, counts)
    row = ((1 + np.sqrt(1 + 8 * k)) / 2).astype(np.int64)
    row -= row * (row - 1) // 2 > k
    row += (row + 1) * row // 2 <= k
    start = np.repeat(flat[:, 0], counts)
    first, second = start + row, start + k - row * (row - 1) // 2

    # The distance between equal points is the constant 0: a distance has
    # no derivative there.
    values = adapter.to_numpy(points)
    apart = (values[first] != values[second]).any(axis=1)
    differences = adapter.gather(points, first[apart]) - adapter.gather(points, second[apart])
    lengths = adapter.row_norms(differences) if transformed is not None else distance._lengths(adapter, differences)
    padded = adapter.concat([adapter.as_tensor(np.zeros(1), like=points), lengths])
    # Where each entry is in padded, after the place of the diagonal, 0.
    position = np.zeros(len(k) + 1, dtype=np.int64)
    position[1:][apart] = 1 + np.arange(int(apart.sum()))

    if isinstance(holder, PointCloudTensor):
        shape = enumeration.shape[:-1]
        return tracked_class(DistanceMatrixTensor)(
            [adapter.gather(padded, position[1:])], np.stack([ends - counts, ends], axis=-1).reshape(*shape, 2),
            sizes=sizes.reshape(shape))

    # A framework array of clouds of n points each gives every matrix in full.
    n = int(holder.shape[-2])
    i, j = np.indices((n, n))
    high, low = np.maximum(i, j), np.minimum(i, j)
    entry = np.arange(len(flat))[:, None, None] * (n * (n - 1) // 2) + high * (high - 1) // 2 + low
    full = adapter.gather(padded, position[np.where(i == j, 0, entry + 1)].ravel())
    return adapter.reshape(full, (*(int(size) for size in holder.shape[:-2]), n, n))


def diam(space: Any) -> Any:
    r"""Compute the diameter of a space: the largest distance between two
    of its points,

    .. math::
        \operatorname{diam}(X) = \max_{x, y \in X} d(x, y).

    Parameters
    ----------
    space : DistanceSpace, DistanceMatrix, or DistanceMatrixTensor
        A point cloud or a batch of point clouds with their distance. A
        ``PointCloud`` and a ``PointCloudTensor`` are such spaces, with the
        Euclidean distance. A distance matrix, or a tensor of them, holds
        the distances themselves.

    Returns
    -------
    float, FloatTensor, or torch.Tensor
        The diameter of a single point cloud or distance matrix as a float,
        and those of a batch as a ``FloatTensor`` of the shape of the batch,
        whose point clouds may have different numbers of points. If the
        points, the distance, or the matrices carry gradients, the diameters
        instead as a PyTorch tensor that carries them: of shape ``(...)``
        for points that are a PyTorch tensor of shape ``(..., n, d)``. A
        space of fewer than two points has diameter 0.

    Raises
    ------
    TypeError
        If ``space`` is of another type, or holds a time series.
    """
    if isinstance(space, DistanceSpace):
        if isinstance(space.points, TimeSeries):
            raise TypeError("diam does not take a time series of point clouds.")
        distances = pairwise_distances(space)
    elif isinstance(space, (DistanceMatrix, DistanceMatrixTensor)):
        distances = space
    else:
        raise TypeError(
            f"diam takes a DistanceSpace, a DistanceMatrix, or a DistanceMatrixTensor (got {type(space).__name__}).")

    adapter = _autodiff._framework(distances)
    if adapter is not None:
        # The matrices in full, of shape (..., n, n).
        return adapter.max_along(adapter.max_along(distances, -1), -1)
    if isinstance(distances, DistanceMatrix):
        return float(distances.to_dense().max(initial=0.0))

    # The largest of the entries of each matrix, of all matrices at once.
    if is_tracked(distances):
        tracked: Any = distances
        entries, enumeration = tracked._tvalues, tracked._enumeration
    else:
        content, enumeration, _ = distances._serial()
        entries = content[:, 0]
    shape = enumeration.shape[:-1]
    flat = enumeration.reshape(-1, 2)
    counts = flat[:, 1] - flat[:, 0]
    ends = np.cumsum(counts)
    rows = np.arange(int(counts.sum()), dtype=np.int64) + np.repeat(flat[:, 0] - (ends - counts), counts)
    matrix_of_entry = np.repeat(np.arange(len(flat)), counts)
    adapter = _autodiff._framework(entries)
    if adapter is not None:
        largest = adapter.segment_max(adapter.gather(entries, rows), matrix_of_entry, len(flat))
        return adapter.reshape(largest, tuple(shape))
    largest = np.zeros(len(flat), dtype=entries.dtype)
    np.maximum.at(largest, matrix_of_entry, entries[rows])
    return FloatTensor(largest.reshape(shape))


def chooses_distance(points: Any) -> bool:
    """Whether the points are in a DistanceSpace that was given a distance,
    and not plain points or a point cloud, which is the space of its own
    points with the Euclidean distance."""
    return isinstance(points, DistanceSpace) and not isinstance(points, EuclideanPoints)


def _points_array(space: DistanceSpace) -> Any:
    """The points of a space as _multiply takes them: an (n, d) NumPy
    array, a framework array, or a PointCloudTensor."""
    return _holder(space.points)
