"""Inner products and linear subspaces of point clouds."""

from __future__ import annotations

from typing import Any, overload

import numpy as np

from . import _autodiff
from .base_tensor import FloatTensor
from . import _sb_cpp as cpp
from .distance_matrix import DistanceMatrixTensor
from ._tracked import is_tracked, tracked_class
from .point_cloud import PointCloud, PointCloudTensor
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
    takes them: those of a framework if an array is a PyTorch, JAX, or
    TensorFlow array, with the NumPy arrays converted to it, and those of
    NumPy otherwise."""
    like = next((array for array in arrays if _autodiff._framework(array) is not None), None)
    if like is None:
        return _NumPy, list(arrays)
    xp = _autodiff._framework(like)
    return xp, [array if _autodiff._framework(array) is not None else xp.as_tensor(array, like=like)
                for array in arrays]


def _matrix(values: Any, name: str) -> Any:
    """A matrix as a float NumPy array, or as it is if it is a PyTorch, JAX,
    or TensorFlow array."""
    values = _autodiff._unwrap(values)
    return values if _autodiff._framework(values) is not None else _float_array(values, name)


def _concrete(matrix: Any) -> np.ndarray | None:
    """The values of a matrix, or None while a framework traces it
    abstractly."""
    adapter = _autodiff._framework(matrix)
    return matrix if adapter is None else adapter.concrete(matrix)


def _public(matrix: Any) -> Any:
    """A matrix as a property returns it: a FloatTensor, or the framework
    array."""
    return matrix if _autodiff._framework(matrix) is not None else FloatTensor(matrix.copy())


def _is_float32(array: Any) -> bool:
    return "float32" in str(array.dtype)


def _as_point_cloud(points: Any) -> Any:
    """A single point cloud from an (n, d) array, FloatTensor, or
    PointCloud, as a PointCloud; a PyTorch, JAX, or TensorFlow array of
    shape (n, d), or a batch of clouds as a PointCloudTensor, as it is."""
    points = _autodiff._unwrap(points)
    if isinstance(points, PointCloudTensor):
        return points
    if _autodiff._framework(points) is not None:
        if len(points.shape) != 2:
            raise ValueError(
                f"A point cloud must have shape (n, d), one point per row (got shape {tuple(points.shape)}).")
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
        and any other real array a ``float64`` one. A PyTorch, JAX, or
        TensorFlow array makes the results of operations that use the inner
        product carry gradients with respect to it.

    Raises
    ------
    ValueError
        If ``gram`` is not a square matrix, not symmetric, or not positive
        definite. A matrix that JAX or ``tf.function`` traces is checked
        for its shape only.
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
        values = _concrete(gram)
        if values is not None:
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
        PyTorch, JAX, or TensorFlow array that the inner product was made
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
        points : PointCloud, FloatTensor, numpy.ndarray, or framework array
            A point cloud of shape ``(n, d)``, one point per row, which may
            be a PyTorch, JAX, or TensorFlow array, or a batch of point
            clouds as a ``PointCloudTensor``.

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
        if self._gram is other._gram:
            return True
        values, other_values = _concrete(self._gram), _concrete(other._gram)
        # Matrices that a framework traces are equal only if they are the same.
        return values is not None and other_values is not None and np.array_equal(values, other_values)

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
        points : PointCloud, FloatTensor, numpy.ndarray, or framework array
            A point cloud of shape ``(n, d)``, one point per row, which may
            be a PyTorch, JAX, or TensorFlow array, or a batch of point
            clouds as a ``PointCloudTensor``.

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

    def _distance_matrices(self, clouds: PointCloudTensor) -> DistanceMatrixTensor:
        """The distances between the points of each cloud, for a distance
        without _euclidean_points."""
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

    def _distance_matrices(self, clouds: PointCloudTensor) -> DistanceMatrixTensor:
        data = clouds._data.materialize() if clouds._has_indexed_storage() else clouds._data
        return DistanceMatrixTensor(cpp.pairwise_lp_distances(data, self._p))

    def _lengths(self, adapter: Any, differences: Any) -> Any:
        return adapter.row_lp_norms(differences, self._p)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distance):
            return NotImplemented
        return isinstance(other, LpDistance) and self._p == other._p

    def __repr__(self) -> str:
        return f"LpDistance({self._p:g})"


class DistanceSpace:
    """Points together with the distance of their space.

    An operation that uses a distance takes a ``DistanceSpace`` in place of
    the plain points and uses its distance. It is usually made by calling a
    :class:`Distance` on the points.

    Parameters
    ----------
    points : PointCloud, FloatTensor, numpy.ndarray, or framework array
        A point cloud of shape ``(n, d)``, one point per row, which may be a
        PyTorch, JAX, or TensorFlow array, or a batch of point clouds as a
        ``PointCloudTensor``.
    distance : Distance
        The distance between the points.

    Raises
    ------
    ValueError
        If the distance does not apply to points of dimension ``d``.
    """

    def __init__(self, points: Any, distance: Distance):
        if not isinstance(distance, Distance):
            raise TypeError(f"Expected a Distance, got {type(distance).__name__}.")
        cloud = _as_point_cloud(points)
        dim = _dim(cloud)
        if dim is not None:
            distance._check_dim(dim)
        self._points = cloud
        self._distance = distance

    @property
    def points(self) -> Any:
        """The points: a ``PointCloud``, the PyTorch, JAX, or TensorFlow
        array of the points, or the ``PointCloudTensor`` of a batch."""
        return self._points

    @property
    def distance(self) -> Distance:
        """The distance of the space."""
        return self._distance

    @property
    def dtype(self) -> Dtype:
        """The dtype of what holds the points, such as ``pcloud64``."""
        if isinstance(self._points, PointCloudTensor):
            return self._points.dtype
        return pcloud32 if _is_float32(self._points) else pcloud64

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._points!r}, {self._distance!r})"


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
    points : PointCloud, FloatTensor, numpy.ndarray, or framework array
        A point cloud of shape ``(n, d)``, one point per row, which may be a
        PyTorch, JAX, or TensorFlow array, or a batch of point clouds as a
        ``PointCloudTensor``.
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
        A PyTorch, JAX, or TensorFlow array makes projections onto the
        subspace carry gradients with respect to the vectors.

    Raises
    ------
    ValueError
        If ``vectors`` is not a matrix, has no rows, or its rows are
        linearly dependent. Vectors that JAX or ``tf.function`` traces are
        checked for their shape only.
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
        values = _concrete(vectors)
        if values is not None and np.linalg.matrix_rank(values) < values.shape[0]:
            raise ValueError("The vectors must be linearly independent.")
        self._vectors = vectors

    @property
    def vectors(self) -> Any:
        """The vectors that span the subspace, as the rows of a ``(k, d)``
        ``FloatTensor``, or the PyTorch, JAX, or TensorFlow array that the
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
        points : PointCloud, PointCloudTensor, FloatTensor, numpy.ndarray, framework array, or InnerProductSpace
            A point cloud of shape ``(n, d)``, one point per row, which may
            be a PyTorch, JAX, or TensorFlow array; a batch of point clouds
            as a ``PointCloudTensor``, each of which is projected; or
            either with its inner product.

        Returns
        -------
        PointCloud, PointCloudTensor, framework array, or InnerProductSpace
            The projected points, of the shape of the input: a
            ``PointCloud`` for a plain point cloud, a ``PointCloudTensor``
            for a batch, and an ``InnerProductSpace`` with the same inner
            product for an ``InnerProductSpace``. If the points, the
            vectors of the subspace, or the Gram matrix of the inner
            product are PyTorch, JAX, or TensorFlow arrays, the projected
            points carry gradients with respect to each of them that is
            one: an array of that framework for a single point cloud, and
            a ``PointCloudTensor`` that carries gradients for a batch.

        Raises
        ------
        ValueError
            If the points do not have the dimension of the vectors of the
            subspace.
        TypeError
            If ``points`` is a :class:`DistanceSpace` that is not an
            ``InnerProductSpace``.
        """
        if isinstance(points, DistanceSpace) and not isinstance(points, InnerProductSpace):
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
    return int(cloud.shape[1])


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
    if not isinstance(points, PointCloudTensor):
        xp, (points, matrix) = _in_one_framework(points, matrix)
        product = xp.matmul(points, matrix)
        return product.astype(np.result_type(points.dtype, matrix.dtype)) if xp is _NumPy else product

    if not is_tracked(points) and _autodiff._framework(matrix) is None:
        data = points._data.materialize() if points._has_indexed_storage() else points._data
        np_dtype = np.float32 if points.dtype is pcloud32 else np.float64
        multiply: Any = cpp.multiply_points  # overloaded on the precision
        return PointCloudTensor(multiply(data, np.ascontiguousarray(matrix, dtype=np_dtype)))

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


def _distance_matrix(distance: Distance, points: np.ndarray) -> Any:
    """The C++ matrix of the distances between the points of one (n, d)
    cloud, for a distance without _euclidean_points."""
    return distance._distance_matrices(PointCloudTensor([points]))[0]._current_matrix()


def _distance_matrices(space: DistanceSpace) -> DistanceMatrixTensor:
    """The distances between the points of each cloud of an untracked
    space, for a distance without _euclidean_points: one matrix for a single
    cloud."""
    points = space.points
    if is_tracked(points):
        raise NotImplementedError(
            f"{space.distance!r} with gradients supports a single point cloud of shape (n, d) only.")
    if not isinstance(points, PointCloudTensor):
        points = PointCloudTensor([np.asarray(points)])
    return space.distance._distance_matrices(points)


def _points_array(space: DistanceSpace) -> Any:
    """The points of a space as _multiply takes them: an (n, d) NumPy
    array, a framework array, or a PointCloudTensor."""
    return _holder(space.points)
