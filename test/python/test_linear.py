import numpy as np
import pytest

import stablebear as sb


def _gram():
    # Symmetric and positive definite: its eigenvalues are 1 and 3.
    return np.array([[2.0, 1.0],
                     [1.0, 2.0]])


def test_inner_product_gives_its_gram_matrix_back():
    ip = sb.InnerProduct(_gram())

    assert isinstance(ip.gram, sb.FloatTensor)
    np.testing.assert_array_equal(np.asarray(ip.gram), _gram())
    assert ip.dim == 2


def test_inner_product_of_float32_matrix_is_float32():
    ip = sb.InnerProduct(_gram().astype(np.float32))

    assert ip.gram.dtype == sb.float32


def test_inner_products_with_equal_matrices_are_equal():
    assert sb.InnerProduct(_gram()) == sb.InnerProduct(_gram())
    assert sb.InnerProduct(_gram()) != sb.InnerProduct(np.eye(2))


@pytest.mark.parametrize("gram, problem", [
    pytest.param(np.ones((2, 3)), "square", id="not_square"),
    pytest.param(np.ones(2), "square", id="not_a_matrix"),
    pytest.param(np.array([[2.0, 1.0], [0.0, 2.0]]), "symmetric", id="not_symmetric"),
    # Its eigenvalues are 3 and -1.
    pytest.param(np.array([[1.0, 2.0], [2.0, 1.0]]), "positive definite", id="indefinite"),
    pytest.param(np.zeros((2, 2)), "positive definite", id="zero"),
])
def test_invalid_gram_matrix_raises(gram, problem):
    with pytest.raises(ValueError, match=problem):
        sb.InnerProduct(gram)


@pytest.mark.parametrize("make_points", [
    pytest.param(lambda points: points, id="array"),
    pytest.param(lambda points: sb.FloatTensor(points), id="float_tensor"),
    pytest.param(lambda points: sb.PointCloud(points), id="point_cloud"),
])
def test_inner_product_space_gives_its_vectors_and_inner_product_back(make_points):
    ip = sb.InnerProduct(_gram())
    points = np.array([[1.0, 10.0],
                       [2.0, 20.0],
                       [3.0, 30.0]])

    space = ip(make_points(points))

    assert isinstance(space, sb.InnerProductSpace)
    assert isinstance(space.vectors, sb.PointCloud)
    np.testing.assert_array_equal(np.asarray(space.vectors), points)
    assert space.inner_product == ip
    assert space.dtype == sb.pcloud64


def test_inner_product_space_of_float32_points_is_pcloud32():
    space = sb.InnerProduct(_gram())(np.array([[1.0, 10.0]], dtype=np.float32))

    assert space.dtype == sb.pcloud32


def test_points_of_another_dimension_than_the_inner_product_raise():
    ip = sb.InnerProduct(_gram())
    with pytest.raises(ValueError, match="dimension 3"):
        ip(np.array([[1.0, 10.0, 100.0]]))


def test_points_that_are_not_a_cloud_raise():
    ip = sb.InnerProduct(_gram())
    with pytest.raises(ValueError, match="shape"):
        ip(np.array([1.0, 10.0]))


def _points():
    return np.array([[1.0, 2.0, 3.0],
                     [-4.0, 5.0, 6.0]])


def test_subspace_gives_its_vectors_and_dimensions_back():
    vectors = np.array([[1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0]])

    subspace = sb.LinearSubspace(vectors)

    assert isinstance(subspace.vectors, sb.FloatTensor)
    np.testing.assert_array_equal(np.asarray(subspace.vectors), vectors)
    assert subspace.dim == 2
    assert subspace.ambient_dim == 3


@pytest.mark.parametrize("vectors, problem", [
    pytest.param(np.array([1.0, 0.0]), "shape", id="not_a_matrix"),
    pytest.param(np.zeros((0, 2)), "shape", id="no_vectors"),
    pytest.param(np.array([[1.0, 2.0], [2.0, 4.0]]), "independent", id="parallel"),
    pytest.param(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), "independent", id="more_than_the_dimension"),
])
def test_invalid_vectors_raise(vectors, problem):
    with pytest.raises(ValueError, match=problem):
        sb.LinearSubspace(vectors)


def test_projection_onto_coordinate_plane_drops_the_other_coordinate():
    plane = sb.LinearSubspace(np.array([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0]]))

    projected = plane.project(_points())

    assert isinstance(projected, sb.PointCloud)
    np.testing.assert_allclose(np.asarray(projected), [[1.0, 2.0, 0.0],
                                                       [-4.0, 5.0, 0.0]])


def test_projection_onto_a_line():
    # The line through (3, 4): the points (5, 0) and (0, 5) have the
    # components 3 and 4 along its unit vector (0.6, 0.8).
    line = sb.LinearSubspace(np.array([[3.0, 4.0]]))

    projected = line.project(np.array([[5.0, 0.0],
                                       [0.0, 5.0]]))

    np.testing.assert_allclose(np.asarray(projected), [[1.8, 2.4],
                                                       [2.4, 3.2]])


@pytest.mark.parametrize("make_points", [
    pytest.param(lambda points: points, id="array"),
    pytest.param(lambda points: sb.FloatTensor(points), id="float_tensor"),
    pytest.param(lambda points: sb.PointCloud(points), id="point_cloud"),
])
def test_projection_takes_every_kind_of_single_cloud(make_points):
    plane = sb.LinearSubspace(np.array([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0]]))

    projected = plane.project(make_points(_points()))

    assert isinstance(projected, sb.PointCloud)
    np.testing.assert_allclose(np.asarray(projected), [[1.0, 2.0, 0.0],
                                                       [-4.0, 5.0, 0.0]])


def test_projecting_twice_equals_projecting_once():
    subspace = sb.LinearSubspace(np.array([[1.0, 2.0, 0.0],
                                           [0.0, 1.0, 3.0]]))

    once = subspace.project(_points())
    twice = subspace.project(once)

    np.testing.assert_allclose(np.asarray(twice), np.asarray(once))


def test_projection_depends_only_on_the_span_of_the_vectors():
    # Both pairs span the plane of the first two coordinates.
    skewed = sb.LinearSubspace(np.array([[2.0, 0.0, 0.0],
                                         [3.0, 5.0, 0.0]]))
    orthonormal = sb.LinearSubspace(np.array([[1.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0]]))

    np.testing.assert_allclose(np.asarray(skewed.project(_points())), np.asarray(orthonormal.project(_points())))


def test_projection_in_inner_product_space_is_orthogonal_in_its_inner_product():
    gram = np.array([[2.0, 1.0, 0.0],
                     [1.0, 2.0, 0.0],
                     [0.0, 0.0, 3.0]])
    vectors = np.array([[1.0, 2.0, 0.0],
                        [0.0, 1.0, 3.0]])
    ip = sb.InnerProduct(gram)

    projected = sb.LinearSubspace(vectors).project(ip(_points()))

    assert isinstance(projected, sb.InnerProductSpace)
    assert projected.inner_product == ip
    residual = _points() - np.asarray(projected.vectors)
    # The inner product of each residual with each vector of the subspace.
    np.testing.assert_allclose(residual @ gram @ vectors.T, np.zeros((2, 2)), atol=1e-12)
    # The residuals are not orthogonal to the vectors in the Euclidean sense.
    assert np.abs(residual @ vectors.T).max() > 0.1


def test_projection_with_identity_inner_product_equals_euclidean_projection():
    subspace = sb.LinearSubspace(np.array([[1.0, 2.0, 0.0],
                                           [0.0, 1.0, 3.0]]))

    in_space = subspace.project(sb.InnerProduct(np.eye(3))(_points()))

    np.testing.assert_allclose(np.asarray(in_space.vectors), np.asarray(subspace.project(_points())))


def test_projection_of_float32_points_onto_float32_subspace_is_float32():
    plane = sb.LinearSubspace(np.array([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0]], dtype=np.float32))

    projected = plane.project(_points().astype(np.float32))

    assert projected.dtype == sb.float32


def test_projection_of_points_of_another_dimension_raises():
    line = sb.LinearSubspace(np.array([[3.0, 4.0]]))
    with pytest.raises(ValueError, match="dimension 3"):
        line.project(_points())
