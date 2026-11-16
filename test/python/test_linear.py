import numpy as np
import pytest

import stablebear as sb

from _frameworks import assert_gradient_matches_finite_differences


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
def test_inner_product_space_gives_its_points_and_inner_product_back(make_points):
    ip = sb.InnerProduct(_gram())
    points = np.array([[1.0, 10.0],
                       [2.0, 20.0],
                       [3.0, 30.0]])

    space = ip(make_points(points))

    assert isinstance(space, sb.InnerProductSpace)
    assert isinstance(space.points, sb.PointCloud)
    np.testing.assert_array_equal(np.asarray(space.points), points)
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
    residual = _points() - np.asarray(projected.points)
    # The inner product of each residual with each vector of the subspace.
    np.testing.assert_allclose(residual @ gram @ vectors.T, np.zeros((2, 2)), atol=1e-12)
    # The residuals are not orthogonal to the vectors in the Euclidean sense.
    assert np.abs(residual @ vectors.T).max() > 0.1


def test_projection_with_identity_inner_product_equals_euclidean_projection():
    subspace = sb.LinearSubspace(np.array([[1.0, 2.0, 0.0],
                                           [0.0, 1.0, 3.0]]))

    in_space = subspace.project(sb.InnerProduct(np.eye(3))(_points()))

    np.testing.assert_allclose(np.asarray(in_space.points), np.asarray(subspace.project(_points())))


def test_projection_of_float32_points_onto_float32_subspace_is_float32():
    plane = sb.LinearSubspace(np.array([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0]], dtype=np.float32))

    projected = plane.project(_points().astype(np.float32))

    assert projected.dtype == sb.float32


def test_projection_of_points_of_another_dimension_raises():
    line = sb.LinearSubspace(np.array([[3.0, 4.0]]))
    with pytest.raises(ValueError, match="dimension 3"):
        line.project(_points())


def _vectors():
    return np.array([[1.0, 2.0, 0.0],
                     [0.0, 1.0, 3.0]])


def _gram3():
    return np.array([[2.0, 1.0, 0.0],
                     [1.0, 2.0, 0.0],
                     [0.0, 0.0, 3.0]])


def _weights():
    # One weight per coordinate of the two projected points of _points.
    return np.array([[1.0, -2.0, 3.0],
                     [0.5, 4.0, -1.5]])


def _framework_gram(fw, a):
    # A symmetric matrix for each a, which is _gram3 at a = (2, 1, 3).
    diagonal = np.diag([1.0, 1.0, 0.0])
    off_diagonal = np.array([[0.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])
    corner = np.diag([0.0, 0.0, 1.0])
    return a[0] * fw.tensor(diagonal) + a[1] * fw.tensor(off_diagonal) + a[2] * fw.tensor(corner)


@pytest.mark.parametrize("which", ["points", "vectors", "both"])
def test_projection_with_framework_arrays_matches_numpy(fw, which):
    points = fw.tensor(_points()) if which in ("points", "both") else _points()
    vectors = fw.tensor(_vectors()) if which in ("vectors", "both") else _vectors()

    projected = sb.LinearSubspace(vectors).project(points)

    assert isinstance(projected, fw.tensor_type)
    expected = sb.LinearSubspace(_vectors()).project(_points())
    np.testing.assert_allclose(fw.to_numpy(projected), np.asarray(expected))


@pytest.mark.parametrize("which", ["points", "gram"])
def test_projection_in_inner_product_space_with_framework_arrays_matches_numpy(fw, which):
    points = fw.tensor(_points()) if which == "points" else _points()
    gram = fw.tensor(_gram3()) if which == "gram" else _gram3()
    ip = sb.InnerProduct(gram)

    projected = sb.LinearSubspace(_vectors()).project(ip(points))

    assert isinstance(projected, sb.InnerProductSpace)
    assert projected.inner_product == ip
    assert isinstance(projected.points, fw.tensor_type)
    expected = sb.LinearSubspace(_vectors()).project(sb.InnerProduct(_gram3())(_points()))
    np.testing.assert_allclose(fw.to_numpy(projected.points), np.asarray(expected.points))


def test_framework_gram_matrix_and_vectors_come_back_as_given(fw):
    gram, vectors = fw.tensor(_gram3()), fw.tensor(_vectors())

    assert sb.InnerProduct(gram).gram is gram
    assert sb.LinearSubspace(vectors).vectors is vectors


def test_invalid_framework_gram_matrix_and_vectors_raise(fw):
    with pytest.raises(ValueError, match="positive definite"):
        sb.InnerProduct(fw.tensor(np.zeros((2, 2))))
    with pytest.raises(ValueError, match="independent"):
        sb.LinearSubspace(fw.tensor(np.array([[1.0, 2.0], [2.0, 4.0]])))


def _losses():
    def wrt_points(fw, x):
        ip = sb.InnerProduct(_gram3())
        return fw.sum(fw.tensor(_weights()) * sb.LinearSubspace(_vectors()).project(ip(x)).points)

    def wrt_vectors(fw, x):
        ip = sb.InnerProduct(_gram3())
        return fw.sum(fw.tensor(_weights()) * sb.LinearSubspace(x).project(ip(_points())).points)

    def wrt_gram(fw, x):
        ip = sb.InnerProduct(_framework_gram(fw, x))
        return fw.sum(fw.tensor(_weights()) * sb.LinearSubspace(_vectors()).project(ip(_points())).points)

    def wrt_vectors_euclidean(fw, x):
        return fw.sum(fw.tensor(_weights()) * sb.LinearSubspace(x).project(_points()))

    return [
        pytest.param(wrt_points, _points(), id="points"),
        pytest.param(wrt_vectors, _vectors(), id="vectors"),
        pytest.param(wrt_gram, [2.0, 1.0, 3.0], id="gram"),
        pytest.param(wrt_vectors_euclidean, _vectors(), id="vectors_euclidean"),
    ]


@pytest.mark.parametrize("loss, x0", _losses())
def test_projection_gradient(fw, loss, x0):
    assert_gradient_matches_finite_differences(fw, lambda x: loss(fw, x), x0)


@pytest.mark.parametrize("loss, x0", _losses())
def test_compiled_projection_gradient_matches_eager(fw, loss, x0):
    if fw.compiled_value_and_grad is None:
        pytest.skip("No compiled mode")
    x0 = np.asarray(x0, dtype=np.float64)

    value, grad = fw.compiled_value_and_grad(lambda x: loss(fw, x), x0)
    eager_value, eager_grad = fw.value_and_grad(lambda x: loss(fw, x), x0)

    np.testing.assert_allclose(value, eager_value)
    np.testing.assert_allclose(grad, eager_grad)


def test_distance_space_gives_its_points_and_distance_back():
    distance = sb.InnerProduct(_gram()).distance
    points = np.array([[1.0, 10.0],
                       [2.0, 20.0]])

    space = distance(points)

    assert type(space) is sb.DistanceSpace
    assert isinstance(space.points, sb.PointCloud)
    np.testing.assert_array_equal(np.asarray(space.points), points)
    assert space.distance == distance
    assert space.dtype == sb.pcloud64


def test_inner_product_space_is_a_distance_space_with_the_distance_of_its_inner_product():
    ip = sb.InnerProduct(_gram())

    space = ip(np.array([[1.0, 10.0]]))

    assert isinstance(space, sb.DistanceSpace)
    assert isinstance(space.distance, sb.Distance)
    assert space.distance == ip.distance


def test_distances_of_equal_inner_products_are_equal():
    assert sb.InnerProduct(_gram()).distance == sb.InnerProduct(_gram()).distance
    assert sb.InnerProduct(_gram()).distance != sb.InnerProduct(np.eye(2)).distance


def test_distance_of_points_of_another_dimension_than_its_inner_product_raises():
    distance = sb.InnerProduct(_gram()).distance
    with pytest.raises(ValueError, match="dimension 3"):
        distance(np.array([[1.0, 10.0, 100.0]]))


def test_projecting_a_distance_space_without_inner_product_raises():
    line = sb.LinearSubspace(np.array([[3.0, 4.0]]))
    space = sb.InnerProduct(_gram()).distance(np.array([[5.0, 0.0]]))
    with pytest.raises(TypeError, match="InnerProductSpace"):
        line.project(space)  # pyright: ignore[reportCallIssue, reportArgumentType]


def _loop_points():
    # Six points around a loop, with all pairwise distances different.
    return np.array([[0.0, 0.0], [2.0, 0.1], [3.1, 1.5], [2.3, 3.2], [0.3, 2.9], [-0.9, 1.3]])


def _homology(X):
    return sb.persistence.compute_persistent_homology(X, max_dim=1, reduced=True, verbose=False)


def _assert_same_barcodes(out, expected):
    assert out.shape == expected.shape
    for dim in range(expected.shape[0]):
        assert out[dim].is_isomorphic_to(expected[dim], atol=1e-12, rtol=1e-12)


def test_persistent_homology_in_inner_product_space_is_that_of_the_transformed_points():
    # x^T G y for G = diag(4, 1) is the Euclidean inner product after
    # doubling the first coordinate.
    ip = sb.InnerProduct(np.diag([4.0, 1.0]))

    out = _homology(ip(_loop_points()))

    _assert_same_barcodes(out, _homology(_loop_points() * np.array([2.0, 1.0])))


def test_persistent_homology_with_identity_inner_product_is_that_of_the_plain_cloud():
    out = _homology(sb.InnerProduct(np.eye(2))(_loop_points()))

    _assert_same_barcodes(out, _homology(_loop_points()))


def test_persistent_homology_of_distance_space_is_that_of_the_inner_product_space():
    ip = sb.InnerProduct(_gram())

    _assert_same_barcodes(_homology(ip.distance(_loop_points())), _homology(ip(_loop_points())))


@pytest.mark.parametrize("which", ["points", "gram"])
def test_persistent_homology_in_inner_product_space_with_framework_arrays_matches_numpy(fw, which):
    points = fw.tensor(_loop_points()) if which == "points" else _loop_points()
    gram = fw.tensor(_gram()) if which == "gram" else _gram()

    out = _homology(sb.InnerProduct(gram)(points))

    assert out._tvalues is not None
    _assert_same_barcodes(out, _homology(sb.InnerProduct(_gram())(_loop_points())))


def _framework_gram2(fw, a):
    # A symmetric matrix for each a, which is _gram at a = (2, 1).
    off_diagonal = np.array([[0.0, 1.0],
                             [1.0, 0.0]])
    return a[0] * fw.tensor(np.eye(2)) + a[1] * fw.tensor(off_diagonal)


def _bars_loss(fw, barcodes):
    bars = barcodes._tvalues
    return fw.sum(bars[:, 1] * bars[:, 1]) + fw.sum(bars[:, 0])


def test_persistent_homology_gradient_wrt_gram_matrix(fw):
    if fw.traces_abstractly:
        pytest.skip("The number of bars needs concrete points")

    def loss(a):
        return _bars_loss(fw, _homology(sb.InnerProduct(_framework_gram2(fw, a))(_loop_points())))

    assert_gradient_matches_finite_differences(fw, loss, [2.0, 1.0])


def test_persistent_homology_in_inner_product_space_gradient_wrt_points(fw):
    if fw.traces_abstractly:
        pytest.skip("The number of bars needs concrete points")

    def loss(P):
        return _bars_loss(fw, _homology(sb.InnerProduct(_gram())(P)))

    assert_gradient_matches_finite_differences(fw, loss, _loop_points())
