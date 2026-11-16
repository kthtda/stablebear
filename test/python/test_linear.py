import numpy as np
import pytest

import stablebear as sb

from _frameworks import assert_gradient_matches_finite_differences, tracked


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
    def loss(a):
        return _bars_loss(fw, _homology(sb.InnerProduct(_framework_gram2(fw, a))(_loop_points())))

    assert_gradient_matches_finite_differences(fw, loss, [2.0, 1.0])


def test_persistent_homology_in_inner_product_space_gradient_wrt_points(fw):
    def loss(P):
        return _bars_loss(fw, _homology(sb.InnerProduct(_gram())(P)))

    assert_gradient_matches_finite_differences(fw, loss, _loop_points())


def _kernel(X, Y):
    return sb.persistence.compute_homological_kernel(X, Y, dim=0, verbose=False)


def _contracted(points):
    # Every pairwise distance shrinks, as the kernel needs of its second cloud.
    return points * np.array([1.0, 0.25])


def test_kernel_in_inner_product_spaces_is_that_of_the_transformed_points():
    # x^T G y for G = diag(4, 1) is the Euclidean inner product after
    # doubling the first coordinate.
    ip = sb.InnerProduct(np.diag([4.0, 1.0]))
    X, Y = _loop_points(), _contracted(_loop_points())
    doubled = np.array([2.0, 1.0])

    out = _kernel(ip(X), ip(Y))

    assert out.shape == (1,)
    assert out[0].is_isomorphic_to(_kernel(X * doubled, Y * doubled)[0], atol=1e-12, rtol=1e-12)


def _cloud3():
    return np.array([[0.0, 0.0, 0.0],
                     [2.0, 0.1, 1.0],
                     [3.1, 1.5, -0.5],
                     [2.3, 3.2, 2.0],
                     [0.3, 2.9, 1.2],
                     [-0.9, 1.3, 3.0]])


def test_kernel_of_cloud_and_its_projection_in_the_same_inner_product_space():
    ip = sb.InnerProduct(_gram3())
    X = ip(_cloud3())

    out = _kernel(X, sb.LinearSubspace(_vectors()).project(X))

    # One bar per point but one, each born before it dies: the distances of
    # the projection are at most those of the cloud.
    bars = out[0].to_numpy()
    assert bars.shape == (5, 2)
    assert np.all(bars[:, 0] <= bars[:, 1])


def test_kernel_of_distance_space_and_plain_cloud_raises():
    ip = sb.InnerProduct(_gram())
    with pytest.raises(TypeError, match="both"):
        _kernel(ip(_loop_points()), _contracted(_loop_points()))
    with pytest.raises(TypeError, match="both"):
        _kernel(_loop_points(), ip(_contracted(_loop_points())))


def test_kernel_of_spaces_with_different_distances_raises():
    X = sb.InnerProduct(_gram())(_loop_points())
    Y = sb.InnerProduct(np.eye(2))(_contracted(_loop_points()))
    with pytest.raises(TypeError, match="equal distances"):
        _kernel(X, Y)


def _projection_kernel_loss(fw, vectors):
    X = sb.InnerProduct(_gram3())(_cloud3())
    bars = _kernel(X, sb.LinearSubspace(vectors).project(X))._tvalues
    # The births are distances of the projection; the deaths do not depend
    # on the subspace.
    return fw.sum(bars[:, 0] * bars[:, 0])


def test_kernel_of_cloud_and_its_projection_gradient_wrt_subspace(fw):
    assert_gradient_matches_finite_differences(fw, lambda V: _projection_kernel_loss(fw, V), _vectors())


def test_kernel_gradient_wrt_gram_matrix(fw):
    X, Y = _loop_points(), _contracted(_loop_points())

    def loss(a):
        ip = sb.InnerProduct(_framework_gram2(fw, a))
        bars = _kernel(ip(X), ip(Y))._tvalues
        return fw.sum(bars[:, 0] * bars[:, 1])

    assert_gradient_matches_finite_differences(fw, loss, [2.0, 1.0])


def _far_triangle():
    # Distances between the points 0-1, 0-2, and 1-2: 7, 10, and 11 for
    # p = 1, and 4, 10, and 7 for p = infinity.
    return np.array([[0.0, 0.0],
                     [3.0, 4.0],
                     [10.0, 0.0]])


def test_lp_distance_gives_its_exponent_back():
    assert sb.LpDistance(1).p == 1.0
    assert sb.LpDistance(float("inf")).p == float("inf")
    assert isinstance(sb.LpDistance(3), sb.Distance)


def test_lp_distances_with_equal_exponents_are_equal():
    assert sb.LpDistance(1) == sb.LpDistance(1.0)
    assert sb.LpDistance(1) != sb.LpDistance(2)
    assert sb.LpDistance(2) != sb.InnerProduct(np.eye(2)).distance


@pytest.mark.parametrize("p", [0.5, 0.0, -1.0, float("nan")])
def test_lp_distance_with_exponent_below_one_raises(p):
    with pytest.raises(ValueError, match="p must be"):
        sb.LpDistance(p)


@pytest.mark.parametrize("p, deaths", [
    # The two shortest of the three distances join the points.
    pytest.param(1.0, [7.0, 10.0], id="taxicab"),
    pytest.param(float("inf"), [4.0, 7.0], id="chebyshev"),
])
def test_persistent_homology_under_lp_distance(p, deaths):
    space = sb.LpDistance(p)(_far_triangle())

    out = sb.persistence.compute_persistent_homology(space, max_dim=0, reduced=True, verbose=False)

    assert out.shape == (1,)
    np.testing.assert_array_equal(out[0].to_numpy(), [[0.0, deaths[0]],
                                                      [0.0, deaths[1]]])


def test_persistent_homology_under_l2_distance_is_that_of_the_plain_cloud():
    _assert_same_barcodes(_homology(sb.LpDistance(2)(_loop_points())), _homology(_loop_points()))


def test_kernel_under_lp_distance_is_that_of_the_distance_matrices():
    X, Y = _loop_points(), _contracted(_loop_points())
    taxicab = sb.LpDistance(1)

    out = _kernel(taxicab(X), taxicab(Y))

    def distances(points):
        return sb.DistanceMatrix(np.abs(points[:, None, :] - points[None, :, :]).sum(axis=-1))

    assert out.shape == (1,)
    assert out[0].is_isomorphic_to(_kernel(distances(X), distances(Y))[0], atol=1e-12, rtol=1e-12)


def _lp_points():
    # Six points around a loop whose pairwise distances all differ for
    # p = 1, 2, 3, and infinity, and still do after _contracted.
    return np.array([[0.15, 0.19], [2.01, 0.1], [2.97, 1.9], [2.47, 3.64], [0.38, 2.86], [-1.17, 1.21]])


@pytest.mark.parametrize("p", [1.0, 3.0, float("inf")])
def test_persistent_homology_under_lp_distance_of_framework_points_matches_numpy(fw, p):
    distance = sb.LpDistance(p)

    out = _homology(distance(fw.tensor(_lp_points())))

    assert out._tvalues is not None
    _assert_same_barcodes(out, _homology(distance(_lp_points())))


@pytest.mark.parametrize("p", [1.0, 3.0])
def test_persistent_homology_under_lp_distance_gradient(fw, p):
    def loss(P):
        return _bars_loss(fw, _homology(sb.LpDistance(p)(P)))

    assert_gradient_matches_finite_differences(fw, loss, _lp_points())


@pytest.mark.parametrize("p", [1.0, 3.0, float("inf")])
def test_kernel_under_lp_distance_of_framework_points_matches_numpy(fw, p):
    distance = sb.LpDistance(p)
    X, Y = _lp_points(), _contracted(_lp_points())

    out = _kernel(distance(fw.tensor(X)), distance(Y))

    assert out._tvalues is not None
    assert out.shape == (1,)
    assert out[0].is_isomorphic_to(_kernel(distance(X), distance(Y))[0], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("p", [1.0, 3.0])
@pytest.mark.parametrize("wrt", ["X", "Y"])
def test_kernel_under_lp_distance_gradient(fw, p, wrt):
    distance = sb.LpDistance(p)
    X, Y = _lp_points(), _contracted(_lp_points())

    def loss(P):
        bars = _kernel(distance(P if wrt == "X" else X), distance(P if wrt == "Y" else Y))._tvalues
        return fw.sum(bars[:, 0] * bars[:, 0]) + fw.sum(bars[:, 1] * bars[:, 1])

    assert_gradient_matches_finite_differences(fw, loss, X if wrt == "X" else Y)


def _batch3():
    # Clouds of 6, 2, and 3 points in 3 dimensions.
    return [_cloud3(), _points(), _cloud3()[[5, 1, 3]] + 0.5]


def _batch2():
    # Clouds of 6, 6, and 4 points in the plane.
    return [_lp_points(), _loop_points(), _lp_points()[[0, 2, 3, 5]] * 1.5]


@pytest.mark.parametrize("gram", [None, _gram3()], ids=["euclidean", "inner_product"])
def test_projection_of_batch_projects_each_cloud(gram):
    subspace = sb.LinearSubspace(_vectors())
    clouds = _batch3()

    def in_space(points):
        return points if gram is None else sb.InnerProduct(gram)(points)

    projected = subspace.project(in_space(sb.PointCloudTensor(clouds)))

    if gram is not None:
        assert isinstance(projected, sb.InnerProductSpace)
        assert projected.dtype == sb.pcloud64
        projected = projected.points
    assert isinstance(projected, sb.PointCloudTensor)
    assert projected.shape == (3,)
    for i, cloud in enumerate(clouds):
        alone = subspace.project(in_space(cloud))
        alone = alone if gram is None else alone.points
        np.testing.assert_allclose(np.asarray(projected[i]), np.asarray(alone), atol=1e-12)


def test_projection_of_batch_of_another_dimension_raises():
    line = sb.LinearSubspace(np.array([[3.0, 4.0]]))
    with pytest.raises(ValueError, match="dimension 3"):
        line.project(sb.PointCloudTensor(_batch3()))


_BATCH_DISTANCES = [
    # A diagonal Gram matrix, under which _contracted still shrinks every
    # distance.
    pytest.param(lambda: sb.InnerProduct(np.diag([4.0, 1.0])).distance, id="inner_product"),
    pytest.param(lambda: sb.LpDistance(1), id="taxicab"),
    pytest.param(lambda: sb.LpDistance(2), id="euclidean"),
]


@pytest.mark.parametrize("make_distance", _BATCH_DISTANCES)
def test_persistent_homology_of_batch_in_distance_space_is_that_of_each_cloud(make_distance):
    distance = make_distance()
    clouds = _batch2()

    out = _homology(distance(sb.PointCloudTensor(clouds)))

    assert out.shape == (3, 2)
    for i, cloud in enumerate(clouds):
        alone = _homology(distance(cloud))
        for dim in range(2):
            assert out[i, dim].is_isomorphic_to(alone[dim], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("make_distance", _BATCH_DISTANCES)
def test_kernel_of_batches_in_distance_spaces_is_that_of_each_pair(make_distance):
    distance = make_distance()
    clouds = _batch2()
    contracted = [_contracted(cloud) for cloud in clouds]

    out = _kernel(distance(sb.PointCloudTensor(clouds)), distance(sb.PointCloudTensor(contracted)))

    assert out.shape == (3,)
    for i in range(3):
        alone = _kernel(distance(clouds[i]), distance(contracted[i]))
        assert out[i].is_isomorphic_to(alone[0], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("which", ["points", "vectors"])
def test_projection_of_batch_with_framework_arrays_matches_untracked(fw, which):
    clouds = sb.PointCloudTensor(_batch3())
    points = tracked(fw, clouds) if which == "points" else clouds
    vectors = fw.tensor(_vectors()) if which == "vectors" else _vectors()

    projected = sb.LinearSubspace(vectors).project(points)

    assert isinstance(projected, sb.PointCloudTensor)
    assert projected._tvalues is not None
    assert projected.shape == (3,)
    expected = sb.LinearSubspace(_vectors()).project(clouds)
    for i in range(3):
        np.testing.assert_allclose(np.asarray(projected[i]), np.asarray(expected[i]), atol=1e-12)


def _batch_weights():
    # One weight per coordinate of the 11 points of _batch3.
    return np.arange(33.0).reshape(11, 3) % 7.0 - 3.0


def test_projection_of_batch_gradient_wrt_vectors(fw):
    def loss(V):
        projected = sb.LinearSubspace(V).project(sb.InnerProduct(_gram3())(sb.PointCloudTensor(_batch3())))
        return fw.sum(fw.tensor(_batch_weights()) * projected.points._tvalues)

    assert_gradient_matches_finite_differences(fw, loss, _vectors())


def test_projection_of_tracked_batch_gradient_wrt_points(fw):
    plain = sb.PointCloudTensor(_batch3())
    content, enumeration = plain._serial()

    def loss(P):
        clouds = type(tracked(fw, plain))([P], enumeration)
        projected = sb.LinearSubspace(_vectors()).project(clouds)
        return fw.sum(fw.tensor(_batch_weights()) * projected._tvalues)

    assert_gradient_matches_finite_differences(fw, loss, content)


def _l1_matrices(clouds):
    return sb.LpDistance(1)._distance_matrices(sb.PointCloudTensor(clouds))


@pytest.mark.parametrize("make_elements", [
    pytest.param(lambda clouds: sb.PointCloudTensor(clouds), id="point_clouds"),
    pytest.param(_l1_matrices, id="distance_matrices"),
])
def test_critical_edges_of_persistent_homology_of_batch(make_elements):
    from stablebear import _sb_cpp as cpp

    ripser = cpp.persistence.PersistenceRipser64
    clouds = _batch2() + [np.zeros((0, 2))]

    bars, enumeration, edges = ripser.persistence_critical_edges(make_elements(clouds)._data, 1, True)

    assert enumeration.shape == (4, 2, 2)
    # The empty cloud has no bars.
    np.testing.assert_array_equal(enumeration[3], [[len(bars), len(bars)],
                                                   [len(bars), len(bars)]])
    for i, cloud in enumerate(clouds[:3]):
        alone_bars, alone_enumeration, alone_edges = ripser.persistence_critical_edges(
            make_elements([cloud])._data, 1, True)
        start, stop = enumeration[i, 0, 0], enumeration[i, 1, 1]
        np.testing.assert_array_equal(bars[start:stop], alone_bars)
        np.testing.assert_array_equal(enumeration[i] - start, alone_enumeration[0])
        # The edges are pairs of points of the element itself.
        np.testing.assert_array_equal(edges[start:stop], alone_edges)
        assert edges[start:stop].max() < len(cloud)


@pytest.mark.parametrize("make_elements", [
    pytest.param(lambda clouds: sb.PointCloudTensor(clouds), id="point_clouds"),
    pytest.param(_l1_matrices, id="distance_matrices"),
])
def test_critical_edges_of_kernel_of_batch(make_elements):
    from stablebear import _sb_cpp as cpp

    kernel = cpp.persistence.HomologicalKernel64
    clouds = _batch2()
    contracted = [_contracted(cloud) for cloud in clouds]

    bars, enumeration, edges = kernel.homological_kernel_critical_edges(
        make_elements(clouds)._data, make_elements(contracted)._data)

    assert enumeration.shape == (3, 2)
    for i, cloud in enumerate(clouds):
        alone_bars, _, alone_edges = kernel.homological_kernel_critical_edges(
            make_elements([cloud])._data, make_elements([contracted[i]])._data)
        start, stop = enumeration[i]
        # One bar per point but one.
        assert stop - start == len(cloud) - 1
        np.testing.assert_array_equal(bars[start:stop], alone_bars)
        np.testing.assert_array_equal(edges[start:stop], alone_edges)
        assert edges[start:stop].max() < len(cloud)


def test_point_cloud_tensor_of_list_with_framework_arrays_carries_gradients(fw):
    first, second, third = _batch3()

    clouds = sb.PointCloudTensor([fw.tensor(first), second, fw.tensor(third)])

    assert isinstance(clouds, sb.PointCloudTensor)
    assert clouds._tvalues is not None
    assert clouds.shape == (3,)
    assert clouds.array_equal(sb.PointCloudTensor(_batch3()))


def test_framework_array_of_three_dimensions_is_a_batch_of_point_clouds(fw):
    # Two clouds of three points in the plane.
    array = np.arange(12.0).reshape(2, 3, 2)

    clouds = sb.PointCloudTensor(fw.tensor(array))

    assert clouds._tvalues is not None
    assert clouds.shape == (2,)
    assert clouds.array_equal(sb.PointCloudTensor(array))


def test_inner_product_space_of_framework_batch_holds_a_point_cloud_tensor(fw):
    array = np.arange(12.0).reshape(2, 3, 2)

    space = sb.InnerProduct(_gram())(fw.tensor(array))

    assert isinstance(space.points, sb.PointCloudTensor)
    assert space.points._tvalues is not None
    assert space.points.shape == (2,)


def test_list_of_framework_point_clouds_of_different_dimensions_raises(fw):
    with pytest.raises(ValueError, match="same d"):
        sb.PointCloudTensor([fw.tensor(_points()), fw.tensor(_loop_points())])


def test_projection_gradient_wrt_one_point_cloud_of_a_batch(fw):
    _, second, third = _batch3()
    # One weight per coordinate of the 11 points of the batch.
    weights = _batch_weights()

    def loss(P):
        clouds = sb.PointCloudTensor([P, second, third])
        return fw.sum(fw.tensor(weights) * sb.LinearSubspace(_vectors()).project(clouds)._tvalues)

    assert_gradient_matches_finite_differences(fw, loss, _batch3()[0])


_TRACKED_BATCH_DISTANCES = [
    pytest.param(None, id="plain"),
    pytest.param(lambda: sb.InnerProduct(_gram()).distance, id="inner_product"),
    pytest.param(lambda: sb.LpDistance(1), id="taxicab"),
]


def _in_distance(make_distance, points):
    return points if make_distance is None else make_distance()(points)


@pytest.mark.parametrize("make_distance", _TRACKED_BATCH_DISTANCES)
def test_persistent_homology_of_tracked_batch_matches_untracked(fw, make_distance):
    clouds = sb.PointCloudTensor(_batch2())

    out = _homology(_in_distance(make_distance, tracked(fw, clouds)))

    expected = _homology(_in_distance(make_distance, clouds))
    assert isinstance(out, sb.persistence.BarcodeTensor)
    assert out._tvalues is not None
    assert out.shape == (3, 2)
    for i in np.ndindex(3, 2):
        assert out[i].is_isomorphic_to(expected[i], atol=1e-12, rtol=1e-12)


def test_persistent_homology_of_tracked_batch_of_one_point_cloud_has_no_batch_axis(fw):
    out = _homology(tracked(fw, sb.PointCloudTensor([_loop_points()])))

    assert out.shape == (2,)
    _assert_same_barcodes(out, _homology(_loop_points()))


@pytest.mark.parametrize("make_distance", _TRACKED_BATCH_DISTANCES)
def test_persistent_homology_of_batch_gradient_wrt_points(fw, make_distance):
    _, second, third = _batch2()

    def loss(P):
        clouds = sb.PointCloudTensor([second, P, third])
        return _bars_loss(fw, _homology(_in_distance(make_distance, clouds)))

    assert_gradient_matches_finite_differences(fw, loss, _lp_points())


def test_persistent_homology_of_projected_batch_gradient_wrt_subspace(fw):
    space = sb.InnerProduct(_gram3())(sb.PointCloudTensor(_batch3()))

    def loss(V):
        return _bars_loss(fw, _homology(sb.LinearSubspace(V).project(space)))

    assert_gradient_matches_finite_differences(fw, loss, _vectors())


def test_persistent_homology_of_batch_gradient_wrt_gram_matrix(fw):
    clouds = sb.PointCloudTensor(_batch2())

    def loss(a):
        return _bars_loss(fw, _homology(sb.InnerProduct(_framework_gram2(fw, a))(clouds)))

    assert_gradient_matches_finite_differences(fw, loss, [2.0, 1.0])


_TRACKED_KERNEL_DISTANCES = [
    pytest.param(None, id="plain"),
    # A diagonal Gram matrix, under which _contracted still shrinks every
    # distance.
    pytest.param(lambda: sb.InnerProduct(np.diag([4.0, 1.0])).distance, id="inner_product"),
    pytest.param(lambda: sb.LpDistance(1), id="taxicab"),
]


@pytest.mark.parametrize("make_distance", _TRACKED_KERNEL_DISTANCES)
@pytest.mark.parametrize("which", ["X", "Y"])
def test_kernel_of_tracked_batches_matches_untracked(fw, make_distance, which):
    X = sb.PointCloudTensor(_batch2())
    Y = sb.PointCloudTensor([_contracted(cloud) for cloud in _batch2()])

    out = _kernel(_in_distance(make_distance, tracked(fw, X) if which == "X" else X),
                  _in_distance(make_distance, tracked(fw, Y) if which == "Y" else Y))

    expected = _kernel(_in_distance(make_distance, X), _in_distance(make_distance, Y))
    assert isinstance(out, sb.persistence.BarcodeTensor)
    assert out._tvalues is not None
    assert out.shape == (3,)
    for i in range(3):
        assert out[i].is_isomorphic_to(expected[i], atol=1e-12, rtol=1e-12)


def test_kernel_of_batch_and_its_projection_gradient_wrt_subspace(fw):
    space = sb.InnerProduct(_gram3())(sb.PointCloudTensor(_batch3()))

    def loss(V):
        bars = _kernel(space, sb.LinearSubspace(V).project(space))._tvalues
        return fw.sum(bars[:, 0] * bars[:, 0])

    assert_gradient_matches_finite_differences(fw, loss, _vectors())


@pytest.mark.parametrize("make_distance", _TRACKED_KERNEL_DISTANCES)
def test_kernel_of_batches_gradient_wrt_points(fw, make_distance):
    other = _lp_points()[[0, 2, 3, 5]] * 1.5
    Y = sb.PointCloudTensor([_contracted(other), _contracted(_lp_points())])

    def loss(P):
        X = sb.PointCloudTensor([other, P])
        bars = _kernel(_in_distance(make_distance, X), _in_distance(make_distance, Y))._tvalues
        return fw.sum(bars[:, 0] * bars[:, 1]) + fw.sum(bars[:, 1] * bars[:, 1])

    assert_gradient_matches_finite_differences(fw, loss, _lp_points())


def test_kernel_of_tracked_batches_of_different_shapes_raises(fw):
    X = tracked(fw, sb.PointCloudTensor(_batch2()))
    Y = sb.PointCloudTensor([_contracted(cloud) for cloud in _batch2()[:2]])
    with pytest.raises(ValueError, match="same shape"):
        _kernel(X, Y)


def test_kernel_of_tracked_batch_and_single_point_cloud_raises(fw):
    X = tracked(fw, sb.PointCloudTensor(_batch2()))
    with pytest.raises(TypeError, match="both"):
        _kernel(X, _contracted(_loop_points()))


def _indexed_clouds():
    # Three subsamples of four of the six points of _cloud3, which share its
    # coordinates.
    return sb.random.subsample(
        sb.PointCloud(_cloud3()), n_points=4, n_samples=3, replace=False, allow_partial="keep",
        discard_duplicates=False, generator=sb.random.Generator(seed=1), verbose=False)


def test_projection_of_indexed_batch_is_that_of_its_dense_copy():
    indexed = _indexed_clouds()
    subspace = sb.LinearSubspace(_vectors())

    projected = subspace.project(indexed)

    expected = subspace.project(indexed.to_dense())
    assert projected.shape == (3,)
    for i in range(3):
        np.testing.assert_allclose(np.asarray(projected[i]), np.asarray(expected[i]), atol=1e-12)


def test_persistent_homology_under_lp_distance_of_indexed_batch_is_that_of_its_dense_copy():
    indexed = _indexed_clouds()
    taxicab = sb.LpDistance(1)

    out = _homology(taxicab(indexed))

    expected = _homology(taxicab(indexed.to_dense()))
    assert out.shape == (3, 2)
    for i in np.ndindex(3, 2):
        assert out[i].is_isomorphic_to(expected[i], atol=1e-12, rtol=1e-12)


def _series3():
    # The clouds of _batch3 at the times 0, 1, and 2.5.
    return sb.timeseries.TimeSeries(sb.PointCloudTensor(_batch3()), times=[0.0, 1.0, 2.5])


def _series2():
    # The clouds of _batch2 at the times 0, 1, and 2.5.
    return sb.timeseries.TimeSeries(sb.PointCloudTensor(_batch2()), times=[0.0, 1.0, 2.5])


@pytest.mark.parametrize("gram", [None, _gram3()], ids=["euclidean", "inner_product"])
def test_projection_of_series_is_the_series_of_the_projections(gram):
    subspace = sb.LinearSubspace(_vectors())

    def in_space(points):
        return points if gram is None else sb.InnerProduct(gram)(points)

    projected = subspace.project(in_space(_series3()))

    if gram is not None:
        assert isinstance(projected, sb.InnerProductSpace)
        projected = projected.points
    assert isinstance(projected, sb.timeseries.TimeSeries)
    assert projected.times.array_equal(_series3().times)
    for i, cloud in enumerate(_batch3()):
        alone = subspace.project(in_space(cloud))
        alone = alone if gram is None else alone.points
        np.testing.assert_allclose(np.asarray(projected.values[i]), np.asarray(alone), atol=1e-12)


@pytest.mark.parametrize("make_distance", _BATCH_DISTANCES)
def test_persistent_homology_of_series_in_distance_space_is_the_series_of_the_barcodes(make_distance):
    distance = make_distance()

    out = _homology(distance(_series2()))

    assert isinstance(out, sb.timeseries.TimeSeries)
    assert out.times.array_equal(_series2().times)
    for i, cloud in enumerate(_batch2()):
        alone = _homology(distance(cloud))
        for dim in range(2):
            assert out.values[i, dim].is_isomorphic_to(alone[dim], atol=1e-12, rtol=1e-12)


def test_distance_space_of_series_that_does_not_hold_point_clouds_raises():
    series = sb.timeseries.TimeSeries([10.0, 20.0], times=[0.0, 1.0])
    with pytest.raises(TypeError, match="point clouds"):
        sb.LpDistance(1)(series)


def test_projection_of_series_onto_framework_subspace_raises(fw):
    with pytest.raises(NotImplementedError, match="gradients"):
        sb.LinearSubspace(fw.tensor(_vectors())).project(_series3())
