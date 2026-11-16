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
