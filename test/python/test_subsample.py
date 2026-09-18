"""Acceptance tests for uniform point-cloud subsampling (issue #229)."""

import numpy as np
import pytest

import stablebear as sb
from stablebear.point_process import subsample


_PCLOUD_DTYPES = [
    pytest.param(sb.pcloud32, np.float32, id="pcloud32"),
    pytest.param(sb.pcloud64, np.float64, id="pcloud64"),
]


def point_cloud(n_pts, dim, dtype):
    return np.arange(n_pts * dim, dtype=dtype).reshape(n_pts, dim)


class PointCloudTensorSizes:
    def __init__(self, clouds, dtype):
        self.clouds = clouds
        self.dtype = dtype


class AtMost:
    def __init__(self, value):
        self.value = value

    def matches(self, actual):
        return actual <= self.value

    def __repr__(self):
        return f"AtMost({self.value})"


class AtLeast:
    def __init__(self, value):
        self.value = value

    def matches(self, actual):
        return actual >= self.value

    def __repr__(self):
        return f"AtLeast({self.value})"


def assert_point_cloud_tensor_sizes(actual, expected):
    def compare_shape(actual_shape, expected_shape):
        assert len(actual_shape) == len(expected_shape)
        for actual_size, expected_size in zip(actual_shape, expected_shape):
            if isinstance(expected_size, (AtMost, AtLeast)):
                assert expected_size.matches(actual_size), (
                    f"expected {expected_size}, got {actual_size}"
                )
            else:
                assert actual_size == expected_size

    def compare(clouds, index=()):
        if isinstance(clouds, tuple):
            compare_shape(actual[index].shape, clouds)
            return ()

        child_shapes = [compare(child, index + (i,)) for i, child in enumerate(clouds)]
        assert all(shape == child_shapes[0] for shape in child_shapes)
        return (len(clouds), *child_shapes[0])

    assert actual.dtype == expected.dtype
    assert actual.shape == compare(expected.clouds)


@pytest.mark.parametrize(("pcloud_dtype", "np_dtype"), _PCLOUD_DTYPES)
class TestSubsample:
    def test_sizes(self, pcloud_dtype, np_dtype):
        clouds = [
            point_cloud(4, 3, np_dtype),
            point_cloud(5, 3, np_dtype) + 100,
        ]
        points = sb.PointCloudTensor(clouds, dtype=pcloud_dtype)

        actual = subsample(
            points,
            n_points=3,
            n_samples=2,
            generator=sb.random.Generator(seed=229),
        )

        assert_point_cloud_tensor_sizes(
            points,
            PointCloudTensorSizes([(4, 3), (5, 3)], pcloud_dtype),
        )
        assert_point_cloud_tensor_sizes(
            actual,
            PointCloudTensorSizes(
                [
                    [(3, 3), (3, 3)],
                    [(3, 3), (3, 3)],
                ],
                pcloud_dtype,
            ),
        )
