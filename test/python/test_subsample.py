"""Acceptance tests for uniform point-cloud subsampling (issue #229)."""

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb
from stablebear.point_process import subsample


_PCLOUD_DTYPES = [
    pytest.param(sb.pcloud32, np.float32, id="pcloud32"),
    pytest.param(sb.pcloud64, np.float64, id="pcloud64"),
]


def point_cloud(n_pts, dim, dtype):
    return np.arange(n_pts * dim, dtype=dtype).reshape(n_pts, dim)


def size_case(*, shape, n_points, replace, allow_partial, expected_n_pts, id):
    return pytest.param(
        shape,
        n_points,
        replace,
        allow_partial,
        expected_n_pts,
        id=id,
    )


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
    def test_write_through_outer_view_materializes_shared_backing_once(
        self, pcloud_dtype, np_dtype
    ):
        coordinates = np.asarray([[1, 2]], dtype=np_dtype)
        points = sb.PointCloudTensor(coordinates, dtype=pcloud_dtype)
        samples = subsample(
            points,
            n_points=2,
            n_samples=2,
            replace=True,
            generator=sb.random.Generator(seed=229),
        )
        separate = subsample(
            points,
            n_points=2,
            n_samples=2,
            replace=True,
            generator=sb.random.Generator(seed=229),
        )

        view = samples[:1]
        sibling = samples[...]
        cell = samples[0]
        other_sample_before = np.asarray(samples[1]).copy()
        source_before = np.asarray(points[()]).copy()
        separate_before = np.asarray(separate[0]).copy()

        cast_dtype = sb.pcloud64 if pcloud_dtype == sb.pcloud32 else sb.pcloud32
        cast = view.astype(cast_dtype)
        cast[0][0, 0] = -7
        assert cast.dtype == cast_dtype
        assert view[0][0, 0] == 1

        view[0][0, 0] = 999

        # Parent, sibling, and an already-extracted cell are aliases of the
        # same logical result cell. Repeated selected rows, other result cells,
        # the input, and independently sampled results are not aliases.
        assert samples[0][0, 0] == 999
        assert sibling[0][0, 0] == 999
        assert cell[0, 0] == 999
        assert samples[0][1, 0] == 1
        npt.assert_array_equal(np.asarray(samples[1]), other_sample_before)
        npt.assert_array_equal(np.asarray(points[()]), source_before)
        npt.assert_array_equal(np.asarray(separate[0]), separate_before)

        # A later write must use the existing materialized backing. If it
        # materialized the original indexed source again, the first write
        # would be lost.
        samples[0][1, 1] = 888
        assert view[0][0, 0] == 999
        assert cell[1, 1] == 888

    @pytest.mark.parametrize(
        ("shape", "n_points", "replace", "allow_partial", "expected_n_pts"),
        [
            size_case(shape=(0, 0), n_points=3, replace=False, allow_partial=True,
                      expected_n_pts=0, id="empty-zero-dimensional", ),
            size_case(shape=(0, 3), n_points=3, replace=False, allow_partial=True,
                      expected_n_pts=0, id="empty", ),
            size_case(shape=(0, 3), n_points=3, replace=True, allow_partial=True,
                      expected_n_pts=0, id="empty-replacement", ),
            size_case(shape=(1, 1), n_points=1, replace=False, allow_partial=False,
                      expected_n_pts=1, id="singleton", ),
            size_case(shape=(1, 3), n_points=4, replace=True, allow_partial=False,
                      expected_n_pts=4, id="singleton-replacement", ),
            size_case(shape=(3, 0), n_points=2, replace=False, allow_partial=False,
                      expected_n_pts=2, id="zero-dimensional", ),
            size_case(shape=(4, 3), n_points=1, replace=False, allow_partial=False,
                      expected_n_pts=1, id="below-population", ),
            size_case(shape=(4, 3), n_points=4, replace=False, allow_partial=False,
                      expected_n_pts=4, id="full-population", ),
            size_case(shape=(4, 3), n_points=7, replace=True, allow_partial=False,
                      expected_n_pts=7, id="above-population-replacement", ),
            size_case(shape=(7, 32), n_points=4, replace=False, allow_partial=False,
                      expected_n_pts=4, id="large-dimension", ),
        ],
    )
    def test_scalar_sizes(
        self,
        pcloud_dtype,
        np_dtype,
        shape,
        n_points,
        replace,
        allow_partial,
        expected_n_pts,
    ):
        points = sb.PointCloudTensor(
            point_cloud(*shape, dtype=np_dtype),
            dtype=pcloud_dtype,
        )

        actual = subsample(
            points,
            n_points=n_points,
            n_samples=2,
            replace=replace,
            allow_partial=allow_partial,
            generator=sb.random.Generator(seed=229),
        )

        assert_point_cloud_tensor_sizes(
            points,
            PointCloudTensorSizes(shape, pcloud_dtype),
        )
        assert_point_cloud_tensor_sizes(
            actual,
            PointCloudTensorSizes(
                [(expected_n_pts, shape[1]), (expected_n_pts, shape[1])],
                pcloud_dtype,
            ),
        )

    def test_ragged_sizes(self, pcloud_dtype, np_dtype):
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

        for index in np.ndindex(*actual.shape):
            sampled = actual._data._get_element(list(index))
            indices = np.asarray(sampled.indices).copy()
            coordinates = np.asarray(sampled.coords).copy()
            npt.assert_array_equal(coordinates, clouds[index[0]])
            npt.assert_array_equal(np.asarray(actual[index]), coordinates[indices])

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

    def test_resampling_indexed_input_matches_dense_without_materializing(
        self, pcloud_dtype, np_dtype, monkeypatch
    ):
        points = sb.PointCloudTensor(
            np.arange(24, dtype=np_dtype).reshape(2, 6, 2), dtype=pcloud_dtype
        )
        selected = points[
            sb.NestedTensor([sb.indices([4, 1, 4, 0]), sb.indices([5, 0, 5, 2])])
        ]
        dense = selected.copy()
        actual_gen = sb.random.Generator(seed=301)
        expected_gen = sb.random.Generator(seed=301)
        options = dict(n_points=3, n_samples=2)

        def reject_intermediate_materialization(*args):
            pytest.fail("resampling must pass indexed input directly to C++")

        with monkeypatch.context() as patch:
            patch.setattr(
                type(selected._data), "materialize", reject_intermediate_materialization
            )
            actual = subsample(selected, generator=actual_gen, **options)
        expected = subsample(dense, generator=expected_gen, **options)
        assert actual.array_equal(expected)
        assert selected._data._get_element([0]).is_indexed

        # Direct indexed dispatch must not alter random-stream allocation.
        actual_next = subsample(points, generator=actual_gen, **options)
        expected_next = subsample(points, generator=expected_gen, **options)
        assert actual_next.array_equal(expected_next)
