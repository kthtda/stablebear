"""Acceptance tests for point-cloud and distance-matrix subsampling."""

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb
from stablebear.random import subsample


_PCLOUD_DTYPES = [
    pytest.param(sb.pcloud32, np.float32, id="pcloud32"),
    pytest.param(sb.pcloud64, np.float64, id="pcloud64"),
]

_DISTMAT_DTYPES = [
    pytest.param(sb.distmat32, np.float32, id="distmat32"),
    pytest.param(sb.distmat64, np.float64, id="distmat64"),
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


@pytest.fixture
def sample_data(make_pcloud_or_distmat_tensor):
    """Three six-point clouds (or matrices) with different internal distances."""
    return make_pcloud_or_distmat_tensor([
        [[0, 0], [1, 0], [0, 2], [3, 1], [4, 3], [2, 5]],
        [[10, 0], [12, 0], [10, 4], [16, 2], [18, 6], [14, 10]],
        [[0, 20], [3, 20], [0, 26], [9, 23], [12, 29], [6, 35]],
    ])


@pytest.fixture
def selected_data(sample_data):
    return sample_data[sb.NestedTensor([
        sb.indices([4, 1, 4, 0]), sb.indices([5, 0, 5, 2]), sb.indices([3, 2, 3, 1]),
    ])]


@pytest.fixture(params=[(sb.pcloud32, np.float32), (sb.pcloud64, np.float64)],
                ids=["pcloud32", "pcloud64"])
def ragged_data(request):
    dtype, np_dtype = request.param
    return sb.PointCloudTensor([
        point_cloud(4, 3, np_dtype), point_cloud(5, 3, np_dtype) + 100,
    ], dtype=dtype)


class TestSubsample:
    @pytest.mark.parametrize(("pcloud_dtype", "np_dtype"), _PCLOUD_DTYPES)
    def test_single_cloud_matches_scalar_tensor(self, pcloud_dtype, np_dtype):
        coordinates = np.array([
            [0, 0],
            [3, 0],
            [0, 4],
        ], dtype=np_dtype)
        cloud = sb.PointCloud(coordinates)
        tensor = sb.PointCloudTensor(coordinates)

        samples = subsample(cloud, n_points=2, n_samples=2,
                            generator=sb.random.Generator(seed=229))
        expected = subsample(tensor, n_points=2, n_samples=2,
                             generator=sb.random.Generator(seed=229))

        assert samples.shape == (2,)
        assert samples.dtype == pcloud_dtype
        assert samples.array_equal(expected)

    @pytest.mark.parametrize(("distmat_dtype", "np_dtype"), _DISTMAT_DTYPES)
    def test_single_matrix_matches_scalar_tensor(self, distmat_dtype, np_dtype):
        distances = np.array([
            [0, 3, 4],
            [3, 0, 5],
            [4, 5, 0],
        ], dtype=np_dtype)
        matrix = sb.DistanceMatrix(distances)
        tensor = sb.DistanceMatrixTensor(distances)

        samples = subsample(matrix, n_points=2, n_samples=2,
                            generator=sb.random.Generator(seed=229))
        expected = subsample(tensor, n_points=2, n_samples=2,
                             generator=sb.random.Generator(seed=229))

        assert samples.shape == (2,)
        assert samples.dtype == distmat_dtype
        assert samples.array_equal(expected)

    @pytest.mark.parametrize("parameter", ["n_points", "n_samples"])
    def test_rejects_noninteger_count(self, parameter):
        source = sb.PointCloudTensor(np.array([[10., 1.]]))
        counts = dict(n_points=1, n_samples=1)
        counts[parameter] = 1.5

        with pytest.raises(TypeError, match=f"{parameter} must be an integer"):
            subsample(source, **counts)

    @pytest.mark.parametrize("parameter", ["n_points", "n_samples"])
    @pytest.mark.parametrize("count", [0, -1])
    def test_rejects_nonpositive_count(self, parameter, count):
        source = sb.PointCloudTensor(np.array([[10., 1.]]))
        counts = dict(n_points=1, n_samples=1)
        counts[parameter] = count

        with pytest.raises(ValueError, match=f"{parameter} must be greater than zero"):
            subsample(source, **counts)

    def test_undersized_last_input_does_not_advance_generator(self):
        source = sb.PointCloudTensor([
            np.array([
                [0., 0.],
                [1., 0.],
                [0., 2.],
                [3., 1.],
            ]),
            np.array([[10., 1.]]),
        ])
        generator = sb.random.Generator(seed=229)
        control_generator = sb.random.Generator(seed=229)

        # The first cloud is large enough; the second must reject the whole call.
        with pytest.raises(ValueError, match="n_points exceeds the number of input points"):
            subsample(source, n_points=3, generator=generator)

        valid_source = source[:1]
        after_failure = subsample(valid_source, n_points=3, n_samples=2,
                                  generator=generator)
        control = subsample(valid_source, n_points=3, n_samples=2,
                            generator=control_generator)
        assert after_failure.array_equal(control)

    def test_batched_sampling_matches_individual_calls_in_row_major_order(self):
        source = sb.PointCloudTensor([
            np.array([
                [10.],
                [20.],
                [30.],
                [40.],
            ]),
            np.array([
                [50.],
                [60.],
                [70.],
                [80.],
            ]),
        ])
        batched_generator = sb.random.Generator(seed=229)
        individual_generator = sb.random.Generator(seed=229)

        batched = subsample(source, n_points=3, n_samples=2,
                            generator=batched_generator)

        # Both samples of cloud 0 precede both samples of cloud 1.
        for cloud, sample in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            individual = subsample(source[cloud:cloud + 1], n_points=3,
                                   generator=individual_generator)
            npt.assert_array_equal(np.asarray(batched[cloud, sample]),
                                   np.asarray(individual[0, 0]))

    def test_sampling_does_not_change_input(self, sample_data):
        before = sample_data.copy()
        subsample(sample_data, n_points=3, n_samples=2, replace=True)
        assert sample_data.array_equal(before)

    def test_writing_input_view_does_not_change_sample(self, make_pcloud_or_distmat_tensor):
        source = make_pcloud_or_distmat_tensor([[
            [0, 0],
            [3, 4],
        ]])
        input_view = source[0]

        # Sample both points so the changed coordinate or distance is included.
        samples = subsample(source, n_points=2,
                            generator=sb.random.Generator(seed=229))
        samples_before = samples.copy()

        input_view[0, 1] = 999
        assert source[0][0, 1] == 999
        assert samples.array_equal(samples_before)

    def test_writing_sample_does_not_change_input(self, sample_data):
        source_before = sample_data.copy()
        result = subsample(sample_data, n_points=3, n_samples=2, replace=True)
        result[0, 0][0, 1] = 999
        assert result[0, 0][0, 1] == 999
        assert sample_data.array_equal(source_before)

    def test_writing_sample_does_not_change_other_samples(self, sample_data):
        result = subsample(sample_data, n_points=3, n_samples=2, replace=True)
        before = result.copy()
        result[0, 0][0, 1] = 999
        assert result[0, 0][0, 1] == 999
        npt.assert_array_equal(np.asarray(result[0, 1]), np.asarray(before[0, 1]))
        assert result[1:].array_equal(before[1:])

    def test_separate_sampling_calls_are_independent(self, sample_data):
        options = dict(n_points=3, n_samples=2, replace=True)
        first_result = subsample(sample_data, generator=sb.random.Generator(seed=229), **options)
        second_result = subsample(sample_data, generator=sb.random.Generator(seed=229), **options)
        second_result_before = second_result.copy()
        first_result[0, 0][0, 1] = 999
        assert first_result[0, 0][0, 1] == 999
        assert second_result.array_equal(second_result_before)

    def test_sample_axis_is_appended(self, sample_data):
        result = subsample(sample_data, n_points=3, n_samples=2)
        assert result.shape == (*sample_data.shape, 2)

    def test_sampling_preserves_dtype(self, sample_data):
        result = subsample(sample_data, n_points=3)
        assert result.dtype == sample_data.dtype

    def test_default_sampling_uses_each_vertex_once_per_full_sample(
        self, make_pcloud_or_distmat_tensor
    ):
        source = make_pcloud_or_distmat_tensor([
            [0, 0],
            [1, 0],
            [0, 2],
            [3, 1],
        ])

        samples = subsample(source, n_points=4, n_samples=2,
                            generator=sb.random.Generator(seed=229))
        indices = samples.indices
        assert indices is not None

        # Each sample starts from all four vertices, with no repeated draw.
        npt.assert_array_equal(np.sort(np.asarray(indices[0])), [0, 1, 2, 3])
        npt.assert_array_equal(np.sort(np.asarray(indices[1])), [0, 1, 2, 3])

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
            size_case(shape=(4, 3), n_points=7, replace=False, allow_partial=True,
                      expected_n_pts=4, id="above-population-partial", ),
            size_case(shape=(4, 3), n_points=7, replace=True, allow_partial=False,
                      expected_n_pts=7, id="above-population-replacement", ),
            size_case(shape=(7, 32), n_points=4, replace=False, allow_partial=False,
                      expected_n_pts=4, id="large-dimension", ),
        ],
    )
    def test_sample_sizes_for_single_input(
        self,
        make_pcloud_or_distmat_tensor,
        shape,
        n_points,
        replace,
        allow_partial,
        expected_n_pts,
    ):
        points = make_pcloud_or_distmat_tensor(point_cloud(*shape, dtype=np.float64))
        actual = subsample(
            points,
            n_points=n_points,
            n_samples=2,
            replace=replace,
            allow_partial=allow_partial,
            generator=sb.random.Generator(seed=229),
        )
        assert actual.shape == (2,)
        indices = actual.indices
        assert indices is not None
        expected_shape = (
            (expected_n_pts, shape[1]) if isinstance(points, sb.PointCloudTensor)
            else (expected_n_pts, expected_n_pts)
        )
        for sample in range(2):
            assert len(indices[sample]) == expected_n_pts
            assert np.asarray(actual[sample]).shape == expected_shape

    @pytest.mark.parametrize(("pcloud_dtype", "np_dtype"), _PCLOUD_DTYPES)
    def test_cloud_duplicate_filter_removes_equal_coordinates(
        self, pcloud_dtype, np_dtype
    ):
        source = sb.PointCloudTensor(np.array([
            [10, 1],
            [10, 1],
            [10, 1],
        ], dtype=np_dtype), dtype=pcloud_dtype)

        filtered = subsample(source, n_points=3, discard_duplicates=True,
                             generator=sb.random.Generator(seed=229))

        npt.assert_array_equal(np.asarray(filtered[0]), [[10, 1]])

    @pytest.mark.parametrize(("distmat_dtype", "np_dtype"), _DISTMAT_DTYPES)
    def test_matrix_duplicate_filter_removes_repeated_vertices(
        self, distmat_dtype, np_dtype
    ):
        source = sb.DistanceMatrixTensor(np.array([[0]], dtype=np_dtype),
                                         dtype=distmat_dtype)

        # Every draw selects the only vertex; filtering leaves it once.
        filtered = subsample(source, n_points=3, replace=True, discard_duplicates=True,
                             generator=sb.random.Generator(seed=229))

        npt.assert_array_equal(np.asarray(filtered[0]), [[0]])

    @pytest.mark.parametrize(("distmat_dtype", "np_dtype"), _DISTMAT_DTYPES)
    def test_matrix_duplicate_filter_preserves_distinct_zero_distance_vertices(
        self, distmat_dtype, np_dtype
    ):
        distances = np.array([
            [0, 0, 5],
            [0, 0, 5],
            [5, 5, 0],
        ], dtype=np_dtype)
        source = sb.DistanceMatrixTensor(distances, dtype=distmat_dtype)

        samples = subsample(source, n_points=3, discard_duplicates=True,
                            generator=sb.random.Generator(seed=229))
        indices = samples.indices
        assert indices is not None
        vertices = np.asarray(indices[0])

        # Vertices 0 and 1 have zero distance but are still distinct vertices.
        npt.assert_array_equal(np.sort(vertices), [0, 1, 2])
        npt.assert_array_equal(np.asarray(samples[0]), distances[np.ix_(vertices, vertices)])

    def test_duplicate_filtering_preserves_generator_advancement(self, sample_data):
        unfiltered_generator = sb.random.Generator(seed=229)
        filtered_generator = sb.random.Generator(seed=229)

        # Seven draws from each six-point input guarantee duplicates to remove.
        subsample(sample_data, n_points=7, replace=True, generator=unfiltered_generator)
        subsample(sample_data, n_points=7, replace=True, discard_duplicates=True,
                  generator=filtered_generator)

        next_unfiltered = subsample(sample_data, n_points=3, generator=unfiltered_generator)
        next_filtered = subsample(sample_data, n_points=3, generator=filtered_generator)
        # Filtering must not change the random stream used by the next call.
        assert next_filtered.array_equal(next_unfiltered)

    def test_ragged_sizes(self, ragged_data):
        actual = subsample(ragged_data, n_points=3, n_samples=2)
        assert actual.shape == (2, 2)
        for index in np.ndindex(*actual.shape):
            assert actual[index].shape == (3, 3)

    def test_each_sample_contains_the_selected_points_from_its_input_cloud(self, ragged_data):
        samples = subsample(ragged_data, n_points=3, n_samples=2)
        drawn_indices = samples.indices
        assert drawn_indices is not None

        for cloud_index in range(2):
            input_coordinates = np.asarray(ragged_data[cloud_index])
            for sample_index in range(2):
                selected_rows = np.asarray(drawn_indices[cloud_index, sample_index])
                expected_coordinates = input_coordinates[selected_rows]
                sampled_coordinates = np.asarray(samples[cloud_index, sample_index])
                npt.assert_array_equal(sampled_coordinates, expected_coordinates)

    @pytest.mark.parametrize(("pcloud_dtype", "np_dtype"), _PCLOUD_DTYPES)
    def test_resampling_cloud_uses_selected_points(self, pcloud_dtype, np_dtype):
        source = sb.PointCloudTensor(
            np.array([[10, 11], [20, 21], [30, 31]], dtype=np_dtype),
            dtype=pcloud_dtype)
        selected = source[sb.NestedTensor(sb.indices([2, 0, 2]))]
        # The middle source point is excluded; the last one appears twice.
        selected_coordinates = np.array([[30, 31], [10, 11], [30, 31]], dtype=np_dtype)

        samples = subsample(selected, n_points=2, n_samples=2,
                            generator=sb.random.Generator(seed=301))
        drawn_indices = samples.indices
        assert drawn_indices is not None
        for sample_index in range(2):
            drawn_rows = np.asarray(drawn_indices[sample_index])
            expected_coordinates = selected_coordinates[drawn_rows]
            npt.assert_array_equal(np.asarray(samples[sample_index]), expected_coordinates)

    @pytest.mark.parametrize(("distmat_dtype", "np_dtype"), _DISTMAT_DTYPES)
    def test_resampling_matrix_uses_selected_vertices(self, distmat_dtype, np_dtype):
        source = sb.DistanceMatrixTensor(
            np.array([[0, 5, 7], [5, 0, 9], [7, 9, 0]], dtype=np_dtype),
            dtype=distmat_dtype)
        selected = source[sb.NestedTensor(sb.indices([2, 0, 2]))]
        # Vertices 0 and 2 here are the same original vertex, so their distance is zero.
        selected_distances = np.array([[0, 7, 0], [7, 0, 7], [0, 7, 0]], dtype=np_dtype)

        samples = subsample(selected, n_points=2, n_samples=2,
                            generator=sb.random.Generator(seed=301))
        drawn_indices = samples.indices
        assert drawn_indices is not None
        for sample_index in range(2):
            drawn_vertices = np.asarray(drawn_indices[sample_index])
            expected_distances = selected_distances[np.ix_(drawn_vertices, drawn_vertices)]
            npt.assert_array_equal(np.asarray(samples[sample_index]), expected_distances)

    def test_resampling_does_not_materialize_input(self, selected_data, monkeypatch):
        def reject_materialization(*args):
            pytest.fail("resampling must read indexed input without materializing it")

        monkeypatch.setattr(type(selected_data._data), "materialize", reject_materialization)
        subsample(selected_data, n_points=3, n_samples=2)
        assert selected_data.indices is not None

    def test_resampling_consumes_the_same_random_streams_as_dense_input(
        self, sample_data, selected_data
    ):
        dense_data = selected_data.copy()
        indexed_generator = sb.random.Generator(seed=301)
        dense_generator = sb.random.Generator(seed=301)

        subsample(selected_data, n_points=3, n_samples=2, generator=indexed_generator)
        subsample(dense_data, n_points=3, n_samples=2, generator=dense_generator)

        # After the same number of draws, the next call should still agree.
        next_after_indexed = subsample(sample_data, n_points=3, generator=indexed_generator)
        next_after_dense = subsample(sample_data, n_points=3, generator=dense_generator)
        assert next_after_indexed.array_equal(next_after_dense)

    def test_writing_resampled_output_does_not_change_input(self, selected_data):
        selected_data_before = selected_data.copy()
        samples = subsample(selected_data, n_points=3, n_samples=2)
        samples[0, 0][0, 1] = 99
        assert samples[0, 0][0, 1] == 99
        assert selected_data.array_equal(selected_data_before)

    def test_writing_resampled_output_does_not_change_other_samples(self, selected_data):
        samples = subsample(selected_data, n_points=3, n_samples=2)
        untouched_sample = np.asarray(samples[0, 1]).copy()
        samples_from_other_inputs = samples[1:].copy()

        samples[0, 0][0, 1] = 99
        assert samples[0, 0][0, 1] == 99
        npt.assert_array_equal(np.asarray(samples[0, 1]), untouched_sample)
        assert samples[1:].array_equal(samples_from_other_inputs)
