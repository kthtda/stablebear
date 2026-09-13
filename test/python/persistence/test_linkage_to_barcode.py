"""Tests for converting linkage matrices to barcodes."""

import warnings

import numpy as np
import pytest

from stablebear.persistence import Barcode, linkage_to_barcode


def assert_linkage_barcode(linkage, expected_reduced, *, reduced):
    bars = list(expected_reduced)
    if not reduced:
        bars.append([0, np.inf])
    expected = Barcode(np.array(bars, dtype=linkage.dtype).reshape(-1, 2))

    original = linkage.copy()
    barcode = linkage_to_barcode(linkage, reduced=reduced)
    np.testing.assert_array_equal(linkage, original)
    assert isinstance(barcode, Barcode)
    assert barcode.to_numpy().dtype == linkage.dtype
    assert barcode.is_isomorphic_to(expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("reduced", [False, True])
class TestLinkageToBarcode:
    def test_single_observation(self, dtype, reduced):
        # SciPy rejects a single observation; we extend its n - 1 row convention
        # by accepting an empty linkage matrix to represent one observation.
        linkage = np.empty((0, 4), dtype=dtype)
        expected = []

        assert_linkage_barcode(linkage, expected, reduced=reduced)

    def test_single_row(self, dtype, reduced):
        linkage = np.array([[0, 1, 1.0, 2]], dtype=dtype)
        expected = [[0, 1.0]]

        assert_linkage_barcode(linkage, expected, reduced=reduced)

    def test_two_rows(self, dtype, reduced):
        linkage = np.array([[0, 1, 1.0, 2], [2, 3, 2.0, 3]], dtype=dtype)
        expected = [[0, 1.0], [0, 2.0]]

        assert_linkage_barcode(linkage, expected, reduced=reduced)

    def test_four_observations_with_simultaneous_merges(self, dtype, reduced):
        linkage = np.array(
            [[0, 1, 1.0, 2], [2, 3, 1.0, 2], [4, 5, 2.0, 4]],
            dtype=dtype,
        )
        expected = [[0, 1.0], [0, 1.0], [0, 2.0]]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert_linkage_barcode(linkage, expected, reduced=reduced)
        assert not caught

    def test_zero_height_merge_followed_by_positive_merge(self, dtype, reduced):
        linkage = np.array([[0, 1, 0.0, 2], [2, 3, 1.0, 3]], dtype=dtype)
        expected = [[0, 1.0]]

        assert_linkage_barcode(linkage, expected, reduced=reduced)

    def test_three_way_merge_at_positive_height(self, dtype, reduced):
        linkage = np.array([[0, 1, 1.0, 2], [2, 3, 1.0, 3]], dtype=dtype)
        expected = [[0, 1.0], [0, 1.0]]

        assert_linkage_barcode(linkage, expected, reduced=reduced)

    def test_all_zero_length_bars_are_omitted(self, dtype, reduced):
        linkage = np.array([[0, 1, 0.0, 2], [2, 3, 0.0, 3]], dtype=dtype)
        expected = []

        assert_linkage_barcode(linkage, expected, reduced=reduced)

    @pytest.mark.parametrize(
        "height",
        [-1.0, np.nan, np.inf, -np.inf],
        ids=["negative", "nan", "positive_infinity", "negative_infinity"],
    )
    def test_invalid_height(self, dtype, reduced, height):
        linkage = np.array([[0, 1, height, 2]], dtype=dtype)

        with pytest.raises(ValueError):
            linkage_to_barcode(linkage, reduced=reduced)

    def test_independent_branches_with_decreasing_heights(self, dtype, reduced):
        linkage = np.array(
            [[0, 1, 3.0, 2], [2, 3, 1.0, 2], [4, 5, 4.0, 4]],
            dtype=dtype,
        )
        expected = [[0, 3.0], [0, 1.0], [0, 4.0]]

        with pytest.warns(UserWarning, match="non-monotone") as caught:
            assert_linkage_barcode(linkage, expected, reduced=reduced)
        assert len(caught) == 1
        assert caught[0].filename == __file__

    def test_inversion_does_not_raise_independent_branch(self, dtype, reduced):
        linkage = np.array(
            [
                [0, 1, 3.0, 2],
                [2, 5, 2.0, 3],
                [3, 4, 1.0, 2],
                [6, 7, 4.0, 5],
            ],
            dtype=dtype,
        )
        expected = [[0, 3.0], [0, 3.0], [0, 1.0], [0, 4.0]]

        with pytest.warns(UserWarning, match="non-monotone") as caught:
            assert_linkage_barcode(linkage, expected, reduced=reduced)
        assert len(caught) == 1
        assert caught[0].filename == __file__

    def test_cascading_inversions(self, dtype, reduced):
        linkage = np.array(
            [[0, 1, 3.0, 2], [2, 4, 2.0, 3], [3, 5, 1.0, 4]],
            dtype=dtype,
        )
        expected = [[0, 3.0], [0, 3.0], [0, 3.0]]

        with pytest.warns(UserWarning, match="non-monotone") as caught:
            assert_linkage_barcode(linkage, expected, reduced=reduced)
        assert len(caught) == 1
        assert caught[0].filename == __file__

    def test_parent_merge_before_child(self, dtype, reduced):
        # A minimal inversion like SciPy's median-linkage example: the final
        # merge must wait until both child clusters exist, at time 3.5.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.is_monotonic.html
        linkage = np.array(
            [[0, 1, 3.0, 2], [2, 3, 3.5, 2], [4, 5, 3.25, 4]],
            dtype=dtype,
        )
        expected = [[0, 3.0], [0, 3.5], [0, 3.5]]

        with pytest.warns(UserWarning, match="non-monotone") as caught:
            assert_linkage_barcode(linkage, expected, reduced=reduced)
        assert len(caught) == 1
        assert caught[0].filename == __file__

    @pytest.mark.parametrize("column", [0, 1], ids=["left", "right"])
    @pytest.mark.parametrize(
        "reference, error",
        [
            (0.5, "existing clusters"),
            (-1, "existing clusters"),
            (2, "existing clusters"),
            (10, "existing clusters"),
            (np.nan, "finite"),
            (np.inf, "finite"),
            (-np.inf, "finite"),
        ],
        ids=[
            "fractional", "negative", "not_yet_created", "out_of_range",
            "nan", "positive_infinity", "negative_infinity",
        ],
    )
    def test_invalid_cluster_reference(self, dtype, reduced, column, reference, error):
        linkage = np.array([[0, 1, 1.0, 2]], dtype=dtype)
        linkage[0, column] = reference

        with pytest.raises(ValueError, match=error):
            linkage_to_barcode(linkage, reduced=reduced)

    def test_reused_observation(self, dtype, reduced):
        linkage = np.array([[0, 1, 1.0, 2], [0, 2, 2.0, 2]], dtype=dtype)

        with pytest.raises(ValueError, match="active"):
            linkage_to_barcode(linkage, reduced=reduced)

    def test_reused_cluster(self, dtype, reduced):
        linkage = np.array(
            [[0, 1, 1.0, 2], [2, 4, 2.0, 3], [3, 4, 3.0, 3]],
            dtype=dtype,
        )

        with pytest.raises(ValueError, match="active"):
            linkage_to_barcode(linkage, reduced=reduced)

    def test_cluster_merged_with_itself(self, dtype, reduced):
        linkage = np.array([[0, 0, 1.0, 2]], dtype=dtype)

        with pytest.raises(ValueError, match="distinct"):
            linkage_to_barcode(linkage, reduced=reduced)

    @pytest.mark.parametrize("count", [-1, 0, 1, 3, 2.5])
    def test_incorrect_observation_merge_count(self, dtype, reduced, count):
        linkage = np.array([[0, 1, 1.0, count]], dtype=dtype)

        with pytest.raises(ValueError, match="cluster count"):
            linkage_to_barcode(linkage, reduced=reduced)

    @pytest.mark.parametrize("count", [2, 3, 5])
    def test_incorrect_cluster_merge_count(self, dtype, reduced, count):
        linkage = np.array(
            [[0, 1, 1.0, 2], [2, 3, 1.0, 2], [4, 5, 2.0, count]],
            dtype=dtype,
        )

        with pytest.raises(ValueError, match="cluster count"):
            linkage_to_barcode(linkage, reduced=reduced)

    @pytest.mark.parametrize(
        "count",
        [np.nan, np.inf, -np.inf],
        ids=["nan", "positive_infinity", "negative_infinity"],
    )
    def test_nonfinite_cluster_count(self, dtype, reduced, count):
        linkage = np.array([[0, 1, 1.0, count]], dtype=dtype)

        with pytest.raises(ValueError, match="finite"):
            linkage_to_barcode(linkage, reduced=reduced)

    def test_noncontiguous_view(self, dtype, reduced):
        storage = np.zeros((2, 8), dtype=dtype)
        linkage = storage[:, ::2]
        linkage[:] = [[0, 1, 1.0, 2], [2, 3, 2.0, 3]]
        expected = [[0, 1.0], [0, 2.0]]

        assert_linkage_barcode(linkage, expected, reduced=reduced)

    def test_negative_strides(self, dtype, reduced):
        storage = np.array([[3, 2.0, 3, 2], [2, 1.0, 1, 0]], dtype=dtype)
        linkage = storage[::-1, ::-1]
        expected = [[0, 1.0], [0, 2.0]]

        assert_linkage_barcode(linkage, expected, reduced=reduced)

    def test_output_independent_of_input(self, dtype, reduced):
        linkage = np.array([[0, 1, 1.0, 2], [2, 3, 2.0, 3]], dtype=dtype)
        barcode = linkage_to_barcode(linkage, reduced=reduced)
        original_bars = barcode.to_numpy().copy()

        linkage[:] = -1

        np.testing.assert_array_equal(barcode.to_numpy(), original_bars)

    def test_readonly_input(self, dtype, reduced):
        linkage = np.array([[0, 1, 1.0, 2], [2, 3, 2.0, 3]], dtype=dtype)
        linkage.flags.writeable = False
        expected = [[0, 1.0], [0, 2.0]]

        assert_linkage_barcode(linkage, expected, reduced=reduced)
        assert not linkage.flags.writeable

    @pytest.mark.parametrize(
        "shape",
        [(), (4,), (0,), (1, 0), (1, 3), (1, 5), (0, 3), (0, 5), (1, 1, 4)],
    )
    def test_wrong_shape(self, dtype, reduced, shape):
        linkage = np.zeros(shape, dtype=dtype)

        with pytest.raises(ValueError, match="shape"):
            linkage_to_barcode(linkage, reduced=reduced)


@pytest.mark.parametrize("reduced", [False, True])
class TestInvalidLinkageTypes:
    @pytest.mark.parametrize(
        "dtype",
        [np.bool_, np.int32, np.int64, np.uint64, np.float16,
         np.complex64, np.complex128, object, np.str_],
    )
    def test_unsupported_dtype(self, reduced, dtype):
        linkage = np.array([[0, 1, 1.0, 2]], dtype=dtype)

        with pytest.raises(TypeError):
            linkage_to_barcode(linkage, reduced=reduced)

    @pytest.mark.parametrize(
        "linkage",
        [None, 1.0, [[0, 1, 1.0, 2]], ((0, 1, 1.0, 2),)],
        ids=["none", "scalar", "list", "tuple"],
    )
    def test_non_array_input(self, reduced, linkage):
        with pytest.raises(TypeError):
            linkage_to_barcode(linkage, reduced=reduced)
