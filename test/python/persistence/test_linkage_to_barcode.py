"""Tests for converting linkage matrices to barcodes."""

import numpy as np
import pytest

from stablebear.persistence import Barcode, linkage_to_barcode


def assert_linkage_barcode(linkage, expected_reduced, *, reduced):
    bars = list(expected_reduced)
    if not reduced:
        bars.append([0, np.inf])
    expected = Barcode(np.array(bars, dtype=linkage.dtype).reshape(-1, 2))

    barcode = linkage_to_barcode(linkage, reduced=reduced)
    assert isinstance(barcode, Barcode)
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

        assert_linkage_barcode(linkage, expected, reduced=reduced)

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

    def test_parent_merge_before_child(self, dtype, reduced):
        # A minimal inversion like SciPy's median-linkage example: the final
        # merge must wait until both child clusters exist, at time 3.5.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.is_monotonic.html
        linkage = np.array(
            [[0, 1, 3.0, 2], [2, 3, 3.5, 2], [4, 5, 3.25, 4]],
            dtype=dtype,
        )
        expected = [[0, 3.0], [0, 3.5], [0, 3.5]]

        assert_linkage_barcode(linkage, expected, reduced=reduced)

    @pytest.mark.parametrize("column", [0, 1], ids=["left", "right"])
    @pytest.mark.parametrize(
        "reference",
        [0.5, -1, 2, 10],
        ids=["fractional", "negative", "not_yet_created", "out_of_range"],
    )
    def test_invalid_cluster_reference(self, dtype, reduced, column, reference):
        linkage = np.array([[0, 1, 1.0, 2]], dtype=dtype)
        linkage[0, column] = reference

        with pytest.raises(ValueError, match="existing clusters"):
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
