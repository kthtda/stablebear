import numpy as np

import stablebear as sb
import stablebear.persistence as pers


def test_empty_barcode(track):
    bc_pts = np.zeros((0, 2))
    bc = pers.Barcode(bc_pts)
    sr = pers.barcode_to_stable_rank(track(pers.BarcodeTensor([bc])))[0]

    expected_sr_pts = np.zeros((1, 2))
    expected_sr = sb.Pcf(expected_sr_pts)

    assert sr == expected_sr


def test_simple_barcode():
    bc_pts = np.array(
        [
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ]
    )
    bc = pers.Barcode(bc_pts)
    sr = pers.barcode_to_stable_rank(bc)

    expected_sr_pts = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    expected_sr = sb.Pcf(expected_sr_pts)

    print(np.asarray(sr))

    assert sr == expected_sr


def test_barcode_with_repeats_and_offsets(track):
    bc_pts = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, 2.0], [0.0, 2.0], [1.0, 3.0]])
    bc = pers.Barcode(bc_pts)
    sr = pers.barcode_to_stable_rank(track(pers.BarcodeTensor([bc])))[0]

    # The lifetimes are 1, 3, 2, 2, and 2. Tracked barcodes give one
    # breakpoint per bar, so compare the values and not the breakpoints.
    np.testing.assert_array_equal(sr([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]),
                                  [5.0, 5.0, 4.0, 4.0, 1.0, 1.0, 0.0, 0.0])


def test_bars_of_equal_lifetime_share_a_breakpoint():
    bc = pers.Barcode(np.array([[0.0, 1.0], [0.0, 3.0], [0.0, 2.0], [0.0, 2.0], [1.0, 3.0]]))
    sr = pers.barcode_to_stable_rank(bc)

    assert sr == sb.Pcf(np.array([[0.0, 5.0], [1.0, 4.0], [2.0, 1.0], [3.0, 0.0]]))


def test_barcode_with_infinite_bars(track):
    bc_pts = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, 2.0], [0.0, np.inf], [1.0, 3.0]])
    bc = pers.Barcode(bc_pts)
    sr = pers.barcode_to_stable_rank(track(pers.BarcodeTensor([bc])))[0]

    # The lifetimes are 1, 3, 2, inf, and 2. Tracked barcodes give a
    # breakpoint at inf, so compare the values and not the breakpoints.
    np.testing.assert_array_equal(sr([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]),
                                  [5.0, 5.0, 4.0, 4.0, 2.0, 2.0, 1.0, 1.0])


def test_tensor_barcode_conversion(track):
    bcs = sb.zeros((5, 6, 7), dtype=sb.barcode64)

    for i in range(bcs.shape[0]):
        for j in range(bcs.shape[1]):
            for k in range(bcs.shape[2]):
                bc_shape = (np.random.randint(1, 20), 2)
                bc_data = np.random.randn(*bc_shape)
                bcs[i, j, k] = pers.Barcode(bc_data)

    srs = pers.barcode_to_stable_rank(track(bcs))

    assert srs.shape == bcs.shape

    for i in range(bcs.shape[0]):
        for j in range(bcs.shape[1]):
            for k in range(bcs.shape[2]):
                expected = pers.barcode_to_stable_rank(bcs[i, j, k])

                print(np.asarray(expected))
                print(np.asarray(srs[i, j, k]))

                assert srs[i, j, k] == expected
