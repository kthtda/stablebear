import numpy as np

import stablebear as sb
import stablebear.persistence as pers


def test_empty_barcode():
    bc_pts = np.zeros((0, 2))
    bc = pers.Barcode(bc_pts)
    sr = pers.barcode_to_stable_rank(bc)

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


def test_barcode_with_repeats_and_offsets():
    bc_pts = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, 2.0], [0.0, 2.0], [1.0, 3.0]])
    bc = pers.Barcode(bc_pts)
    sr = pers.barcode_to_stable_rank(bc)

    expected_sr_pts = np.array([[0.0, 5.0], [1.0, 4.0], [2.0, 1.0], [3.0, 0.0]])
    expected_sr = sb.Pcf(expected_sr_pts)

    print(np.asarray(sr))

    assert sr == expected_sr


def test_barcode_with_infinite_bars():
    bc_pts = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, 2.0], [0.0, np.inf], [1.0, 3.0]])
    bc = pers.Barcode(bc_pts)
    sr = pers.barcode_to_stable_rank(bc)

    expected_sr_pts = np.array([[0.0, 5.0], [1.0, 4.0], [2.0, 2.0], [3.0, 1.0]])
    expected_sr = sb.Pcf(expected_sr_pts)

    print(np.asarray(sr))

    assert sr == expected_sr


def test_tensor_barcode_conversion():
    bcs = sb.zeros((5, 6, 7), dtype=sb.barcode64)

    for i in range(bcs.shape[0]):
        for j in range(bcs.shape[1]):
            for k in range(bcs.shape[2]):
                bc_shape = (np.random.randint(1, 20), 2)
                bc_data = np.random.randn(*bc_shape)
                bcs[i, j, k] = pers.Barcode(bc_data)

    srs = pers.barcode_to_stable_rank(bcs)

    assert srs.shape == bcs.shape

    for i in range(bcs.shape[0]):
        for j in range(bcs.shape[1]):
            for k in range(bcs.shape[2]):
                expected = pers.barcode_to_stable_rank(bcs[i, j, k])

                print(np.asarray(expected))
                print(np.asarray(srs[i, j, k]))

                assert srs[i, j, k] == expected
