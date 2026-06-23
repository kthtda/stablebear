import numpy as np
import numpy.testing as npt

import stablebear as sb
import stablebear.persistence as pers


def test_empty_barcode():
    bc = pers.Barcode(np.zeros((0, 2)))
    apf = pers.barcode_to_accumulated_persistence(bc)

    expected = sb.Pcf(np.zeros((1, 2)))
    assert apf == expected


def test_single_bar():
    # Bar [1, 3): lifetime=2, midpoint=2
    # APF: 0 for t < 2, 2 for t >= 2
    bc = pers.Barcode(np.array([[1.0, 3.0]]))
    apf = pers.barcode_to_accumulated_persistence(bc)

    expected = sb.Pcf(np.array([[0.0, 0.0], [2.0, 2.0]]))
    assert apf == expected


def test_infinite_bar_ignored():
    # Infinite bars have infinite midpoint, so they should not contribute
    bc = pers.Barcode(np.array([[0.0, np.inf], [1.0, 3.0]]))
    apf = pers.barcode_to_accumulated_persistence(bc)

    # Same as single bar [1, 3)
    expected = sb.Pcf(np.array([[0.0, 0.0], [2.0, 2.0]]))
    assert apf == expected


def test_figure1_h0():
    """H0 from Figure 1 of Biscio and Moller (2019).

    Bars: [0, 0.5), [0, 0.62), [0, oo)
    Lifetimes: 0.5, 0.62, and inf
    Midpoints: 0.25, 0.31, and inf
    APF_0: 0 -> 0.5 at t=0.25 -> 1.12 at t=0.31
    """
    bc = pers.Barcode(np.array([[0.0, 0.5], [0.0, 0.62]]))
    apf = pers.barcode_to_accumulated_persistence(bc)

    pts = apf.to_numpy()
    npt.assert_allclose(pts[:, 0], [0.0, 0.25, 0.31], atol=1e-6)
    npt.assert_allclose(pts[:, 1], [0.0, 0.5, 1.12], atol=1e-6)


def test_figure1_h1():
    """H1 from Figure 1 of Biscio and Moller (2019).

    Bars: [0, 0.5) x3, [0.62, 0.75) x1
    Lifetimes: 0.5, 0.5, 0.5, 0.13
    Midpoints: 0.25, 0.25, 0.25, 0.685
    APF_1: 0 -> 1.5 at t=0.25 -> 1.63 at t=0.685
    """
    bc = pers.Barcode(np.array([
        [0.0, 0.5], [0.0, 0.5], [0.0, 0.5], [0.62, 0.75],
    ]))
    apf = pers.barcode_to_accumulated_persistence(bc)

    pts = apf.to_numpy()
    npt.assert_allclose(pts[:, 0], [0.0, 0.25, 0.685], atol=1e-6)
    npt.assert_allclose(pts[:, 1], [0.0, 1.5, 1.63], atol=1e-6)


def test_max_death():
    """max_death should give the same result as manually removing bars that die after it."""
    bc_full = pers.Barcode(np.array([
        [0.0, 0.5], [0.0, 0.5], [0.0, 0.5], [0.62, 0.75],
    ]))
    bc_truncated = pers.Barcode(np.array([
        [0.0, 0.5], [0.0, 0.5], [0.0, 0.5],
    ]))

    apf_with_max_death = pers.barcode_to_accumulated_persistence(bc_full, max_death=0.6)
    apf_manual = pers.barcode_to_accumulated_persistence(bc_truncated)

    assert apf_with_max_death == apf_manual


def test_tensor_barcode_conversion():
    bcs = sb.zeros((3, 4), dtype=sb.barcode64)

    for i in range(bcs.shape[0]):
        for j in range(bcs.shape[1]):
            n_bars = np.random.randint(1, 10)
            births = np.sort(np.abs(np.random.randn(n_bars)))
            deaths = births + np.abs(np.random.randn(n_bars)) + 0.01
            bcs[i, j] = pers.Barcode(np.column_stack([births, deaths]))

    apfs = pers.barcode_to_accumulated_persistence(bcs)

    assert apfs.shape == bcs.shape

    for i in range(bcs.shape[0]):
        for j in range(bcs.shape[1]):
            expected = pers.barcode_to_accumulated_persistence(bcs[i, j])
            assert apfs[i, j] == expected
