import numpy as np

import stablebear as sb


# ---------------------------------------------------------------------------
# Bug #6: self-aliasing basic-slice assignment (a[:] = a[::-1], a[1:] = a[:-1])
# silently corrupted data, because Tensor::assign_from walked element-by-element
# with no overlap handling and read back values it had already overwritten.
# NumPy materializes an overlapping RHS into a temporary; match that.
# ---------------------------------------------------------------------------


def test_reverse_in_place_1d():
    a = sb.FloatTensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    a[:] = a[::-1]
    na = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    na[:] = na[::-1]
    assert np.asarray(a).tolist() == na.tolist() == [4.0, 3.0, 2.0, 1.0]


def test_shift_down_2d():
    b = sb.FloatTensor(np.arange(12, dtype=np.float32).reshape(4, 3))
    b[1:] = b[:-1]
    nb = np.arange(12, dtype=np.float32).reshape(4, 3)
    nb[1:] = nb[:-1]
    assert np.array_equal(np.asarray(b), nb)


def test_forward_overlap_1d():
    a = sb.FloatTensor(np.arange(6, dtype=np.float64))
    a[1:] = a[:-1]
    na = np.arange(6, dtype=np.float64)
    na[1:] = na[:-1]
    assert np.asarray(a).tolist() == na.tolist() == [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]


def test_int_tensor_self_alias():
    a = sb.IntTensor(np.array([1, 2, 3, 4], dtype=np.int64))
    a[:] = a[::-1]
    na = np.array([1, 2, 3, 4], dtype=np.int64)
    na[:] = na[::-1]
    assert np.asarray(a).tolist() == na.tolist() == [4, 3, 2, 1]


def test_copy_rhs_control_still_correct():
    c = sb.FloatTensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    c[:] = c[::-1].copy()
    nc = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    nc[:] = nc[::-1].copy()
    assert np.asarray(c).tolist() == nc.tolist() == [4.0, 3.0, 2.0, 1.0]


def test_non_aliased_assignment_unchanged():
    d = sb.FloatTensor(np.array([1.0, 2.0, 3.0, 4.0]))
    e = sb.FloatTensor(np.array([10.0, 20.0, 30.0, 40.0]))
    d[:] = e[::-1]
    nd = np.array([1.0, 2.0, 3.0, 4.0])
    ne = np.array([10.0, 20.0, 30.0, 40.0])
    nd[:] = ne[::-1]
    assert np.asarray(d).tolist() == nd.tolist() == [40.0, 30.0, 20.0, 10.0]
    assert np.asarray(e).tolist() == ne.tolist() == [10.0, 20.0, 30.0, 40.0]  # source untouched


def test_partial_overlap_disjoint_halves():
    """Same buffer, disjoint regions -- must still match numpy."""
    a = sb.FloatTensor(np.arange(6, dtype=np.float64))
    a[:3] = a[3:]
    na = np.arange(6, dtype=np.float64)
    na[:3] = na[3:]
    assert np.asarray(a).tolist() == na.tolist()
