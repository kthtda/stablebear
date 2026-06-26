import numpy as np
import pytest

import stablebear as sb


# ---------------------------------------------------------------------------
# Bug #19: split(tensor, 0) performed an integer modulo-by-zero
# (axis_size % n_sections) and aborted the interpreter with SIGFPE.
# array_split already guarded n_sections == 0; split now does too, raising a
# clean, catchable ValueError.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "t",
    [
        pytest.param(sb.FloatTensor(np.arange(6, dtype=np.float64)), id="float"),
        pytest.param(sb.IntTensor(np.arange(6, dtype=np.int64)), id="int"),
        pytest.param(sb.zeros((6,)), id="pcf"),
        pytest.param(sb.FloatTensor(np.zeros((0,))), id="empty"),
    ],
)
def test_split_zero_sections_raises(t):
    with pytest.raises(ValueError):
        sb.split(t, 0)


def test_split_zero_sections_explicit_axis_raises():
    t = sb.FloatTensor(np.arange(12, dtype=np.float64).reshape(3, 4))
    with pytest.raises(ValueError):
        sb.split(t, 0, axis=1)


def test_array_split_zero_sections_still_raises():
    with pytest.raises(ValueError):
        sb.array_split(sb.FloatTensor(np.arange(6.0)), 0)


# ---------------------------------------------------------------------------
# Bug #20 / #73: concatenate / split / array_split rejected a negative axis
# with a raw pybind "incompatible function arguments" error (the bindings take
# an unsigned axis). Negative axes now resolve from the end, as in NumPy.
# ---------------------------------------------------------------------------


def test_concatenate_negative_axis_resolves_like_numpy():
    a = sb.FloatTensor(np.arange(6.0).reshape(2, 3))
    got = np.asarray(sb.concatenate([a, a], axis=-1))
    ref = np.concatenate([np.asarray(a), np.asarray(a)], axis=-1)
    np.testing.assert_array_equal(got, ref)
    assert got.shape == (2, 6)


@pytest.mark.parametrize("splitter", [sb.split, sb.array_split])
def test_split_negative_axis_resolves_like_numpy(splitter):
    m = sb.FloatTensor(np.arange(12.0).reshape(2, 6))
    parts = splitter(m, 3, axis=-1)
    assert [tuple(p.shape) for p in parts] == [(2, 2), (2, 2), (2, 2)]


def test_split_out_of_range_axis_raises():
    m = sb.FloatTensor(np.arange(6.0).reshape(2, 3))
    with pytest.raises(IndexError):
        sb.split(m, 1, axis=5)


# ---------------------------------------------------------------------------
# Bug #21: a negative split-index entry raised a low-level pybind TypeError
# (split_indices takes unsigned indices). Negative entries now offset from the
# end (clamped at 0), matching numpy.split / numpy.array_split.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("splitter, np_splitter",
                         [(sb.split, np.split), (sb.array_split, np.array_split)])
@pytest.mark.parametrize("indices", [[-2], [-7], [2, -1], [-3, -1]])
def test_negative_split_indices_offset_like_numpy(splitter, np_splitter, indices):
    base = np.arange(7.0)
    got = [list(np.asarray(p).ravel()) for p in splitter(sb.FloatTensor(base), indices)]
    ref = [list(x) for x in np_splitter(base, indices)]
    assert got == ref


# ---------------------------------------------------------------------------
# Bug #74: joining tensors of different dtype/precision leaked the raw pybind
# overload dump. There is no implicit upcasting, so report a clear error.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", [sb.concatenate, sb.stack])
def test_join_mismatched_dtype_raises_clean(op):
    f32 = sb.FloatTensor(np.arange(3.0, dtype=np.float32))
    f64 = sb.FloatTensor(np.arange(3.0, dtype=np.float64))
    with pytest.raises(TypeError) as excinfo:
        op([f32, f64])
    msg = str(excinfo.value)
    assert "incompatible function arguments" not in msg
    assert "same dtype" in msg


@pytest.mark.parametrize("op", [sb.concatenate, sb.stack])
def test_join_cross_family_raises_clean(op):
    f = sb.FloatTensor(np.arange(3.0))
    p = sb.zeros((3,), dtype=sb.pcf64)
    with pytest.raises(TypeError) as excinfo:
        op([f, p])
    assert "same dtype" in str(excinfo.value)
