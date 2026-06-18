import numpy as np
import pytest

import stablebear as sb


# ---------------------------------------------------------------------------
# Bug #45: NumericTensor lacked __bool__, so bool() fell back to __len__ and
# reported the first-dim length, not the value -- every non-empty numeric
# tensor was truthy, multi-element tensors failed to raise, and a 0-d tensor
# raised "len() of unsized object". __bool__ now lives on the Tensor base and
# matches NumPy: size-1 -> element truthiness; otherwise ValueError.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "TensorType, np_dtype",
    [
        (sb.IntTensor, np.int64),
        (sb.IntTensor, np.int32),
        (sb.FloatTensor, np.float64),
        (sb.FloatTensor, np.float32),
    ],
)
def test_size1_truthiness_matches_value(TensorType, np_dtype):
    assert bool(TensorType(np.array([0], dtype=np_dtype))) is False
    assert bool(TensorType(np.array([5], dtype=np_dtype))) is True
    assert bool(TensorType(np.array([-3], dtype=np_dtype))) is True


def test_zero_d_truthiness():
    z = sb.zeros((), dtype=sb.float64)
    z[()] = 0.0
    assert bool(z) is False
    z[()] = 2.5
    assert bool(z) is True


@pytest.mark.parametrize(
    "t",
    [
        pytest.param(sb.FloatTensor(np.array([0.0, 0.0])), id="multi-float"),
        pytest.param(sb.IntTensor(np.array([1, 2, 3])), id="multi-int"),
        pytest.param(sb.FloatTensor(np.zeros((0,))), id="empty"),
        pytest.param(sb.FloatTensor(np.zeros((2, 2))), id="2d"),
    ],
)
def test_multi_element_bool_raises(t):
    with pytest.raises(ValueError, match="ambiguous"):
        bool(t)


def test_numpy_parity():
    assert bool(np.array([0])) is False
    with pytest.raises(ValueError):
        bool(np.array([0.0, 0.0]))
    assert bool(sb.IntTensor(np.array([0]))) is False
    with pytest.raises(ValueError):
        bool(sb.FloatTensor(np.array([0.0, 0.0])))


def test_bool_tensor_still_works():
    """BoolTensor (the comparison result) keeps its behavior via inheritance."""
    assert bool(sb.FloatTensor(np.array([1.0])) == sb.FloatTensor(np.array([1.0]))) is True
    assert bool(sb.BoolTensor(np.array([False]))) is False
    with pytest.raises(ValueError):
        bool(sb.BoolTensor(np.array([True, False])))


def test_if_statement_reflects_value():
    """The headline footgun: ``if tensor:`` must reflect the value, not length."""
    if sb.IntTensor(np.array([0])):
        pytest.fail("a size-1 zero tensor should be falsy")
    if not sb.IntTensor(np.array([7])):
        pytest.fail("a size-1 nonzero tensor should be truthy")
