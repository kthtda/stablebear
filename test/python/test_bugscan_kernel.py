import numpy as np
import pytest

import stablebear as sb


def _ramp_tensor(dtype):
    """A 1-D tensor of 3 PCFs with values 1, 2, 3 on [0, 2)."""
    npdt = np.float64 if dtype is sb.pcf64 else np.float32
    X = sb.zeros((3,), dtype=dtype)
    for i in range(3):
        X[i] = sb.Pcf(np.array([[0.0, float(i + 1)], [2.0, 0.0]], dtype=npdt))
    return X


@pytest.mark.parametrize("dtype", [sb.pcf64, sb.pcf32])
def test_l2_kernel_on_reversed_view_matches_materialized_copy(dtype):
    """l2_kernel on a negative-step PCF view must compute the correct kernel.

    BUG (issue #23): l2_kernel segfaulted on a negative-step PCF tensor view
    because Tensor1dValueIterator stored its stride as unsigned size_t, so a
    negative stride wrapped to a huge value and the pairwise-integration task
    iterated out of bounds. Element access on the reversed view already worked,
    so the data was fine -- only the view passed into the kernel path crashed.
    """
    X = _ramp_tensor(dtype)
    rev = X[::-1]

    # Sanity: the reversed view itself is fine -- element access works.
    assert [rev[k].to_numpy()[0, 1] for k in range(3)] == [3.0, 2.0, 1.0]

    K = np.asarray(sb.l2_kernel(rev).to_dense())
    expected = np.array([[18.0, 12.0, 6.0], [12.0, 8.0, 4.0], [6.0, 4.0, 2.0]])
    assert np.allclose(K, expected), (dtype, K)


@pytest.mark.parametrize("dtype", [sb.pcf64, sb.pcf32])
def test_pdist_on_reversed_view_matches_materialized_copy(dtype):
    """pdist on a negative-step PCF view must compute the correct distances.

    Same root cause as the l2_kernel crash in issue #23: both ops go through
    the CpuPairwiseIntegrationTask / Tensor1dValueIterator path.
    """
    X = _ramp_tensor(dtype)
    rev = X[::-1]

    D = np.asarray(sb.pdist(rev).to_dense())
    expected = np.array([[0.0, 2.0, 4.0], [2.0, 0.0, 2.0], [4.0, 2.0, 0.0]])
    assert np.allclose(D, expected), (dtype, D)


@pytest.mark.parametrize("dtype", [sb.pcf64, sb.pcf32])
def test_positive_step_strided_view_kernel(dtype):
    """A positive-step strided view must also reduce/kernel correctly.

    Guards the other sign of the stride fix: the iterator now stores a signed
    stride, so both strided and reversed views walk the right elements.
    """
    npdt = np.float64 if dtype is sb.pcf64 else np.float32
    X = sb.zeros((5,), dtype=dtype)
    for i in range(5):
        X[i] = sb.Pcf(np.array([[0.0, float(i + 1)], [2.0, 0.0]], dtype=npdt))

    view = X[::2]  # values 1, 3, 5
    assert [view[k].to_numpy()[0, 1] for k in range(3)] == [1.0, 3.0, 5.0]

    K = np.asarray(sb.l2_kernel(view).to_dense())
    # K[i,j] = 2 * v_i * v_j  (integral of constant product over [0, 2))
    v = np.array([1.0, 3.0, 5.0])
    expected = 2.0 * np.outer(v, v)
    assert np.allclose(K, expected), (dtype, K)
