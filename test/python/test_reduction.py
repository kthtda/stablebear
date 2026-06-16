import numpy as np
import pytest

import stablebear as sb


def test_mean_of_1d_returns_tensor():
    A = sb.zeros((10,))
    avg = sb.mean(A)

    assert isinstance(avg, sb.PcfTensor)
    assert avg.shape == (1,)
    assert np.array_equal(avg[0].to_numpy(), np.zeros((1, 2)))


def test_mean_of_simple():
    A = sb.zeros((2,))

    A[0] = sb.Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))

    avg = sb.mean(A)
    avg_data = avg[0].to_numpy()

    assert avg_data.shape == (2, 2)
    assert avg_data[0][0] == pytest.approx(0.0, 1e-4)
    assert avg_data[0][1] == pytest.approx(0.5, 1e-4)
    assert avg_data[1][0] == pytest.approx(2.0, 1e-4)
    assert avg_data[1][1] == pytest.approx(1.5, 1e-4)


def test_mean_f64():
    A = sb.zeros((2,), dtype=sb.pcf64)
    A[0] = sb.Pcf(np.array([[0.0, 4.0], [2.0, 0.0]], dtype=np.float64))
    A[1] = sb.Pcf(np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64))

    avg = sb.mean(A)
    assert isinstance(avg, sb.PcfTensor)
    avg_data = avg[0].to_numpy()
    assert avg_data[0][1] == pytest.approx(3.0, abs=1e-7)


def test_mean_2d_dim0():
    X = sb.zeros((2, 3))
    X[0, 0] = sb.Pcf(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32))
    X[1, 0] = sb.Pcf(np.array([[0.0, 4.0], [1.0, 0.0]], dtype=np.float32))

    avg = sb.mean(X, dim=0)
    assert avg.shape == (3,)
    # Column 0: mean of 2 and 4 on [0,1) -> 3
    assert avg[0](0.5) == pytest.approx(3.0, abs=1e-5)


def test_mean_2d_dim1():
    X = sb.zeros((2, 3))
    X[0, 0] = sb.Pcf(np.array([[0.0, 6.0], [1.0, 0.0]], dtype=np.float32))
    X[0, 1] = sb.Pcf(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32))
    X[0, 2] = sb.Pcf(np.array([[0.0, 4.0], [1.0, 0.0]], dtype=np.float32))

    avg = sb.mean(X, dim=1)
    assert avg.shape == (2,)
    # Row 0: mean of 6, 2, 4 on [0,1) -> 4
    assert avg[0](0.5) == pytest.approx(4.0, abs=1e-5)


def test_resolve_pcf_inputs_invalid():
    from stablebear.base_tensor import _resolve_pcf_inputs
    from stablebear.reductions import _REDUCTIONS_BACKEND_MAP

    with pytest.raises(ValueError, match="not supported"):
        _resolve_pcf_inputs(_REDUCTIONS_BACKEND_MAP, "not a tensor")


def test_max_time_1d():
    A = sb.zeros((2,))
    A[0] = sb.Pcf(np.array([[0.0, 1.0], [5.0, 0.0]], dtype=np.float32))
    A[1] = sb.Pcf(np.array([[0.0, 1.0], [3.0, 0.0]], dtype=np.float32))

    result = sb.max_time(A)
    assert isinstance(result, sb.FloatTensor)
    assert float(result) == pytest.approx(5.0, abs=1e-5)


def test_max_time_f64():
    A = sb.zeros((2,), dtype=sb.pcf64)
    A[0] = sb.Pcf(np.array([[0.0, 1.0], [7.0, 0.0]], dtype=np.float64))
    A[1] = sb.Pcf(np.array([[0.0, 1.0], [3.0, 0.0]], dtype=np.float64))

    result = sb.max_time(A)
    assert isinstance(result, sb.FloatTensor)
    assert float(result) == pytest.approx(7.0, abs=1e-10)


def test_max_time_2d():
    X = sb.zeros((2, 3))
    X[0, 0] = sb.Pcf(np.array([[0.0, 1.0], [10.0, 0.0]], dtype=np.float32))
    X[0, 1] = sb.Pcf(np.array([[0.0, 1.0], [6.0, 0.0]], dtype=np.float32))
    X[0, 2] = sb.Pcf(np.array([[0.0, 1.0], [4.0, 0.0]], dtype=np.float32))
    X[1, 0] = sb.Pcf(np.array([[0.0, 1.0], [3.0, 0.0]], dtype=np.float32))
    X[1, 1] = sb.Pcf(np.array([[0.0, 1.0], [8.0, 0.0]], dtype=np.float32))
    X[1, 2] = sb.Pcf(np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32))

    result = sb.max_time(X, dim=0)
    assert result.shape == (3,)
    result_np = np.asarray(result)
    assert result_np[0] == pytest.approx(10.0, abs=1e-5)
    assert result_np[1] == pytest.approx(8.0, abs=1e-5)
    assert result_np[2] == pytest.approx(4.0, abs=1e-5)

    result1 = sb.max_time(X, dim=1)
    assert result1.shape == (2,)
    result1_np = np.asarray(result1)
    assert result1_np[0] == pytest.approx(10.0, abs=1e-5)
    assert result1_np[1] == pytest.approx(8.0, abs=1e-5)


def test_max_time_with_default_pcf():
    """Default-constructed PCFs (single breakpoint at t=0) should not crash."""
    X = sb.zeros((3,))
    result = sb.max_time(X)
    assert float(result) == pytest.approx(0.0)


def test_max_time_mixed_default_and_nondefault():
    """max_time should reflect the non-zero PCFs, not be dragged down by defaults."""
    X = sb.zeros((3,))
    X[1] = sb.Pcf(np.array([[0.0, 1.0], [5.0, 0.0]], dtype=np.float32))

    result = sb.max_time(X)
    assert float(result) == pytest.approx(5.0, abs=1e-5)


def test_float_on_single_element_float_tensor():
    t = sb.FloatTensor(np.array([3.14]))
    assert float(t) == pytest.approx(3.14)


def test_int_on_single_element_float_tensor():
    t = sb.FloatTensor(np.array([7.9]))
    assert int(t) == 7


def test_float_rejects_multi_element_tensor():
    t = sb.FloatTensor(np.array([1.0, 2.0]))
    with pytest.raises(TypeError, match="single-element"):
        float(t)
