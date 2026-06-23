import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


def test_add_f32():
    a = sb.Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    b = sb.Pcf(np.array([[0.0, 4.0], [2.0, 5.0]], dtype=np.float32))

    c = a + b

    assert isinstance(c, sb.Pcf)
    result = c.to_numpy()
    assert result[0, 1] == pytest.approx(5.0, abs=1e-5)
    assert result[1, 1] == pytest.approx(8.0, abs=1e-5)


def test_add_f64():
    a = sb.Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64))
    b = sb.Pcf(np.array([[0.0, 4.0], [2.0, 5.0]], dtype=np.float64))

    c = a + b

    assert isinstance(c, sb.Pcf)
    result = c.to_numpy()
    assert result[0, 1] == pytest.approx(5.0, abs=1e-10)
    assert result[1, 1] == pytest.approx(8.0, abs=1e-10)


def test_add_mismatched_types_raises():
    a = sb.Pcf(np.array([[0.0, 1.0]], dtype=np.float32))
    b = sb.Pcf(np.array([[0.0, 1.0]], dtype=np.float64))

    with pytest.raises(TypeError, match="Mismatched PCF types"):
        a + b


def test_add_does_not_modify_operands():
    a = sb.Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    b = sb.Pcf(np.array([[0.0, 4.0], [2.0, 5.0]], dtype=np.float32))

    a_orig = a.to_numpy().copy()
    b_orig = b.to_numpy().copy()

    _ = a + b

    npt.assert_array_equal(a.to_numpy(), a_orig)
    npt.assert_array_equal(b.to_numpy(), b_orig)
