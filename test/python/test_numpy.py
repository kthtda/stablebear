import numpy as np

import stablebear as sb
import stablebear._sb_cpp as cpp


def test_np_to_pcf_has_correct_type():
    # Strictly-increasing breakpoint times (t=0, 1); the test only checks dtype
    # inference, but the times must be valid for the strict Pcf constructor.
    X32 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    X64 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    f32 = sb.Pcf(X32)
    assert isinstance(f32._data, cpp.Pcf_f32_f32)
    assert f32.ttype == sb.float32
    assert f32.vtype == sb.float32

    f64 = sb.Pcf(X64)
    assert isinstance(f64._data, cpp.Pcf_f64_f64)
    assert f64.ttype == sb.float64
    assert f64.vtype == sb.float64

    f32_64 = sb.Pcf(X32, dtype=sb.pcf64)
    assert isinstance(f32_64._data, cpp.Pcf_f64_f64)
    assert f32_64.ttype == sb.float64
    assert f32_64.vtype == sb.float64

    f64_32 = sb.Pcf(X64, dtype=sb.pcf32)
    assert isinstance(f64_32._data, cpp.Pcf_f32_f32)
    assert f64_32.ttype == sb.float32
    assert f64_32.vtype == sb.float32

