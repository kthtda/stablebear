"""NumPy gives C integer types of one size scalar types of their own: C
``long`` is ``np.long``, not ``np.int32``, on Windows, and C ``long long`` is
``np.longlong``, not ``np.int64``, on Linux. Arrays of them take the dtype of
their size, as arrays of ``np.int32`` and ``np.int64`` do."""

import numpy as np
import pytest

import stablebear as sb

# C long is 4 bytes on Windows and 8 bytes on Linux and macOS.
_LONG = sb.int32 if np.dtype(np.long).itemsize == 4 else sb.int64
_ULONG = sb.uint32 if np.dtype(np.ulong).itemsize == 4 else sb.uint64


@pytest.mark.parametrize("c_type, dtype", [
    pytest.param(np.longlong, sb.int64, id="longlong"),
    pytest.param(np.ulonglong, sb.uint64, id="ulonglong"),
    pytest.param(np.long, _LONG, id="long"),
    pytest.param(np.ulong, _ULONG, id="ulong"),
])
def test_int_tensor_of_a_c_integer_type_takes_the_dtype_of_its_size(c_type, dtype):
    tensor = sb.IntTensor(np.array([1, 2, 3], dtype=c_type))

    assert tensor.dtype == dtype


@pytest.mark.parametrize("c_type, dtype", [
    pytest.param(np.longlong, sb.int64, id="longlong"),
    pytest.param(np.long, _LONG, id="long"),
])
def test_pcf_of_a_c_integer_type_takes_the_dtype_of_its_size(c_type, dtype):
    f = sb.Pcf(np.array([[0, 10],
                         [1, 20],
                         [3, 30]], dtype=c_type))

    assert f.ttype == dtype
    assert f(2) == 20
