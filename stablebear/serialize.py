import numpy as np

from . import _sb_cpp as cpp
from .base_tensor import PcfTensor
from .typing import _assert_valid_dtype, pcf32, pcf64


def from_serial_content(
    content: np.ndarray, enumeration: np.ndarray, dtype=None
) -> PcfTensor:
    """Creates a `Tensor` of PCFs from serial numpy data.

    Parameters
    ----------
    content : np.ndarray
        ``(m, 2)`` array of points, where `m` is the sum of lengths of the individual PCFs
    enumeration : np.ndarray
        ``(n_1, n_2, ..., n_k, 2)`` array of `(start, end)` pointers into the content array.
    dtype : datatype
        Sets the ``dtype`` of the resulting PCF ``Array``. If ``None``, uses the ``dtype`` of the supplied ``content`` array. By default, ``None``.

    Returns
    -------
    PcfTensor
        `PcfTensor` of shape ``(n_1, n_2, ..., n_k)``, where element ``(i_1, i_2, ..., i_k)`` is a `Pcf` with points ``content[start, stop]`` with ``start=enumeration[i_1,...,i_k, 0]`` and ``stop=enumeration[i_1,...,i_k, 1]``.
    """

    if dtype is None:
        if content.dtype == np.float32:
            dtype = pcf32
        elif content.dtype == np.float64:
            dtype = pcf64
        else:
            raise TypeError("content must have dtype either np.float32 or np.float64.")

    if dtype == pcf32 and content.dtype != np.float32:
        content = content.astype(np.float32)
    elif dtype == pcf64 and content.dtype != np.float64:
        content = content.astype(np.float64)

    _assert_valid_dtype(dtype, [pcf32, pcf64])

    if dtype == pcf32:
        return PcfTensor(cpp.make_from_serial_content_32(content, enumeration))
    elif dtype == pcf64:
        return PcfTensor(cpp.make_from_serial_content_64(content, enumeration))
