import numpy as np

from .. import _sb_cpp as cpp
from .._tensor_base import Tensor, _tensor_from_nested
from ..typing import barcode32, barcode64
from .barcode import Barcode

cpp_p = cpp.persistence

_BARCODE_CPP_TO_DTYPE = {
    cpp_p.Barcode32Tensor: barcode32,
    cpp_p.Barcode64Tensor: barcode64,
}


class BarcodeTensor(Tensor):
    def __init__(self, data):
        super().__init__()
        if isinstance(data, BarcodeTensor):
            data = data._data
        elif isinstance(data, (list, tuple)):
            data = _tensor_from_nested(data, {
                cpp_p.Barcode32: cpp_p.Barcode32Tensor,
                cpp_p.Barcode64: cpp_p.Barcode64Tensor,
            })
        elif not isinstance(data, (cpp_p.Barcode32Tensor, cpp_p.Barcode64Tensor)):
            raise TypeError(f"Cannot create BarcodeTensor from {type(data)}")
        self._data = data
        self.dtype = _BARCODE_CPP_TO_DTYPE[type(self._data)]

    def _to_py_tensor(self, data):
        return BarcodeTensor(data)

    def to_numpy(self):
        """Return the per-cell barcodes as ``(n_i, 2)`` NumPy arrays.

        Barcodes are ragged (each cell may hold a different number of bars), so
        they cannot be packed into one dense numeric array. For a 0-d tensor the
        single ``(n, 2)`` array is returned; otherwise the result is an
        ``object`` ndarray of shape ``self.shape`` whose entries are the
        per-cell ``(n_i, 2)`` arrays.
        """
        if self.ndim == 0:
            return Barcode(self._data._get_element([])).to_numpy()
        shape = tuple(self.shape)
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape):
            out[idx] = Barcode(self._data._get_element(list(idx))).to_numpy()
        return out

    def tolist(self):
        """Return the per-cell barcodes as a nested list of ``(n_i, 2)`` arrays.

        For a 0-d tensor the single ``(n, 2)`` array is returned; otherwise the
        nesting mirrors ``self.shape``.
        """
        shape = tuple(self.shape)

        def _build(prefix):
            depth = len(prefix)
            if depth == len(shape):
                return Barcode(self._data._get_element(list(prefix))).to_numpy()
            return [_build(prefix + (i,)) for i in range(shape[depth])]

        return _build(())

    def _decay_value(self, val):
        if isinstance(val, np.ndarray):
            return Barcode(val)._data
        return val._data

    def _represent_element(self, element):
        return Barcode(element)

    def _get_valid_setitem_dtypes(self):
        return [BarcodeTensor, Barcode, np.ndarray]
