#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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

    def _decay_value(self, val):
        if isinstance(val, np.ndarray):
            return Barcode(val)._data
        return val._data

    def _represent_element(self, element):
        return Barcode(element)

    def _get_valid_setitem_dtypes(self):
        return [BarcodeTensor, Barcode, np.ndarray]
