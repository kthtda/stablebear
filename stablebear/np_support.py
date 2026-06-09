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

from .tensor import PcfContainerLike, PcfTensor
from .typing import _validate_dtype, pcf32, pcf64


def numpy_type(fs: PcfContainerLike):
    if isinstance(fs, PcfTensor):
        _validate_dtype(fs.dtype, [pcf32, pcf64])
        if fs.dtype == pcf32:
            return np.float32
        elif fs.dtype == pcf64:
            return np.float64

    raise NotImplementedError(
        "Data type not supported (please file an issue if you think this is in error)."
    )
