#    Copyright 2024-2026 Bjorn Wehlin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


__before = set(dir())

from . import random, system
from ._tensor_base import Shape
from .comparison import allclose
from .distance import cdist, lp_distance, pdist
from .inner_product import l2_kernel
from .io import load, save
from .norms import lp_norm
from .functional import Pcf, iterate_rectangles
from .reductions import max_time, mean
from .serialize import from_serial_content
from .distance_matrix import DistanceMatrix, DistanceMatrixTensor
from .symmetric_matrix import SymmetricMatrix, SymmetricMatrixTensor
from .tensor import (
    BoolTensor,
    FloatTensor,
    IntPcfTensor,
    IntTensor,
    PcfTensor,
    PointCloudTensor,
)
from .tensor_create import array_split, concatenate, split, stack, zeros
from .typing import (
    dtype,
    barcode32,
    barcode64,
    boolean,
    distmat32,
    distmat64,
    float32,
    float64,
    int32,
    int64,
    pcf32,
    pcf32i,
    pcf64,
    pcf64i,
    pcloud32,
    pcloud64,
    symmat32,
    symmat64,
    uint32,
    uint64,
)

import types as _types
__all__ = sorted(
    name for name in set(dir()) - __before - {"__before"}
    if not name.startswith("_")
    and not isinstance(globals()[name], _types.ModuleType)
) + ["random", "system"]
del __before, _types
