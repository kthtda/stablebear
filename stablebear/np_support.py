import numpy as np

from .base_tensor import PcfContainerLike, PcfTensor
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
