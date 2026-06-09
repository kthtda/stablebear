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

import numpy as np

from . import _sb_cpp as cpp
from .async_task import _run_task
from .symmetric_matrix import SymmetricMatrix
from .tensor import PcfContainerLike, _resolve_pcf_inputs
from .typing import pcf32, pcf64


_INNER_PRODUCT_BACKEND_MAP = {pcf32: cpp.InnerProduct_f32_f32, pcf64: cpp.InnerProduct_f64_f64}


def l2_kernel(fs: PcfContainerLike, verbose=False) -> SymmetricMatrix:
    r"""Compute the pairwise :math:`L_2` kernel matrix for a 1-D tensor of PCFs.

    For a tensor :math:`(f_0, f_1, \ldots, f_{n-1})`, returns an
    :math:`n \times n` matrix :math:`K` where

    .. math::
        K_{ij} = \langle f_i, f_j \rangle_{L_2}
               = \int_0^\infty f_i(t) \, f_j(t) \, dt.

    Parameters
    ----------
    fs : PcfContainerLike
        A 1-D tensor of PCFs.
    verbose : bool, optional
        Show progress information during computation, by default False.

    Returns
    -------
    SymmetricMatrix
        A compressed symmetric kernel matrix.

    Raises
    ------
    ValueError
        If ``fs`` is not 1-dimensional.
    """
    backend, X = _resolve_pcf_inputs(_INNER_PRODUCT_BACKEND_MAP, fs)

    if len(X.shape) != 1:
        raise ValueError("1d tensor expected.")
    task, sm_or_dense = backend.l2(X._data)
    _run_task(lambda: task, verbose=verbose)

    if isinstance(sm_or_dense, np.ndarray):
        return SymmetricMatrix.from_dense(sm_or_dense)
    else:
        return SymmetricMatrix(sm_or_dense)
