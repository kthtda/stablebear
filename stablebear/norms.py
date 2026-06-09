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

from . import _sb_cpp as cpp
from .async_task import _run_task
from .tensor import FloatTensor, PcfContainerLike, _resolve_pcf_inputs
from .typing import pcf32, pcf64


_NORMS_BACKEND_MAP = {pcf32: cpp.Norms_f32_f32, pcf64: cpp.Norms_f64_f64}


def lp_norm(fs: PcfContainerLike, p=1, verbose=False) -> FloatTensor:
    r"""Computes the :math:`L_p` norm of each PCF in `fs`. For example, if `fs` is an :math:`m \times n` array with elements indexed as :math:`f_{ij}`, :math:`0 \leq i < m, 0 \leq j < n`, we compute

      .. math::
        \begin{pmatrix}
          \Vert f_{11} \Vert_p & \Vert f_{12} \Vert_p & \cdots & \Vert f_{1n} \Vert_p \\
          \Vert f_{21} \Vert_p & \Vert f_{22} \Vert_p & \cdots & \Vert f_{2n} \Vert_p \\
          \vdots & \vdots & \ddots & \vdots & \\
          \Vert f_{m1} \Vert_p & \Vert f_{m2} \Vert_p & \cdots & \Vert f_{mn} \Vert_p \\
        \end{pmatrix},

      where

      .. math::
        \Vert f_{ij} \Vert_p = \left(\int_0^\infty |f_i(t)|^p\, dt\right)^{1/p}.

      Parameters
      ----------
      fs : PcfContainerLike
          PCFs whose norms are to be computed.
      p : int, optional
          :math:`p` parameter in the :math:`L_p` norm, by default 1
      verbose : bool, optional
          Print additional information during the computation, by default False

      Returns
      -------
      FloatTensor
        Tensor of the same shape as `fs` with :math:`L_p` norms of the input functions.
      """

    if p < 1:
        raise ValueError("p must be >= 1.")

    backend, X = _resolve_pcf_inputs(_NORMS_BACKEND_MAP, fs)

    if p == 1:
        task, out = backend.lpnorm_l1(X._data)
    else:
        task, out = backend.lpnorm_lp(X._data, float(p))

    _run_task(lambda: task, verbose=verbose)

    return FloatTensor(out)
