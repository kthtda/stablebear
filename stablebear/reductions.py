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
from .functional.pcf import Pcf
from .base_tensor import (
    FloatTensor,
    PcfTensor,
    PcfContainerLike,
    _resolve_pcf_inputs,
)
from .typing import pcf32, pcf64


_REDUCTIONS_BACKEND_MAP = {pcf32: cpp.Reductions_f32_f32, pcf64: cpp.Reductions_f64_f64}


def _to_tensor(outFs):
    if isinstance(outFs, (cpp.Pcf32Tensor, cpp.Pcf64Tensor)):
        return PcfTensor(outFs)
    elif isinstance(outFs, (cpp.Float32Tensor, cpp.Float64Tensor)):
        return FloatTensor(outFs)
    else:
        raise ValueError(
            "Invalid output type (this is probably a bug -- please report it!)."
        )


def _resolve_dim(dim: int, ndim: int) -> int:
    """Normalize a reduction ``dim`` to a non-negative axis (NumPy semantics).

    Negative ``dim`` counts from the last axis (``dim=-1`` is the last axis),
    matching NumPy and the rest of the tensor API (``t[-1]``, ``stack(axis=-1)``).
    Raises ``IndexError`` when ``dim`` is out of range for a tensor of rank
    ``ndim``.
    """
    resolved = dim + ndim if dim < 0 else dim
    if not 0 <= resolved < ndim:
        raise IndexError(
            f"dim {dim} is out of range for tensor with {ndim} dimension(s)"
        )
    return resolved


def mean(fs: PcfContainerLike, dim: int = 0):
    r"""Compute the pointwise mean of a PCF tensor along the given dimension.

    The mean is computed pointwise in time: for functions
    :math:`f_1, f_2, \ldots, f_n` being reduced, the resulting function
    :math:`\bar{f}` satisfies

    .. math::
        \bar{f}(t) = \frac{1}{n} \sum_{i=1}^{n} f_i(t)

    for all :math:`t`.

    See :ref:`tensors-how-dim-works` for a detailed explanation of dimension
    reduction semantics.

    Parameters
    ----------
    fs : PcfContainerLike
        A ``PcfTensor`` with dtype ``pcf32`` or ``pcf64``.
    dim : int, optional
        Dimension along which to reduce, by default 0. Negative values count
        from the last axis (NumPy semantics: ``dim=-1`` is the last axis).

    Returns
    -------
    PcfTensor
        A ``PcfTensor`` with the reduced dimension removed.

    Raises
    ------
    IndexError
        If ``dim`` is out of range for the input tensor's number of dimensions.
    """
    backend, tensor = _resolve_pcf_inputs(_REDUCTIONS_BACKEND_MAP, fs)
    return _to_tensor(backend.mean(tensor._data, _resolve_dim(dim, tensor.ndim)))


def max_time(fs: PcfContainerLike, dim: int = 0):
    r"""Compute the maximum breakpoint time along the given dimension.

    For each PCF :math:`f_i` with breakpoints
    :math:`(t_0^{(i)}, t_1^{(i)}, \ldots, t_{n_i-1}^{(i)})`, let
    :math:`T_i = t_{n_i-1}^{(i)}` be the last breakpoint. For functions
    :math:`f_1, f_2, \ldots, f_n` being reduced, this returns

    .. math::
        \max(T_1, T_2, \ldots, T_n).

    The result is numeric, not a PCF.

    See :ref:`tensors-how-dim-works` for a detailed explanation of dimension
    reduction semantics.

    Parameters
    ----------
    fs : PcfContainerLike
        A ``PcfTensor`` with dtype ``pcf32`` or ``pcf64``.
    dim : int, optional
        Dimension along which to reduce, by default 0. Negative values count
        from the last axis (NumPy semantics: ``dim=-1`` is the last axis).

    Returns
    -------
    FloatTensor
        A numeric tensor with the reduced dimension removed.

    Raises
    ------
    IndexError
        If ``dim`` is out of range for the input tensor's number of dimensions.
    ValueError
        If the reduction dimension is empty (size 0). ``max`` has no identity
        over an empty range, so there is no value to return.
    """
    backend, tensor = _resolve_pcf_inputs(_REDUCTIONS_BACKEND_MAP, fs)
    return _to_tensor(backend.max_time(tensor._data, _resolve_dim(dim, tensor.ndim)))
