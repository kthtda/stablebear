from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import numpy as np

from . import _autodiff
from . import _sb_cpp as cpp
from .async_task import _run_task
from .base_tensor import FloatTensor, PcfContainerLike, _resolve_pcf_inputs
from .typing import pcf32, pcf64

if TYPE_CHECKING:
    import torch


_NORMS_BACKEND_MAP = {pcf32: cpp.Norms_f32_f32, pcf64: cpp.Norms_f64_f64}


def _run(op, X, *args, verbose=False):
    task, out = op(X._data, *args)
    _run_task(lambda: task, verbose=verbose)
    return FloatTensor(out)


def _lp_norm_vjp(grad_out, fs, p, verbose=False):
    backend, X = _resolve_pcf_inputs(_NORMS_BACKEND_MAP, fs)
    dnorm_dp = _run(backend.lpnorm_lp, X, float(p), True, verbose=verbose)
    task, grads = backend.lpnorm_lp_grad(X._data, float(p))
    _run_task(lambda: task, verbose=verbose)
    content, enumeration = cpp.to_serial_content(grads)  # (d/dt, d/dv) per breakpoint
    sizes = np.diff(enumeration.reshape(-1, 2), axis=1).ravel()
    dnorm_dfs = np.repeat(np.reshape(grad_out, -1), sizes)[:, None] * content
    return (dnorm_dfs, np.sum(grad_out * np.asarray(dnorm_dp)))


@overload
def lp_norm(fs: PcfContainerLike, p: float = 1, verbose: bool = False) -> FloatTensor: ...
@overload
def lp_norm(fs: PcfContainerLike, p: torch.Tensor, verbose: bool = False) -> torch.Tensor: ...
@_autodiff.differentiable("fs", "p", vjp=_lp_norm_vjp)
def lp_norm(fs: PcfContainerLike, p: Any = 1, verbose: bool = False) -> Any:
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
      p : float or torch.Tensor, optional
          :math:`p` parameter in the :math:`L_p` norm, by default 1. If `p`
          is a scalar ``torch.Tensor``, the norms are returned as a
          ``torch.Tensor`` that carries gradients with respect to `p`.
      verbose : bool, optional
          Print additional information during the computation, by default False

      Returns
      -------
      FloatTensor or torch.Tensor
        Tensor of the same shape as `fs` with :math:`L_p` norms of the input functions.
      """

    if np.ndim(p) > 0:
        raise ValueError(f"p must be a single number (got {np.size(p)} numbers).")
    if p < 1:
        raise ValueError("p must be >= 1.")

    backend, X = _resolve_pcf_inputs(_NORMS_BACKEND_MAP, fs)

    if p == 1:
        return _run(backend.lpnorm_l1, X, verbose=verbose)
    return _run(backend.lpnorm_lp, X, float(p), verbose=verbose)
