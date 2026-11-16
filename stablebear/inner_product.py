import numpy as np

from typing import Any

from . import _autodiff
from . import _sb_cpp as cpp
from ._tracked import is_tracked
from .async_task import _run_task
from .distance import _pairwise_out, _pairwise_weights
from .symmetric_matrix import SymmetricMatrix
from .base_tensor import PcfContainerLike, _resolve_pcf_inputs
from .typing import pcf32, pcf64


_INNER_PRODUCT_BACKEND_MAP = {pcf32: cpp.InnerProduct_f32_f32, pcf64: cpp.InnerProduct_f64_f64}


def _l2_kernel_vjp(grad_out, fs, verbose=False):
    _, X = _resolve_pcf_inputs(_INNER_PRODUCT_BACKEND_MAP, fs)
    return (cpp.l2_kernel_grad(X._data, _pairwise_weights(grad_out, X)),)


@_autodiff.differentiable("fs", vjp=_l2_kernel_vjp, out=_pairwise_out)
def _tracked_l2_kernel(fs: Any, verbose: bool = False) -> Any:
    """l2_kernel of PCFs that carry gradients, as an (n, n) framework array."""
    return l2_kernel(fs, verbose).to_dense()


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
    if is_tracked(fs):
        tracked: Any = fs
        if tracked.ndim != 1:
            raise ValueError("1d tensor expected.")
        return _tracked_l2_kernel(fs, verbose)

    backend, X = _resolve_pcf_inputs(_INNER_PRODUCT_BACKEND_MAP, fs)

    if len(X.shape) != 1:
        raise ValueError("1d tensor expected.")
    task, sm_or_dense = backend.l2(X._data)
    _run_task(lambda: task, verbose=verbose)

    if isinstance(sm_or_dense, np.ndarray):
        return SymmetricMatrix(sm_or_dense)
    else:
        return SymmetricMatrix(sm_or_dense)
