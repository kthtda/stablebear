from typing import Any

import numpy as np

from . import _autodiff
from . import _sb_cpp as cpp
from ._tracked import is_tracked
from .async_task import _run_task
from .distance_matrix import DistanceMatrix
from .functional.pcf import Pcf, _has_matching_types
from .base_tensor import FloatTensor, PcfContainerLike, _resolve_pcf_inputs
from .typing import float32, float64, pcf32, pcf64


_DISTANCE_BACKEND_MAP = {pcf32: cpp.Distance_f32_f32, pcf64: cpp.Distance_f64_f64}

_VTYPE_TO_DISTANCE_BACKEND = {
    float32: cpp.Distance_f32_f32,
    float64: cpp.Distance_f64_f64,
}


def lp_distance(f: Pcf, g: Pcf, p: float = 1) -> float:
    r"""Compute the :math:`L_p` distance between two PCFs.

    .. math::
        \Vert f - g \Vert_p
        = \left(\int_0^\infty |f(t) - g(t)|^p\, dt\right)^{1/p}

    Parameters
    ----------
    f : Pcf
        First piecewise constant function.
    g : Pcf
        Second piecewise constant function.
    p : float, optional
        The :math:`p` parameter in the :math:`L_p` distance (must be
        :math:`\geq 1`), by default 1.

    Returns
    -------
    float
        The :math:`L_p` distance between *f* and *g*.

    Raises
    ------
    ValueError
        If ``p < 1``.
    TypeError
        If *f* and *g* have different dtypes, or are integer PCFs.
    """
    if p < 1:
        raise ValueError("p must be >= 1.")

    if not isinstance(f, Pcf) or not isinstance(g, Pcf):
        raise TypeError("Both f and g must be Pcf objects.")

    if not _has_matching_types(f, g):
        raise TypeError("f and g must have the same dtype.")

    backend = _VTYPE_TO_DISTANCE_BACKEND.get(f.vtype)
    if backend is None:
        raise TypeError(
            f"lp_distance is not supported for this PCF type (vtype={f.vtype})."
        )

    if p == 1:
        return float(backend.lp_distance_l1(f._data, g._data))
    else:
        return float(backend.lp_distance_lp(f._data, g._data, float(p)))


def _pairwise_weights(grad_out, X) -> np.ndarray:
    """What cpp.pdist_grad and cpp.l2_kernel_grad take as weights for the
    gradient of a symmetric (n, n) result: each entry counts for both of its
    functions."""
    grad_out = np.asarray(grad_out)
    return np.ascontiguousarray(grad_out + grad_out.T, dtype=np.float32 if X.dtype is pcf32 else np.float64)


def _pdist_vjp(grad_out, fs, p, verbose=False):
    _, X = _resolve_pcf_inputs(_DISTANCE_BACKEND_MAP, fs)
    return (cpp.pdist_grad(X._data, _pairwise_weights(grad_out, X), float(p)),)


@_autodiff.differentiable("fs", vjp=_pdist_vjp)
def _tracked_pdist(fs: Any, p: float, verbose: bool = False) -> Any:
    """pdist of PCFs that carry gradients, as an (n, n) framework array."""
    return pdist(fs, p, verbose).to_dense()


def pdist(fs: PcfContainerLike, p: float = 1, verbose: bool = False) -> DistanceMatrix:
    r"""Compute the pairwise :math:`L_p` distance matrix for a 1-D tensor of PCFs.

    For a tensor :math:`(f_0, f_1, \ldots, f_{n-1})`, returns an
    :math:`n \times n` matrix :math:`D` where

    .. math::
        D_{ij} = \Vert f_i - f_j \Vert_p.

    Parameters
    ----------
    fs : PcfContainerLike
        A 1-D tensor of PCFs.
    p : float, optional
        The :math:`p` parameter in the :math:`L_p` distance (must be
        :math:`\geq 1`), by default 1.
    verbose : bool, optional
        Show progress information during computation, by default False.

    Returns
    -------
    DistanceMatrix
        A compressed symmetric distance matrix. If the PCFs of `fs` carry
        gradients, the distances instead as a PyTorch
        array of shape ``(n, n)``, which carries gradients with respect to the
        breakpoint times and values of the PCFs.

    Raises
    ------
    ValueError
        If ``fs`` is not 1-dimensional.

    Notes
    -----
    The PCFs carry gradients when their times or values depend on PyTorch tensors (see :class:`~stablebear.functional.pcf.Pcf`).
    The gradient with respect to the first time of a PCF and to its last
    value, which holds on an infinite interval, is 0. Where breakpoints of
    two PCFs share a time, the derivative of their distance with respect to
    that time is that of moving both breakpoints. The gradient of the
    distance between two equal PCFs is 0.
    """
    if p < 1:
        raise ValueError("p must be >= 1.")

    if is_tracked(fs):
        tracked: Any = fs
        if tracked.ndim != 1:
            raise ValueError("1d tensor expected.")
        return _tracked_pdist(fs, p, verbose)

    backend, X = _resolve_pcf_inputs(_DISTANCE_BACKEND_MAP, fs)

    if len(X.shape) != 1:
        raise ValueError("1d tensor expected.")

    if p == 1:
        task, dm_or_dense = backend.pdist_l1(X._data)
    else:
        task, dm_or_dense = backend.pdist_lp(X._data, float(p))

    _run_task(lambda: task, verbose=verbose)

    return DistanceMatrix(dm_or_dense)


def cdist(X: PcfContainerLike, Y: PcfContainerLike, p: float = 1, verbose: bool = False) -> FloatTensor:
    r"""Compute the pairwise :math:`L_p` distances between two tensors of PCFs.

    For tensors :math:`X` of shape :math:`(m_1, \ldots, m_n)` and :math:`Y` of
    shape :math:`(k_1, \ldots, k_l)`, returns a tensor of shape
    :math:`(m_1, \ldots, m_n, k_1, \ldots, k_l)` where

    .. math::
        D_{i_1, \ldots, i_n, j_1, \ldots, j_l}
        = \Vert X_{i_1, \ldots, i_n} - Y_{j_1, \ldots, j_l} \Vert_p.

    Parameters
    ----------
    X : PcfContainerLike
        A tensor of PCFs (any shape).
    Y : PcfContainerLike
        A tensor of PCFs (any shape, same dtype as X).
    p : float, optional
        The :math:`p` parameter in the :math:`L_p` distance (must be
        :math:`\geq 1`), by default 1.
    verbose : bool, optional
        Show progress information during computation, by default False.

    Returns
    -------
    FloatTensor
        A tensor of shape ``(*X.shape, *Y.shape)`` containing pairwise distances.
    """
    if p < 1:
        raise ValueError("p must be >= 1.")

    backend, Xt, Yt = _resolve_pcf_inputs(_DISTANCE_BACKEND_MAP, X, Y)

    if p == 1:
        task, out = backend.cdist_l1(Xt._data, Yt._data)
    else:
        task, out = backend.cdist_lp(Xt._data, Yt._data, float(p))

    _run_task(lambda: task, verbose=verbose)

    return FloatTensor(out)
