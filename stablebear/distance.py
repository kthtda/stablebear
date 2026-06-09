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
from .distance_matrix import DistanceMatrix
from .functional.pcf import Pcf, _has_matching_types
from .tensor import FloatTensor, PcfContainerLike, _resolve_pcf_inputs
from .typing import float32, float64, pcf32, pcf64


_DISTANCE_BACKEND_MAP = {pcf32: cpp.Distance_f32_f32, pcf64: cpp.Distance_f64_f64}

_VTYPE_TO_DISTANCE_BACKEND = {
    float32: cpp.Distance_f32_f32,
    float64: cpp.Distance_f64_f64,
}


def lp_distance(f: Pcf, g: Pcf, p=1) -> float:
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


def pdist(fs: PcfContainerLike, p=1, verbose=False) -> DistanceMatrix:
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
        A compressed symmetric distance matrix.

    Raises
    ------
    ValueError
        If ``fs`` is not 1-dimensional.
    """
    if p < 1:
        raise ValueError("p must be >= 1.")

    backend, X = _resolve_pcf_inputs(_DISTANCE_BACKEND_MAP, fs)

    if len(X.shape) != 1:
        raise ValueError("1d tensor expected.")

    if p == 1:
        task, dm_or_dense = backend.pdist_l1(X._data)
    else:
        task, dm_or_dense = backend.pdist_lp(X._data, float(p))

    _run_task(lambda: task, verbose=verbose)

    return DistanceMatrix(dm_or_dense)


def cdist(X: PcfContainerLike, Y: PcfContainerLike, p=1, verbose=False) -> FloatTensor:
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
