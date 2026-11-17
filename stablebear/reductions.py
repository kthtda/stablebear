from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import numpy as np

from . import _autodiff
from . import _sb_cpp as cpp
from ._tracked import is_tracked
from .functional.pcf import Pcf
from .base_tensor import (
    FloatTensor,
    PcfTensor,
    PcfContainerLike,
    _resolve_pcf_inputs,
)
from .typing import pcf32, pcf64
from .base_tensor import IntTensor
from .distance_matrix import DistanceMatrix, DistanceMatrixTensor

if TYPE_CHECKING:
    import torch


_REDUCTIONS_BACKEND_MAP = {pcf32: cpp.Reductions_f32_f32, pcf64: cpp.Reductions_f64_f64}


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


def _tracked_sum(fs: Any, dim: int, as_mean: bool) -> PcfTensor:
    """The sum, or mean, along ``dim`` of a PcfTensor that carries
    gradients, as a tensor that carries them.

    A PCF is the running sum of its jumps, so the sum of several is the
    running sum of the jumps of all of them in the order of their times. All
    sums take one sort and one running sum.
    """
    from .base_tensor import _TrackedPcfTensor

    content, enumeration = fs._serial()
    shape = tuple(enumeration.shape[:-1])
    axis = _resolve_dim(dim, len(shape))
    count = shape[axis]
    if count == 0:
        # No members, so no gradients either: the zero PCFs.
        plain = fs.detach()
        return mean(plain, dim) if as_mean else sum(plain, dim)
    like = fs._tvalues if fs._tvalues is not None else fs._ttimes
    adapter = _autodiff._framework(like)

    reduced = shape[:axis] + shape[axis + 1:]
    n_sums = int(np.prod(reduced))
    # The sum that each PCF is a member of.
    sum_of_pcf = np.broadcast_to(np.expand_dims(np.arange(n_sums).reshape(reduced), axis), shape).ravel()

    # The breakpoints of all PCFs, PCF by PCF, and the jump at each.
    flat = enumeration.reshape(-1, 2)
    sizes = flat[:, 1] - flat[:, 0]
    ends = np.cumsum(sizes)
    position = np.arange(int(ends[-1])) - np.repeat(ends - sizes, sizes)
    rows = np.repeat(flat[:, 0], sizes) + position
    values = fs._tvalues if fs._tvalues is not None else adapter.as_tensor(content[:, 1], like=like)
    before = adapter.gather(values, np.maximum(rows - 1, 0))
    jumps = adapter.gather(values, rows) - before * adapter.as_tensor((position > 0).astype(content.dtype), like=like)

    # By sum, and by time within each.
    sum_of_row = np.repeat(sum_of_pcf, sizes)
    order = np.lexsort((content[rows, 0], sum_of_row))
    times, sum_of_row = content[rows[order], 0], sum_of_row[order]

    # The running sum over all sums, less what it was when each began.
    running = adapter.concat([adapter.as_tensor(np.zeros(1, dtype=content.dtype), like=like),
                              adapter.cumsum(adapter.gather(jumps, order))])
    per_sum = np.bincount(sum_of_row, minlength=n_sums)
    begins = np.repeat(np.cumsum(per_sum) - per_sum, per_sum)

    # Of the breakpoints of a sum at one time, the last holds its value.
    keep = np.ones(len(times), dtype=bool)
    keep[:-1] = (sum_of_row[1:] != sum_of_row[:-1]) | (times[1:] != times[:-1])
    kept = np.flatnonzero(keep)
    result = adapter.gather(running, kept + 1) - adapter.gather(running, begins[kept])

    # The last value of a sum is the sum of the last values of its members,
    # taken directly: the running sum is off by its rounding, and a sum of
    # PCFs that end at 0 must end at exactly 0 to have a finite norm.
    last_values = adapter.segment_sums(adapter.gather(values, flat[:, 1] - 1), sum_of_pcf, n_sums)
    is_last = np.ones(len(kept), dtype=bool)
    is_last[:-1] = sum_of_row[kept[1:]] != sum_of_row[kept[:-1]]
    index = np.arange(len(kept))
    index[is_last] = len(kept) + sum_of_row[kept[is_last]]
    result = adapter.gather(adapter.concat([result, last_values]), index)
    if as_mean:
        result = result / count
    result_times = times[kept] if fs._ttimes is None else adapter.gather(fs._ttimes, rows[order[kept]])

    ends = np.cumsum(np.bincount(sum_of_row[kept], minlength=n_sums))
    starts = np.concatenate([[0], ends[:-1]])
    return _TrackedPcfTensor([result_times, result], np.stack([starts, ends], axis=-1).reshape(*(reduced or (1,)), 2))


def sum(fs: PcfContainerLike, dim: int = 0) -> PcfTensor:
    r"""Compute the pointwise sum of PCFs along the given dimension.

    For functions :math:`f_1, \ldots, f_n` along the reduced dimension,
    the result satisfies :math:`s(t) = \sum_{i=1}^{n} f_i(t)` for all
    :math:`t`. An empty sum is the identically zero PCF.

    See :ref:`tensors-how-dim-works` for dimension reduction semantics.

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
        A tensor of the same dtype with the reduced dimension removed.
        Reducing a one-dimensional tensor returns shape ``(1,)``.

    Raises
    ------
    IndexError
        If ``dim`` is out of range for the input tensor's number of dimensions.

    Notes
    -----
    A tensor that carries gradients gives one that carries them, with
    respect to the values and the times of its PCFs.
    """
    if is_tracked(fs):
        return _tracked_sum(fs, dim, as_mean=False)
    backend, tensor = _resolve_pcf_inputs(_REDUCTIONS_BACKEND_MAP, fs)
    return PcfTensor(backend.sum(tensor._data, _resolve_dim(dim, tensor.ndim)))


def mean(fs: PcfContainerLike, dim: int = 0) -> PcfTensor:
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
        Reducing a one-dimensional tensor returns shape ``(1,)``.

    Raises
    ------
    IndexError
        If ``dim`` is out of range for the input tensor's number of dimensions.

    Notes
    -----
    A tensor that carries gradients gives one that carries them, with
    respect to the values and the times of its PCFs.
    """
    if is_tracked(fs):
        return _tracked_sum(fs, dim, as_mean=True)
    backend, tensor = _resolve_pcf_inputs(_REDUCTIONS_BACKEND_MAP, fs)
    return PcfTensor(backend.mean(tensor._data, _resolve_dim(dim, tensor.ndim)))


def max_time(fs: PcfContainerLike, dim: int = 0) -> FloatTensor:
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
    return FloatTensor(backend.max_time(tensor._data, _resolve_dim(dim, tensor.ndim)))


def _full_matrices(matrices: Any) -> Any:
    """The matrices of a DistanceMatrixTensor of shape S, which all have n
    points, in full: a NumPy array of shape (*S, n, n), or a framework array
    if the tensor carries gradients."""
    if is_tracked(matrices):
        entries, enumeration, sizes = matrices._tvalues, matrices._enumeration, matrices._sizes
    else:
        content, enumeration, sizes = matrices._serial()
        entries = content[:, 0]
    if sizes.size == 0 or (sizes != sizes.flat[0]).any():
        raise ValueError(
            "max of a DistanceMatrixTensor needs matrices of one size, as its result holds a number for each "
            f"point of each matrix (got sizes from {int(sizes.min()) if sizes.size else 0} to "
            f"{int(sizes.max()) if sizes.size else 0}).")
    n = int(sizes.flat[0])
    # Entry (i, j) of each matrix among its entries below the diagonal, by
    # rows, after a 0 for the diagonal.
    i, j = np.indices((n, n))
    high, low = np.maximum(i, j), np.minimum(i, j)
    starts = enumeration.reshape(-1, 2)[:, 0]
    index = np.where(i == j, 0, 1 + starts[:, None, None] + high * (high - 1) // 2 + low).ravel()
    shape = (*enumeration.shape[:-1], n, n)
    adapter = _autodiff._framework(entries)
    if adapter is None:
        return np.concatenate([np.zeros(1, dtype=entries.dtype), entries])[index].reshape(shape)
    padded = adapter.concat([adapter.as_tensor(np.zeros(1), like=entries), entries])
    return adapter.reshape(adapter.gather(padded, index), shape)


@overload
def max(x: DistanceMatrix | DistanceMatrixTensor | FloatTensor, dim: int = 0) -> FloatTensor: ...
@overload
def max(x: IntTensor, dim: int = 0) -> IntTensor: ...
@overload
def max(x: torch.Tensor, dim: int = 0) -> torch.Tensor: ...
def max(x: Any, dim: int = 0) -> Any:
    r"""Compute the largest number along the given dimension.

    For a distance matrix of :math:`n` points, the result holds, for each
    point, its largest distance to another point, so the maximum of the
    result is the largest distance of the matrix::

        farthest = sb.max(D)          # shape (n,)
        diameter = sb.max(farthest)   # shape (1,)

    See :ref:`tensors-how-dim-works` for a detailed explanation of dimension
    reduction semantics.

    Parameters
    ----------
    x : DistanceMatrix, DistanceMatrixTensor, FloatTensor, IntTensor, or torch.Tensor
        A distance matrix, which reduces as its :math:`n \times n` matrix, or
        numbers. A ``DistanceMatrixTensor`` of shape ``shape``, whose
        matrices must all have :math:`n` points, reduces as numbers of shape
        ``(*shape, n, n)`` along one of the last two dimensions, such as
        ``dim=-1``, to the largest distance from each point of each matrix.
    dim : int, optional
        Dimension along which to reduce, by default 0. Negative values count
        from the last axis. A distance matrix is symmetric, so both of its
        dimensions give the same result.

    Returns
    -------
    FloatTensor, IntTensor, or torch.Tensor
        The largest numbers, with the reduced dimension removed. Reducing a
        one-dimensional tensor returns shape ``(1,)``. A PyTorch tensor
        gives a PyTorch tensor, which carries its gradients, and so does a
        ``DistanceMatrixTensor`` that carries gradients.

    Raises
    ------
    IndexError
        If ``dim`` is out of range for the number of dimensions of ``x``.
    ValueError
        If the reduction dimension is empty (size 0), or the matrices of a
        ``DistanceMatrixTensor`` have different sizes.
    NotImplementedError
        If ``dim`` is a dimension of the shape of a ``DistanceMatrixTensor``,
        and not one of its matrices.
    TypeError
        If ``x`` is of another type.
    """
    if isinstance(x, DistanceMatrixTensor):
        values = _full_matrices(x)
        if _resolve_dim(dim, len(values.shape)) < len(values.shape) - 2:
            raise NotImplementedError(
                f"max along dim {dim} of a DistanceMatrixTensor of shape {tuple(x.shape)} would compare its "
                "matrices with each other, which is not supported. Pass dim=-1 for the largest distance from "
                "each point of each matrix.")
        return max(values if _autodiff._framework(values) is not None else FloatTensor(values), dim)

    adapter = _autodiff._framework(x)
    if adapter is not None:
        values = x
    elif isinstance(x, DistanceMatrix):
        # ponytail: the full matrix, twice the stored entries; reduce the
        # compressed entries in C++ if matrices get large.
        values = x.to_dense()
    elif isinstance(x, (FloatTensor, IntTensor)):
        values = np.asarray(x)
    else:
        raise TypeError(
            "max takes a DistanceMatrix, a DistanceMatrixTensor, a FloatTensor, an IntTensor, or a PyTorch tensor "
            f"(got {type(x).__name__}).")
    axis = _resolve_dim(dim, len(values.shape))
    if values.shape[axis] == 0:
        raise ValueError("max of an empty dimension has no value.")
    if adapter is not None:
        return adapter.reshape(adapter.max_along(values, axis), tuple(values.shape[:axis]) + tuple(values.shape[axis + 1:]) or (1,))
    largest = np.atleast_1d(values.max(axis=axis))
    return IntTensor(largest) if isinstance(x, IntTensor) else FloatTensor(largest)
