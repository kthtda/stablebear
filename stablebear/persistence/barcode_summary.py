from typing import Any, overload

import numpy as np

from .. import _autodiff
from .. import _sb_cpp as cpp
from .._tracked import is_tracked
from ..async_task import _run_task
from ..base_tensor import PcfTensor, _get_backend, _TrackedPcfTensor
from ..functional.pcf import Pcf
from ..functional.plf import Plf
from ..timeseries.series import TimeSeries
from ..typing import barcode32, barcode64, float32, float64
from .barcode import Barcode
from .ph_tensor import BarcodeTensor, _TrackedBarcodeTensor

cpp_p = cpp.persistence

_BACKEND_MAP = {
    barcode32: cpp_p.PersistenceBarcodeSummary32,
    barcode64: cpp_p.PersistenceBarcodeSummary64,
}


def _barcode_to_pcf(bc, task_method, verbose=False, **kwargs):
    """Shared implementation for barcode-to-PCF conversions."""
    if isinstance(bc, TimeSeries):
        if bc.dtype not in _BACKEND_MAP:
            raise TypeError(f"Expected a time series of barcodes, got {bc.dtype}")
        task = getattr(_BACKEND_MAP[bc.dtype], task_method)(bc._data, **kwargs)
        _run_task(lambda: task, verbose=verbose)
        return TimeSeries._from_cpp(task.result())

    backend, X = _get_backend(bc, _BACKEND_MAP)

    task = getattr(backend, task_method)(X._data, **kwargs)
    _run_task(lambda: task, verbose=verbose)
    out = PcfTensor(task.result())

    if isinstance(X, Barcode):
        # A single barcode gives a 0-d result holding its PCF.
        return out[()]

    if len(out.shape) == 2 and out.shape[0] == 1:
        out = out[0, :]

    return out


def _owners(enumeration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each barcode in C order its number of bars, and for each bar the
    flat index of its barcode, from the enumeration of BarcodeTensor._serial."""
    flat = enumeration.reshape(-1, 2)
    counts = flat[:, 1] - flat[:, 0]
    return counts, np.repeat(np.arange(len(counts)), counts)


def _enumeration(counts: np.ndarray, shape: tuple) -> np.ndarray:
    """The enumeration of consecutive blocks of the given sizes, with shape
    shape + (2,)."""
    ends = np.cumsum(counts)
    return np.stack([ends - counts, ends], axis=-1).reshape(*shape, 2)


def _tracked_stable_rank(bc: Any) -> PcfTensor:
    """The stable ranks of a BarcodeTensor whose bars are a framework tensor.

    With n bars, the stable rank is n on [0, l_1), then n - 1 on [l_1, l_2), and
    so on down to 0 after l_n, for the sorted lifetimes l_1 <= ... <= l_n. So its
    times are framework operations on the bars, which the framework
    differentiates (sort included), and its values are constant. Equal
    lifetimes give repeated times, which describe the same function.
    """
    bars = bc._tvalues
    adapter = _autodiff._framework(bars)
    enumeration = bc._enumeration
    counts, owners = _owners(enumeration)
    lifetimes = bars[:, 1] - bars[:, 0]
    order = cpp.argsort_segments(
        np.ascontiguousarray(adapter.to_numpy(lifetimes)), np.ascontiguousarray(enumeration.reshape(-1, 2)))
    sorted_lifetimes = adapter.gather(lifetimes, order)

    # Each stable rank starts with a breakpoint at time 0, then one per bar,
    # with the values n, ..., 0 for a barcode of n bars.
    times, sizes, position = _prepend_zeros(adapter, bars, sorted_lifetimes, counts)
    values = (np.repeat(counts, sizes) - position).astype(np.float32 if bc.dtype == barcode32 else np.float64)
    return _TrackedPcfTensor([times, values], _enumeration(sizes, tuple(enumeration.shape[:-1])))


def _prepend_zeros(adapter: Any, like: Any, x: Any, counts: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray]:
    """The framework array x, of consecutive blocks of the given sizes, with
    a zero before each block, as for PCFs that start with a breakpoint at
    time 0. Also returns the sizes of the new blocks and, for each row, its
    position in its block."""
    n = len(counts)
    sizes = counts + 1
    starts = np.cumsum(sizes) - sizes
    position = np.arange(int(sizes.sum())) - np.repeat(starts, sizes)
    # Row k of the zeros for the first row of block k, and then the rows of
    # x, which follow the zeros, in order.
    rows = np.where(position == 0, np.repeat(np.arange(n), sizes),
                    n + np.arange(len(position)) - np.repeat(np.arange(1, n + 1), sizes))
    zeros = adapter.as_tensor(np.zeros(n), like=like)
    return adapter.gather(adapter.concat([zeros, x]), rows), sizes, position


def _bar_values(bc: Any) -> np.ndarray:
    """The current values of the bars of a BarcodeTensor that carries
    gradients, as an (m, 2) array."""
    return _autodiff._framework(bc._tvalues).to_numpy(bc._tvalues)


def _tracked_betti_curve(bc: Any) -> PcfTensor:
    """The Betti curves of a BarcodeTensor whose bars are a framework tensor.

    The curve steps up at each birth and down at each death. So its times
    are the births and deaths of the barcode, sorted, which the framework
    differentiates, and its values, the running count of the steps, are
    constant. Equal times describe the same function as one step would.
    """
    bars = bc._tvalues
    adapter = _autodiff._framework(bars)
    enumeration = bc._enumeration
    counts, _ = _owners(enumeration)
    concrete = _bar_values(bc)
    n_bars = len(concrete)

    # The events of each barcode, as birth and death of each of its bars.
    events = np.ascontiguousarray(concrete.reshape(-1))
    order = cpp.argsort_segments(events, np.ascontiguousarray(2 * enumeration.reshape(-1, 2)))
    is_death = order % 2
    sorted_events = adapter.gather(adapter.concat([bars[:, 0], bars[:, 1]]), is_death * n_bars + order // 2)

    times, sizes, position = _prepend_zeros(adapter, bars, sorted_events, 2 * counts)
    # The steps of each barcode sum to zero, so one running sum serves all.
    values = np.zeros(len(position), dtype=np.float32 if bc.dtype == barcode32 else np.float64)
    values[position > 0] = np.cumsum(1 - 2 * is_death)
    return _TrackedPcfTensor([times, values], _enumeration(sizes, tuple(enumeration.shape[:-1])))


def _tracked_accumulated_persistence(bc: Any, max_death: float) -> PcfTensor:
    """The accumulated persistence functions of a BarcodeTensor whose bars
    are a framework tensor.

    The function steps up by the lifetime of each bar at its midpoint. So
    its times are the sorted midpoints and its values the running sums of
    the lifetimes in that order, both framework operations on the bars.
    Which bars count, the finite ones that die by max_death, depends on
    their values.
    """
    bars = bc._tvalues
    adapter = _autodiff._framework(bars)
    enumeration = bc._enumeration
    counts, owners = _owners(enumeration)
    concrete = _bar_values(bc)

    keep = np.isfinite(concrete[:, 1]) & (concrete[:, 1] <= max_death)
    kept_counts = np.bincount(owners[keep], minlength=len(counts))
    kept = adapter.gather(bars, np.flatnonzero(keep))
    midpoints = (kept[:, 0] + kept[:, 1]) / 2
    order = cpp.argsort_segments(
        np.ascontiguousarray((concrete[keep, 0] + concrete[keep, 1]) / 2),
        np.ascontiguousarray(_enumeration(kept_counts, (len(counts),))))

    times, sizes, position = _prepend_zeros(adapter, bars, adapter.gather(midpoints, order), kept_counts)
    # The running sum over all barcodes, less what it was when each began.
    sums = adapter.concat([
        adapter.as_tensor(np.zeros(1), like=bars), adapter.cumsum(adapter.gather(kept[:, 1] - kept[:, 0], order))])
    begins = np.repeat(np.cumsum(kept_counts) - kept_counts, sizes)
    values = adapter.gather(sums, begins + position) - adapter.gather(sums, begins)
    return _TrackedPcfTensor([times, values], _enumeration(sizes, tuple(enumeration.shape[:-1])))


@overload
def barcode_to_stable_rank(bc: Barcode, verbose: bool = False) -> Pcf: ...


@overload
def barcode_to_stable_rank(bc: BarcodeTensor, verbose: bool = False) -> PcfTensor: ...


@overload
def barcode_to_stable_rank(bc: TimeSeries, verbose: bool = False) -> TimeSeries: ...


def barcode_to_stable_rank(
    bc: Barcode | BarcodeTensor | TimeSeries, verbose=False
):
    r"""Convert barcodes to stable rank functions.

    The stable rank of a barcode is the PCF that counts, for each
    :math:`t \geq 0`, the number of bars with length (death minus birth)
    strictly greater than :math:`t` :footcite:`Chacholski2020`.

    Parameters
    ----------
    bc : Barcode, BarcodeTensor, or TimeSeries
        A single barcode or a container of barcodes.
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    Pcf, PcfTensor, or TimeSeries
        A single ``Pcf`` if the input is a single ``Barcode``, otherwise the
        same kind of container as the input, with the same shape.

    Notes
    -----
    The bars of a ``BarcodeTensor`` can carry gradients of PyTorch, as when it comes from
    :func:`~stablebear.persistence.homological_kernel.compute_homological_kernel`
    of arrays of one of them. The result is then a ``PcfTensor`` whose
    breakpoint times carry gradients with respect to the bars. Besides its
    breakpoint at time 0, it has one per bar, so bars of equal length give
    breakpoints with equal times.
    """
    if isinstance(bc, _TrackedBarcodeTensor):
        return _tracked_stable_rank(bc)
    if is_tracked(bc):
        # A single barcode that carries gradients, as a tensor of one.
        return _tracked_stable_rank(BarcodeTensor([bc]))[0]
    return _barcode_to_pcf(
        bc, "spawn_barcode_to_stable_rank_task", verbose
    )


@overload
def barcode_to_betti_curve(bc: Barcode, verbose: bool = False) -> Pcf: ...


@overload
def barcode_to_betti_curve(bc: BarcodeTensor, verbose: bool = False) -> PcfTensor: ...


@overload
def barcode_to_betti_curve(bc: TimeSeries, verbose: bool = False) -> TimeSeries: ...


def barcode_to_betti_curve(
    bc: Barcode | BarcodeTensor | TimeSeries, verbose=False
):
    r"""Convert barcodes to Betti curves.

    The Betti curve is the PCF that counts, for each filtration value
    :math:`t \geq 0`, the number of bars alive at :math:`t`
    (i.e., bars with birth :math:`\leq t <` death) :footcite:`Umeda2017,Chazal2021`.

    Parameters
    ----------
    bc : Barcode, BarcodeTensor, or TimeSeries
        A single barcode or a container of barcodes.
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    Pcf, PcfTensor, or TimeSeries
        A single ``Pcf`` if the input is a single ``Barcode``, otherwise the
        same kind of container as the input, with the same shape.

    Notes
    -----
    The bars of a ``BarcodeTensor`` can carry gradients of PyTorch, as when it comes from
    :func:`~stablebear.persistence.homological_kernel.compute_homological_kernel`
    of arrays of one of them. The result is then a ``PcfTensor`` whose
    breakpoint times carry gradients with respect to the bars. Besides its
    breakpoint at time 0, it has one per birth and one per death, so
    births and deaths at equal times give breakpoints with equal times.
    """
    if isinstance(bc, _TrackedBarcodeTensor):
        return _tracked_betti_curve(bc)
    if is_tracked(bc):
        return _tracked_betti_curve(BarcodeTensor([bc]))[0]
    return _barcode_to_pcf(
        bc, "spawn_barcode_to_betti_curve_task", verbose
    )


@overload
def barcode_to_accumulated_persistence(bc: Barcode, max_death: float = float("inf"), verbose: bool = False) -> Pcf: ...


@overload
def barcode_to_accumulated_persistence(bc: BarcodeTensor, max_death: float = float("inf"), verbose: bool = False) -> PcfTensor: ...


@overload
def barcode_to_accumulated_persistence(bc: TimeSeries, max_death: float = float("inf"), verbose: bool = False) -> TimeSeries: ...


def barcode_to_accumulated_persistence(
    bc: Barcode | BarcodeTensor | TimeSeries,
    max_death: float = float("inf"),
    verbose: bool = False,
):
    r"""Convert barcodes to accumulated persistence functions.

    The accumulated persistence function (APF) is defined as

    .. math::

       \mathrm{APF}(t) = \sum_{i=1}^{N} \ell_i \, \mathbf{1}_{m_i \leq t}

    where :math:`N` is the number of bars, :math:`\ell_i = d_i - b_i` is the
    lifetime of bar :math:`i`, and :math:`m_i = (b_i + d_i) / 2` is its
    midpoint :footcite:`Biscio2019`.

    When ``max_death`` is finite, only bars with :math:`d_i \leq`
    ``max_death`` are included (Equation 2 in the paper).

    Parameters
    ----------
    bc : Barcode, BarcodeTensor, or TimeSeries
        A single barcode or a container of barcodes.
    max_death : float, optional
        If finite, exclude bars whose death time exceeds this value.
        By default ``inf`` (all finite bars included).
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    Pcf, PcfTensor, or TimeSeries
        A single ``Pcf`` if the input is a single ``Barcode``, otherwise the
        same kind of container as the input, with the same shape.

    Notes
    -----
    The bars of a ``BarcodeTensor`` can carry gradients of PyTorch, as when it comes from
    :func:`~stablebear.persistence.homological_kernel.compute_homological_kernel`
    of arrays of one of them. The result is then a ``PcfTensor`` whose
    breakpoint times and values carry gradients with respect to the bars.
    Besides its breakpoint at time 0, it has one per included bar, so bars
    with equal midpoints give breakpoints with equal times.
    """
    _autodiff.require_number(max_death, "max_death")
    if isinstance(bc, _TrackedBarcodeTensor):
        return _tracked_accumulated_persistence(bc, max_death)
    if is_tracked(bc):
        return _tracked_accumulated_persistence(BarcodeTensor([bc]), max_death)[0]
    return _barcode_to_pcf(
        bc, "spawn_barcode_to_accumulated_persistence_task", verbose,
        max_death=max_death
    )


def _plf_vjp(f: Plf, x: np.ndarray, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The gradients of sum(grad * f(x)) with respect to the times x, shaped
    like x, and to the breakpoint times and values of f, each (n,). At a
    breakpoint, the derivatives are those of the segment after it. At an
    infinite x, f takes its limit: the derivatives are 0, except 1 with
    respect to the last value where the limit is that value.
    """
    pts = f.to_numpy().astype(np.float64)
    n = len(pts)
    if n == 1:
        return np.zeros(len(x)), np.zeros(1), np.array([grad.sum()])
    finite = ~np.isinf(x)
    xf, g = x[finite], grad[finite]
    # Segment k from breakpoint k to k + 1; after the last breakpoint, the
    # last segment continues.
    k = np.minimum(np.searchsorted(pts[:, 0], xf, side="right") - 1, n - 2)
    h = pts[k + 1, 0] - pts[k, 0]
    slope = (pts[k + 1, 1] - pts[k, 1]) / h
    w = (xf - pts[k, 0]) / h
    d_x = np.zeros(len(x))
    d_x[finite] = g * slope

    def scatter(left, right):
        return np.bincount(k, weights=g * left, minlength=n) + np.bincount(k + 1, weights=g * right, minlength=n)

    d_times = scatter(-slope * (1 - w), -slope * w)
    d_values = scatter(1 - w, w)
    if pts[-1, 1] == pts[-2, 1]:
        d_values[-1] += grad[~finite].sum()
    return d_x, d_times, d_values


def _transformed_bars_vjp(grad_out, bc, f):
    bars = np.asarray(bc._serial()[0], dtype=np.float64).reshape(-1)
    grad = np.asarray(grad_out, dtype=np.float64).reshape(-1)
    d_bars, d_times, d_values = _plf_vjp(f, bars, grad)
    return d_bars.reshape(-1, 2), np.column_stack([d_times, d_values])


@_autodiff.differentiable("bc", "f", vjp=_transformed_bars_vjp, returns=BarcodeTensor)
def _transformed_bars(bc: BarcodeTensor, f: Plf) -> Any:
    """The transformed barcodes of a BarcodeTensor, empty bars included."""
    return _transform_bars(bc, f, verbose=False, keep_empty=True)


def _tracked_transform_bars(bc: BarcodeTensor, f: Plf) -> BarcodeTensor:
    """transform_bars of a BarcodeTensor where the bars, f, or both depend
    on framework tensors."""
    transformed = _transformed_bars(bc, f)
    bars, enumeration = transformed._tvalues, transformed._enumeration
    adapter = _autodiff._framework(bars)
    concrete = adapter.to_numpy(bars)
    counts, owners = _owners(enumeration)
    keep = concrete[:, 0] < concrete[:, 1]
    kept_counts = np.bincount(owners[keep], minlength=len(counts))
    return _TrackedBarcodeTensor(
        [adapter.gather(bars, np.flatnonzero(keep))], _enumeration(kept_counts, tuple(enumeration.shape[:-1])))


@overload
def transform_bars(bc: Barcode, f: Plf, verbose: bool = False) -> Barcode: ...


@overload
def transform_bars(bc: BarcodeTensor, f: Plf, verbose: bool = False) -> BarcodeTensor: ...


@overload
def transform_bars(bc: TimeSeries, f: Plf, verbose: bool = False) -> TimeSeries: ...


def transform_bars(bc: Barcode | BarcodeTensor | TimeSeries, f: Plf, verbose: bool = False):
    r"""Transform the endpoints of every bar by a PLF.

    Each bar :math:`(b, d)` becomes :math:`(f(b), f(d))`. An infinite
    endpoint becomes the limit of :math:`f`: :math:`\infty` or
    :math:`-\infty` if the last segment of :math:`f` rises or falls, or its
    last value if the segment is flat. Bars with :math:`f(b) \geq f(d)`,
    which :math:`f` can give where it is flat or falls, are removed.

    The bars and :math:`f` can carry gradients: the bars when they come
    from :func:`~stablebear.persistence.homological_kernel.compute_homological_kernel`
    of PyTorch tensors, and :math:`f` when its times or
    values are such tensors (see :class:`~stablebear.functional.plf.Plf`).
    Then `bc` must be a ``Barcode`` or a ``BarcodeTensor``, and the bars
    of the result carry gradients with respect to both. At a
    breakpoint of :math:`f`, the derivatives are those of the segment after
    it.

    Parameters
    ----------
    bc : Barcode, BarcodeTensor, or TimeSeries
        A single barcode or a container of barcodes.
    f : Plf
        The function to apply to the endpoints. It is converted to the
        precision of the barcodes.
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    Barcode, BarcodeTensor, or TimeSeries
        The same kind of object as `bc`, with the same shape, less the
        removed bars.

    Raises
    ------
    ValueError
        If an endpoint is negative (a PLF is defined on
        :math:`[0, \infty)`).
    NotImplementedError
        If `bc` is a time series while the bars or `f` carry gradients.

    Examples
    --------
    >>> import stablebear as sb
    >>> bc = sb.persistence.Barcode([[0.0, 1.0], [0.5, float("inf")]])
    >>> f = sb.Plf([[0.0, 0.0], [1.0, 2.0]])
    >>> sb.persistence.transform_bars(bc, f).to_numpy()
    array([[ 0.,  2.],
           [ 1., inf]])
    """
    if not isinstance(f, Plf):
        raise TypeError(f"Expected a Plf, got {type(f).__name__}.")
    if is_tracked(f) or is_tracked(bc):
        if isinstance(bc, Barcode):
            # A single barcode, as a tensor of one.
            return _tracked_transform_bars(BarcodeTensor([bc]), f)[0]
        if not isinstance(bc, BarcodeTensor):
            raise NotImplementedError(
                "transform_bars with gradients supports a Barcode or a BarcodeTensor only.")
        return _tracked_transform_bars(bc, f)
    return _transform_bars(bc, f, verbose=verbose, keep_empty=False)


def _transform_bars(bc, f, verbose, keep_empty):
    if isinstance(bc, TimeSeries):
        if bc.dtype not in _BACKEND_MAP:
            raise TypeError(f"Expected a time series of barcodes, got {bc.dtype}")
        backend, data = _BACKEND_MAP[bc.dtype], bc._data
    else:
        backend, X = _get_backend(bc, _BACKEND_MAP)
        data = X._data
    f = f.astype(float32 if bc.dtype == barcode32 else float64)

    task = backend.spawn_transform_bars_task(data, f._data, keep_empty=keep_empty)
    _run_task(lambda: task, verbose=verbose)
    if isinstance(bc, TimeSeries):
        return TimeSeries._from_cpp(task.result())
    if isinstance(bc, Barcode):
        return BarcodeTensor(task.result())[()]
    return BarcodeTensor(task.result())
