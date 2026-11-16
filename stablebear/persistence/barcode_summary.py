from typing import overload

import numpy as np

from .. import _autodiff
from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..base_tensor import PcfTensor, _get_backend
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


def _tracked_stable_rank(bc: _TrackedBarcodeTensor) -> PcfTensor:
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
    concrete = adapter.concrete(lifetimes)
    if concrete is not None:
        sorted_lifetimes = adapter.gather(lifetimes, np.lexsort((concrete, owners)))
    elif len(counts) == 1:
        sorted_lifetimes = adapter.sort(lifetimes)
    else:
        raise NotImplementedError(
            "Stable ranks of several barcodes whose bars JAX or tf.function traces abstractly are not supported.")

    # Each stable rank starts with a breakpoint at time 0, then one per bar.
    n_barcodes = len(counts)
    starts = np.cumsum(counts + 1) - (counts + 1)
    zero_rows = starts
    bar_rows = np.delete(np.arange(int(np.sum(counts + 1))), zero_rows)
    order = np.empty(n_barcodes + len(owners), dtype=np.int64)
    order[zero_rows] = np.arange(n_barcodes)
    order[bar_rows] = n_barcodes + np.arange(len(owners))
    zeros = adapter.as_tensor(np.zeros(n_barcodes), like=bars)
    times = adapter.gather(adapter.concat([zeros, sorted_lifetimes]), order)

    values = np.concatenate([np.arange(n, -1, -1) for n in counts]) if n_barcodes else np.zeros(0)
    content = np.column_stack([_autodiff._concrete_column(times, len(order), adapter), values]).astype(
        np.float32 if bc.dtype == barcode32 else np.float64)
    out = _autodiff._pcfs_from_content(content, _enumeration(counts + 1, tuple(enumeration.shape[:-1])))
    out._ttimes = times
    return out


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
    """
    if isinstance(bc, _TrackedBarcodeTensor):
        return _tracked_stable_rank(bc)
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
    """
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
    """
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


def _transformed_bars_out(bc, f):
    n_bars = len(bc._serial()[0])
    return (n_bars, 2), np.float32 if bc.dtype == barcode32 else np.float64


@_autodiff.differentiable("bc", "f", vjp=_transformed_bars_vjp, out=_transformed_bars_out)
def _transformed_bars(bc: BarcodeTensor, f: Plf) -> np.ndarray:
    """The (m, 2) transformed bars of a BarcodeTensor in the order of
    _serial, empty ones included."""
    out = _transform_bars(bc, f, verbose=False, keep_empty=True)
    assert isinstance(out, BarcodeTensor)
    return out._serial()[0]


def _tracked_transform_bars(bc: BarcodeTensor, f: Plf) -> BarcodeTensor:
    """transform_bars of a BarcodeTensor where the bars, f, or both depend
    on framework tensors."""
    bars = _transformed_bars(bc, f)
    adapter = _autodiff._framework(bars)
    concrete = adapter.concrete(bars)
    if concrete is None:
        raise NotImplementedError(
            "transform_bars of bars or a PLF that JAX or tf.function traces abstractly is not supported: "
            "which bars it removes depends on their values.")
    _, enumeration = bc._serial()
    counts, owners = _owners(enumeration)
    keep = concrete[:, 0] < concrete[:, 1]
    kept_counts = np.bincount(owners[keep], minlength=len(counts))
    return _TrackedBarcodeTensor(
        adapter.gather(bars, np.flatnonzero(keep)), _enumeration(kept_counts, tuple(enumeration.shape[:-1])))


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
    of PyTorch, JAX, or TensorFlow arrays, and :math:`f` when its times or
    values are such arrays (see :class:`~stablebear.functional.plf.Plf`).
    Then `bc` must be a ``BarcodeTensor``, and the result is one of the
    same shape whose bars carry gradients with respect to both. At a breakpoint of :math:`f`, the derivatives are those of the
    segment after it. Which bars are removed depends on their values, which
    JAX and ``tf.function`` do not provide while they trace, so there this
    raises ``NotImplementedError``.

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
        If the bars or `f` carry gradients that JAX or ``tf.function``
        traces, or `bc` is not a ``BarcodeTensor`` while they carry
        gradients.

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
    if f._tracked_parts() is not None or isinstance(bc, _TrackedBarcodeTensor):
        if not isinstance(bc, BarcodeTensor):
            raise NotImplementedError(
                "transform_bars with gradients supports a BarcodeTensor only; "
                "wrap a single Barcode as BarcodeTensor([bc]).")
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
