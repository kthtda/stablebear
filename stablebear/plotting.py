import matplotlib.pyplot as plt
import numpy as np

from . import distributions as _distributions
from ._validation import _positive_integer
from .persistence.barcode import Barcode
from .persistence.ph_tensor import BarcodeTensor
from .reductions import max_time as max_time_reduction
from .base_tensor import IntPcfTensor, PcfContainerLike, PcfTensor, _to_tensor_pcf


def plot(f: PcfContainerLike, fmt="", ax=None, auto_label=False, max_time=None, **kwargs):
    """Plot one or more PCFs using matplotlib's step function.

    Parameters
    ----------
    f : PcfContainerLike
        A single ``Pcf``, a list of ``Pcf`` objects, or a 1-D PCF tensor.
    fmt : str, optional
        A matplotlib format string (e.g. ``'r--'``), by default ``''``.
    ax : matplotlib axes, optional
        Axes to plot on. If ``None``, uses ``matplotlib.pyplot`` directly.
    auto_label : bool, optional
        If ``True`` and ``f`` is a tensor, label each PCF as ``f0``, ``f1``,
        etc. By default ``False``.
    max_time : float, optional
        Extend the plot so the final constant segment reaches this time.
        If ``None``, single PCFs are not extended and tensors extend to
        the latest breakpoint across all elements.
    **kwargs
        Additional keyword arguments passed to ``matplotlib.pyplot.step``
        (e.g. ``color``, ``linewidth``, ``alpha``, ``label``).

    Raises
    ------
    ValueError
        If ``f`` is a tensor with more than one dimension.
    """
    ax = plt if ax is None else ax

    def plot_single_(f, maxtime, **kwargs1):
        X = np.array(f)
        if maxtime is not None and X[-1, 0] != maxtime:
            X = np.vstack((X, [maxtime, X[-1, 1]]))
        ax.step(X[:, 0], X[:, 1], fmt, where="post", **kwargs, **kwargs1)

    if isinstance(f, (list, tuple)):
        f = _to_tensor_pcf(f)

    if isinstance(f, (PcfTensor, IntPcfTensor)):
        if len(f.shape) != 1:
            squeezed = f.squeeze()
            if len(squeezed.shape) != 1:
                raise ValueError(f"Expected 1-dimensional array (got array with {f.shape})")
            return plot(squeezed, ax=ax, max_time=max_time, auto_label=auto_label, **kwargs)
        if f.shape[0] == 0:
            # Nothing to plot: an empty tensor has no PCFs. Bail out before
            # reducing with max_time, which has no identity over an empty range
            # and would otherwise raise on the (vacuous) reduction.
            return
        if max_time is not None:
            mt = max_time
        elif isinstance(f, PcfTensor):
            mt = float(max_time_reduction(f))
        else:
            mt = max(float(np.asarray(f[i])[-1, 0]) for i in range(f.shape[0]))
        for i in range(f.shape[0]):
            kw = {"label": f"f{i}"} if auto_label else {}
            plot_single_(f[i], mt, **kw)
    else:
        plot_single_(f, max_time)


def _finite_vector(value, size, message):
    """Return ``value`` as a float array of ``size`` finite real entries."""
    if np.iscomplexobj(value):
        raise ValueError(message)
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(message)
    return value


def plot_distance_weight_heatmap(
    distribution, query=(0.0, 0.0), extent=None,
    resolution=512, ax=None, **kwargs,
):
    """Plot distribution weights applied to distance from a query point.

    Parameters
    ----------
    distribution : Distribution
        A distribution from :mod:`stablebear.distributions`.
    query : array_like, optional
        Two finite coordinates ``(x, y)``, by default the origin.
    extent : array_like, optional
        Plot bounds ``(xmin, xmax, ymin, ymax)``. If omitted, uses
        ``distribution.plot_range()`` to choose a square centered on ``query``:
        its half-width is the largest absolute endpoint plus a 10% margin.
        Bounds must be finite and strictly increasing on each axis.
    resolution : int, optional
        Positive number of pixels on each axis, by default 512. Weights are
        evaluated at pixel centers within ``extent``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If omitted, uses the current axes.
    **kwargs
        Styling arguments passed to ``Axes.imshow``, such as ``cmap``,
        ``alpha``, ``norm``, ``vmin``, and ``vmax``. Defaults are
        ``cmap="viridis"``, ``aspect="equal"``, and ``interpolation="nearest"``.
        The image origin is fixed to ``"lower"``.

    Returns
    -------
    matplotlib.image.AxesImage
        The heatmap, suitable for passing to ``fig.colorbar``.

    Notes
    -----
    Colors represent ``distribution.weight(distance)``. No normalization over
    the grid is performed, and the image is not a probability density over the
    plane. This function adds no colorbar, labels, or query marker.
    """
    if not isinstance(distribution, _distributions.Distribution):
        raise TypeError("distribution must be a Distribution")
    resolution = _positive_integer(resolution, "resolution")
    query = _finite_vector(query, 2, "query must contain two finite real coordinates")
    if extent is None:
        radius = 1.1 * max(abs(bound) for bound in distribution.plot_range())
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("cannot choose a finite plot region; pass extent explicitly")
        extent = (query[0] - radius, query[0] + radius,
                  query[1] - radius, query[1] + radius)
    extent = _finite_vector(extent, 4, "extent must contain four finite real bounds")
    xmin, xmax, ymin, ymax = extent
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("extent must satisfy xmin < xmax and ymin < ymax")

    fractions = (np.arange(resolution) + 0.5) / resolution
    x = (1 - fractions) * xmin + fractions * xmax
    y = (1 - fractions) * ymin + fractions * ymax
    distances = np.hypot(x[None, :] - query[0], y[:, None] - query[1])
    weights = distribution.weight(distances)

    ax = plt.gca() if ax is None else ax
    kwargs.setdefault("cmap", "viridis")
    kwargs.setdefault("aspect", "equal")
    kwargs.setdefault("interpolation", "nearest")
    return ax.imshow(weights, origin="lower", extent=extent, **kwargs)


def _resolve_barcode_colors(n, color=None, ax=None):
    """Resolve a color specification into one color per barcode.

    Parameters
    ----------
    n : int
        Number of barcodes to plot.
    color : color-like object, list of color-like objects or None, optional
        ``None`` takes the next ``n`` colors from the axes' property cycle, a
        single color is repeated for all barcodes, and a sequence of colors is
        cycled across barcodes.
    ax : matplotlib axes, optional
        Axes whose property cycle supplies default colors. Only used when
        ``color`` is ``None``; defaults to the current axes.

    Returns
    -------
    list of color-like objects
        One color per barcode to be plotted.

    Raises
    ------
    ValueError
        If color is neither None, a color-like object, nor a list of color-like objects.
    """
    from matplotlib.colors import is_color_like

    if color is None:
        # Advance the axes' own cycle, as ax.plot does, so set_prop_cycle and
        # repeated calls on the same axes behave as in matplotlib.
        ax = plt.gca() if ax is None else ax
        return [ax._get_lines.get_next_color() for _ in range(n)]
    if is_color_like(color):
        cycle = [color]
    else:
        try:
            cycle = list(color)
        except TypeError:
            cycle = []
        if len(cycle) == 0 or not all(is_color_like(c) for c in cycle):
            raise ValueError(
                f"'color' must be a color or a non-empty sequence of colors; got {color!r}"
            )
    return [cycle[i % len(cycle)] for i in range(n)]


def _infinite_bar_end(bars):
    """Return the x value where infinite bars in ``bars`` are drawn to end."""
    deaths = bars[:, 1]
    finite = np.isfinite(deaths)
    if finite.any():
        return deaths[finite].max() * 1.15
    return bars[:, 0].max() + 1.0


def plot_barcode(bc, ax=None, y_offset=0, **kwargs):
    """Plot a persistence barcode as horizontal line segments.

    Each bar is drawn as a horizontal segment from birth to death.
    Bars with infinite death are drawn as arrows extending to the right
    edge of the plot. For a ``BarcodeTensor``, infinite bars in all of its
    barcodes end at the same x.

    Parameters
    ----------
    bc : Barcode, BarcodeTensor or NumPy array
        A single ``Barcode``, a 1-D ``BarcodeTensor`` or an (n, 2) NumPy array
        of ``(birth, death)`` pairs. For a tensor,
        the barcodes are stacked vertically in order.
    ax : matplotlib axes, optional
        Axes to plot on. If ``None``, uses the current axes.
    y_offset : int, optional
        Starting y position for the first bar. Useful when stacking
        multiple barcodes on the same axes.
    **kwargs
        Additional keyword arguments passed to
        ``matplotlib.collections.LineCollection``
        (e.g. ``linewidth``, ``alpha``, ``label``), except for ``color``,
        described below. ``colors`` is not accepted, because the bars are
        sorted before drawing, and neither is ``cmap``.

        ``color`` is a single color, or a sequence with one color per
        barcode that is cycled if shorter than the number of barcodes.
        A tuple of three or four numbers is one RGB(A) color, as in
        matplotlib. To color barcodes from a colormap, pass its colors, such
        as ``mpl.colormaps["viridis"](np.linspace(0, 1, n))`` or
        ``mpl.colormaps["tab20"].colors``. A single ``Barcode`` or array is
        treated as a one-barcode tensor, so it uses the first color of a
        sequence. If ``color`` is omitted, each barcode takes the next color
        from the axes' property cycle, continuing across calls on the same
        axes.

    Returns
    -------
    int
        The next available y position (for stacking).

    Raises
    ------
    TypeError
        If ``bc`` is not a ``Barcode``, ``BarcodeTensor``, or ``numpy.ndarray``,
        if ``bc`` is a ``numpy.ndarray`` containing neither floats nor ints,
        or if keyword ``colors`` or ``cmap`` is passed.
    ValueError
        If ``bc`` is a numpy array that is not of shape ``(n, 2)``, a
        ``BarcodeTensor`` with more than one dimension, or if ``color`` is
        neither a color nor a non-empty sequence of colors.
    """
    from matplotlib.collections import LineCollection

    for name in ("colors", "cmap"):
        # LineCollection would accept 'colors' (per-segment), but the bars are
        # sorted internally so per-segment colors would land in scrambled
        # order, and 'cmap' has no effect on it. Only 'color' has
        # well-defined semantics here.
        if name in kwargs:
            raise TypeError(
                f"plot_barcode() does not accept '{name}'; use 'color' "
                "(a single color, or a sequence with one color per barcode)"
            )

    if isinstance(bc, BarcodeTensor):
        if len(bc.shape) != 1:
            squeezed = bc.squeeze()
            if len(squeezed.shape) != 1:
                raise ValueError(f"Expected 1-dimensional tensor (got shape {bc.shape})")
            return plot_barcode(squeezed, ax=ax, y_offset=y_offset, **kwargs)
        colors = _resolve_barcode_colors(bc.shape[0], kwargs.pop("color", None), ax=ax)
        # Infinite bars in every barcode end at the same x, computed from all
        # bars in the tensor.
        all_bars = np.concatenate([np.asarray(bc[i]).reshape(-1, 2) for i in range(bc.shape[0])])
        if len(all_bars) > 0:
            kwargs["_inf_end"] = _infinite_bar_end(all_bars)
        y = y_offset
        for i in range(bc.shape[0]):
            y = plot_barcode(bc[i], ax=ax, y_offset=y, color=colors[i], **kwargs)
        return y

    if not isinstance(bc, (Barcode, np.ndarray)):
        raise TypeError(f"Expected Barcode or numpy.ndarray; got {type(bc)}")

    # A single barcode behaves like a 1-element tensor for color purposes:
    # a sequence of colors resolves to its first color, and no
    # color takes the next one from the axes' cycle once the axes exist.
    # Explicit colors are validated here, before any axes is created.
    color = kwargs.pop("color", None)
    if color is not None:
        color = _resolve_barcode_colors(1, color)[0]
    inf_end = kwargs.pop("_inf_end", None)

    if isinstance(bc, np.ndarray):
        if bc.ndim != 2 or bc.shape[1] != 2:
            raise ValueError(f"Input should have shape (n, 2). Supplied input has shape {bc.shape}")
        if not (np.issubdtype(bc.dtype, np.floating) or np.issubdtype(bc.dtype, np.integer)):
            raise TypeError(f"Unsupported dtype {bc.dtype}; expected float or integer values")

    ax = plt.gca() if ax is None else ax
    if color is None:
        color = _resolve_barcode_colors(1, ax=ax)[0]
    kwargs["color"] = color

    bars = np.asarray(bc)

    if len(bars) == 0:
        return y_offset

    # Sort bars lexicographically by (birth, length), descending
    lengths = np.where(np.isfinite(bars[:, 1]), bars[:, 1] - bars[:, 0], np.inf)
    order = np.lexsort((lengths, bars[:, 0]))[::-1]
    bars = bars[order]

    defaults = {"linewidth": 1.5}
    defaults.update(kwargs)
    lw = defaults.get("linewidth", 1.5)

    # Infinite bars extend to this point
    xmax = _infinite_bar_end(bars) if inf_end is None else inf_end

    # Build all bar segments in one pass, using sequential y-positions
    segments = []
    inf_y_positions = []
    for i in range(len(bars)):
        y = y_offset + i
        birth = bars[i, 0]
        death = bars[i, 1]
        if np.isfinite(death):
            segments.append([(birth, y), (death, y)])
        else:
            segments.append([(birth, y), (xmax, y)])
            inf_y_positions.append(y)

    arrow_color = None
    if segments:
        lc = LineCollection(segments, **defaults)
        ax.add_collection(lc)
        arrow_color = tuple(lc.get_colors()[0])

    # Add arrowheads for infinite bars
    for y in inf_y_positions:
        ax.annotate("",
            xy=(xmax, y),
            xytext=(xmax - (xmax * 0.02), y),
            arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=lw))

    ax.autoscale_view()
    ax.yaxis.set_ticks([])

    return y_offset + len(bars)


def _time_axis_unit(ax, unit, requested):
    """The time unit of the x axis of ``ax``, storing ``unit`` on it first if
    it has none. ``requested`` is the unit asked for, if any, which must
    match a unit the axis already has."""
    if ax.xaxis.get_converter() is not None:
        raise TypeError("The x axis holds dates or other values that are not times in a unit")
    current = ax.xaxis.get_units()
    if current is None:
        # Axes made with sharex share this unit.
        ax.xaxis.set_units(unit)
        return unit
    if requested is not None and requested != current:
        raise ValueError(f"The time axis is in {current!r}, not {requested!r}")
    return current


def plot_timeseries(series, ax=None, *, unit=None, relative=False, markers=False, **kwargs):
    """Plot a time series as the step function it evaluates to.

    Each channel is drawn from its first observation to the end of its
    domain, holding each value until the next observation, at the times
    where it lies in the series it was cut from: a window is drawn where it
    was cut, at its :attr:`~stablebear.timeseries.TimeSeries.origin` plus its
    own times, unless ``relative`` is true.

    Parameters
    ----------
    series : TimeSeries or TimeSeriesTensor
        A time series with one channel of scalar observations, or a 1-D
        tensor of them, such as windows, drawn one line each, each taking
        the next color of the axes unless ``color`` is given.
    ax : matplotlib axes, optional
        Axes to draw on. If ``None``, uses the current axes.
    unit : str, optional
        The time unit of the x axis, one of the units of
        :class:`~stablebear.timeseries.Times`. Defaults to the unit the axis
        already has, or else the unit of ``series``.
    relative : bool, default False
        Whether to draw each time series at its own times, from 0, instead
        of where it lies in the series it was cut from. Windows are then
        drawn on top of each other, to compare their shapes.
    markers : bool, default False
        Whether to mark each observation with a dot, which also shows a
        last value that holds only at the end of the domain.
    **kwargs
        Passed to ``Axes.plot``, such as ``color``, ``linewidth``, or
        ``label``.

    Returns
    -------
    list of matplotlib.lines.Line2D
        The lines drawn.

    Raises
    ------
    ValueError
        If ``unit`` differs from the unit the x axis already has, the
        observations are not scalars, or a tensor has more than one
        dimension.
    TypeError
        If the x axis holds dates or other values that are not times.

    Notes
    -----
    The x axis keeps one time unit, stored on it by the first time series
    drawn, and each time series is converted into it exactly, so series in
    different units line up. The axis is labeled ``time (<unit>)`` unless it
    already has a label.
    """
    from .timeseries.series import TimeSeriesTensor

    ax = plt.gca() if ax is None else ax
    if isinstance(series, TimeSeriesTensor):
        tensor = series.squeeze() if len(series.shape) != 1 else series
        if len(tensor.shape) != 1:
            raise ValueError(f"Expected a 1-D tensor of time series, got shape {series.shape}")
        return [line for k in range(tensor.shape[0])
                for line in plot_timeseries(tensor[k], ax=ax, unit=unit, relative=relative, markers=markers, **kwargs)]
    requested = None if unit is None else series.to(unit).unit
    series = series.to(_time_axis_unit(ax, requested or series.unit, requested))
    offset = 0.0 if relative else series.origin
    times = np.asarray(series.times) + offset
    values = np.asarray(series.values)
    if values.ndim != 1:
        raise ValueError(f"Can only plot scalar observations, not observations of shape {series.shape}")
    if markers:
        kwargs = {"marker": "o", "markevery": list(range(len(times))), **kwargs}
    end = series.domain.end + offset
    if end > times[-1]:
        # The last value holds until the end of the domain.
        times = np.append(times, end)
        values = np.append(values, values[-1])
    lines = ax.plot(times, values, drawstyle="steps-post", **kwargs)
    if not ax.get_xlabel():
        ax.set_xlabel(f"time ({series.unit})")
    return lines
