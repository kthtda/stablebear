# Time series dates and plotting spec (draft)

Status: draft for discussion. Nothing implemented.

## Problem

`stablebear.plotting` plots PCFs, barcodes, and distribution heatmaps, but not
time series. Today every time series figure is hand-rolled:

- `docs/_static/gen_timeseries_snap_fig.py` has a private `_plot_steps` helper
  (step segments, filled/hollow ends), window bars (`_plot_window_rows`), and
  ad hoc per-channel subplots.
- `docs/tutorial_notebooks/timeseries_tides.ipynb` converts timestamps to
  dates and calls `ax.plot(stamps, levels)` directly.

Users repeat `np.asarray(series.times.to("s"))`, `np.asarray(series.values)`,
and `ax.step(..., where="post")` boilerplate, and easily get the step semantics
wrong (`where="pre"`, or dropping the final step of a window that extends past
its last observation).

## Goals

1. One call draws a scalar time series as the step function `evaluate`
   defines.
2. Multi-channel and vector-valued series are plottable without manual
   unpacking.
3. A 1-D `TimeSeriesTensor` (e.g. windows) can be overlaid, like `plot` does
   for a 1-D `PcfTensor`.
4. The time axis is unit-correct from the first iteration: it is labeled with
   its unit, every series drawn on it is converted exactly into that unit, and
   axes shared with `sharex` agree on it (see "Time axis").
5. Conventions match the existing module: `ax=None` means current axes,
   `**kwargs` forwarded to matplotlib, colors from the axes' property cycle.

## Non-goals (v1)

- Calendar steps. Window sizes, steps, and delays have a fixed length; months
  and years are rejected. See Q9.
- Non-numeric elements (point clouds, barcodes, PCFs, matrices). Raise
  `TypeError`; users evaluate and plot the elements with existing functions.
- Delay-embedding scatter plots. `ax.plot(*np.asarray(points).T, "o")` is
  already one line.
- Interactive/plotly backends.
- Query markers, domain shading, open/closed endpoint circles. These are
  docs-figure annotations, not general plotting; they stay in the docs script.
- Window bars. The docs figures keep their own bar code; windows themselves
  are drawn in place through their origin (see "Prerequisite: origin").

## Prerequisite: domain accessor

A window's domain extends past its last observation (to `size`), and an
open-start window cannot be evaluated at 0. The C++ series knows its domain
and whether each end is closed (`TimeSeries::domain()` in
`include/sbear/timeseries.hpp` returns the ends and two closed flags, and
windowing sets them from `closed`), but Python exposes none of it. Plotting
needs the domain end to draw the final step to the right place.

Add to `stablebear.timeseries`:

```python
class Domain(NamedTuple):
    start: float
    end: float
    start_closed: bool
    end_closed: bool
```

and to `TimeSeries`:

```python
@property
def domain(self) -> Domain:
    """The interval on which every channel can be evaluated, in `unit`."""
```

- The values mirror the C++ `TimeSeries::domain()`, converted to `unit`. For
  several channels, this is the intersection, the same interval `evaluate`
  checks. Per-channel domains come from `series[i].domain`.
- A `NamedTuple`, so `start, end, *_ = series.domain` and `series.domain.end`
  both work.
- A series built from observations has both ends closed. A window's ends
  follow its `closed` setting, and a partial window's cut end follows the
  series' end.
- `TimeSeriesTensor` gets no `domain`; use `windows[k].domain`.

This is a public API addition and needs its own tests and docs (the
`TimeSeries` docstring already describes domains in prose and should point
to the property).

## Prerequisite: origin

A window's times start at 0. Its place in the series it was cut from is
recorded nowhere on the window, only in the bounds from
`Window(..., return_bounds=True)`. To draw a window where it was, each
channel stores an origin: the time, in the original series, that its time 0
stands for.

### Why per channel

`windows()` in `include/sbear/timeseries/windowing.hpp` shifts each channel
separately, by `shift = max(channel.time(lo), from)`. Channels snap with
their own timestamps and tolerances, so one window's channels can be shifted
by slightly different amounts, and the bound `a_k` is not exactly any of
them. A single origin per series would misplace a channel by up to its snap
distance; a per-channel origin is exact.

### C++

- `Channel` gains `float64_t m_origin`, in the channel's unit, default 0,
  with `origin()` and `with_origin(float64_t)`.
- `windows()` sets each window channel's origin to
  `channel.origin() + shift`. A window of a window therefore points into the
  first series, not the intermediate window.
- Everything that keeps a channel's times keeps its origin: delay embeddings,
  `until`, `with_start_closed`, element-wise operations, channel selection,
  and `from_channels`.
- Converting a channel to another unit converts its origin by the same exact
  factor.
- `TimeSeries::origin()` is not added; origins are per channel, as `unit`
  and `snap_tolerance` are.

The original time of a window observation is `origin + t`. Windowing stores
`t - shift`, clamped with `nextafter` to keep times increasing, so
`origin + t` can differ from the original timestamp by a few ulps. That is
far below anything a plot shows, and evaluation does not use the origin at
all.

### Python

```python
@property
def origin(self) -> float | FloatTensor:
    """The time in the original series that time 0 stands for, in `unit`."""
```

- One channel: a float. Several channels: a float64 `FloatTensor` with one
  entry per channel, as `snap_tolerance` does.
- 0 for a series built from observations.
- Read-only. `TimeSeries(values, times)` takes no origin argument, and there
  is no setter; only windowing sets an origin.
- `array_equal` ignores the origin and otherwise stays as it is: same
  timestamps, values, unit, snap tolerance, and domain in every channel,
  everything that changes evaluation. The origin changes only where a series
  is drawn, so two windows with the same contents cut from different places
  are `array_equal`. Callers that care about placement compare `origin` as
  well. No separate comparison function is added. See Q6.
- `TimeSeriesTensor.array_equal` ignores origins the same way, element by
  element: two tensors of windows with the same contents cut from different
  places are `array_equal`.
- C++ `TimeSeries::operator==` and `Channel::operator==` compare the origin
  too, so C++ round-trip tests catch a lost origin. The Python binding of
  `array_equal` therefore stops using `operator==`
  (`src/python/py_timeseries_bind.hpp:67`) and calls a C++ comparison that
  skips the origin.
- The window bounds stay as they are: `a_k` as computed, before snapping, in
  the series' unit. The origin is the snapped shift, so the two can differ
  by a snap distance; the `Window` docstring says so.

### Save, load, and pickle

Time series are unreleased (`## Unreleased` in `CHANGELOG.md`), and they
first appear in file format version 3. The origin is added to the version 3
channel record directly, with no version bump. Files written by earlier
development builds will no longer load; that is acceptable before release.
Pickling goes through the same binary format and needs no separate change.

## Prerequisite: epoch on `Times`

`Times` gains an optional epoch: the calendar instant that time 0 stands for.
A timestamp `t` is then the date `epoch + t`, and a window observation is
`epoch + origin + t`. Without an epoch, times are plain durations, as today.

### Python

```python
class Times:
    def __init__(self, times, unit: str = "s", *, epoch=None): ...

    @property
    def epoch(self) -> np.datetime64 | None: ...
```

- `epoch`: anything `np.datetime64` accepts, such as `"2023-01-01T00:00"`, a
  `np.datetime64`, or a naive `datetime.datetime`. Its resolution must be one
  of the `Times` units, attoseconds to days; `W`, `M`, and `Y` are rejected
  because months and years have no fixed length. A timezone-aware `datetime`
  raises `ValueError`, because `np.datetime64` has no timezone and only warns
  while dropping it; convert to a naive UTC time first. `NaT` and values that
  are not dates raise `TypeError`/`ValueError`.
- `times` may also be a `datetime64` array. The epoch then defaults to its
  first element, and each timestamp is `(stamp - epoch)` in `unit`, computed
  in integer `timedelta64` arithmetic and divided by the unit once, so whole
  multiples of the unit are exact. Passing `epoch` explicitly uses that
  epoch instead. This replaces the tides notebook's
  `(stamps - stamps[0]).astype(float)` step.
- `Times.to(unit)` keeps the epoch.
- `Times.__eq__` compares the epoch, count and resolution: the same numbers
  from different epochs are different instants, and an epoch kept at day
  resolution (`"2023-01-01"`) differs from the same instant kept at
  microseconds (`datetime(2023, 1, 1)`). The `epoch` values themselves, as
  `np.datetime64`, compare as instants in NumPy.
- `repr` shows `epoch=...` when set.

`TimeSeries` exposes it as `series.epoch`, a `np.datetime64` or `None`. It
is a single value even for several channels: `from_channels` and
`with_epoch` keep every channel of a series on the same epoch, and nothing
else sets one, so the channels never disagree.

### `with_epoch`

```python
Times.with_epoch(epoch) -> Times
TimeSeries.with_epoch(epoch) -> TimeSeries
```

- Returns a copy with the epoch replaced; the timestamps, origins, and
  everything else are unchanged. It relabels time 0, it does not shift any
  times. `None` removes the epoch.
- Accepts the same values as the `epoch` constructor argument, with the same
  errors.
- `TimeSeries.with_epoch` sets every channel to the same epoch, so the result
  always satisfies the `from_channels` rule. It shares observations with the
  original and copies none, as `to` does.
- This is how a series built without an epoch, or a window or delay
  embedding of one, gets one afterwards:
  `windows[k].with_epoch("2023-01-01")`. Because a window keeps the epoch of
  the series it was cut from, calling `with_epoch` before windowing is
  usually simpler.
- `TimeSeriesTensor` gets no `with_epoch`; set it on the series before
  windowing, or per element.
- C++: `Times::with_epoch(std::optional<Epoch>)` and
  `TimeSeries::with_epoch(std::optional<Epoch>)`.

### Propagation

- `TimeSeries.to(unit)`, channel selection, element-wise operations, delay
  embeddings, and windows keep each channel's epoch. A window's date is
  `epoch + origin + t`, so its epoch is that of the series it was cut from,
  not the date of its start.
- `from_channels` requires all channels to have the same epoch, or none;
  otherwise `ValueError`. Aligning channels with different epochs would shift
  queries per channel, which is a separate feature (Q8).
- `array_equal` ignores the epoch, as it ignores the origin: neither changes
  evaluation, only placement. Callers that care compare `epoch` directly.

### C++

- `Times` gains `std::optional<Epoch> m_epoch`, where `Epoch` is an `int64_t`
  count and a `TimeUnit`, the exact value of a `datetime64` at its own
  resolution. No float is involved, so an epoch round-trips exactly. Two
  epochs are equal when they have the same count at the same resolution;
  the same instant at another resolution is a different epoch, which keeps
  the comparison exact without converting between resolutions.
- Epochs follow NumPy `datetime64` and C++ `std::chrono::sys_time`: the
  Gregorian calendar, and 86400 seconds in every day, so leap seconds are
  not counted. Across a leap second, `epoch + t` can differ from UTC by up
  to a second, as in NumPy, pandas, and Python's `datetime`.
- `Epoch` can be built from a `std::chrono::sys_time` at any resolution
  that is a `TimeUnit`, such as `Epoch(sys_days{2023y / January / 1})`.
- `Times::to` copies it; `Times::operator==` compares it.
- Windowing, which builds new `Times` for each window channel, copies the
  source channel's epoch.
- `TimeSeries::operator==` compares epochs; the comparison used by Python
  `array_equal` skips them, with the origin.
- Save/load: the version 3 `Times` record gains a presence flag, the count,
  and the unit. Time series are unreleased, so there is no version bump, as
  for the origin.

The `datetime64` array constructor is Python-only; C++ takes numbers and an
optional `Epoch`.

## Dates as time points

Data often come with dates and times as their time points. With an epoch on
`Times`, a series can take dates wherever it takes times, and give dates
back. Numbers keep working as they do now; dates are converted to numbers in
the series' unit at the boundary, in Python, and the C++ core keeps working
on numbers.

### Building

`Times(stamps, unit="min")` with a `datetime64` array (see "Prerequisite:
epoch on `Times`"). `TimeSeries(values, times=stamps)` with a `datetime64`
array, rather than a `Times`, builds `Times(stamps, unit="s")`: the default
unit, as for plain arrays. Pass a `Times` to choose the unit.

### Queries

`series(t)` and `evaluate(t)` also accept, for a series with an epoch:

- a `np.datetime64`, a naive `datetime.datetime`, or a `datetime.date`
  (including subclasses such as `pandas.Timestamp`);
- an array-like of `datetime64`.

Strings are not accepted, so that `series("2023-01-01")` cannot be confused
with a unit; wrap them in `np.datetime64`. A timezone-aware `datetime`
raises `ValueError`, as for epochs.

Conversion, per channel: the query date `q` is at time
`t = (q - epoch) - origin` in the channel's unit. `q - epoch` is computed in
integer `timedelta64` arithmetic, at the finer of the two resolutions, and
converted to the unit by one division; subtracting the origin is the only
other rounding. Snapping absorbs rounding as for any computed query. For a
window, the origin moves the query to the window's own times, so a date
evaluates a window at that date, as it evaluates the series it was cut from.

Errors:

- A date query on a series without an epoch: `ValueError` that suggests
  `with_epoch`.
- A date query with `unit`: `TypeError`; a date has no unit.
- A date outside the domain: `ValueError` naming the date and the domain as
  dates, not the converted numbers.
- Mixing dates and numbers in one array: `TypeError`.

Numeric queries are unchanged: still times in the series' own frame, from 0
for a window.

### Durations

`Window(size, step)` and `TimeDelayEmbedding(dimension, delay)` also accept
`np.timedelta64` and `datetime.timedelta` for `size`, `step`, and `delay`.
These do not need an epoch. They are kept as given and converted to the
series' unit when applied to a series, by the same integer-then-one-division
rule, since the unit is not known when the `Window` is built.
`Window(size=np.timedelta64(25, "h"), step=np.timedelta64(3, "h"))` then
means the same on a series in minutes as on one in hours. `repr` shows the
duration as given. Months and years (`M`, `Y`) are rejected: no fixed length.

### Dates back

```python
Times.to_datetime64(resolution: str = "us") -> np.ndarray
```

- Returns `epoch + t` for each timestamp as a `datetime64[resolution]` array,
  rounded to the nearest step. `resolution` is any `Times` unit. The default
  `"us"` matches the plotting conversion and covers about 290,000 years
  either side of the epoch; `"ns"` covers 1678 to 2262.
- Without an epoch: `ValueError` that suggests `with_epoch`.
- Window dates come from `bounds` (below).

### `bounds`

```python
def bounds(self, *, dates: bool = False,
           resolution: str = "us") -> Domain | tuple[Domain, ...]:
```

Each channel's domain placed in the series it was cut from: the `Domain`
from "Prerequisite: domain accessor", with `origin` added to `start` and
`end`, and the closed flags unchanged.

- One channel: a `Domain`. Several channels: a tuple with one `Domain` per
  channel, in channel order. Channels are not intersected here, because each
  channel's origin can differ after snapping; `domain` gives the shared
  interval in the series' own frame.
- `start` and `end` are in the series' unit. With `dates=True` they are
  `datetime64[resolution]` values, `epoch + origin + start` and likewise for
  `end`, rounded as in `to_datetime64`; without an epoch, `ValueError` that
  suggests `with_epoch`.
- For a series built from observations, the origin is 0, so `bounds()`
  equals `domain` for each channel.
- For windows, `[w.bounds(dates=True) for w in windows]` gives each window's
  dates. These are the snapped bounds each channel actually holds; the
  bounds from `Window(..., return_bounds=True)` remain the unsnapped `a_k`
  and `a_k + size` shared by all channels.
- `TimeSeriesTensor` gets no `bounds`; loop over the elements.

## Time axis

Each axes has one time unit for its x axis. The plotting functions below
follow these rules:

1. **Resolving the unit.** If the x axis already has a time unit, that unit is
   used. Otherwise the unit is the `unit` argument if given, else the unit of
   the series being drawn, and it is stored on the axis.
2. **Conversion.** Times are converted into the axis unit with `Times.to`, which
   converts by an exact integer factor. A series in `ms` drawn on an axis in
   `s` lands at the same physical times as a series in `s`.
3. **Conflicts.** If `unit` is given and differs from the axis unit, raise
   `ValueError` naming both units. If the x axis has a non-time converter, such
   as a date axis, raise `TypeError`.
4. **Label.** Set the x label to `time (<unit>)`, using the unit's symbol as
   `Times` spells it, only if the label is empty. A label set by the user is
   kept.
5. **Shared axes.** Axes created with `sharex` share the unit: a series in
   `ms` drawn under one in `s` on a shared axis is converted to `s`.
6. **Ticks.** Keep matplotlib's default numeric locator and formatter. With
   the unit in the label, tick values are plain numbers in that unit.

Mechanism: store the unit string with `ax.xaxis.set_units(unit)` and read it
with `ax.xaxis.get_units()`. Verified on matplotlib 3.10.9: with no converter
set, `set_units` stores the string, propagates it to `sharex` axes, and does
not change how numeric data are drawn. A date axis is detected by
`ax.xaxis.get_converter()` not being `None`.

### Date axes

A series whose times have an epoch is drawn on a date axis, at
`epoch + origin + t`, instead of on a numeric axis in a unit. This is what
the tides notebook does by hand with its `to_dates` helper. There is no
`epoch` argument; the epoch comes from the series.

- `relative=True` draws `t` on a numeric axis as usual, ignoring epoch and
  origin, so windows can still be overlaid.
- Conversion: convert `origin + t` to microseconds, round to the nearest
  integer, and add them to the epoch, cast to `datetime64[us]`. Microseconds
  cover about 290,000 years either side of the epoch and match the
  resolution of matplotlib dates, which are stored as float days, so finer
  units lose nothing that matplotlib could have drawn.
- The axis becomes a date axis through matplotlib's own `datetime64`
  converter, so locators, formatters, and converters registered by the user,
  such as `ConciseDateConverter`, apply, and `sharex` axes share it.
- A date axis has no time unit. `plot_timeseries` raises `ValueError` when
  given `unit` for a series with an epoch (unless `relative=True`). A series
  without an epoch on a date axis raises `TypeError`, and a series with an
  epoch on an axis that has a time unit raises `ValueError`.
- No x label is set: the epoch's timezone is not known.
- Series with different epochs can share a date axis; each is placed at its
  own dates.

### Unit-carrying times (not used)

Not used: a matplotlib unit converter that lets `ax.plot(series.times, ...)`
carry units by itself. matplotlib skips converters for numeric arrays
(`munits._is_natively_supported`), so `Times` would need boxed, non-numeric
elements. That is a larger design and slower for long series.

Known limitation: data drawn with plain `ax.plot` on the same axes is taken to
be in the axis unit already, and calling `ax.xaxis.set_units` after drawing
does not rescale lines already drawn. Choose the unit with `unit=` on the
first call.

## API

All in `stablebear/plotting.py`.

### `plot_timeseries`

```python
def plot_timeseries(series, ax=None, *, unit=None, relative=False,
                    markers=False, **kwargs) -> list[Line2D]:
```

- `series`: a `TimeSeries` or a 1-D `TimeSeriesTensor`. There is no channel
  argument: to draw one channel of a series with several, pass `series[i]`.
- `ax`: for a series with one channel, an `Axes`; `None` uses the current
  axes. For several channels, see "Several channels".
- `unit`: time unit for the x axis; resolved as in "Time axis".
- `relative`: if `False` (default), draw each channel at `origin + t`, so a
  window lands where it was in the original series and lines up with that
  series on the same axes. If `True`, draw at `t`, so windows are overlaid
  from 0 for comparing their shapes. For a series built from observations
  the origin is 0 and both draw the same.
- `markers`: if `True`, mark each observation with a dot at
  `(t_i, x_i)`.
- `**kwargs`: forwarded to `Axes.plot` (`color`, `linewidth`, `label`, ...).

Drawing:

- Each scalar channel is drawn with `drawstyle="steps-post"` from the first
  observation to the channel's domain end. If the domain end is after the last
  observation (windows), the last value is extended to it, as `plot` does with
  `max_time`.
- Vector observations of shape `(d,)`: one line per component, labeled
  `x[0]`, `x[1]`, ... when `label` is not given. Higher-rank observations
  raise `ValueError`.
- Non-numeric dtypes raise `TypeError` naming the dtype.
- The x axis follows "Time axis".
- Returns the created `Line2D` objects as one flat list, in channel order,
  then component order. Each line's `axes` tells which row it is on.

`TimeSeriesTensor` input: one line per element, all on the same axes (or rows,
for several channels), each element taking the next color from the cycle
unless `color` is given. With the default `relative=False`, windows are drawn
side by side where they were in the original series, overlapping where the
windows overlap; with `relative=True`, they are overlaid from 0. Tensors with
more than one dimension are squeezed, then rejected, as in `plot`.

#### Several channels

A series with several channels is drawn with one row per channel, in channel
order, on axes that share the x axis.

- `ax=None`: create a new figure with
  `plt.subplots(n_channels, 1, sharex=True, squeeze=False)`. Rows are not
  taken from the current axes, because one axes cannot hold several rows.
- `ax` given: a sequence of exactly `n_channels` axes, used in order. A single
  `Axes`, or a sequence of another length, raises `ValueError` that suggests
  `series[i]` for drawing one channel on one axes. The axes are used as given;
  they are not made to share x.
- Every row uses the same time unit, resolved once from the first row by the
  rules in "Time axis", with `series.unit` (the series' unit, not the
  channel's) as the default. Each channel is converted into it from its own
  unit. For a series with an epoch, every row is a date axis.
- The x label is set only on the last row, by the rules in "Time axis". No y
  labels are set; channels have no names.
- `kwargs` such as `color` and `label` apply to every row.
- Rows may hold observations of different shapes: a scalar channel next to a
  vector channel gets one line in its row and `d` lines in the other.

## Docs

- `docs/plotting.rst`: new "Time series" section with a scalar series, a
  two-channel series in different units (one call, one row per channel), and
  windows drawn over their series and overlaid with `relative=True`.
- `docs/timeseries_windows.rst`: mention `origin`, and how it differs from
  the bounds. Each figure gets light/dark images
  and a `.. dropdown:: Show code` with `literalinclude` from
  `docs/_static/gen_plotting_gallery.py`.
- `docs/timeseries.rst`: one-line pointer to the new section.
- `docs/api_plotting.rst`: update the module summary line.
- Tides notebook: build `Times` from the `datetime64` stamps (epoch set
  automatically), draw the water levels with `plot_timeseries`, and replace
  the `to_dates` helper with `bounds(dates=True)` for window dates. Keep the cells
  that plot values computed per window (norms, tidal range), which are not
  time series, and the distance-matrix image.
- `docs/timeseries_units.rst` or `docs/timeseries.rst`: a section on epochs
  and building `Times` from dates.
- Changelog entry.

Docs figure scripts keep their own annotation code (filled/hollow endpoints,
query squares); they may call `plot_timeseries` for the underlying line.

## Tests (`test/python/test_plotting_timeseries.py`)

Follow `test/TESTING.md`. Use the Agg backend and inspect returned artists'
data, not images.

1. Scalar series: line x/y data equal the step vertices from the observations;
   final point at the domain end.
2. Window (domain end past last observation): last value extended to `size`.
3. Time axis:
   - `unit="ms"` on a series in seconds: x data scaled by exactly 1000; axis
     unit `ms`; label `time (ms)`.
   - A series in `ms` drawn after one in `s` on the same axes: converted to
     `s`, label unchanged.
   - Same with two `sharex` axes, each drawn on separately.
   - `unit="ms"` on an axis already in `s`: `ValueError`.
   - Axes with a date converter: `TypeError`.
   - A user-set x label is kept.
4. Date axes:
   - A series in `h` with epoch `2023-01-01T00:00`: x data equal the
     `datetime64` values `epoch + t` (compare with `mdates.date2num`).
   - Windows of it: each at `epoch + origin + t`; with `relative=True`, on a
     numeric axis from 0.
   - A series in `ns` whose times are not whole microseconds: rounded to the
     nearest microsecond.
   - `unit` with an epoch series: `ValueError`; with `relative=True`: allowed.
   - A series without epoch on a date axis: `TypeError`. A series with epoch
     on an axis with a time unit: `ValueError`.
   - Two `sharex` axes with the same epoch line up; two series with
     different epochs on one date axis land at their own dates.
5. Vector channel `(d=3,)`: three lines, default labels `x[0]`..`x[2]`; an
   explicit `label` is used as-is.
6. Several channels:
   - Three channels, `ax=None`: a new figure with 3 rows sharing x; each row
     holds its channel's line; the return value lists them in channel order.
   - Channels in `s` and `ms` on a series in `s`: both rows in `s`, with the
     `ms` channel's times divided by exactly 1000.
   - x label on the last row only.
   - `ax` as a sequence of 3 axes: used in order; a single `Axes` or 2 axes:
     `ValueError`.
   - A scalar channel next to a vector channel `(d=2,)`: 1 and 2 lines.
   - `series[1]` on a single axes draws channel 1 only.
   - A series with an epoch and several channels: every row is a date axis.
7. `TimeSeriesTensor` of 3 windows: 3 lines; colors advance the axes' cycle.
   By default each line runs from the window's origin to origin plus its
   domain end, and its points match the original series' points there; with
   `relative=True` each runs from 0.
8. Non-numeric dtype (point cloud series): `TypeError`.
9. Empty `TimeSeriesTensor`: no artists, no error.
10. `domain` property:
    - A series built from observations: first and last timestamps, both
      closed.
    - Several channels: the intersection, with the closed flags of the ends
      that bound it.
    - Windows for each `closed` setting (`"left"`, `"right"`, `"both"`,
      `"none"`): ends at 0 and `size`, with the matching flags.
    - A partial window: cut end at the series' end, with the series' flag.
    - Unit conversion via `to`.
    - Field access and tuple unpacking.
11. `origin` (C++ and Python):
    - 0 for a series built from observations.
    - Windows: origin equals the start bound when no snapping happens; with
      a start that snaps onto an observation, the observation's time.
    - Two channels that snap differently: different origins, each placing
      its channel's observations at their original times.
    - A window of a window: origin into the first series.
    - Kept by delay embeddings, element-wise operations, channel selection,
      and `from_channels`; converted exactly by `to`.
    - `origin + t` within a few ulps of the original timestamps.
    - Save/load and pickle round trips keep `origin` (compared directly).
    - Two windows with the same contents from different places:
      `array_equal` is `True`, origins differ; C++ `operator==` is `false`.
12. `Times` epoch (`test/python/test_timeseries.py` and C++):
    - `epoch` as `str`, `np.datetime64`, and naive `datetime` give the same
      value; a timezone-aware `datetime`: `ValueError`; `W`/`M`/`Y`
      resolution, `NaT`, and non-dates rejected.
    - Default `None`; `to` keeps it; `__eq__` compares it; `repr` shows it.
    - From a `datetime64[m]` array with `unit="min"`: epoch is the first
      stamp, timestamps are exact whole minutes; with `unit="h"`, divided
      once; with an explicit epoch, offsets from it.
    - Kept by windows, delay embeddings, channel selection, element-wise
      operations, and `TimeSeries.to`; `series.epoch` is one value for a
      series with several channels.
    - `from_channels` with equal epochs: kept; different, or some `None`:
      `ValueError`.
    - `array_equal` ignores it; C++ `operator==` does not.
    - Save/load and pickle round trips keep it exactly, including
      attosecond and day resolutions.
    - `with_epoch` on `Times` and `TimeSeries`: sets, replaces, and removes
      (`None`) the epoch; timestamps, values, and origin unchanged; every
      channel of a multi-channel series gets it; invalid epochs raise as in
      the constructor; on a window, the plot moves to the new dates.
13. Dates as time points (`test/python/test_timeseries.py`):
    - `TimeSeries(values, times=stamps)` with `datetime64` stamps: unit `s`,
      epoch the first stamp.
    - A date query equals the numeric query at `q - epoch`, for scalar
      `datetime64`, `datetime`, `date`, and a `datetime64` array; strings
      rejected.
    - On a window: a date query equals the parent series at that date.
    - A date one ulp-scale rounding below a timestamp snaps to it.
    - Errors: no epoch, `unit` with a date, outside the domain (message
      shows dates), mixed dates and numbers, timezone-aware `datetime`.
    - `Window` and `TimeDelayEmbedding` with `timedelta64` and `timedelta`:
      same result as the numeric equivalent, on series in `min` and in `h`;
      `M`/`Y` rejected; `repr` shows the duration.
    - `to_datetime64`: default `us`, other resolutions, rounding to nearest,
      no epoch raises.
14. `bounds`:
    - A series built from observations: equal to `domain`, per channel.
    - A window: `start` and `end` are `origin` plus its domain ends; closed
      flags match `domain`; with no snapping, equal to the `return_bounds`
      row.
    - Two channels that snap differently: different bounds per channel.
    - `dates=True` with an epoch: `epoch + origin + start/end`, at the
      default and another resolution; without an epoch: `ValueError`.
    - Tides-style use: window start dates from `bounds(dates=True)` match the
      dates computed from `return_bounds` and the epoch.

## Open questions

- Q1. Resolved: no channel argument; `series[i]` selects a channel, and
  several channels are drawn as rows with a shared x axis.
- Q2. Resolved: the function is `plot_timeseries`, separate from `plot`.
- Q3. Resolved: each channel stores an origin (see "Prerequisite: origin").
- Q4. Resolved: `Times` stores an optional epoch (see "Prerequisite: epoch
  on `Times`"); the plotting `epoch` argument is dropped.
- Q6. Resolved: `array_equal` keeps comparing snap tolerance, domain, and
  unit as it does today; only the origin is left out. The unit is compared
  as stored, without converting.
- Q7. Resolved: queries, window sizes and steps, and delays accept dates and
  durations (see "Dates as time points").
- Q8. Resolved: `from_channels` raises `ValueError` for channels with
  different epochs, or with some epochs `None`. Aligning them is left for
  later.
- Q9. Resolved: only fixed-length steps. `Window` sizes and steps, and
  delays, are numbers or fixed-length durations (`as` to `D`, and `W`);
  months and years raise `ValueError`.
- Q10. Resolved: `TimeSeries.bounds` gives each channel's bounds in the
  original series, as numbers or dates (see "`bounds`").
  `Window(..., return_bounds=True)` is unchanged.
