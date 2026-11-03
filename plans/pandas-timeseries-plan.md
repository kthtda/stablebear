# pandas input and durations for time series: implementation plan

Progress: 52 of 58 items, 106 of 117 points (91%)

Specs: [pandas-timeseries-spec.md](pandas-timeseries-spec.md) (phases A, B,
and H to L) and [timeseries-durations-spec.md](timeseries-durations-spec.md)
(phases C to G)

## Phase A: pandas values in the existing time series API

- [x] **A1.** [1] (033f5e862) Add pandas to the `test` group in `pyproject.toml` and to
  `docs/requirements.txt`, but not to the runtime dependencies.
  Tests: in a subprocess with `sys.modules["pandas"] = None`, stablebear
  imports and builds, windows, and embeds a `TimeSeries`.
- [x] **A2.** [1] (352d6ce08) Keep the resolution of `pd.Timestamp` in `DateTime` and of
  `pd.Timedelta` in `_duration` (`stablebear/timeseries/date_time.py`), via
  their `to_datetime64` and `to_timedelta64` methods, found by attribute so
  pandas is not imported.
  Tests: `DateTime` of a nanosecond `Timestamp` has unit `"ns"`;
  `Window(size=pd.Timedelta("1ns"))` on a series in `"ns"`;
  `TimeDelayEmbedding(delay=pd.Timedelta("3h"))` equals one with
  `np.timedelta64(3, "h")`.
- [x] **A3.** Timezone-aware dates become naive UTC.
  - [x] **A3a.** [1] (79218327d) `DateTime` converts an aware `datetime` or `Timestamp`
    to UTC instead of raising; update its docstring.
    Tests: `datetime` at +02:00 equals the UTC value two hours earlier;
    aware `Timestamp` likewise.
  - [x] **A3b.** [2] (93943a310) `Times` accepts a `DatetimeIndex` or a `Series` of
    dates, naive or aware, so `TimeSeries(s, s.index)` works for both;
    update the `Times` and `TimeSeries` docstrings.
    Tests: aware index equals the naive UTC index; `Series` of dates equals
    its index; `NaT` raises.
- [x] **A4.** [1] (725c6b968) `TimeSeries.evaluate` takes a `DatetimeIndex` or a `Series`
  of dates as dates, not numbers.
  Tests: equals evaluating `index.to_numpy()`; aware index equals its UTC
  equivalent; aware `Timestamp`.

## Phase B: `TimeSeries.from_pandas`

- [x] **B1.** [2] (8596a71fe) `TimeSeries.from_pandas(obj, unit="s", *,
  snap_tolerance=1e-14)` for a `Series` with a `DatetimeIndex`,
  `TimedeltaIndex`, or numeric index, with its docstring.
  Tests: datetime index equals `TimeSeries(values, Times(index))`, epoch
  the first stamp; timedelta index in `unit="ms"`;
  `RangeIndex` has no epoch; `NaN` kept; int and bool dtypes kept.
## Phase C: `TimeDelta`

- [x] **C1.** [2] (f620e0328, 1121a3b92) `TimeDelta(value, unit=None)` in
  `stablebear/timeseries/date_time.py`, exported from
  `stablebear.timeseries`: `count`, `unit`, `to_timedelta64`, `==`, `hash`,
  `repr`, pickling; with its docstring, which `api_timeseries.rst` picks
  up from the module. A count is an integer or a float.
  Tests: from `np.timedelta64`, `datetime.timedelta`, and `pd.Timedelta` at
  their resolutions; `TimeDelta(3, "h")` and `TimeDelta(1.5, "h")`;
  `to_timedelta64` at its default and a given resolution; `NaT`, months,
  and `NaN` raise; pickling round trip.
- [x] **C2.** [2] (5f7112a5b) Arithmetic on `TimeDelta` and `DateTime`, in the finer
  unit by exact factors.
  Tests: `TimeDelta(1, "h") + TimeDelta(30, "min")` is 90 minutes;
  negation, `* 3`, and `* 0.5`; `DateTime + TimeDelta(1.5, "h")`; `DateTime - DateTime` and `DateTime + TimeDelta`;
  comparisons across resolutions.
- [x] **C3.** [1] (3d7136917) `_duration` keeps a `TimeDelta` and `_in_unit` converts its
  count to the series' unit by an exact factor, so `Window` and
  `TimeDelayEmbedding` take it without rounding.
  Tests: `Window(TimeDelta(2, "h"))` equals `Window(np.timedelta64(2, "h"))`
  on a series in hours; a first window of `TimeDelta(1 / 3, "s")` on a
  series in nanoseconds holds an observation at 333333100 ns.
  Note: `_duration` now turns every duration into a `TimeDelta`, so
  `Window.size`, `Window.step`, and `TimeDelayEmbedding.delay` already are
  one for duration input, and their reprs show it; E6 is left with numbers.
- [x] **C4.** Arrays of durations and dates in one unit, as `TimeDeltaTensor`
  and `DateTimeTensor`.
  - [x] **C4a.** [3] (5d5d0c882) `TimeDelta(counts, unit)` for a `FloatTensor`,
    `IntTensor`, `ndarray`, or nested list, kept as a `FloatTensor` or
    `IntTensor`, and a `timedelta64` array at its resolution: `count`,
    `shape`, `ndim`, `len`, iteration, indexing, `to_timedelta64`,
    `array_equal`, `repr`, and pickling; no hash; update the docstring.
    Tests: integer counts stay an `IntTensor`; a `timedelta64[m]` array is
    in minutes; indexing one element gives a scalar `TimeDelta`;
    `to_timedelta64` of float counts rounds to microseconds; hashing
    raises; pickling round trip.
  - [x] **C4b.** [2] (6ee6aeb71) Elementwise arithmetic and comparisons on array
    `TimeDelta`, in the finer unit by exact factors; `==` and `!=` give a
    `BoolTensor`; `DateTime` arithmetic rejects array counts.
    Tests: array plus scalar in another unit; `* 0.5`; `<` against a
    scalar; `==` of the same lengths in different units is all false;
    `DateTime + array` raises `TypeError`.
  - [x] **C4d.** [3] (56199584d) `DateTime` holds an array of instants: from a
    `datetime64` array, a `DatetimeIndex` or `Series` of dates, or a
    sequence of values a scalar accepts, kept as an `IntTensor` at their
    resolution; `shape`, `ndim`, `len`, iteration, indexing,
    `to_datetime64`, comparisons, `==` and `!=` as `BoolTensor`,
    `array_equal`, `repr`, and pickling; no hash; an array epoch raises.
    Tests: a `datetime64[m]` array is in minutes; strings at their finest
    resolution; an aware `DatetimeIndex` is its UTC instants; indexing one
    element gives a scalar `DateTime`; `<` against a scalar; hashing
    raises; pickling round trip; `Times(..., epoch=array)` raises
    `TypeError`.
  - [x] **C4e.** [2] (9a4e16cd1) `DateTime` and `TimeDelta` arithmetic on arrays,
    replacing the `TypeError` of C4b.
    Tests: `DateTime + TimeDelta([30, 90], "min")`; array `DateTime` minus
    a scalar `DateTime` is an array `TimeDelta`; array minus a scalar
    `TimeDelta` with a float count.
  - [x] **C4f.** [3] (1cef3c791) Move the arrays of `TimeDelta` into a new
    `TimeDeltaTensor`, so that a `TimeDelta` holds one duration. Update `evaluate`,
    `Times`, the `DateTime` arithmetic, and the docstrings to use it.
    Tests: the C4a, C4b, C4e, and D3 tests on `TimeDeltaTensor`; a
    `TimeDeltaTensor` from a `TimeDelta` list equals one from its
    `timedelta64` array.
    Note: a `TimeDelta` still has the `shape`, `ndim`, and `len` of
    `_Counts`, which `DateTime` arrays need; C4g can move them to the
    tensors.
  - [x] **C4g.** [3] (9991218e7) Move the arrays of `DateTime` into a new
    `DateTimeTensor`, so that a `DateTime` holds one instant. Update the
    arithmetic, `docs/timeseries_dates.rst`, and the docstrings.
    Tests: the C4d and C4e tests on `DateTimeTensor`; a `DateTimeTensor`
    epoch raises.
  Note: C4f and C4g come after E3 in time; they split the array forms out
  of the scalars before E4 gives a `TimeDeltaTensor` as `origin`.

## Phase D: durations as input

- [x] **D1.** [2] (2e0ef181b) `Times` accepts durations: a `timedelta64` array, a
  sequence of `TimeDelta`, or a `TimedeltaIndex`, converted to `unit`
  exactly; `TimeSeries` takes them as `times`. Durations and dates are by
  default in their own resolution, not seconds.
  Tests: `timedelta64[m]` is in minutes; a `TimeDelta` list is in its
  finest unit; `timedelta64[ms]` in `unit="s"` gives exact seconds;
  `datetime64[m]` is in minutes; `TimedeltaIndex` equals its array.
- [x] **D2.** [2] (432122132) `TimeSeries.evaluate` accepts durations, single or arrays,
  as times from time 0, with or without an epoch.
  Tests: `TimeDelta(90, "min")` on a series in hours equals 1.5; a
  `timedelta64` array; a window evaluated at a duration from its start.
- [x] **D3.** [3] (6c9281336) `TimeSeriesTensor.evaluate` accepts durations, and
  `evaluate` and `Times` take an array `TimeDelta` as an array of
  durations.
  Tests: windows at durations equal windows at their times in seconds;
  `TimeDelta(sb.FloatTensor([[30, 90], [240, 60]]), "min")` on a series in
  hours equals evaluating `[[0.5, 1.5], [4.0, 1.0]]`;
  `Times(TimeDelta(sb.IntTensor([0, 90]), "min"))` equals
  `Times(np.array([0, 90], dtype="timedelta64[m]"))`.
  Note: absorbed D4. C++ converts each series to the durations' own unit,
  since the series of a tensor can differ in unit.
- [x] **D5.** [1] (9091b61df) `Times` and `TimeSeries` take a `DateTimeTensor` as
  dates, as its `datetime64` array.
  Tests: `Times(dates, unit="h")` and `Times(dates)` equal the `Times` of
  `dates.to_datetime64()`; a series with a `DateTimeTensor` as times has
  its first date as epoch.
  Note: added during F4, whose examples build series from dates.
- [x] **D6.** An epoch without `Times`.
  - [x] **D6a.** [1] (4fd939cd4) `TimeSeries(values, times, *, epoch=None)` counts
    dates from `epoch` and gives durations their time 0 at it.
    Tests: dates with an epoch keep their instants and count from it,
    exact in whole units; durations with an epoch equal the series
    `with_epoch` it; a `DateTimeTensor` epoch raises `TypeError`.
  - [x] **D6b.** [3] (e2eca5eea) `TimeSeries.from_channels` counts dated channels of
    different epochs from the earliest, shifting each channel's times and
    domain end in C++.
    Tests: channels whose first dates differ combine, and evaluate at a
    date to the values of each channel there; a window channel keeps
    where it lies; channels with and without an epoch raise; the epochs
    at different resolutions combine.
- [x] **D7.** [1] (62cc4ece0) `TimeSeries.evaluate` takes a `DateTime` and a
  `DateTimeTensor` as dates, as it takes a `datetime64` array.
  Tests: a `DateTime` and a `DateTimeTensor` give the values at their
  `datetime64` dates; a `DateTimeTensor` keeps its shape.
- [x] **D8.** [2] (a70059601) A tuple `(count, unit)` is a duration wherever one is
  taken: `TimeDelta(count, unit)`, or `TimeDeltaTensor(count, unit)` for
  a sequence or array of counts. This covers `Window`,
  `TimeDelayEmbedding`, `evaluate` on series and tensors, and `times` of
  `TimeSeries`.
  Tests: each of these with a tuple equals it with the `TimeDelta` or
  `TimeDeltaTensor`; a tuple of two dates is still two dates; a tuple
  with an unknown unit raises.
- [x] **D9.** [1] (c921bd4ed) `TimeDelta` and `TimeDeltaTensor` take the unit aliases
  that series take, such as `"hours"`, through one C++ lookup, and keep
  the short name.
  Tests: `TimeDelta` and `TimeDeltaTensor` in `"hours"`, `"minutes"`, and
  `"\u00b5s"` are in `"h"`, `"min"`, and `"us"`; `(1.5, "hours")` equals
  `(1.5, "h")`;
  `"m"` and `"fortnight"` raise.

## Phase E: durations and dates as output

- [x] **E1.** [2] (2f30d0933) `np.asarray(times)` gives `datetime64` or `timedelta64` at
  the finer of the unit and microseconds; add `Times.to_floats(unit=None)`
  and use it in `plot_timeseries` and other internal callers.
  Tests: dated and undated times; nanosecond unit kept; `to_floats` in
  another unit.
- [x] **E2.** `TimesTensor`, and times of several channels and of tensors
  of series in it.
  - [x] **E2a.** [3] (70b224f9b) C++: an empty default `sb::Times`, and
    `Tensor<sb::Times>` registered in both modules. Python: `TimesTensor`
    with dtype `sb.times` in `stablebear/typing.py`, indexing to `Times`,
    assignment, `copy`, `array_equal`; with its docstring.
    Tests: a tensor of two `Times` in different units indexes back to
    each; assigning a `Times`; `copy` is independent; `array_equal`
    compares units and epochs.
  - [x] **E2b.** [2] (870e78f63) Save and load a `TimesTensor`.
    Tests: round trip with and without epochs, in two units.
  - [x] **E2c.** [1] (6a2d71e44) `TimeSeries.times` of several channels is a
    `TimesTensor` of shape `(n_channels,)`.
    Tests: two channels with different times and units.
  - [x] **E2d.** [2] (a1ee2bca8) `TimeSeriesTensor.times` is a `TimesTensor` of its
    shape, with a last axis of channels for series of several channels.
    Tests: the times of windows equal each window's `times`; windows of
    two channels have shape `(n_windows, 2)`.
- [x] **E3.** [2] (924050b7f) `TimeSeries.domain` has `DateTime` or `TimeDelta` ends;
  a dated window's domain is where it lies.
  Tests: undated series; dated series; dated window; closed ends kept.
- [x] **E4.** [2] (7ec48f8f9) `TimeSeries.origin` is a `TimeDelta` or a `TimeDeltaTensor`;
  remove `TimeSeries.bounds` and update its callers and docs.
  Tests: window origins, including after snapping; several channels.
- [x] **E5.** [2] (f04d7b62f) `Window(..., return_bounds=True)` gives a
  `TimeDeltaTensor` of shape `(n_windows, 2)` in the series' unit, or a
  `DateTimeTensor` for a series with an epoch.
  Tests: dated and undated series; partial windows.
- [x] **E6.** [1] (19485daf8) `Window.size`, `Window.step`, and
  `TimeDelayEmbedding.delay` are `TimeDelta`; update the reprs.
  Tests: from `np.timedelta64` and `pd.Timedelta`; repr.
  Note: C3 already made durations `TimeDelta` and showed them in the
  reprs, so this item adds the tests. Numbers stay floats until G3, since
  a number has no unit before the window meets a series.
- [x] **E7.** [3] (dd05ac8e1) The channels of a window share its origin, the computed
  start `a_k`, in `include/sbear/timeseries/windowing.hpp`; a snapped
  observation stays at time 0. Update the window docstring, the origin
  part of `docs/timeseries_windows.rst`, and `docs/timeseries_snapping.rst`.
  Tests: a start that snaps keeps `3 * 0.3` as origin and holds the
  observation at 0.9 at time 0; channels that snap differently share an
  origin; dated bounds of windows of such a window are their dates.
- [x] **E8.** [3] (563faadce) `TimeSeries.times` is a `TimeDeltaTensor` from time 0 in
  the series' unit, with or without an epoch. For several channels it
  raises `ValueError` naming `series[i].times`, and
  `TimeSeriesTensor.times` is removed. `TimeSeries.datetimes` gives
  `epoch + origin + times` as a `DateTimeTensor`. Update
  `plot_timeseries` and other internal callers.
  Tests: undated and dated series; `datetimes` of a dated series and a
  window; a window's times start at 0; several channels and a series
  without an epoch raise.
  Note: replaces E2c and E2d.
- [x] **E9.** [3] (aea5e201a) Tests, docstrings, and internal code build series from
  `TimeDeltaTensor` and `DateTimeTensor` and set epochs with `epoch=`,
  instead of `Times`.
  Tests: the `Times` tests move to `TimeSeries` where they test its times,
  and are removed where they test only `Times`.
- [x] **E10.** Remove `Times`.
  - [x] **E10a.** [2] (e153dbb9d) The guide pages, the figure scripts in
    `docs/_static/`, and the tide gauge tutorial build series from
    `TimeDeltaTensor` and dates, and point to `TimeSeries.datetimes`
    instead of `Times.to_datetime64`. F4 still rewrites the pages.
  - [x] **E10b.** [2] (0aeb67eb9) Remove `Times`, `TimesTensor`, and `sb.times` from the
    Python API, and the registration of `Tensor<sb::Times>` in C++, with
    the saving and loading of a `TimesTensor`.
    Tests: the E2 tests are removed.
    Note: undoes E1's `Times` conversions and E2.

## Phase F: tests and docs use durations

Tests and docs write durations as `TimeDelta` and dates as `DateTime`, or
as `TimeDeltaTensor` and `DateTimeTensor` for several. `np.timedelta64`,
`datetime.timedelta`, `pd.Timedelta`, and `datetime64` stay only in tests
that check they are accepted, and in one place per guide page that says
so.

- [x] **F1.** [3] (d697d7a7b) `test/python/test_windowing.py` builds series from
  `TimeDelta` durations and passes `TimeDelta` to `Window` and
  `evaluate`.
- [x] **F2.** [3] (bb8279d1c) `test/python/test_timeseries.py` likewise.
- [x] **F3.** [2] (e5342aa45) `test_time_delay_embedding.py`, `test_plotting_timeseries.py`,
  `test_timeseries_pandas.py`, `persistence/test_timeseries_persistence.py`,
  `test_io_roundtrip.py`, and `test_integration.py` likewise.
- [x] **F4.** [3] (2e5f0d1ca) The time series guide pages (`docs/timeseries*.rst`) use
  `TimeDelta` and `DateTime`; `timeseries_units.rst` explains the unit as
  storage.
  Note: depends on G, which settles series without a unit; do it after
  G4.
- [x] **F5.** [1] (cd6a198e7) The tide gauge tutorial uses `TimeDelta` for its window
  sizes and delays, and a `TimeDeltaTensor` for its evaluation grid.

## Phase G: numbers mean a series without a unit

- [x] **G1.** [3] (1547f1f8a) Series without a unit: C++ gains a `TimeUnit` that
  converts only to itself, saved and loaded as any unit. `TimeSeries` of
  numbers and `from_pandas` of a numeric index without `unit` give one,
  with `unit` `None`; `epoch=` and `with_epoch` on it raise
  `ValueError`, `to(unit)` reads its numbers as counts of `unit`, and
  `from_channels` of channels with and without a unit raises.
  Tests: `TimeSeries(values, [0, 1, 2]).unit is None`; a numeric index
  without `unit`; round trip of a series without a unit; `to("s")` on a
  series without a unit; the raises.
  Note: replaces rejecting numbers as times, which raised `TypeError`.
- [x] **G2.** [1] (019db8043) `evaluate` on series and tensors: a series without a
  unit takes numbers, and raises `TypeError` for durations, dates, and
  `unit=`; a series with a unit takes `series(t, unit)` as
  `TimeDelta(t, unit)` or `TimeDeltaTensor(t, unit)`, and raises for
  numbers without `unit`, naming `series(t, unit)`.
  Tests: both kinds of series, single and many times; tensor likewise.
  Note: keeps the `unit` argument, which this item first removed.
- [x] **G3.** [1] (364d763ae) `Window` and `TimeDelayEmbedding` take numbers for a
  series without a unit and durations for one with a unit; the other kind
  raises `TypeError` when applied.
  Tests: `Window(size=3.0)` on both kinds; `TimeDelayEmbedding(2, 1.0)`
  likewise; durations on a series without a unit raise.
- [x] **G4.** [2] (1547f1f8a) A series without a unit gives numbers: `times` is a
  `FloatTensor`, `domain` has float ends, `origin` is a float or a
  `FloatTensor`, and `return_bounds` gives a `FloatTensor`;
  `plot_timeseries` labels its axis "time" and raises on an axis with a
  unit.
  Tests: each output; a window of a series without a unit; the plot
  label and the raise.
  Note: done with G1, since the docs mix, plot, and bound series of
  numbers.
- [x] **G5.** [3] (364d763ae) The unit is storage: `TimeSeries.to(unit)` converts each
  channel's times, end, origin, and view lags and source by one exact
  factor (`Channel::to`), `from_channels` converts every channel into the
  series' unit, and a series requires all channels in its unit. The
  conversions between series and channel units go, and a saved series
  writes its unit once, not per channel. The time-series golden corpus is
  removed, to be regenerated after G6.
  Tests: `to` converts times, origin, and an end after the last
  observation; a converted embedding evaluates as before; `from_channels`
  stores channels in its unit.
  Note: done with G3. The tests that compared windows and embeddings of
  a duration with those of a number in the series' unit are removed.
- [x] **G6.** [3] (5d26cd412) A series holds its unit and epoch once: `Times` keeps
  only values and origin, `TimeSeries` holds the unit and epoch, and
  `Channel::to` becomes a scaling by a factor `TimeSeries::to` computes.
  The native series constructor takes values, unit, and epoch instead of
  a `cpp.Times`. A saved series writes its epoch once. Regenerate the
  time-series golden corpus.
  Tests: existing round trips and epoch tests; the regenerated corpus.

## Phase H: `TimeSeries.from_pandas` for DataFrames and bad input

- [x] **H1.** [2] (c0f0ad307) A `DataFrame` gives one channel per column, sharing the
  index's times, each column converted as a `Series` is, then in one
  common dtype.
  Tests: channel `i` equals `from_pandas(df.iloc[:, i])`; int and float
  columns give float64 channels; `pd.NA` in a column becomes `NaN`; a string column raises `TypeError`; no
  columns raises `ValueError`.
- [x] **H2.** [1] (0d3fe8618) Reject bad input with errors that say what to do.
  Tests: `NaT` in the index; unsorted index names `sort_index()`;
  duplicates name `index.duplicated()`; `PeriodIndex` and `MultiIndex`
  raise `TypeError`; an ndarray raises `TypeError`.

## Phase I: back to pandas

- [x] **I2.** [3] (fe283b97f) `TimeSeries.to_pandas(resolution=None)` for one channel of
  scalar numbers or booleans gives a `Series` indexed by a `DatetimeIndex`
  of `epoch + origin + times`, else a `TimedeltaIndex` of `times`, by default at the
  finer of the unit and microseconds; a unit finer than nanoseconds needs
  `resolution="ns"`. Add
  `stablebear/_pandas.py` with the import that raises `ImportError` saying
  pandas is needed.
  Tests: round trip of a `Series` with a `DatetimeIndex` (times, epoch,
  values, dtype); an undated series gives a `TimedeltaIndex`; a picosecond
  unit raises without `resolution` and rounds to nanoseconds with
  `resolution="ns"`; `resolution="h"` raises; a window's index starts
  at its start date; a PCF series raises `TypeError`; with
  `sys.modules["pandas"] = None` the `ImportError` says pandas is needed.
  Note: absorbed I1, `Times.to_pandas`, since `Times` is removed (E10).
  A series without a unit gives an `Index` of its numbers and takes no
  `resolution`.
- [ ] **I3.** [2] Several channels give a `DataFrame` indexed by the union of
  the channels' times, each column holding its channel's value in effect
  there, and `NaN` outside the channel's domain.
  Tests: round trip of a `DataFrame`; channels with different times hold
  their values and give `NaN` outside their domains; integer channels in
  their domains stay integers.

## Phase J: windows and embeddings of pandas input

- [ ] **J1.** [2] `Window` and `TimeDelayEmbedding` convert a pandas `Series`
  or `DataFrame` with `TimeSeries.from_pandas`, and raise `TypeError` on
  other objects that are not a `TimeSeries`; update their docstrings and
  type hints.
  Tests: windows of a `Series` with `size=pd.Timedelta("3h")` equal those of
  `from_pandas(s)`, also with `return_bounds=True`; embedding of a
  `DataFrame` equals that of `from_pandas(df)`; an ndarray raises
  `TypeError` naming `TimeSeries(values, times)`.

## Phase K: a clearer error for NumPy ufuncs on tensors

- [ ] **K1.** [2] Replace `__array_ufunc__ = None` in
  `stablebear/_tensor_base.py` by a method that calls the tensor's
  reflected operator for the binary ufuncs that have one, so
  `ndarray + tensor` still works, and otherwise raises `TypeError`
  suggesting `np.asarray(tensor)` for numeric tensors.
  Tests: `ndarray + FloatTensor` is a `FloatTensor`; `ndarray < FloatTensor`
  is a `BoolTensor`; `np.sin(FloatTensor)` raises naming `np.asarray`;
  `np.sin` of a `PcfTensor` raises.

## Phase L: pandas in the user guide and tutorial

- [ ] **L1.** [2] A "From pandas" page, `docs/timeseries_pandas.rst`, in the
  time series guide: `from_pandas` for a `Series` and a `DataFrame`,
  windows of a `Series`, time zones, `NaN`, and `to_pandas`.
- [ ] **L2.** [2] The tide gauge tutorial
  (`docs/tutorial_notebooks/timeseries_tides.ipynb`) loads its data with
  `pd.read_csv` and `TimeSeries.from_pandas`; its window sizes and delays
  stay `TimeDelta`, as F5 left them.
- [ ] **L3.** [1] `CHANGELOG.md` entries for the new pandas API, the
  changed `DateTime` time zone behavior, and durations instead of numbers.
