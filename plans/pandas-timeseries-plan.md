# pandas input and durations for time series: implementation plan

Progress: 6 of 36 items, 8 of 63 points (13%)

Specs: [pandas-timeseries-spec.md](pandas-timeseries-spec.md) (phases A, B,
and H to K) and [timeseries-durations-spec.md](timeseries-durations-spec.md)
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
- [ ] **B2.** [2] A `DataFrame` gives one channel per column, sharing the
  index's times, with values from `df.to_numpy()`.
  Tests: channel `i` equals `from_pandas(df.iloc[:, i])`; int and float
  columns give float64 channels; a string column raises `TypeError`; no
  columns raises `ValueError`.
- [ ] **B3.** [1] Reject bad input with errors that say what to do.
  Tests: `NaT` in the index; unsorted index names `sort_index()`;
  duplicates name `index.duplicated()`; `PeriodIndex` and `MultiIndex`
  raise `TypeError`; an ndarray raises `TypeError`.

## Phase C: `TimeDelta`

- [ ] **C1.** [2] `TimeDelta(value, unit=None)` in
  `stablebear/timeseries/date_time.py`, exported from
  `stablebear.timeseries`: `count`, `unit`, `to_timedelta64`, `==`, `hash`,
  `repr`, pickling; with its docstring and a line in `api_timeseries.rst`.
  Tests: from `np.timedelta64`, `datetime.timedelta`, and `pd.Timedelta` at
  their resolutions; `TimeDelta(3, "h")`; `NaT`, months, and a fractional
  count raise; pickling round trip.
- [ ] **C2.** [2] Arithmetic on `TimeDelta` and `DateTime`, following
  `np.timedelta64`.
  Tests: `TimeDelta(1, "h") + TimeDelta(30, "min")` is 90 minutes;
  negation and `* 3`; `DateTime - DateTime` and `DateTime + TimeDelta`;
  comparisons across resolutions.
- [ ] **C3.** [1] `_duration` accepts `TimeDelta`, so `Window` and
  `TimeDelayEmbedding` take it.
  Tests: `Window(TimeDelta(2, "h"))` equals `Window(np.timedelta64(2, "h"))`
  on a series in hours.

## Phase D: durations as input

- [ ] **D1.** [2] `Times` accepts durations: a `timedelta64` array, a
  sequence of `TimeDelta`, or a `TimedeltaIndex`, converted to `unit`
  exactly; `TimeSeries` takes them as `times`.
  Tests: `timedelta64[ms]` in `unit="s"` gives exact seconds; `TimeDelta`
  list equals the array; `TimedeltaIndex` equals its array.
- [ ] **D2.** [2] `TimeSeries.evaluate` accepts durations, single or arrays,
  as times from time 0, with or without an epoch.
  Tests: `TimeDelta(90, "min")` on a series in hours equals 1.5; a
  `timedelta64` array; a window evaluated at a duration from its start.
- [ ] **D3.** [2] `TimeSeriesTensor.evaluate` accepts durations.
  Tests: windows at `np.arange(3) * np.timedelta64(1, "h")` equal windows
  at `[0.0, 1.0, 2.0]` in hours.

## Phase E: durations and dates as output

- [ ] **E1.** [2] `np.asarray(times)` gives `datetime64` or `timedelta64` at
  the finer of the unit and microseconds; add `Times.to_floats(unit=None)`
  and use it in `plot_timeseries` and other internal callers.
  Tests: dated and undated times; nanosecond unit kept; `to_floats` in
  another unit.
- [ ] **E2.** [2] `TimeSeries.times` of several channels is a tuple of
  `Times`.
  Tests: two channels with different times and units.
- [ ] **E3.** [2] `TimeSeries.domain` has `DateTime` or `TimeDelta` ends;
  a dated window's domain is where it lies.
  Tests: undated series; dated series; dated window; closed ends kept.
- [ ] **E4.** [2] `TimeSeries.origin` is a `TimeDelta` or a tuple of them;
  remove `TimeSeries.bounds` and update its callers and docs.
  Tests: window origins, including after snapping; several channels.
- [ ] **E5.** [2] `Window(..., return_bounds=True)` gives a `datetime64` or
  `timedelta64` array of shape `(n_windows, 2)`.
  Tests: dated and undated series; partial windows.
- [ ] **E6.** [1] `Window.size`, `Window.step`, and
  `TimeDelayEmbedding.delay` are `TimeDelta`; update the reprs.
  Tests: from `np.timedelta64` and `pd.Timedelta`; repr.

## Phase F: tests and docs use durations

- [ ] **F1.** [3] `test/python/test_windowing.py` builds series from
  durations and passes durations to `Window` and `evaluate`.
- [ ] **F2.** [3] `test/python/test_timeseries.py` likewise.
- [ ] **F3.** [2] `test_time_delay_embedding.py`, `test_plotting_timeseries.py`,
  `test_timeseries_pandas.py`, `persistence/test_timeseries_persistence.py`,
  `test_io_roundtrip.py`, and `test_integration.py` likewise.
- [ ] **F4.** [3] The time series guide pages (`docs/timeseries*.rst`) use
  durations and dates; `timeseries_units.rst` explains the unit as
  storage.
- [ ] **F5.** [1] The tide gauge tutorial uses durations for its window
  sizes, delays, and evaluation grid.

## Phase G: numbers are rejected

- [ ] **G1.** [2] `Times` needs `unit` for numbers; `TimeSeries` and
  `from_pandas` reject numbers without a unit, naming `Times(numbers,
  unit)`.
  Tests: `TimeSeries(values, [0, 1, 2])` raises; `Times([0, 1, 2])` raises;
  numeric index without `unit` raises.
- [ ] **G2.** [1] `evaluate` on series and tensors rejects numbers; remove
  its `unit` argument.
  Tests: `series(1.5)` raises `TypeError`; tensor likewise.
- [ ] **G3.** [1] `Window` and `TimeDelayEmbedding` reject numbers.
  Tests: `Window(3.0)` and `TimeDelayEmbedding(2, 1.0)` raise `TypeError`.

## Phase H: back to pandas

- [ ] **H1.** [1] `Times.to_pandas(resolution=None)`: a `DatetimeIndex`
  with an epoch, else a `TimedeltaIndex`, by default at the resolution of
  `np.asarray(times)`, at most nanoseconds. Add `stablebear/_pandas.py` with
  the import that raises `ImportError` naming `pip install pandas`.
  Tests: dated times; undated times; a picosecond unit gives nanoseconds;
  `resolution="h"` raises; with `sys.modules["pandas"] = None`
  the `ImportError` names `pip install pandas`.
- [ ] **H2.** [2] `TimeSeries.to_pandas(resolution=None)` for one channel of
  scalar numbers or booleans gives a `Series` indexed by
  `times.to_pandas()`.
  Tests: round trip of a `Series` with a `DatetimeIndex` (times, epoch,
  values, dtype); a window's index starts
  at its start date; a PCF series raises `TypeError`.
- [ ] **H3.** [2] Several channels give a `DataFrame` indexed by the union of
  the channels' times, with `NaN` where a channel has no observation.
  Tests: round trip of a `DataFrame`; channels with different times give
  the union with `NaN`.

## Phase I: windows and embeddings of pandas input

- [ ] **I1.** [2] `Window` and `TimeDelayEmbedding` convert a pandas `Series`
  or `DataFrame` with `TimeSeries.from_pandas`, and raise `TypeError` on
  other objects that are not a `TimeSeries`; update their docstrings and
  type hints.
  Tests: windows of a `Series` with `size=pd.Timedelta("3h")` equal those of
  `from_pandas(s)`, also with `return_bounds=True`; embedding of a
  `DataFrame` equals that of `from_pandas(df)`; an ndarray raises
  `TypeError` naming `TimeSeries(values, times)`.

## Phase J: a clearer error for NumPy ufuncs on tensors

- [ ] **J1.** [2] Replace `__array_ufunc__ = None` in
  `stablebear/_tensor_base.py` by a method that calls the tensor's
  reflected operator for the binary ufuncs that have one, so
  `ndarray + tensor` still works, and otherwise raises `TypeError`
  suggesting `np.asarray(tensor)` for numeric tensors.
  Tests: `ndarray + FloatTensor` is a `FloatTensor`; `ndarray < FloatTensor`
  is a `BoolTensor`; `np.sin(FloatTensor)` raises naming `np.asarray`;
  `np.sin` of a `PcfTensor` raises.

## Phase K: pandas in the user guide and tutorial

- [ ] **K1.** [2] A "From pandas" page, `docs/timeseries_pandas.rst`, in the
  time series guide: `from_pandas` for a `Series` and a `DataFrame`,
  windows of a `Series`, time zones, `NaN`, and `to_pandas`.
- [ ] **K2.** [2] The tide gauge tutorial
  (`docs/tutorial_notebooks/timeseries_tides.ipynb`) loads its data with
  `pd.read_csv` and `TimeSeries.from_pandas`, and uses `pd.Timedelta` for
  window sizes and delays.
- [ ] **K3.** [1] `CHANGELOG.md` entries for the new pandas API, the
  changed `DateTime` time zone behavior, and durations instead of numbers.
