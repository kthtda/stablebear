# pandas and NumPy input for time series: implementation plan

Progress: 3 of 16 items, 3 of 25 points (12%)

Spec: [pandas-timeseries-spec.md](pandas-timeseries-spec.md)

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
- [ ] **A3.** Timezone-aware dates become naive UTC.
  - [x] **A3a.** [1] (79218327d) `DateTime` converts an aware `datetime` or `Timestamp`
    to UTC instead of raising; update its docstring.
    Tests: `datetime` at +02:00 equals the UTC value two hours earlier;
    aware `Timestamp` likewise.
  - [ ] **A3b.** [2] `Times` accepts a `DatetimeIndex` or a `Series` of
    dates, naive or aware, so `TimeSeries(s, s.index)` works for both;
    update the `Times` and `TimeSeries` docstrings.
    Tests: aware index equals the naive UTC index; `Series` of dates equals
    its index; `NaT` raises.
- [ ] **A4.** [1] `TimeSeries.evaluate` takes a `DatetimeIndex` or a `Series`
  of dates as dates, not numbers.
  Tests: equals evaluating `index.to_numpy()`; aware index equals its UTC
  equivalent; aware `Timestamp`.

## Phase B: `TimeSeries.from_pandas`

- [ ] **B1.** [2] `TimeSeries.from_pandas(obj, unit="s", *, epoch=None,
  snap_tolerance=1e-14)` for a `Series` with a `DatetimeIndex`,
  `TimedeltaIndex`, or numeric index, with its docstring.
  Tests: datetime index equals `TimeSeries(values, Times(index))`, epoch
  the first stamp; `epoch=` overrides it; timedelta index in `unit="ms"`;
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

## Phase C: back to pandas

- [ ] **C1.** [1] `Times.to_pandas(resolution="us")`: a `DatetimeIndex`
  with an epoch, else a float64 `Index`. Add `stablebear/_pandas.py` with
  the import that raises `ImportError` naming `pip install pandas`.
  Tests: dated times give `to_datetime64(resolution)`; undated give the
  timestamps; `resolution="h"` raises; with `sys.modules["pandas"] = None`
  the `ImportError` names `pip install pandas`.
- [ ] **C2.** [2] `TimeSeries.to_pandas(resolution="us")` for one channel of
  scalar numbers or booleans gives a `Series` indexed by
  `times.to_pandas()`.
  Tests: round trip of a `Series` with a `DatetimeIndex` at
  `resolution="ns"` (times, epoch, values, dtype); a window's index starts
  at its start date; a PCF series raises `TypeError`.
- [ ] **C3.** [2] Several channels give a `DataFrame` indexed by the union of
  the channels' times, with `NaN` where a channel has no observation.
  Tests: round trip of a `DataFrame`; channels with different times give
  the union with `NaN`.

## Phase D: windows and embeddings of pandas input

- [ ] **D1.** [2] `Window` and `TimeDelayEmbedding` convert a pandas `Series`
  or `DataFrame` with `TimeSeries.from_pandas`, and raise `TypeError` on
  other objects that are not a `TimeSeries`; update their docstrings and
  type hints.
  Tests: windows of a `Series` with `size=pd.Timedelta("3h")` equal those of
  `from_pandas(s)`, also with `return_bounds=True`; embedding of a
  `DataFrame` equals that of `from_pandas(df)`; an ndarray raises
  `TypeError` naming `TimeSeries(values, times)`.

## Phase E: a clearer error for NumPy ufuncs on tensors

- [ ] **E1.** [2] Replace `__array_ufunc__ = None` in
  `stablebear/_tensor_base.py` by a method that calls the tensor's
  reflected operator for the binary ufuncs that have one, so
  `ndarray + tensor` still works, and otherwise raises `TypeError`
  suggesting `np.asarray(tensor)` for numeric tensors.
  Tests: `ndarray + FloatTensor` is a `FloatTensor`; `ndarray < FloatTensor`
  is a `BoolTensor`; `np.sin(FloatTensor)` raises naming `np.asarray`;
  `np.sin` of a `PcfTensor` raises.

## Phase F: pandas in the user guide and tutorial

- [ ] **F1.** [2] A "From pandas" page, `docs/timeseries_pandas.rst`, in the
  time series guide: `from_pandas` for a `Series` and a `DataFrame`,
  windows of a `Series`, time zones, `NaN`, and `to_pandas`.
- [ ] **F2.** [2] The tide gauge tutorial
  (`docs/tutorial_notebooks/timeseries_tides.ipynb`) loads its data with
  `pd.read_csv` and `TimeSeries.from_pandas`, and uses `pd.Timedelta` for
  window sizes and delays.
- [ ] **F3.** [1] `CHANGELOG.md` entries for the new API and the changed
  `DateTime` time zone behavior.
