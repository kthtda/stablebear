# Time series general release: checklist

Progress: 1 of 29 items, 1 of 46 points (2%)

From the release review of `issue-258-pandas` on 2026-11-08, in which each
problem below was reproduced twice. The items correct behavior that the
[durations spec](timeseries-durations-spec.md),
[pandas spec](pandas-timeseries-spec.md),
[functions spec](timeseries-functions-spec.md), and
[point cloud spec](timeseries-point-clouds-spec.md) describe; an item that
changes a decision in one of them updates that spec first. Phase A comes
before the merge. The rest can land on the branch or after the merge, but
before 0.6.0, which follows the 0.6.0 betas directly, with no further
pre-release. Files saved by the betas need not load.

Decisions for the maintainer: B4a (equality) and F2 (iterating a series).

## Phase A: CI and the docs build pass

- [x] **A1.** [1] (a4c09c865) Read the Docs builds with Sphinx 9, with the "All
  members, including inherited" dropdowns: `docs/_ext/inherited_members.py`
  adds each dropdown in an `autodoc-process-docstring` handler and moves it
  after the members of its class in a `doctree-read` handler, instead of
  subclassing `ClassDocumenter`, which the new autodoc of Sphinx 9 does not
  call. The branch turns on `fail_on_warning`, and Read the Docs builds with
  Sphinx 9.1.1 on Python 3.12, where registering the subclass warns
  "directive 'autoclass' is already registered".
  `autodoc_use_legacy_class_based` would also work, but Sphinx keeps the
  legacy autodoc only for backward compatibility.
  Tests: `-W --keep-going` builds pass with Sphinx 9.1.1 on Python 3.12 and
  Sphinx 8.1.3 on Python 3.10, with the same 19 dropdowns in the same places
  as before.
- [ ] **A2.** [1] Nothing assumes pandas indexes are in nanoseconds; pandas
  3, which pip installs on Python 3.11 and newer and so in most CI jobs,
  gives microseconds. `test_timeseries_pandas.py:135` compares instants
  instead of `DateTime`s of one resolution. `docs/timeseries_pandas.rst`
  (lines 21-22 and 30), the `TimeSeries` and `from_pandas` docstrings
  (`series.py:427`, `:547`), cell 4 of the tide notebook, and
  `plans/pandas-timeseries-spec.md:36` say the unit is the resolution of the
  index, or the example passes `unit=`.
  Tests: the pandas tests pass with pandas 2.3.3 on Python 3.10 and pandas
  3.1.1 on Python 3.13.
- [ ] **A3.** [1] The PR for this branch, closing #258 and #263 to #267,
  passes the full CI matrix: Linux, Windows with MSVC, macOS with AppleClang,
  Python 3.10 to 3.15, and the free-threaded builds. CI has never run on the
  branch.

## Phase B: no silently wrong results or unreadable files

- [ ] **B1.** [2] A `timedelta64` or `datetime64` given with a unit raises
  `TypeError` instead of being read as counts of the unit (#256).
  `_array_counts` (`date_time.py:721`) takes only dtype kinds `i`, `u`, and
  `f`, and `TimeDelta(value, unit)` (`date_time.py:599`) rejects
  `np.timedelta64`, `np.datetime64`, and booleans before its `Integral`
  check. `_from_pair` sends every `(count, unit)` tuple through these, so
  this covers times, calls, `cut_windows`, `delay_embed`, and
  `to_point_clouds`. The errors point to `TimeDeltaTensor(value)` without a
  unit. `from_pandas` rejects a column of durations as values. Close #256
  with this fix.
  Today: `TimeDeltaTensor(np.array([0, 1], "m8[h]"), "min")` gives
  `[0, 1] min`.
  Tests: a `timedelta64` array, a `TimedeltaIndex`, a list of
  `np.timedelta64`, and `np.timedelta64(5, "ns")` with a unit raise; an
  `(array, unit)` tuple raises in `TimeSeries` and in a call; a column of
  durations raises in `from_pandas`.
- [ ] **B2.** [2] A series that saves also loads: `Channel::to` and
  `Channel::shifted` (`timeseries.tpp:221`, `:246`) treat the end as
  `cut_channel` does, dropping a closed end that lands at or before the last
  time and moving an open one to `nextafter(last, +inf)`, and `shifted`
  keeps the times strictly increasing.
  Today: 9 of 127 delay embeddings of windows of a 10 Hz series (size 2 s,
  step 0.3 s) save after `.to("min")` but do not load, and do not unpickle
  in `multiprocessing` or joblib.
  Tests: every such embedding saves, loads, and pickles after `.to("min")`
  and `.to("d")`; `from_channels` of the windows of two series with
  different epochs saves and loads, with both anchors.
- [ ] **B3.** Dates are counted from an epoch exactly, without int64
  overflow at fine resolutions.
  - [ ] **B3a.** [3] Python: one helper computes `date - epoch` in the
    series unit exactly, in Python integers or in whole coarse units plus a
    remainder. `_time_of_date` (calls and `to_point_clouds`),
    `_times_from_datetimes` (construction and `from_pandas`), and
    `_on_one_epoch` (`from_channels`) use it.
    Today: a yearly series from 1500 to 2100, called at
    `pd.date_range("2090-01-01", periods=3, freq="YS")` in ns, gives 1505,
    1506, and 1507, and `from_pandas` of a yearly ns index from 1700 to 2008
    raises "Timestamps must be strictly increasing".
    Tests: ns queries on series longer than 292 and 584 years; an epoch
    before 1678 with ns queries; `from_pandas` of a 400-year ns index;
    `from_channels` of a channel from 1659 in s and one from 1990 in ns
    counts from 1659.
  - [ ] **B3b.** [2] C++: `time_since` (`timeseries.hpp:241-276`) computes
    the offset without an int64 intermediate at the finer resolution, so a
    tensor of series agrees with B3a instead of raising `OverflowError`, and
    a unit with multiplier 0 throws instead of dividing by it.
    Tests: a tensor of a series with an epoch in 1659 called at ns dates;
    one series and a tensor of it agree in unit `"as"`.
  - [ ] **B3c.** [2] Arithmetic and comparisons of `DateTime`,
    `DateTimeTensor`, and `TimeDeltaTensor` across resolutions raise
    `OverflowError` when a result leaves int64, instead of wrapping.
    Tests: the start of a domain in 1659 plus `pd.Timedelta(days=10)`;
    `TimeDeltaTensor([350 * 365], "d") + TimeDelta(1, "ns")`; a date in 1659
    in s against one in 1990 in ns.
- [ ] **B4.** Equality of durations and dates.
  - [ ] **B4a.** [1] Decide how `==` and `hash` compare `TimeDelta`,
    `DateTime`, and their tensors, and record it in the durations spec
    (lines 49 and 97). Recommended: by value, as `<` already does and as
    numpy, pandas, and `datetime` do, with `array_equal` comparing count and
    unit. A change after the release would silently flip results.
    Today: `TimeDelta(1, "h") == TimeDelta(60, "min")` is false while `<=`
    and `>=` are true, `series.datetimes[0] == series.epoch` is false, and
    an epoch from pandas equals a `DateTime` in one pandas version but not
    the other.
  - [ ] **B4b.** [3] Implement B4a. Update `docs/timeseries_dates.rst:42-46`
    and the notes of the four classes.
    Tests: 1 h equals 60 min and hashes the same;
    `TimeDelta(1, "h") == np.timedelta64(1, "h")`; `datetimes[0]` of a
    dated series equals its epoch; the length of a window from pandas
    equals `TimeDelta(2, "h")`; `array_equal` still tells 1 h from 60 min;
    the asserts at `test_timeseries.py:101`, `:197`, and `:375` change.

## Phase C: dates and pandas at the edges

- [ ] **C1.** [1] A date outside the domain with digits finer than a
  microsecond raises `ValueError`: the precision search in
  `_check_date_in_domain` (`series.py:1087`) falls back to the resolution of
  the dates, instead of letting `StopIteration` escape, which silently ends
  `map()`.
  Tests: such a `Timestamp` on an ns series, called and through
  `to_point_clouds`; `list(map(series, dates))` raises.
- [ ] **C2.** [3] A date at a closed end of the domain is in the domain
  despite rounding: placing a date snaps it to a closed end within the
  channel's snap tolerance, in `_check_date_in_domain` and `_evaluate_date`
  and in the C++ placement of dates for tensors and `to_point_clouds`.
  Errors on the tensor path name dates instead of times in the unit.
  Today: 761 of 2150 windows of 6-minute data stored in hours, stepping by
  6 minutes, reject their own start date, as in "2023-01-01T00:00:00.300 is
  outside the domain [2023-01-01T00:00:00.300, ...)".
  Tests: every window and delay embedding evaluates at its `domain.start`
  and its `datetimes`, with a step that is inexact in hours; the closed end
  of a channel that `from_channels` shifted; the same through `sb.tensor`
  and `to_point_clouds(times=...)`.
- [ ] **C3.** [1] `to_pandas` builds its index from times rounded to the
  output resolution instead of from raw floats, so one instant held as two
  floats one ulp apart, as `from_channels` can shift it, gives one label.
  Today: two 10-minute channels with different epochs in unit `"h"` raise
  "cannot reindex on an axis with duplicate labels".
  Tests: that series; channels with different times keep `NaN` outside
  their domains.
- [ ] **C4.** [2] `TimeSeries(values, times)` takes lists of `datetime`,
  `pd.Timestamp`, `DateTime`, or date strings as dates, through
  `DateTimeTensor`, and lists of `(count, unit)` tuples as durations,
  through `TimeDeltaTensor`, as calling a series does. A `(dates, unit)`
  tuple raises `TypeError`.
  Today: a list of `datetime` raises "float() argument must be a string or a
  real number".
  Tests: each list equals its tensor; a list of dates and durations raises;
  a `(dates, unit)` tuple raises.
- [ ] **C5.** [1] Calls take dates in the same forms everywhere: a list of
  `DateTime` in a call of a series, a tuple of date strings in a call of a
  `TimeSeriesTensor`, and `datetime64` with a multiple, such as
  `datetime64[15m]`, in tensor date queries (`_date_counts`,
  `series.py:78-82`).
  Tests: each form gives the values at its dates, on a series and on a
  tensor.

## Phase D: windows and views

- [ ] **D1.** [3] Window bounds compare tolerantly in both directions, in
  `detail::cut_channel` (`windowing.hpp:35-39`) and where windows are
  checked to fit (`:132-134`, `:159-161`): an observation within the snap
  reach of an open end is at the end, one within reach of a computed start
  is at the start, and the reach scales with the operands of the computed
  bound. This changes which observations some windows hold, so it comes
  before the release.
  Today: windows of 1.5 s every 0.1 s on 10 Hz data give 49 windows of 16
  observations instead of 15, and the same data in milliseconds gives a
  different number of windows.
  Tests: `np.arange(n) / 10` and `np.linspace` times give windows of equal
  sizes; an observation on a bound between two half-open windows is in one;
  seconds and milliseconds give the same windows.
- [ ] **D2.** [2] A view of a view takes from the original channel:
  `Channel::take` (`timeseries.tpp:186`) composes the indices of a view
  with one lag through its base, so chains stay one level deep and earlier
  levels are freed.
  Today: about 25,000 chained `restrict` calls segfault when freed in the
  Debug build, and the memory of such chains grows quadratically.
  Tests: 100,000 chained `restrict` calls evaluate and are freed.
- [ ] **D3.** [2] Converting or shifting a view with one lag leaves its
  source alone (`Channel::to` and `Channel::shifted`,
  `timeseries.tpp:211-254`), since it reads the source by index only, and
  `plot_timeseries` (`plotting.py:410`, `:517-520`) reads the unit name
  without converting and converts at most once.
  Today: `plot_timeseries(windows, relative=True, unit="h")` on a million
  minutes of data takes 57 s instead of 0.28 s.
  Tests: a converted window and an embedding of it evaluate as before; a
  plot of windows with `unit=` is unchanged.

## Phase E: dates without Python loops

- [ ] **E1.** [2] `TimeDeltaTensor.to_timedelta64` (`date_time.py:422-425`)
  rounds exactly in NumPy instead of with one `Fraction` per element:
  integer counts by a checked multiply or a half-even integer division,
  and float counts by `rint` of a TwoProduct with a correction at ties,
  below 2^52. A plain `np.rint(c * f)` is not exact.
  Today: about 4 us per element, so `to_pandas`, `datetimes`, and plots of
  dated series take about 4 s per million observations, all in Python.
  Tests: random integer and float counts, including values near ties, give
  the results of the scalar `TimeDelta` path.
- [ ] **E2.** [1] `TimeSeries.__call__` and `to_point_clouds` check for a
  `DateTimeTensor` before converting the query with `np.asarray`
  (`series.py:159-171`, `:1020`, `:1151-1166`), and an empty
  `DateTimeTensor` counts as dates.
  Today: calling with a `DateTimeTensor` is 11 to 16 times slower than with
  its `datetime64` array, and an empty one raises "This series has a unit".
  Tests: an empty `DateTimeTensor` gives an empty result; a
  `DateTimeTensor` and its `datetime64` array give equal results.

## Phase F: the public API before it is frozen

- [ ] **F1.** [1] `TimeSeries.__call__` and `TimeSeriesTensor.__call__`
  (`series.py:968`, `:1220`) return `Any`, as `AGENTS.md` says for results
  that depend on the number of channels.
  Today: pyright types `series(t)` as `NestedTensor | Unknown` and flags
  `float(series(t))` and `series(t) + 1.0`.
  Tests: `test/typing/check_types.py` uses `float(series(t))`,
  `series(times) * 2.0`, and `windows(times) * 2.0`.
- [ ] **F2.** [2] Iterating a `TimeSeries` is defined and documented.
  Recommended: `__iter__` raises `TypeError` naming `series[i]` for channels
  and `series.values` for observations, `__array__` raises the same way, and
  `len` keeps counting observations, raising `TypeError` for several
  channels.
  Today: a series of one channel and 4 observations has `len` 4 but
  iterates once, `list()` of several channels raises `ValueError`, and
  `np.asarray(series)` fails with "exceed the maximum number of dimension".
  Tests: iterating, `list()`, and `np.asarray` raise `TypeError` for one
  channel and for several; `len` of one channel.
- [ ] **F3.** [1] `TimeSeriesTensor` is exported from
  `stablebear.timeseries`, and its constructor takes a list of series as
  `sb.tensor` does; `docs/conf.py:83`, `api_timeseries.rst:10`,
  `timeseries_windows.rst:53`, and `tensors.rst:220` name it there.
  Tests: `TimeSeriesTensor([a, b])` equals `sb.tensor([a, b])`.
- [ ] **F4.** [1] `to_point_clouds` takes `n_points` instead of `n`, as
  `noisy_sin` and `PointCloud.n_points` say.
  Tests: the grid tests pass `n_points`.
- [ ] **F5.** [1] `t in domain` tells whether a time lies in a `Domain`,
  instead of whether it equals one of the tuple's fields, and a `Domain`
  checks that its ends are of one kind.
  Tests: times inside, outside, and at open and closed ends; ends of two
  kinds raise.

## Phase G: guide pages

- [ ] **G1.** [1] The From pandas page (`docs/timeseries_pandas.rst:63-68`)
  sends windows and embeddings through `to_point_clouds` before
  `compute_persistent_homology`, which raises for them, and the dtype error
  of `compute_persistent_homology` for a tensor of series names
  `to_point_clouds`.
- [ ] **G2.** [1] The Dates page (`docs/timeseries_dates.rst:71-75`) says
  that `from_channels` counts channels with different epochs from the
  earliest, and that channels with and without an epoch do not combine.
- [ ] **G3.** [1] The examples run as written:
  `sb.stack(TimeSeries.from_channels([a, b], unit="s"))` in
  `timeseries_channels.rst:244-247`, the imports of the block in
  `timeseries_snapping.rst:59-66`, and `dated` and `domain` in the
  `restrict` recipe of `timeseries_windows.rst:300`. The `TimeSeries`
  docstring (`series.py:399-404`) names the operations that take a series
  instead of claiming every element-wise one.
- [ ] **G4.** [1] `docs/saving.rst` lists series and tensors of series
  among what can be saved, and the `TimeSeries` docstring
  (`series.py:380-382`) no longer says each channel keeps its unit.

Left out, since these can come after the release without breaking the API
or the format:
- Reading corrupted files: an epoch without a unit or at NaT, an infinite
  end that windowing follows until memory runs out, and a corrupted length
  that makes `sb.load` allocate gigabytes, as for every tensor type since
  before 0.6.0b3.
- `sb.load` reports any failure to read a tensor, such as a truncated file,
  as "Expected format type SingleObject" (`io.py:260-272`).
- Windows hold their own times and indices, about 19 bytes per held
  observation.
- `restrict` of a series with an epoch, and #268.
- `py.typed` in wheels; pyright with numpy 2.5 stubs; `pd.NA` in boolean and
  object columns; `to_pandas(resolution)` truncating its epoch; Python loops
  over channels for date queries; one write per element when saving;
  `DateTimeTensor` from strings one at a time; the GIL held while building
  and converting a series.
- The README and the docs index do not mention time series.
- #97 can be closed as obsolete.
