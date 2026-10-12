# Time series (#242): implementation plan

This plan implements #242 on `issue-242-timeseries`, branched from `main` at
`05ce5ba4a`: unit-bearing timestamps, time series made of channels that may
differ in value type, time grid, and unit, persistence of snapshot series
(workflow 1), and windowed delay embeddings sampled into point clouds
(workflow 2), with serialization and documentation. Useful parts of the
`origin/timeseries` branch are recycled where they fit (see "Recycling").

Channels with their own value types, time grids, and units go beyond the
issue, which defers a channel-combination API; we added them to this release.

## How to work through this plan

Work through the items in order. Each item depends only on items above it and
is one review unit: a decision, or a small change with at most two or three
focused tests. Keep new work uncommitted for review, and check an item off
only once we have reviewed it. If a decision goes against its recommendation,
revise the affected items before starting them.

Read [TESTING.md](../test/TESTING.md) before writing tests and
[WRITING.md](../docs/WRITING.md) before writing documentation. Build and test
commands are in [AGENTS.md](../AGENTS.md#testing). C++ changes need a CMake
build and install; for Python-only items, run pytest from `test/` with
`PYTHONPATH=..` to test the source tree.

Each test item gives its input, the expected result, and **Catches:**, the bug
that makes the test fail. Tags: **(decision)** a choice to make before the
items that follow; **(existing)** confirm existing coverage without adding a
test; **(optional)** safe to skip without losing important coverage. Check off
a skipped optional item with the note "skipped".

**Test budget.** The items add about 60 test functions, many of them
parametrized, plus three optional ones. Keep that number from growing. Before
adding a test that is not listed, check whether a listed test's
parametrization covers it, and prefer leaving a guarantee to the design and
its docstring over testing every combination. Do not test what the public API
cannot observe. No C++ unit tests are planned: every guarantee here is
observable from Python (TESTING.md principle 4).

## Conventions

- **Files.** Source: `include/sbear/timeseries.hpp` (`Times`, `Channel<T>`),
  `include/sbear/timeseries/series_task.hpp` (analysis adapter),
  `include/sbear/io/timeseries_io.hpp`, `src/python/py_timeseries.{hpp,cpp}`,
  and the package `stablebear/timeseries/` with `series.py` (`Times`,
  `TimeSeries`, channels) and `derived.py` (`TimeDelayEmbedding`, `Window`,
  `RegularSampler`). Tests: `test/python/test_timeseries.py` (storage,
  access, evaluation, units, channels, serialization),
  `test/python/test_timeseries_derived.py` (embedding, windows, sampling), and
  `test/python/persistence/test_timeseries_persistence.py` (workflow 1). Items
  that touch another file name it.
- **Data.** Use timestamps, delays, and window sizes that are exact in binary
  floating point (multiples of 0.25), so expected values do not depend on
  rounding; item 19 tests rounding on purpose. Give observations distinct
  values (10, 20, 30, ...) so that mix-ups show.
- **Channels A and B.** Several items use channel A, with times
  `Times([0.0, 1.0, 2.5], unit="s")` and values `[10.0, 20.0, 30.0]`, and
  channel B, with times `Times([500.0, 1500.0, 3000.0], unit="ms")` and values
  `[-1.0, -2.0, -3.0]`, combined as
  `TimeSeries.from_channels({"a": a, "b": b}, unit="s")`.
- **Precisions.** Parametrize over 32 and 64 bits only where the code
  dispatches on precision: construction per element type, serialization, and
  the native persistence entry points. Elsewhere use 64 bits.
- **Comparisons.** Compare numeric values exactly; the inputs are exact.
  Compare barcodes with `is_isomorphic_to(..., atol=0, rtol=0)` and PCF
  tensors with `array_equal`.
- **Parameters.** Spell out `dimension`, `delay`, `size`, `step`, `n_points`,
  `unit`, `max_dim`, and `reduced` (TESTING.md principle 18) unless the test
  checks a default.
- **Comments.** Test comments describe the input or what is checked. The
  **Catches:** analysis stays in this plan.

## Design at a glance

This assumes the recommended option of each decision.

- `sb::Times` holds finite, strictly increasing float64 timestamps and a unit
  from a fixed set with exact factors in seconds.
- `sb::Channel<T>` stores recorded observations of one element type: `Times`,
  a flat `std::vector<T>` in row-major order, a `value_shape`, and a domain.
  Observation `i` is the block `[i * k, (i + 1) * k)` with
  `k = prod(value_shape)`. Point counts, matrix sizes, and bar counts are
  internal to each element and may change over time. Like `NestedTensor`, one
  registration template binds it for every tensor element type in the library
  (19 types). Several values of one type on one grid, such as two cameras
  sampled together, are a value axis of one channel.
- `TimeSeries` is the public object: an ordered set of named channels and a
  query unit. Its domain is the intersection of the channels' domains.
  `series[t]` and `evaluate(t)` look up each channel on its own grid, in its
  own unit, and return the value for one channel or an ordered dict for
  several. `series[a:b]` is a window, and `series["name"]` selects channels.
- Derived channels are lazy and store no values: delay embeddings and windows
  hold their sources and parameters and ask them for values when evaluated.
- `sb::SeriesTask` runs an existing batch task (`RipserTask`,
  `RipserDistMatTask`, `BarcodeSummaryTask`) once over all channels of one
  element type, concatenating their arrays and splitting the result, and
  returns channels with the input's times and domains.
- `stablebear.timeseries` wraps the native types. Evaluation and derived
  channels use NumPy on the stored arrays.

## Recycling

The `origin/timeseries` branch predates the rename (`mpcf`/`masspcf` became
`sb`/`stablebear`). Read its files with `git show origin/timeseries:<path>`
and copy ideas and small blocks, not whole files.

- `include/mpcf/timeseries.hpp`: `snap_in_place` and `default_snap_tol` give
  the snapping rule of item 19, and its row-major `m_values` with
  `n_channels` is the flat layout of `Channel<T>` (item 11). The offsets from
  `start_time` and `time_step` and the interpolation variants do not fit.
- `include/mpcf/algorithms/embed_time_delay.hpp`: the oldest-to-newest
  component order and the valid start `start + (dimension - 1) * delay`
  (items 39-41). Not its windows, which are anchored at the end and keep a
  partial first window, and not its tuples, which it stores only at original
  observation times; the issue rules out both.
- `include/mpcf/io/timeseries_io.hpp`: the outline of the writer and reader
  for items 25-26 and 52, with new payloads.
- `src/python/py_timeseries.cpp`: the array-copy and length-check patterns for
  items 10-14.
- `masspcf/persistence/homology.py` (the `TensorTimeSeries` branch): one
  batched Ripser call over all snapshots. Item 27 keeps the idea and moves it
  into native code.
- `docs/ts_*.rst`: background wording for item 55; the API differs.

Not recycled: datetime support, interpolation strategies, `TimeSeriesTensor`,
the scikit-learn transformers, `l2_kernel` cross kernels, and the `tensor_eval`
parallel threshold.

## Decisions

- [x] **1. Domain and evaluation.** (decision: (a), closed domain for series built from observations, half-open for windows) Evaluation
  uses the previous value: at time `t`, the observation with the largest
  timestamp `<= t`. Queries outside the domain raise `ValueError`. Options for
  the domain of a channel:
  (a) A channel built from observations has the closed domain
  `[t_0, t_{n-1}]`, so every observation can be evaluated. Windows have
  half-open domains `[a, b)`, and channels derived from a window keep its
  half-open end. The channel stores one flag for this.
  (b) Every domain is half-open: `[t_0, end)` with an optional `end=` that
  defaults to `t_{n-1}`. One rule and no flag, but by default the last
  observation cannot be evaluated, and a singleton series cannot be evaluated
  at all.
  Recommendation: (a). With either option, `domain` is a `(start, end)`
  tuple, and an empty channel has no domain (`domain is None`) and cannot be
  evaluated.

- [x] **2. Floating-point timestamps.** (decision: (a), snap within `1e-9 * max(1, |t|)`, no user parameter) Delays
  and window starts are computed in floating point, so a query can land just
  before the observation it should reach: `0.3 - 0.1` is less than `0.2`.
  Options:
  (a) Snap a query to an observation that lies above it by at most
  `1e-9 * max(1, |t|)`, as `snap_in_place` does on the branch. No user
  parameter; domain checks stay exact.
  (b) No snapping; document that times and delays should be exactly
  representable.
  Recommendation: (a).

- [x] **3. Time units.** (decision: (a), fixed set `as` through `d` with exact integer conversion factors) `Times(values, unit="s")`; plain timestamp
  arrays mean seconds. Options:
  (a) A fixed set covering physics time scales: `"as"`, `"fs"`, `"ps"`,
  `"ns"`, `"us"`, `"ms"`, `"s"`, `"min"`, `"h"`, and `"d"`. Store each unit as
  a power of ten times an integer (1 or 60, 3600, 86400), and convert by
  multiplying or dividing by an exact integer factor rather than by a
  rounded one such as `1e-15`; dividing by an exact factor rounds once.
  Adding `"zs"` or `"ys"` later is one table entry. `Times.to(unit)`
  converts, and a series converts each query from its query unit into each
  channel's unit. Single-channel parameters (delays, window sizes, query
  times) are in the series' query unit.
  (b) Any nonempty string, with no conversion.
  Recommendation: (a).

- [x] **4. Ownership.** (decision: (a), copy in and copy out; stored series are immutable) Options:
  (a) Copy in, copy out for stored channels. The constructor copies its
  values, materializing indexed point-cloud and distance-matrix tensors;
  `values`, `observation(i)`, and evaluation return copies; `Times.values` is
  a read-only array. Stored channels are then immutable, so series and
  derived channels can hold references to them without copying them and
  without seeing later changes.
  (b) Views, with documented write-through.
  Recommendation: (a). Mark the copies with a `ponytail:` comment; views can
  come later if profiling shows a cost.

- [ ] **5. Native storage and where the code lives.** (decision)
  Recommendation: follow the `NestedTensor` pattern
  (`include/sbear/nested_tensor.hpp`, `register_nested_tensor_element` in
  `src/python/py_tensor.cpp`, `stablebear/nested_tensor.py`).
  - `sb::Channel<T>` holds `Times`, a flat `std::vector<T>`, `value_shape`,
    and the domain, not a `Tensor<T>`, so its layout, gathering, and slicing
    stay under our control. Copies share storage, which is safe because
    stored channels are immutable (decision 4).
  - One `register_timeseries_element<T>` call per element type binds it for
    bool, float32/64, int32/64, uint32/64, pcf32/64, pcf32i/64i,
    pcloud32/64, distmat32/64, symmat32/64, and barcode32/64.
  - Python has one channel wrapper whose `dtype` is the element dtype, with a
    dtype-keyed table for the native class and for the tensor wrapper that
    `values` returns.
  - Serialization marks the element type's own format with a new
    `TensorFormatFlag::TimeSeries`, so one writer and reader serve all
    element types, as `TensorFormatFlag::Nested` does.
  - Python with NumPy implements evaluation (one shared lookup:
    `searchsorted` plus snapping, then reading an observation, identical for
    every element type) and derived channels.
  Alternative: store a `Tensor<T>`, which reuses tensor slicing at first but
  ties the layout to tensor views and strides.

- [ ] **6. Analysis adapter.** (decision) Recommendation:
  `sb::SeriesTask<TaskT, InT, OutT>` derives from an existing batch task. A
  base declared before the task (base-from-member) holds the input channels
  and the output tensor. It concatenates the flat arrays of all input
  channels of one element type into one tensor of shape
  `(sum of n_times * k, ...)` reshaped per channel as needed, so that the task
  runs once; `result()` splits the output by channel offsets and returns
  channels with the inputs' times, domains, and value shapes extended by the
  task (for persistence, `value_shape + (max_dim + 1,)`). The task supplies
  scheduling, progress, cancellation, and error capture, and the base keeps
  inputs and output alive until completion (`BarcodeSummaryTask` stores only a
  reference to its input). A series whose channels have several element types
  runs one task per type. Each operation declares its contract with a
  `make_*_series_task` overload, and pybind11 overloads on the native channel
  type select the implementation. A singleton channel keeps its time axis and
  an empty channel gives a typed empty result. Alternative: Python spawns the
  existing tasks per channel and rebuilds the series; no new C++, but dispatch
  and result construction stay in Python, against the issue.

- [ ] **7. Public names and accepted values.** (decision) Recommendation: the
  issue's names, except that `TimeDelayEmbedding` drops `unit="time"` until a
  second delay mode exists.
  - `stablebear.timeseries` exports `Times(values, unit="s")`,
    `TimeSeries(values, times=None)` for one channel,
    `TimeSeries.from_channels(channels, unit=None)` for several,
    `TimeDelayEmbedding(dimension, delay)`,
    `Window(size, step=None, overlap=None)`, and `RegularSampler(n_points)`
    with `.times(series)`.
  - `from_channels` takes a dict of names to one-channel series or a list of
    them (named `"0"`, `"1"`, ...). `unit` defaults to the channels' shared
    unit and is required when their units differ.
  - A `TimeSeries` exposes `channel_names`, `unit`, `domain`, `times` (a
    `Times` for one channel, a `NestedTensor` with one leaf per channel in the
    query unit for several), `to(unit)`, `evaluate(t)`, `series[t]` (the same
    as `evaluate(t)`; an int `t` is a time), `series[a:b]` (a window, see item
    43), and `series["name"]` or `series[["a", "b"]]` (channel selection). A
    one-channel series also exposes `values`, `dtype`, `shape`,
    `value_shape`, `len()`, and `observation(i)`; on several channels these
    raise `ValueError` asking for a channel to be selected.
  - For one channel, `evaluate` returns that channel's value; for several, an
    ordered dict keyed by channel name.
  - `values` may be a tensor of any element type (`FloatTensor`, `IntTensor`,
    `BoolTensor`, `PcfTensor`, `IntPcfTensor`, `PointCloudTensor`,
    `DistanceMatrixTensor`, `SymmetricMatrixTensor`, `BarcodeTensor`) or a
    numeric array-like (float32 stays float32, booleans become bool, other
    numbers become float64). Axis 0 is time. Anything else raises
    `TypeError`, including strings, `NestedTensor`, and sequences containing
    stablebear objects, which `np.asarray` would silently turn into numeric
    arrays.
  - `TimeSeries` is generic (`TimeSeries[PointCloud]`) in Python
    3.10-compatible annotations.
  - An empty list gives an empty float64 channel, so a typed empty channel of
    point clouds is built from `sb.zeros((0,), dtype=sb.pcloud64)`.

- [ ] **8. Derived channels are lazy.** (decision) Recommendation: delay
  embeddings and windows store no values. A derived channel holds its source
  channel and parameters, exposes the same interface as a stored channel
  except recorded observations, and evaluates its source when asked. A window
  of a series cuts every channel at the same `[a, b)` in the query unit and
  works for any element type; its value at `a` comes from the source even
  when the last observation lies before the window, and its `times` are the
  source's timestamps inside `[a, b)`. An embedding requires numeric channels,
  takes its delay in the query unit, and has no `times`. Internal lookups skip
  the domain check, because shifted and converted times may round just
  outside it. Alternative: materialize derived channels at the union of the
  times where any component changes, which costs memory and loses the
  original timestamps.

- [ ] **9. Saving series.** (decision) Recommendation: save a series as a
  tree. Leaves are stored channels of any element type (a native variant over
  `Channel<T>`); nodes record derived channels (kind, parameters), channel
  names, and the query unit. A loaded tree evaluates as the original without
  the original objects. Alternative: materialize derived channels on saving,
  which brings back the union of time grids that decision 8 avoids.

## Stored channels

- [ ] **10. `Times`.** `sb::Times` in `include/sbear/timeseries.hpp` validates
  its timestamps and unit and converts units. The new `stablebear/timeseries/`
  package wraps it with `values` (a read-only float64 array), `unit`, `len()`,
  `==`, `repr`, and `to(unit)`. Tests:
  - `Times([0.0, 1.5, 4.0], unit="ms")` has float64 values
    `[0.0, 1.5, 4.0]` and unit `"ms"`. **Catches:** a dropped or replaced
    unit.
  - Invalid input raises `ValueError`, parametrized over the timestamps
    `[0.0, 0.0]`, `[1.0, 0.0]`, `[0.0, nan]`, `[0.0, inf]`, and
    `[[0.0, 1.0]]`, and over `unit="sec"` with valid timestamps.
    **Catches:** duplicate, decreasing, nonfinite, or 2-D timestamps
    accepted; unknown units accepted.
  - Assigning `times.values[0] = 5.0` raises `ValueError`. **Catches:**
    timestamps that can be reordered after validation.
  - Parametrized: `Times([0.0, 1.5], unit="min").to("s")` gives
    `[0.0, 90.0]`; `Times([0.0, 1.5], unit="fs").to("as")` gives
    `[0.0, 1500.0]`; `Times([0.0, 2.25], unit="as").to("fs")` gives
    `[0.0, 0.00225]`. Each result has the target unit. **Catches:** an
    inverted factor, a rounded factor such as `1e-3` (which gives
    `0.0022500000000000003`), the unit label not updated.

- [ ] **11. Numeric one-channel series.** `sb::Channel<T>` with its
  validation, `register_timeseries_element` for float32 and float64, and
  Python `TimeSeries(values, times)` with `times`, `unit`, `values`, `dtype`,
  `shape`, `value_shape`, `len()`, and `repr`. This is the largest item.
  Tests:
  - Plain timestamps mean seconds, and value axes are kept: values
    `[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]` with
    `times=[0.0, 1.5, 4.0]` give `unit == "s"`, `shape == (3, 2, 2)`,
    `value_shape == (2, 2)`, and the same values. This test checks the
    default unit. **Catches:** a missing default unit, flattened value axes.
  - An explicit unit is kept:
    `TimeSeries([1.0, 2.0], times=Times([0.0, 0.5], unit="ms"))` has unit
    `"ms"`. **Catches:** `Times` reduced to a plain array.
  - One timestamp per observation: values of shape `(3, 2)` with 2 or 6
    timestamps raise `ValueError`. **Catches:** counting scalar entries
    instead of the leading axis, or no check.
  - Value precision: float32 values give `sb.float32`; float64 and int64
    values give `sb.float64`. **Catches:** float32 widened, integer values
    stored as integers.

- [ ] **12. Every element type.** Register the remaining 17 element types and
  dispatch on the tensor type in Python. Test, parametrized over the 19
  dtypes: a two-observation tensor whose elements differ (for objects, also in
  size: clouds of 1 and 2 points, 2x2 and 3x3 matrices, barcodes with 1 and 2
  bars, PCFs with 2 and 3 breakpoints) with `times=[0.0, 1.0]` gives a series
  with the tensor's `dtype`, `shape == (2,)`, and values equal to the input.
  **Catches:** a type bound to the wrong native class or precision, elements
  of different sizes rejected.

- [ ] **13. (optional) Indexed tensors are copied.** A point-cloud tensor that
  holds a selection (indexed with `sb.indices`) gives a series whose
  `values.indices is None`. **Catches:** an indexed tensor passed to a binding
  that expects ordinary storage.

- [ ] **14. Packed tables.** With `times` omitted, a 2-D table holds
  timestamps in seconds in column 0 and values in the remaining columns. Tests
  with the issue's table
  `data = np.array([[0.0, 10.0, 20.0], [1.3, 11.0, 21.0], [4.0, 12.0, 22.0]])`:
  - `TimeSeries(data)` has the same times, unit, and values as
    `TimeSeries(data[:, 1:], times=data[:, 0])`: times `[0.0, 1.3, 4.0]` and
    values of shape `(3, 2)`. **Catches:** timestamps left among the values,
    row numbers used as times.
  - `TimeSeries(data, times=[0.0, 1.0, 2.0])` keeps all three columns as
    values. **Catches:** column 0 removed even though times were given.
  - Parametrized `ValueError`: `[[0.0], [1.0]]` (no value column),
    `[1.0, 2.0]` (1-D), `np.zeros((2, 2, 2))` (3-D), and
    `[[1.0, 10.0], [0.0, 11.0]]` (decreasing timestamps in column 0).
    **Catches:** an empty value array, times inferred from other ranks,
    column-0 timestamps that skip validation.

- [ ] **15. Unsupported values.** Test, parametrized `TypeError`: a list of
  two equal-size `sb.PointCloud`s, `np.array(["a", "b"])`, and a
  `NestedTensor`. **Catches:** object sequences silently converted to numeric
  arrays, strings or nested tensors passed to the native constructor.

- [ ] **16. Observations and ownership.** `series.observation(i)` returns a
  copy of recorded observation `i`: a scalar for scalar values, otherwise the
  element or tensor type of the values (`PointCloud`, `BarcodeTensor`, ...).
  Tests with values `[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]` and
  `times=[0.0, 1.0, 2.0]`:
  - `observation(1)` is `[3.0, 4.0]`, `observation(-1)` is `[5.0, 6.0]`, and
    `observation(3)` raises `IndexError`. **Catches:** indexing along a value
    axis, negative indices counted from the wrong end, no bounds check.
  - After `obs = series.observation(0)` and `obs[0] = 99.0`,
    `obs[0] == 99.0` and `series.observation(0)` is still `[1.0, 2.0]`.
    **Catches:** observations returned as views into the series (decision 4).

- [ ] **17. Evaluation.** `series.evaluate(t)` and `series[t]` return the
  previous value (decision 1), as `observation(i)` would. Tests with times
  `[0.0, 1.5, 4.0]` and values `[10.0, 20.0, 30.0]`:
  - `series[t]` at 0.0, 1.0, 1.5, 3.75, and 4.0 gives 10, 10, 20, 20, and 30,
    and `series[1]` (an int) gives 10, the value at time 1. **Catches:**
    next-value, nearest, or linear interpolation; an open end; an int read as
    a position.
  - `domain == (0.0, 4.0)`, and `series[-0.5]` and `series[4.5]` raise
    `ValueError`. **Catches:** extrapolation outside the domain, NaN instead
    of an error.

- [ ] **18. Empty and singleton series.** Tests:
  - `TimeSeries(sb.zeros((0,), dtype=sb.pcloud64), times=[])` has
    `len() == 0`, dtype pcloud64, and `domain is None`, and `series[0.0]`
    raises `ValueError`. **Catches:** an invented domain, a crash on empty
    storage.
  - `TimeSeries([7.0], times=[2.0])` has domain `(2.0, 2.0)` and
    `series[2.0] == 7.0`. **Catches:** a singleton series rejected or
    impossible to evaluate.

- [ ] **19. Snapping.** Implement decision 2 in the shared lookup. Test with
  times `[0.0, 0.25, 0.5]` and values `[1.0, 2.0, 3.0]`:
  `series[np.nextafter(0.25, 0.0)]` is 2.0 and `series[0.25 - 1e-6]` is 1.0.
  **Catches:** no snapping, or a tolerance wide enough to change values at
  ordinary times.

- [ ] **20. Query unit.** `series.to(unit)` returns the series with another
  query unit; the stored channels are unchanged. Test:
  `TimeSeries([10.0, 20.0], times=Times([0.0, 0.5], unit="s")).to("ms")` has
  unit `"ms"`, times `[0.0, 500.0]`, domain `(0.0, 500.0)`, and `[250.0]`
  equal to 10. **Catches:** queries or the domain left in the old unit.

## Several channels

- [ ] **21. Building multi-channel series.** `TimeSeries.from_channels`
  (decision 7). Tests:
  - `from_channels({"a": a, "cam": clouds})`, where `clouds` is a point-cloud
    series in seconds, has `channel_names == ("a", "cam")`, unit `"s"`, and
    `series["cam"].dtype` equal to the clouds' dtype; a list `[a, clouds]`
    gives the names `("0", "1")`. **Catches:** channel order or names lost,
    channels of different types rejected.
  - Parametrized `ValueError`: channels A and B without `unit=`, an empty
    dict, and a multi-channel series as a channel. **Catches:** a silently
    chosen unit, an empty series, nested channel sets.

- [ ] **22. Domain and times of several channels.** Test with
  `from_channels({"a": a, "b": b}, unit="s")`: domain `(0.5, 2.5)`;
  `series[0.25]` raises `ValueError`; `times` is a `NestedTensor` whose leaves
  are `[0.0, 1.0, 2.5]` and `[0.5, 1.5, 3.0]`. The end is closed only if every
  channel ending there is closed. **Catches:** the union of domains,
  timestamps reported in each channel's own unit.

- [ ] **23. Evaluating several channels.** Test with the series of item 22:
  `series[1.25] == {"a": 20.0, "b": -1.0}`, `series[1.5] == {"a": 20.0,
  "b": -2.0}`, and the keys come in channel order. **Catches:** channel B read
  in the wrong unit (it would answer at 1.25 ms), one grid assumed for all
  channels, channel order lost.

- [ ] **24. Selecting channels.** Tests with the series of item 22:
  - `series["b"]` is a one-channel series with unit `"s"` and
    `series["b"][1.5] == -2.0`; `series[["b", "a"]]` has
    `channel_names == ("b", "a")`. **Catches:** the query unit lost on
    selection, selection order ignored.
  - `series.values` and `series.observation(0)` raise `ValueError`, and
    `series["c"]` raises `KeyError`. **Catches:** per-channel accessors that
    silently pick a channel.

## Serialization of stored series

- [ ] **25. Save, load, and pickle one channel.** `include/sbear/io/timeseries_io.hpp`
  writes a stored channel: unit, timestamps, domain (with its closedness, or
  none), `value_shape`, then the elements with the existing element writers,
  under the element type's format with `TensorFormatFlag::TimeSeries`
  (decision 5). Register it in `StreamableObject` and `read_any_object`;
  loading goes through the validating constructor. No `FormatVersion` bump is
  needed; older readers report an unhandled object type. In Python,
  `TimeSeries` uses `_BinaryIoMixin`, and `save` and `load` dispatch it. Test,
  parametrized over the 19 dtypes and over `save`/`load` and `pickle`: item
  12's series with `Times(..., unit="ms")` loads with the same `dtype`,
  times, unit, domain, and values. **Catches:** a type flag mapped to the
  wrong type or precision, a dropped unit or domain, timestamps narrowed to
  32 bits.

- [ ] **26. Saving several channels.** The series tree of decision 9 for
  stored channels. Test, parametrized over `save`/`load` and `pickle`: item
  21's series of `a` and `clouds` loads with the same channel names, unit,
  channel dtypes, and values at 0.0 and 1.0. **Catches:** names, order, or the
  query unit lost; a channel loaded as another type.

## Snapshot persistence (workflow 1)

Items 27-35 use three clouds with different point counts and different
barcodes: `[[0, 0], [1, 0]]`, the unit square
`[[0, 0], [1, 0], [1, 1], [0, 1]]` (one degree-1 bar), and
`[[0, 0], [2, 0], [5, 0]]`, with `times=Times([0.0, 1.3, 4.0], unit="ms")`.

- [ ] **27. `SeriesTask` and point-cloud persistence.** Add `sb::SeriesTask`
  (decision 6) in `include/sbear/timeseries/series_task.hpp`, a
  `make_ripser_series_task` overload for point-cloud channels, and bindings in
  `src/python/persistence/py_ripser.cpp`, with each task class exposing
  `result()`. The registered task base is not the first base, so the
  pybind11 classes need `py::multiple_inheritance()`.
  `compute_persistent_homology` forwards a `TimeSeries`, keeping the existing
  complex and distance checks; update its signature and docstring. Tests:
  - Parametrized over pcloud32 and pcloud64: the result has the input's times
    and unit, `shape == (3, 2)`, and the matching barcode dtype, and each
    `result.observation(i)` is isomorphic to
    `compute_persistent_homology(clouds[i], max_dim=1)`. **Catches:** lost or
    permuted times, a dropped unit, a flattened homology axis, snapshots
    paired with the wrong barcodes.
  - A series of one cloud gives `shape == (1, 2)`. **Catches:** the existing
    singleton squeeze applied to series.

- [ ] **28. Several values per observation.** Test: a point-cloud channel
  whose values have shape `(3, 2)` (two clouds per time) gives
  `value_shape == (2, 2)`, and its values are isomorphic to
  `compute_persistent_homology` of the `(3, 2)` `PointCloudTensor`.
  **Catches:** value axes dropped or mixed with the time axis.

- [ ] **29. Distance-matrix persistence.** The distance-matrix overload and
  binding. Test, parametrized over distmat32 and distmat64: a series of a 2x2
  and a 3x3 matrix, built with `sb.zeros((2,), dtype=...)` and element
  assignment, keeps its times, and each observation is isomorphic to
  `compute_persistent_homology(matrix, max_dim=1)`. **Catches:** distance
  matrices sent down the point-cloud path or to the wrong precision, matrices
  of different sizes rejected.

- [ ] **30. Typed empty series.** Test: an empty pcloud32 series gives a
  barcode32 series of shape `(0, 2)` for `max_dim=1`, with unit `"s"`.
  **Catches:** an untyped or wrongly shaped empty result, failure on empty
  input.

- [ ] **31. Keyword data argument and parameter forwarding.** Test:
  `compute_persistent_homology(X=series, max_dim=0, reduced=True)` has shape
  `(3, 1)` and no infinite bar. **Catches:** `max_dim` or `reduced` dropped on
  the series path, keyword calls failing.

- [ ] **32. Persistence of mixed channels.** One task per element type across
  channels (decision 6). Test: `from_channels({"cam": clouds, "dm": matrices})`
  with item 29's matrices at `Times([0.5, 2.0], unit="ms")` gives a series with
  the same channel names, each channel with its input's times, and each
  channel equal to the persistence of that channel alone. **Catches:**
  channels swapped or merged, results split at the wrong offsets, one
  channel's times given to another.

- [ ] **33. Unsupported channels.** Test, parametrized `TypeError`:
  `compute_persistent_homology` of a numeric series, of a barcode series, and
  of `from_channels({"cam": clouds, "temp": numeric})` (the message names
  `"temp"`), and `barcode_to_stable_rank` of a point-cloud series.
  **Catches:** a numeric series silently read as point clouds, an unsupported
  channel passed to a native task or silently dropped.

- [ ] **34. Stable rank of a series.** A `make_stable_rank_series_task`
  overload, its binding in `src/python/persistence/py_barcode_summary.cpp`,
  and a series branch in `_barcode_to_pcf`. Test: a barcode series of shape
  `(2, 2)` with four distinct barcodes gives a PCF series with the same times
  and shape whose values array-equal `barcode_to_stable_rank` of the input
  `BarcodeTensor`. **Catches:** times dropped, cells mixed up, the input
  released before the task finishes.

- [ ] **35. Betti curves and accumulated persistence.** The other two
  overloads and bindings, forwarding `max_death`. Tests:
  - Parametrize item 34's test over the three summary functions.
    **Catches:** a summary bound to another summary's task.
  - `barcode_to_accumulated_persistence(series, max_death=2.5)` array-equals
    the function applied to the `BarcodeTensor` with `max_death=2.5`, and
    differs from the result without `max_death`. **Catches:** `max_death`
    dropped on the series path.

- [ ] **36. Progress, errors, and the GIL.** (existing) The series path runs
  through `_run_task`, so the existing progress, error-propagation, and
  GIL-release tests (`test_gil_release.py`, `test_async_task.py`) cover it,
  and the existing persistence tests cover calls without series. Confirm; add
  no test.

- [ ] **37. (optional) CPU limits.** Persistence of the three-cloud series
  under `sb.system.limit_cpus(1)` equals the result with all threads; restore
  the limit afterwards. **Catches:** results that depend on scheduling.

## Delay embeddings, windows, and sampling (workflow 2)

Items 42-45 use the source series with times `[0.0, 0.5, 1.5, 2.5, 4.0]` and
values `[1.0, 2.0, 3.0, 4.0, 5.0]`.

- [ ] **38. `PointCloudTensor` from a `FloatTensor`.** Accept a `FloatTensor`
  whose last two axes are points and coordinates, as an ndarray already is.
  Test in `test/python/test_point_cloud_tensors.py`, parametrized over float32
  and float64: a `FloatTensor` of shape `(2, 3, 2)` gives a tensor of shape
  `(2,)` with the matching pcloud dtype, and `np.asarray(clouds[1])` equals
  the second `(3, 2)` block. **Catches:** the current `TypeError`, the wrong
  precision.

- [ ] **39. Delay embedding.** `TimeDelayEmbedding(dimension, delay)` returns a
  lazy series (decision 8) whose value at `t` is
  `(x(t - (m - 1) * delay), ..., x(t))`, so the value shape becomes
  `(m,) + value_shape`. The domain starts at `start + (m - 1) * delay` and
  keeps the source's end and its closedness. Tests with times
  `Times([0.0, 1.0, 2.5, 4.0], unit="ms")`, values
  `[10.0, 20.0, 30.0, 40.0]`, `dimension=3`, and `delay=1.0`:
  - The series at 2.0, 2.75, 3.25, 3.75, and 4.0 gives `[10, 20, 20]`,
    `[10, 20, 30]`, `[20, 20, 30]`, `[20, 30, 30]`, and `[20, 30, 40]`. The
    values at 3.25 and 3.75 differ from the value at the last observation
    time, 2.5. **Catches:** newest-first order, the same delay for every
    component, values held from the last observation time.
  - The embedded series has domain `(2.0, 4.0)`, unit `"ms"`, and
    `value_shape == (3,)`, and `[1.5]` raises `ValueError`. **Catches:**
    components that refer to times before the series starts.

- [ ] **40. Vector values and embedding errors.** Tests:
  - Times `[0.0, 1.0, 2.0]`, values `[[1, -1], [2, -2], [3, -3]]`,
    `dimension=2`, and `delay=1.0` give `value_shape == (2, 2)` and
    `[2.0] == [[2, -2], [3, -3]]`. **Catches:** value axes flattened or
    transposed.
  - Parametrized: `dimension=0`, `delay=0.0`, and `delay=-1.0` raise
    `ValueError`; `dimension=3` and `delay=1.0` on times `[0.0, 1.0]` raise
    `ValueError` (too short); a barcode series raises `TypeError`.
    **Catches:** invalid parameters accepted, an empty domain created
    silently, object values embedded.

- [ ] **41. Embedding several channels.** Each channel is embedded with the
  delay converted from the query unit into its own unit. Test:
  `TimeDelayEmbedding(dimension=2, delay=0.5)` of the series of item 22 has
  domain `(1.0, 2.5)` and gives `{"a": [20, 20], "b": [-1, -2]}` at 1.5.
  **Catches:** the delay applied in channel B's milliseconds.

- [ ] **42. Window bounds.** `Window(size, step=None, overlap=None)` returns a
  list of lazy start-aligned windows `[a, a + size)` with
  `a = start + k * step` while `a + size <= end`, where
  `step = size - overlap` and the default step is `size`. Incomplete windows
  are dropped. Test with the source series, parametrized over the arguments
  and the window domains: `size=2.0, step=1.0` and `size=2.0, overlap=1.0`
  give `[(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)]`; `size=2.0` gives
  `[(0.0, 2.0), (2.0, 4.0)]`; `size=5.0` gives `[]`. **Catches:** windows
  anchored at the end, partial windows kept, the overlap used as the step, an
  overlapping default.

- [ ] **43. Window contents and time slicing.** `series[a:b]` is the window
  `[a, b)`, the same object `Window` produces. Tests with the source series:
  - `w = series[1.0:3.0]` has times `[1.5, 2.5]` (original timestamps),
    `w[1.0] == 2.0` (the observation at 0.5), and `w[2.75] == 4.0`.
    **Catches:** the value before the first observation in the window missing,
    timestamps moved to the window start.
  - `w[3.0]` raises `ValueError`, and `series[1.0:3.0:0.5]` raises
    `ValueError`. **Catches:** a closed window end, a step silently ignored.

- [ ] **44. Window parameter errors.** Test, parametrized `ValueError`:
  `size=0.0`, `size=-1.0`, `step=0.0`, `overlap=2.0` and `overlap=-0.5` with
  `size=2.0`, and `step` together with `overlap`. **Catches:** windows that
  never advance, conflicting parameters accepted.

- [ ] **45. Embedding a window.** Test: `TimeDelayEmbedding(dimension=2,
  delay=1.0)` of `series[1.0:3.0]` has domain `(2.0, 3.0)`; `[2.0]` is
  `[2, 3]` (with the value at 1.0 from the source) and `[2.75]` is `[3, 4]`;
  `[1.5]` and `[3.0]` raise `ValueError`. **Catches:** history taken from
  before the window, the half-open end lost.

- [ ] **46. Windows of several channels.** Test:
  `Window(size=1.0, step=1.0)` of the series of item 22 gives windows with
  domains `(0.5, 1.5)` and `(1.5, 2.5)`; the second gives
  `{"a": 20.0, "b": -2.0}` at 1.5 and has `times` leaves `[]` and `[1.5]`.
  **Catches:** channels cut at different bounds, channel B cut in
  milliseconds.

- [ ] **47. Sampler query times.** `RegularSampler(n_points).times(series)`
  returns `Times` with `a + (b - a) * k / n_points` for
  `k = 0, ..., n_points - 1`, in the series' unit. The end is never sampled,
  even for a closed domain. Test: `RegularSampler(n_points=4).times(w)` for
  `w = series[1.0:3.0]` gives `[1.0, 1.5, 2.0, 2.5]` with the window's unit.
  **Catches:** the end point included, spacing `(b - a) / (n - 1)`.

- [ ] **48. Sampling one channel.** `RegularSampler(n_points)(windows)`
  evaluates each series at its query times and returns a `FloatTensor` of
  shape `(len(windows), n_points, *value_shape)`. Test: the three windows of
  `Window(size=2.0, step=1.0)` with `n_points=2` give
  `[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]`. **Catches:** windows stacked out of
  order, values at window starts missing.

- [ ] **49. Sampling several channels.** Numeric channels with equal
  `value_shape` sample to shape
  `(len(windows), n_points, n_channels, *value_shape)`. Test: item 46's windows
  with `n_points=2` give `[[[10, -1], [20, -1]], [[20, -2], [20, -2]]]`.
  **Catches:** the channel axis placed elsewhere, channels evaluated on one
  grid.

- [ ] **50. Sampler errors.** Test, parametrized: `n_points=0`, an empty list,
  and channels with different value shapes raise `ValueError`; a point-cloud
  series raises `TypeError`. **Catches:** an empty result without a value
  shape, ragged stacking, object values sampled into a `FloatTensor`.

- [ ] **51. The delay workflow end to end.** Test in
  `test/python/test_integration.py`: times `[0.0, 0.5, ..., 3.5]` and values
  `[0, 1, 0, -1, 0, 1, 0, -1]` through
  `TimeDelayEmbedding(dimension=2, delay=0.5)`,
  `Window(size=1.0, step=1.0)` (three windows),
  `RegularSampler(n_points=4)`, `PointCloudTensor`, and
  `compute_persistent_homology(..., max_dim=1)` give samples of shape
  `(3, 4, 2)`, three clouds, and barcodes of shape `(3, 2)`. **Catches:**
  steps that do not compose.

- [ ] **52. Saving derived series.** Derived-channel nodes in the tree of
  decision 9. Test, parametrized over `save`/`load` and `pickle`: the window
  `[1.5, 2.5)` of item 41's embedding loads with the same channel names,
  domain, unit, and values at 1.5 and 2.0, and `[2.5]` still raises
  `ValueError`. **Catches:** a node kind, parameter, or query unit lost on
  saving, the half-open end lost.

- [ ] **53. Golden files.** Pin both formats (TESTING.md principle 11) with a
  generator modeled on `test/golden/serialization/generate_point_cloud_v3.py`,
  which records explicit expected values. Write a barcode32 one-channel series
  of shape `(2, 2)`, item 26's two-channel series, and item 52's derived
  series as `.sb` and pickle files, and teach `test_serialization_goldens.py`
  to compare series (channel names, unit, domain, values at listed times).
  **Catches:** a later change to a payload that breaks saved files.

## Documentation and release

Code items update the docstrings they touch (AGENTS.md). These items add the
pages.

- [ ] **54. API reference.** Add `docs/api_timeseries.rst` to the toctree in
  `docs/stablebear.rst`. Check that the docstrings state the contracts of
  decisions 1-4 and 7-8: domains and their closedness, previous-value
  evaluation and snapping, units, query units and conversion, channels and
  time indexing, copies and lazy derived channels, and the meaning of each
  axis. Check that `compute_persistent_homology`, the three summaries,
  `save`, and `load` document series input and output.

- [ ] **55. User guide.** A `docs/timeseries.rst` guide linked from
  `docs/userguide.rst`: times, units, domains, and channels, then two small
  examples. The first follows a point-cloud series through persistence to
  stable ranks; the second combines two numeric channels in different units
  and takes them through a delay embedding, windows, and sampling to
  persistence. Link each operation to its API entry on first mention. Figures
  are not needed; if one is added, follow the dropdown rule in AGENTS.md.

- [ ] **56. Changelog.** A New features entry in `CHANGELOG.md` linking #242,
  under the heading of the release that ships it (to be decided; 0.5.1 is in
  progress).

## Final checks

- [ ] **57. Test run.** After a CMake build and install, run the new test
  files, then the full Python suite and `sb_test` from `test/`.
- [ ] **58. Docs build.** Run `make html` in `docs/` (with `LC_ALL=C.UTF-8` if
  the locale is missing), check for warnings, and read the new pages.
- [ ] **59. Push.** Push `issue-242-timeseries`, open a PR against `main`, and
  remove this plan before merging.

## Coverage of the acceptance criteria

| Criterion in #242 | Items |
| --- | --- |
| Time, domain, evaluation, shape, and ownership contracts; empty and singleton behavior | 1-9, 18 |
| `Times`, native storage and access, default seconds, explicit units, one timestamp per observation | 10-12, 16-17, 19-20 |
| Floating-point arrays with explicit times or a packed table | 11, 14-15 |
| Native operation contract, dispatch, and adapter; calls, parameters, CPU limits | 27, 31-32, 36-37 |
| Persistence and barcode summaries of series | 27-35 |
| Delay embedding and windows | 39-46 |
| Regular sampling into `PointCloudTensor` and through persistence | 38, 47-51 |
| Save, load, and pickle for both workflows | 25-26, 52-53 |
| Tests and documented examples | all test items, 54-55 |
| Channels with their own types, grids, and units (added) | 3, 20-24, 32, 41, 46, 49 |

## Out of scope

As in the issue: datetime timestamps, sample-count delays and windows, other
samplers, interpolation, partial windows, gap policies, tensors of series,
reductions over time, provenance, and normalization. Also deferred here:
`NestedTensor` values, `Tensor<TimeSeries>`, views instead of copies for
stored channels, evaluation in C++, and the branch's notebooks and
scikit-learn transformers.
