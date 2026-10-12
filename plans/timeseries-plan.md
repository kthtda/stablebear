# Time series (#242): implementation plan

This plan implements #242 on `issue-242-timeseries`, branched from `main` at
`05ce5ba4a`: unit-bearing timestamps, time series made of channels with their
own time grids and units, persistence of snapshot series (workflow 1), and
windowed delay embeddings sampled into point clouds (workflow 2), with
serialization and documentation. Useful parts of the `origin/timeseries`
branch are recycled where they fit (see "Recycling").

Where the plan departs from the issue:

- Channels with their own time grids and units are in this release; the issue
  defers a channel-combination API.
- Every channel holds one observation per time, a tensor of a fixed `shape`
  (often `()`). Persistence of a channel `"cam"`
  returns one channel per homology degree (`"cam/H0"`, `"cam/H1"`) instead of
  a value axis of length `max_dim + 1`.
- A delay embedding adds a leading delay axis to each observation, and the
  sampler flattens observations into points and returns a
  `PointCloudTensor` directly, so `PointCloudTensor` needs no `FloatTensor`
  constructor.

## How to work through this plan

Work through the items in order. Each item depends only on items above it and
is one review unit: a decision, or a small change with at most two or three
focused tests. Keep new work uncommitted for review, and check an item off
only once we have reviewed it. If a decision goes against its recommendation,
revise the affected items before starting them.

Read [TESTING.md](../test/TESTING.md) before writing tests and
[WRITING.md](../docs/WRITING.md) before writing documentation. Build and test
commands are in [AGENTS.md](../AGENTS.md#testing). Nearly every item changes
C++, so build and install with CMake before running its tests.

Each test item gives its input, the expected result, and **Catches:**, the bug
that makes the test fail. Tags: **(decision)** a choice to make before the
items that follow; **(existing)** confirm existing coverage without adding a
test; **(optional)** safe to skip without losing important coverage. Check off
a skipped optional item with the note "skipped".

**Test budget.** The items add about 55 test functions, many of them
parametrized, plus two optional ones. Keep that number from growing. Before
adding a test that is not listed, check whether a listed test's
parametrization covers it, and prefer leaving a guarantee to the design and
its docstring over testing every combination. Do not test what the public API
cannot observe. No C++ unit tests are planned: every guarantee here is
observable from Python (TESTING.md principle 4).

## Conventions

- **Files.** Source: `include/sbear/timeseries.hpp` (`Times`, channels,
  `TimeSeries<T>`), `include/sbear/timeseries/derived.hpp` (windows,
  embeddings, sampling), `include/sbear/timeseries/series_task.hpp` (analysis
  adapter), `include/sbear/io/timeseries_io.hpp`,
  `src/python/py_timeseries.{hpp,cpp}`, and the package
  `stablebear/timeseries/` with thin wrappers. Tests:
  `test/python/test_timeseries.py` (storage, access, evaluation, units,
  channels, serialization), `test/python/test_timeseries_derived.py`
  (embedding, windows, sampling), and
  `test/python/persistence/test_timeseries_persistence.py` (workflow 1). Items
  that touch another file name it.
- **Data.** Use timestamps, delays, and window sizes that are exact in binary
  floating point (multiples of 0.25), so expected values do not depend on
  rounding; item 20 tests rounding on purpose. Give observations distinct
  values (10, 20, 30, ...) so that mix-ups show.
- **Channels A and B.** Several items use channel A, with times
  `Times([0.0, 1.0, 2.5], unit="s")` and values `[10.0, 20.0, 30.0]`, and
  channel B, with times `Times([500.0, 1500.0, 3000.0], unit="ms")` and values
  `[-1.0, -2.0, -3.0]`, combined as
  `TimeSeries.from_channels({"a": a, "b": b}, unit="s")`.
- **Clouds.** Workflow 1 uses three clouds with different point counts and
  different barcodes: `[[0, 0], [1, 0]]`, the unit square
  `[[0, 0], [1, 0], [1, 1], [0, 1]]` (one degree-1 bar), and
  `[[0, 0], [2, 0], [5, 0]]`, with `times=Times([0.0, 1.3, 4.0], unit="ms")`.
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
- A stored channel holds a shared `Times`, a flat `std::vector<T>`, a
  `shape`, and a domain. Each observation is a tensor of shape `shape`
  (`()` for one element per time, `(d,)` for a point, `(h, w)` for an image),
  stored as `n * prod(shape)` values and presented as a tensor only when
  accessed. Point counts, matrix sizes, and bar counts are internal to each
  element and may change over time.
- `sb::TimeSeries<T>` is an ordered set of named channels of one element type
  `T` and a query unit. Its domain is the intersection of the channels'
  domains. Like `NestedTensor`, one registration template binds it for every
  tensor element type in the library (19 types).
- Everything runs in C++: lookups, snapping, domain checks, unit conversion,
  windows, delay embeddings, sampling, and analysis. Python only binds
  arguments and wraps results.
- `series[t]` looks up each channel on its own grid, in its own unit, and
  returns the value for one channel or an ordered dict for several.
  `series[a:b]` is a window, and `series["name"]` selects channels.
- Windows and delay embeddings are lazy channels: native nodes that hold
  their source and parameters and compute values when evaluated. A delay
  embedding of a numeric channel of shape `S` has shape `(m, *S)`.
- `RegularSampler` samples each window of each numeric channel into a point
  cloud, flattening each observation into a point of dimension
  `prod(shape)`, and returns a `PointCloudTensor` of shape
  `(n_windows, n_channels)`.
- `sb::SeriesTask` runs an existing batch task (`RipserTask`,
  `RipserDistMatTask`, `BarcodeSummaryTask`) once over all channels of a
  series and returns a series with the inputs' times and domains.

## Recycling

The `origin/timeseries` branch predates the rename (`mpcf`/`masspcf` became
`sb`/`stablebear`). Read its files with `git show origin/timeseries:<path>`
and copy ideas and small blocks, not whole files.

- `include/mpcf/timeseries.hpp`: `snap_in_place` and `default_snap_tol` for
  item 20, `find_interval` for the previous-value lookup, and its row-major
  `m_values` with `n_channels` for the flat layout (items 11-12). The
  offsets from `start_time` and `time_step` and the interpolation variants do
  not fit.
- `include/mpcf/algorithms/embed_time_delay.hpp`: the evaluation loop of
  `embed_time_delay_range` (oldest-to-newest components, valid start
  `start + (dimension - 1) * delay`) for the lazy embedding (items 38-40).
  Not its windows, which are anchored at the end and keep a partial first
  window, and not its tuples, which it stores only at original observation
  times; the issue rules out both.
- `include/mpcf/io/timeseries_io.hpp`: the outline of the writer and reader
  for items 26-27 and 51, with new payloads.
- `src/python/py_timeseries.cpp`: the array-copy and length-check patterns for
  items 10-15.
- `masspcf/persistence/homology.py` (the `TensorTimeSeries` branch): one
  batched Ripser call over all snapshots. Item 28 keeps the idea and moves it
  into native code.
- `docs/ts_*.rst`: background wording for item 54; the API differs.

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
  defaults to `t_{n-1}`.
  With either option, `domain` is a `(start, end)` tuple, and an empty channel
  has no domain (`domain is None`) and cannot be evaluated.

- [x] **2. Floating-point timestamps.** (decision: (a), snap within `1e-9 * max(1, |t|)`, no user parameter) Delays
  and window starts are computed in floating point, so a query can land just
  before the observation it should reach: `0.3 - 0.1` is less than `0.2`.
  Options:
  (a) Snap a query to an observation that lies above it by at most
  `1e-9 * max(1, |t|)`, as `snap_in_place` does on the branch. No user
  parameter; domain checks stay exact.
  (b) No snapping; document that times and delays should be exactly
  representable.

- [x] **3. Time units.** (decision: (a), fixed set `as` through `d` with exact integer conversion factors) `Times(values, unit="s")`; plain timestamp
  arrays mean seconds. Options:
  (a) A fixed set covering physics time scales: `"as"`, `"fs"`, `"ps"`,
  `"ns"`, `"us"`, `"ms"`, `"s"`, `"min"`, `"h"`, and `"d"`. Store each unit as
  a power of ten times an integer (1 or 60, 3600, 86400), and convert by
  multiplying or dividing by an exact integer factor rather than by a
  rounded one such as `1e-15`; dividing by an exact factor rounds once.
  Adding `"zs"` or `"ys"` later is one table entry. `Times.to(unit)`
  converts, and a series converts each query from its query unit into each
  channel's unit. Delays, window sizes, and query times are in the series'
  query unit.
  (b) Any nonempty string, with no conversion.

- [x] **4. Ownership.** (decision: (a), copy in and copy out; stored series are immutable) Options:
  (a) Copy in, copy out for stored channels. The constructor copies its
  values, materializing indexed point-cloud and distance-matrix tensors;
  `values`, `observation(i)`, and evaluation return copies; `Times.values` is
  a read-only array. Stored channels are then immutable, so series and
  derived channels can hold references to them without copying them and
  without seeing later changes.
  (b) Views, with documented write-through.

- [ ] **5. Native storage and where the code lives.** (decision)
  Recommendation, from our discussion:
  - All evaluation runs in C++: the previous-value lookup with snapping,
    domain checks, unit conversion, windows, delay embeddings, and sampling.
    Python wrappers only bind arguments and wrap results.
  - A stored channel holds a shared `Times`, a flat `std::vector<T>` (not a
    `Tensor<T>`, so its layout, gathering, and slicing stay under our
    control), a `shape`, and a domain. Each observation is a tensor of shape
    `shape`, stored as `prod(shape)` consecutive values and built as a
    `Tensor<T>` only when accessed. Storing a `Tensor<T>` per observation
    instead would cost 80 bytes and 2 to 4 heap allocations per time point.
    Values are kept apart from times rather than as `(time, value)` pairs,
    because the
    batched analysis needs a contiguous array of values. A `Pcf` cannot hold a
    channel: it is defined on `[0, inf)` and only for arithmetic values.
  - `sb::TimeSeries<T>` holds named channels of one element type. Following
    the `NestedTensor` pattern (`include/sbear/nested_tensor.hpp`,
    `register_nested_tensor_element` in `src/python/py_tensor.cpp`,
    `stablebear/nested_tensor.py`), one `register_timeseries_element<T>` call
    per element type binds it for bool, float32/64, int32/64, uint32/64,
    pcf32/64, pcf32i/64i, pcloud32/64, distmat32/64, symmat32/64, and
    barcode32/64. Python has one `TimeSeries` class whose `dtype` is the
    element dtype, with a dtype-keyed table for the native class.
  - Serialization marks the element type's own format with a new
    `TensorFormatFlag::TimeSeries`, so one writer and reader serve all
    element types, as `TensorFormatFlag::Nested` does.

- [ ] **6. Analysis adapter.** (decision) Recommendation:
  `sb::SeriesTask<TaskT, InT, OutT>` derives from an existing batch task. A
  base declared before the task (base-from-member) holds the input series and
  the output tensor. It concatenates the values of all channels into one
  tensor, so the task runs once over the whole series, and `result()` splits
  the output by channel offsets. Persistence of a channel `"cam"` with
  `max_dim = k` returns channels `"cam/H0"` to `"cam/Hk"` with cam's times and
  domain; the barcode summaries keep channel names. The task supplies
  scheduling, progress, cancellation, and error capture, and the base keeps
  inputs and output alive until completion (`BarcodeSummaryTask` stores only a
  reference to its input). Each operation declares its contract with a
  `make_*_series_task` overload, and pybind11 overloads on the native series
  type select the implementation. A singleton channel keeps its observation
  and an empty channel gives a typed empty result. Alternative: Python spawns
  the existing tasks per channel and rebuilds the series; no new C++, but
  dispatch and result construction stay in Python, against the issue.

- [ ] **7. Public names and accepted values.** (decision) Recommendation:
  - `stablebear.timeseries` exports `Times(values, unit="s")`,
    `TimeSeries(values, times=None)` for one channel (named `"0"`),
    `TimeSeries.from_channels(channels, unit=None)` for several,
    `TimeDelayEmbedding(dimension, delay)` (without the issue's `unit="time"`
    until a second delay mode exists), `Window(size, step=None, overlap=None)`,
    and `RegularSampler(n_points)` with `.times(series)`.
  - `from_channels` takes a dict of names to one-channel series of one element
    type, or a list of them (named `"0"`, `"1"`, ...). `unit` defaults to the
    channels' shared unit and is required when their units differ.
  - A `TimeSeries` exposes `channel_names`, `dtype`, `unit`, `domain`, `times`
    (a `Times` for one channel, a `NestedTensor` with one leaf per channel in
    the query unit for several), `to(unit)`, `evaluate(t)`, `series[t]` (the
    same as `evaluate(t)`; an int `t` is a time), `series[a:b]` (a window),
    and `series["name"]` or `series[["a", "b"]]` (channel selection). A
    one-channel series also exposes `values`, `shape`, `len()`,
    and `observation(i)`; on several channels these raise `ValueError` asking
    for a channel to be selected.
  - For one channel, `evaluate` returns the observation: a scalar or element
    (`PointCloud`, `Barcode`, ...) for `shape == ()`, otherwise a tensor of
    that shape. For several channels it returns an ordered dict keyed by
    channel name.
  - `values` may be a tensor of any element type (`FloatTensor`,
    `IntTensor`, `BoolTensor`, `PcfTensor`, `IntPcfTensor`,
    `PointCloudTensor`, `DistanceMatrixTensor`, `SymmetricMatrixTensor`,
    `BarcodeTensor`) or a numeric array-like (float32 stays float32, booleans
    become bool, other numbers become float64) of shape `(n, *shape)`; axis 0
    is time. Anything else raises `TypeError`: strings, `NestedTensor`, and
    sequences containing stablebear objects, which `np.asarray` would silently
    turn into numeric arrays.
  - `RegularSampler(n_points)(windows)` returns a `PointCloudTensor` of shape
    `(len(windows), n_channels)`; each cloud has `n_points` points of
    dimension `prod(shape)` of its channel. Joining channels into one cloud is
    a later extension.
  - `TimeSeries` is generic (`TimeSeries[PointCloud]`) in Python
    3.10-compatible annotations.
  - An empty list gives an empty float64 channel, so a typed empty channel of
    point clouds is built from `sb.zeros((0,), dtype=sb.pcloud64)`.

- [ ] **8. Derived channels are lazy.** (decision) Recommendation: windows and
  delay embeddings are native nodes that store no values. An embedding of a
  channel of shape `S` has shape `(m, *S)`, newest last. A node holds its
  source channel and parameters, exposes the same interface as a stored
  channel except recorded observations, and evaluates its source when asked.
  A window cuts every channel at the same `[a, b)` in the query unit and works
  for any element type; its value at `a` comes from the source even when the
  last observation lies before the window, and its `times` are the source's
  timestamps inside `[a, b)`. An embedding requires a numeric channel, takes
  its delay in the query unit, and has no `times`. Internal lookups skip the
  domain check, because shifted and converted times may round just outside
  it. Alternative: materialize derived channels at the union of the times
  where any component changes, which costs memory and loses the original
  timestamps.

- [ ] **9. Saving series.** (decision) Recommendation: save a series as a tree
  with its channel names and query unit. Each channel is a stored channel or
  a node (window or embedding, with its parameters) over another channel. A
  loaded tree evaluates as the original without the original objects.
  Alternative: materialize derived channels on saving, which brings back the
  union of time grids that decision 8 avoids.

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

- [ ] **11. Scalar series.** The stored channel and `sb::TimeSeries<T>` with
  their validation, `register_timeseries_element` for float32 and float64,
  and Python `TimeSeries(values, times)` with `channel_names`, `times`,
  `unit`, `values`, `dtype`, `shape`, `len()`, and `repr`. This is the largest
  item. Tests:
  - Plain timestamps mean seconds: values `[10.0, 20.0, 30.0]` with
    `times=[0.0, 1.5, 4.0]` give `unit == "s"`, `channel_names == ("0",)`,
    `shape == ()`, `len() == 3`, and the same values. This test checks the
    defaults. **Catches:** a missing default unit or channel name.
  - An explicit unit is kept:
    `TimeSeries([1.0, 2.0], times=Times([0.0, 0.5], unit="ms"))` has unit
    `"ms"`. **Catches:** `Times` reduced to a plain array.
  - One timestamp per observation: values of shape `(3,)` with 2 or 4
    timestamps raise `ValueError`. **Catches:** a length mismatch accepted.
  - Value precision: float32 values give `sb.float32`; float64 and int64
    values give `sb.float64`. **Catches:** float32 widened, integer values
    stored as integers.

- [ ] **12. Observation shapes.** Values of shape `(n, *shape)` form one
  channel whose observations have shape `shape`. Tests:
  - Values `[[1, -1], [2, -2], [3, -3]]` with `times=[0.0, 1.0, 2.0]` give
    `shape == (2,)`, `len() == 3`, and `series[1.5]` equal to `[2, -2]`.
    **Catches:** points split into channels or counted as observations,
    coordinates transposed.
  - Values `[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]` with `times=[0.0, 1.0]`
    give `shape == (2, 2)` and `series[1.0]` equal to `[[5, 6], [7, 8]]`.
    **Catches:** higher-rank observations flattened or transposed.

- [ ] **13. Every element type.** Register the remaining 17 element types and
  dispatch on the tensor type in Python. Test, parametrized over the 19
  dtypes: a two-observation 1-D tensor whose elements differ (for objects,
  also in size: clouds of 1 and 2 points, 2x2 and 3x3 matrices, barcodes with
  1 and 2 bars, PCFs with 2 and 3 breakpoints) with `times=[0.0, 1.0]` gives a
  series with the tensor's `dtype`, `len() == 2`, and values equal to the
  input. **Catches:** a type bound to the wrong native class or precision,
  elements of different sizes rejected.

- [ ] **14. (optional) Indexed tensors are copied.** A point-cloud tensor that
  holds a selection (indexed with `sb.indices`) gives a series whose
  `values.indices is None`. **Catches:** an indexed tensor passed to a binding
  that expects ordinary storage.

- [ ] **15. Packed tables.** With `times` omitted, a 2-D table holds
  timestamps in seconds in column 0 and the coordinates of one point channel
  in the remaining columns. Tests with the issue's table
  `data = np.array([[0.0, 10.0, 20.0], [1.3, 11.0, 21.0], [4.0, 12.0, 22.0]])`:
  - `TimeSeries(data)` has the same times, unit, and values as
    `TimeSeries(data[:, 1:], times=data[:, 0])`: times `[0.0, 1.3, 4.0]` and
    `shape == (2,)`. **Catches:** timestamps left among the coordinates, row
    numbers used as times.
  - `TimeSeries(data, times=[0.0, 1.0, 2.0])` has `shape == (3,)`.
    **Catches:**
    column 0 removed even though times were given.
  - Parametrized `ValueError`: `[[0.0], [1.0]]` (no value column),
    `[1.0, 2.0]` (1-D), `np.zeros((2, 2, 2))` (3-D), and
    `[[1.0, 10.0], [0.0, 11.0]]` (decreasing timestamps in column 0).
    **Catches:** an empty point, times inferred from other ranks, column-0
    timestamps that skip validation.

- [ ] **16. Unsupported values.** Test, parametrized `TypeError`: a list of
  two equal-size `sb.PointCloud`s, `np.array(["a", "b"])`, and a
  `NestedTensor`. **Catches:** object sequences silently converted to numeric
  arrays, strings or nested tensors passed to the native constructor.

- [ ] **17. Observations and ownership.** `series.observation(i)` returns a
  copy of recorded observation `i`. Tests with values
  `[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]` and `times=[0.0, 1.0, 2.0]`:
  - `observation(1)` is `[3.0, 4.0]`, `observation(-1)` is `[5.0, 6.0]`, and
    `observation(3)` raises `IndexError`. **Catches:** indexing along the
    coordinates, negative indices counted from the wrong end, no bounds check.
  - After `obs = series.observation(0)` and `obs[0] = 99.0`,
    `obs[0] == 99.0` and `series.observation(0)` is still `[1.0, 2.0]`.
    **Catches:** observations returned as views into the series (decision 4).

- [ ] **18. Evaluation.** `series.evaluate(t)` and `series[t]` return the
  previous value (decision 1). Tests with times `[0.0, 1.5, 4.0]` and values
  `[10.0, 20.0, 30.0]`:
  - `series[t]` at 0.0, 1.0, 1.5, 3.75, and 4.0 gives 10, 10, 20, 20, and 30,
    and `series[1]` (an int) gives 10, the value at time 1. **Catches:**
    next-value, nearest, or linear interpolation; an open end; an int read as
    a position.
  - `domain == (0.0, 4.0)`, and `series[-0.5]` and `series[4.5]` raise
    `ValueError`. **Catches:** extrapolation outside the domain, NaN instead
    of an error.

- [ ] **19. Empty and singleton series.** Tests:
  - `TimeSeries(sb.zeros((0,), dtype=sb.pcloud64), times=[])` has
    `len() == 0`, dtype pcloud64, and `domain is None`, and `series[0.0]`
    raises `ValueError`. **Catches:** an invented domain, a crash on empty
    storage.
  - `TimeSeries([7.0], times=[2.0])` has domain `(2.0, 2.0)` and
    `series[2.0] == 7.0`. **Catches:** a singleton series rejected or
    impossible to evaluate.

- [ ] **20. Snapping.** Implement decision 2 in the native lookup. Test with
  times `[0.0, 0.25, 0.5]` and values `[1.0, 2.0, 3.0]`:
  `series[np.nextafter(0.25, 0.0)]` is 2.0 and `series[0.25 - 1e-6]` is 1.0.
  **Catches:** no snapping, or a tolerance wide enough to change values at
  ordinary times.

- [ ] **21. Query unit.** `series.to(unit)` returns the series with another
  query unit; the stored channels are unchanged. Test:
  `TimeSeries([10.0, 20.0], times=Times([0.0, 0.5], unit="s")).to("ms")` has
  unit `"ms"`, times `[0.0, 500.0]`, domain `(0.0, 500.0)`, and `[250.0]`
  equal to 10. **Catches:** queries or the domain left in the old unit.

## Several channels

- [ ] **22. Building multi-channel series.** `TimeSeries.from_channels`
  (decision 7). Tests:
  - `from_channels({"a": a, "b": b}, unit="s")` has
    `channel_names == ("a", "b")` and unit `"s"`, and `from_channels([a, b],
    unit="s")` has the names `("0", "1")`. **Catches:** channel order or
    names lost.
  - Parametrized: channels A and B without `unit=`, an empty dict, and a
    multi-channel series as a channel raise `ValueError`; A together with a
    point-cloud channel raises `TypeError`. **Catches:** a silently chosen
    unit, an empty series, nested channel sets, mixed element types.

- [ ] **23. Domain and times of several channels.** Test with the series of
  channels A and B: domain `(0.5, 2.5)`; `series[0.25]` raises `ValueError`;
  `times` is a `NestedTensor` whose leaves are `[0.0, 1.0, 2.5]` and
  `[0.5, 1.5, 3.0]`. The end is closed only if every channel ending there is
  closed. **Catches:** the union of domains, timestamps reported in each
  channel's own unit.

- [ ] **24. Evaluating several channels.** Test with the series of channels A
  and B: `series[1.25] == {"a": 20.0, "b": -1.0}`,
  `series[1.5] == {"a": 20.0, "b": -2.0}`, and the keys come in channel order.
  **Catches:** channel B read in the wrong unit (it would answer at 1.25 ms),
  one grid assumed for all channels, channel order lost.

- [ ] **25. Selecting channels.** Tests with the series of channels A and B:
  - `series["b"]` is a one-channel series with unit `"s"` and
    `series["b"][1.5] == -2.0`; `series[["b", "a"]]` has
    `channel_names == ("b", "a")`. **Catches:** the query unit lost on
    selection, selection order ignored.
  - `series.values` and `series.observation(0)` raise `ValueError`, and
    `series["c"]` raises `KeyError`. **Catches:** per-channel accessors that
    silently pick a channel.

## Serialization of stored series

- [ ] **26. Save, load, and pickle one channel.**
  `include/sbear/io/timeseries_io.hpp` writes a stored channel: unit,
  timestamps, domain (with its closedness, or none), `shape`, then the
  values with the existing element writers, under the
  element type's format with `TensorFormatFlag::TimeSeries` (decision 5).
  Register it in `StreamableObject` and `read_any_object`; loading goes
  through the validating constructor. No `FormatVersion` bump is needed;
  older readers report an unhandled object type. In Python, `TimeSeries` uses
  `_BinaryIoMixin`, and `save` and `load` dispatch it. Test, parametrized over
  the 19 dtypes, a float64 channel with `shape == (2, 2)`, and a barcode
  channel with `shape == (2,)`, and over `save`/`load` and `pickle`: the
  series loads with the same `dtype`, `shape`,
  channel name, times, unit, domain, and values. **Catches:** a type flag
  mapped to the wrong type or precision; a dropped unit, domain, or shape;
  timestamps narrowed to 32 bits.

- [ ] **27. Saving several channels.** The tree of decision 9 for stored
  channels. Test, parametrized over `save`/`load` and `pickle`: the series of
  channels A and B loads with the same channel names, query unit, domain, and
  values at 0.5 and 1.5. **Catches:** names, order, or the query unit lost; a
  channel's own unit lost.

## Snapshot persistence (workflow 1)

- [ ] **28. `SeriesTask` and point-cloud persistence.** Add `sb::SeriesTask`
  (decision 6) in `include/sbear/timeseries/series_task.hpp`, a
  `make_ripser_series_task` overload for point-cloud series, and bindings in
  `src/python/persistence/py_ripser.cpp`, with each task class exposing
  `result()`. The registered task base is not the first base, so the
  pybind11 classes need `py::multiple_inheritance()`.
  `compute_persistent_homology` forwards a `TimeSeries`, keeping the existing
  complex and distance checks; update its signature and docstring. Tests:
  - Parametrized over pcloud32 and pcloud64, for the series of the three
    clouds: the result has `channel_names == ("0/H0", "0/H1")`, the matching
    barcode dtype, and the input's times and unit in each channel, and
    `result["0/Hk"].observation(i)` is isomorphic to degree `k` of
    `compute_persistent_homology(clouds[i], max_dim=1)`. **Catches:** lost
    or permuted times, a dropped unit, degrees mixed up, snapshots paired with
    the wrong barcodes.
  - A series of one cloud gives two channels with one observation each.
    **Catches:** the existing singleton squeeze applied to series.

- [ ] **29. Several point-cloud channels.** Test: `from_channels` of `"cam"`
  (the three clouds) and `"lidar"` (the first two clouds at
  `Times([0.5, 2.0], unit="ms")`) gives channels `"cam/H0"`, `"cam/H1"`,
  `"lidar/H0"`, and `"lidar/H1"`, each with its source's times and barcodes
  equal to the persistence of that source alone. **Catches:** channels
  swapped or merged, results split at the wrong offsets, one channel's times
  given to another.

- [ ] **30. Distance-matrix persistence.** The distance-matrix overload and
  binding. Test, parametrized over distmat32 and distmat64: a series of a 2x2
  and a 3x3 matrix, built with `sb.zeros((2,), dtype=...)` and element
  assignment, keeps its times, and each observation is isomorphic to
  `compute_persistent_homology(matrix, max_dim=1)` in the matching degree.
  **Catches:** distance matrices sent down the point-cloud path or to the
  wrong precision, matrices of different sizes rejected.

- [ ] **31. Typed empty series.** Test: an empty pcloud32 series gives
  barcode32 channels `"0/H0"` and `"0/H1"` with no observations for
  `max_dim=1`, and unit `"s"`. **Catches:** an untyped or missing empty
  result, failure on empty input.

- [ ] **32. Keyword data argument and parameter forwarding.** Test:
  `compute_persistent_homology(X=series, max_dim=0, reduced=True)` has
  `channel_names == ("0/H0",)` and no infinite bar. **Catches:** `max_dim` or
  `reduced` dropped on the series path, keyword calls failing.

- [ ] **33. Unsupported series.** Test, parametrized `TypeError`:
  `compute_persistent_homology` of a numeric series and of a barcode series,
  and `barcode_to_stable_rank` of a point-cloud series. **Catches:** a numeric
  series silently read as point clouds, an unsupported series passed to a
  native task.

- [ ] **34. Stable rank of a series.** A `make_stable_rank_series_task`
  overload, its binding in `src/python/persistence/py_barcode_summary.cpp`,
  and a series branch in `_barcode_to_pcf`. Test: a series of barcode
  channels `"x"` and `"y"` with two distinct barcodes each gives a PCF series
  with the same channel names and times whose channel values array-equal
  `barcode_to_stable_rank` of each channel's `BarcodeTensor`. **Catches:**
  times or names dropped, channels mixed up, the input released before the
  task finishes.

- [ ] **35. Betti curves and accumulated persistence.** The other two
  overloads and bindings, forwarding `max_death`. Tests:
  - Parametrize item 34's test over the three summary functions.
    **Catches:** a summary bound to another summary's task.
  - `barcode_to_accumulated_persistence(series, max_death=2.5)` array-equals
    the function applied to each channel's `BarcodeTensor` with
    `max_death=2.5`, and differs from the result without `max_death`.
    **Catches:** `max_death` dropped on the series path.

- [ ] **36. Progress, errors, and the GIL.** (existing) The series path runs
  through `_run_task`, so the existing progress, error-propagation, and
  GIL-release tests (`test_gil_release.py`, `test_async_task.py`) cover it,
  and the existing persistence tests cover calls without series. Confirm; add
  no test.

- [ ] **37. (optional) CPU limits.** Persistence of the three-cloud series
  under `sb.system.limit_cpus(1)` equals the result with all threads; restore
  the limit afterwards. **Catches:** results that depend on scheduling.

## Delay embeddings, windows, and sampling (workflow 2)

Items 41-48 use the source series with times `[0.0, 0.5, 1.5, 2.5, 4.0]` and
values `[1.0, 2.0, 3.0, 4.0, 5.0]`.

- [ ] **38. Delay embedding.** `TimeDelayEmbedding(dimension, delay)` returns
  a lazy series (decision 8) of points: for a scalar channel `x`, the point at
  `t` is `(x(t - (m - 1) * delay), ..., x(t))`, so `shape == (m,)`. The domain
  starts at `start + (m - 1) * delay` and keeps the source's end and its
  closedness. Tests with times `Times([0.0, 1.0, 2.5, 4.0], unit="ms")`,
  values `[10.0, 20.0, 30.0, 40.0]`, `dimension=3`, and `delay=1.0`:
  - The series at 2.0, 2.75, 3.25, 3.75, and 4.0 gives `[10, 20, 20]`,
    `[10, 20, 30]`, `[20, 20, 30]`, `[20, 30, 30]`, and `[20, 30, 40]`. The
    points at 3.25 and 3.75 differ from the point at the last observation
    time, 2.5. **Catches:** newest-first order, the same delay for every
    component, points held from the last observation time.
  - The embedding has domain `(2.0, 4.0)`, unit `"ms"`, and
    `shape == (3,)`, and
    `[1.5]` raises `ValueError`. **Catches:** components that refer to times
    before the series starts.

- [ ] **39. Embedding shaped observations, and embedding errors.** A channel
  of shape `S` embeds into shape `(m, *S)`, oldest first. Tests:
  - Times `[0.0, 1.0, 2.0]`, values `[[1, -1], [2, -2], [3, -3]]`,
    `dimension=2`, and `delay=1.0` give `shape == (2, 2)` and `[2.0]` equal
    to `[[2, -2], [3, -3]]`. **Catches:** the delay axis placed last,
    newest-first order.
  - Parametrized: `dimension=0`, `delay=0.0`, and `delay=-1.0` raise
    `ValueError`; `dimension=3` and `delay=1.0` on times `[0.0, 1.0]` raise
    `ValueError` (too short); a barcode series raises `TypeError`.
    **Catches:** invalid parameters accepted, an empty domain created
    silently, object values embedded.

- [ ] **40. Embedding several channels.** Each channel is embedded with the
  delay converted from the query unit into its own unit. Test:
  `TimeDelayEmbedding(dimension=2, delay=0.5)` of the series of channels A and
  B has domain `(1.0, 2.5)` and gives `{"a": [20, 20], "b": [-1, -2]}` at
  1.5. **Catches:** the delay applied in channel B's milliseconds.

- [ ] **41. Window bounds.** `Window(size, step=None, overlap=None)` returns a
  list of lazy start-aligned windows `[a, a + size)` with
  `a = start + k * step` while `a + size <= end`, where
  `step = size - overlap` and the default step is `size`. Incomplete windows
  are dropped. Test with the source series, parametrized over the arguments
  and the window domains: `size=2.0, step=1.0` and `size=2.0, overlap=1.0`
  give `[(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)]`; `size=2.0` gives
  `[(0.0, 2.0), (2.0, 4.0)]`; `size=5.0` gives `[]`. **Catches:** windows
  anchored at the end, partial windows kept, the overlap used as the step, an
  overlapping default.

- [ ] **42. Window contents and time slicing.** `series[a:b]` is the window
  `[a, b)`, the same object `Window` produces. Tests with the source series:
  - `w = series[1.0:3.0]` has times `[1.5, 2.5]` (original timestamps),
    `w[1.0] == 2.0` (the observation at 0.5), and `w[2.75] == 4.0`.
    **Catches:** the value before the first observation in the window missing,
    timestamps moved to the window start.
  - `w[3.0]` raises `ValueError`, and `series[1.0:3.0:0.5]` raises
    `ValueError`. **Catches:** a closed window end, a step silently ignored.

- [ ] **43. Window parameter errors.** Test, parametrized `ValueError`:
  `size=0.0`, `size=-1.0`, `step=0.0`, `overlap=2.0` and `overlap=-0.5` with
  `size=2.0`, and `step` together with `overlap`. **Catches:** windows that
  never advance, conflicting parameters accepted.

- [ ] **44. Embedding a window.** Test: `TimeDelayEmbedding(dimension=2,
  delay=1.0)` of `series[1.0:3.0]` has domain `(2.0, 3.0)`; `[2.0]` is
  `[2, 3]` (with the value at 1.0 from the source) and `[2.75]` is `[3, 4]`;
  `[1.5]` and `[3.0]` raise `ValueError`. **Catches:** history taken from
  before the window, the half-open end lost.

- [ ] **45. Windows of several channels.** Test:
  `Window(size=1.0, step=1.0)` of the series of channels A and B gives windows
  with domains `(0.5, 1.5)` and `(1.5, 2.5)`; the second gives
  `{"a": 20.0, "b": -2.0}` at 1.5 and has `times` leaves `[]` and `[1.5]`.
  **Catches:** channels cut at different bounds, channel B cut in
  milliseconds.

- [ ] **46. Sampler query times.** `RegularSampler(n_points).times(series)`
  returns `Times` with `a + (b - a) * k / n_points` for
  `k = 0, ..., n_points - 1`, in the series' unit. The end is never sampled,
  even for a closed domain. Test: `RegularSampler(n_points=4).times(w)` for
  `w = series[1.0:3.0]` gives `[1.0, 1.5, 2.0, 2.5]` with the window's unit.
  **Catches:** the end point included, spacing `(b - a) / (n - 1)`.

- [ ] **47. Sampling one channel.** `RegularSampler(n_points)(windows)`
  evaluates each window at its query times and returns a `PointCloudTensor`
  of shape `(len(windows), n_channels)` whose clouds have `n_points` points;
  each observation is flattened into a point of dimension `prod(shape)`, and a
  scalar channel gives 1-D points. Test: the three windows of
  `Window(size=2.0, step=1.0)` with `n_points=2` give shape `(3, 1)` and the
  clouds `[[1], [2]]`, `[[2], [3]]`, and `[[3], [4]]`, of dtype pcloud64.
  **Catches:** windows stacked out of order, values at window starts missing,
  the wrong precision.

- [ ] **48. Sampling several channels.** Test: `Window(size=1.0, step=1.0)` of
  the series of channels A and B with `n_points=2` gives shape `(2, 2)` with
  the clouds `[[10], [20]]` and `[[-1], [-1]]` in the first window and
  `[[20], [20]]` and `[[-2], [-2]]` in the second. **Catches:** channels
  swapped, channels evaluated on one grid.

- [ ] **49. Sampler errors.** Test, parametrized: `n_points=0` and an empty
  list raise `ValueError`; a point-cloud series raises `TypeError`.
  **Catches:** an empty result without a type, object values sampled into
  point clouds.

- [ ] **50. The delay workflow end to end.** Test in
  `test/python/test_integration.py`: times `[0.0, 0.5, ..., 3.5]` and values
  `[0, 1, 0, -1, 0, 1, 0, -1]` through
  `TimeDelayEmbedding(dimension=2, delay=0.5)`,
  `Window(size=1.0, step=1.0)` (three windows), `RegularSampler(n_points=4)`,
  and `compute_persistent_homology(..., max_dim=1)` give clouds of shape
  `(3, 1)` with four 2-D points each and barcodes of shape `(3, 1, 2)`.
  **Catches:** steps that do not compose.

- [ ] **51. Saving derived series.** Window and embedding nodes in the tree of
  decision 9. Test, parametrized over `save`/`load` and `pickle`: the window
  `[1.5, 2.5)` of item 40's embedding loads with the same channel names,
  domain, unit, and values at 1.5 and 2.0, and `[2.5]` still raises
  `ValueError`. **Catches:** a node kind, parameter, or query unit lost on
  saving, the half-open end lost.

- [ ] **52. Golden files.** Pin the formats (TESTING.md principle 11) with a
  generator modeled on `test/golden/serialization/generate_point_cloud_v3.py`,
  which records explicit expected values. Write item 28's barcode32 result,
  the series of channels A and B, and item 51's derived series as `.sb` and
  pickle files, and teach `test_serialization_goldens.py` to compare series
  (channel names, unit, domain, values at listed times). **Catches:** a later
  change to a payload that breaks saved files.

## Documentation and release

Code items update the docstrings they touch (AGENTS.md). These items add the
pages.

- [ ] **53. API reference.** Add `docs/api_timeseries.rst` to the toctree in
  `docs/stablebear.rst`. Check that the docstrings state the contracts of
  decisions 1-4 and 6-8: domains and their closedness, previous-value
  evaluation and snapping, units, query units and conversion, channels and
  their names, time indexing, observation shapes and flattening into
  points, copies and lazy derived
  channels. Check that `compute_persistent_homology`, the three summaries,
  `save`, and `load` document series input and output.

- [ ] **54. User guide.** A `docs/timeseries.rst` guide linked from
  `docs/userguide.rst`: times, units, domains, and channels, then two small
  examples. The first follows a point-cloud series through persistence to
  stable ranks; the second takes two numeric channels in different units
  through a delay embedding, windows, and sampling to persistence. Link each
  operation to its API entry on first mention. Figures are not needed; if one
  is added, follow the dropdown rule in AGENTS.md.

- [ ] **55. Changelog.** A New features entry in `CHANGELOG.md` linking #242,
  under the heading of the release that ships it (to be decided; 0.5.1 is in
  progress).

## Final checks

- [ ] **56. Test run.** After a CMake build and install, run the new test
  files, then the full Python suite and `sb_test` from `test/`.
- [ ] **57. Docs build.** Run `make html` in `docs/` (with `LC_ALL=C.UTF-8` if
  the locale is missing), check for warnings, and read the new pages.
- [ ] **58. Push.** Push `issue-242-timeseries`, open a PR against `main`, and
  remove this plan before merging.

## Coverage of the acceptance criteria

| Criterion in #242 | Items |
| --- | --- |
| Time, domain, evaluation, shape, and ownership contracts; empty and singleton behavior | 1-9, 19 |
| `Times`, native storage and access, default seconds, explicit units, one timestamp per observation | 10-13, 17-18, 20-21 |
| Floating-point arrays with explicit times or a packed table | 11-12, 15-16 |
| Native operation contract, dispatch, and adapter; calls, parameters, CPU limits | 28-29, 32, 36-37 |
| Persistence and barcode summaries of series | 28-35 |
| Delay embedding and windows | 38-45 |
| Regular sampling into point clouds and through persistence | 46-50 |
| Save, load, and pickle for both workflows | 26-27, 51-52 |
| Tests and documented examples | all test items, 53-54 |
| Channels with their own grids and units (added) | 3, 21-25, 29, 40, 45, 48 |

## Out of scope

As in the issue: datetime timestamps, sample-count delays and windows, other
samplers, interpolation, partial windows, gap policies, tensors of series,
reductions over time, provenance, and normalization. Also deferred here:
joining channels into one point cloud when sampling, series with channels of
different element types, observations whose shape changes over time
(a later per-observation tensor storage mode), `NestedTensor` values,
`Tensor<TimeSeries>`, views instead of copies for stored channels,
and the branch's notebooks and scikit-learn transformers.
