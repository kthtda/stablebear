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
  (often `()`).
- `times` is always given explicitly; a table's column 0 is never read as
  timestamps.
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

**Test budget.** The items add about 60 test functions, many of them
parametrized, plus three optional ones. Keep that number from growing. Before
adding a test that is not listed, check whether a listed test's
parametrization covers it, and prefer leaving a guarantee to the design and
its docstring over testing every combination. Do not test what the public API
cannot observe. No C++ unit tests are planned: every guarantee here is
observable from Python (TESTING.md principle 4).

## Conventions

- **Files.** Source: `include/sbear/timeseries.hpp` (`Times`, channels,
  `TimeSeries<T>`), `include/sbear/timeseries/derived.hpp` (windows,
  embeddings, sampling), `include/sbear/ufunc.hpp` (operations and
  `UfuncTask`), `include/sbear/io/timeseries_io.hpp`,
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
  `TimeSeries.from_channels([a, b], unit="s")` (channels 0 and 1).
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
- A stored channel holds a shared `Times`, a flat `std::vector<T>`, and a
  `shape`. Each observation is a tensor of shape `shape`
  (`()` for one element per time, `(d,)` for a point, `(h, w)` for an image),
  stored as `n * prod(shape)` values and presented as a tensor only when
  accessed. Point counts, matrix sizes, and bar counts are internal to each
  element and may change over time.
- `sb::TimeSeries<T>` is an ordered list of channels of one element type
  `T` and a query unit. It can be evaluated from the latest first timestamp
  to the earliest last timestamp of its channels, both included; there is
  no stored domain. Like `NestedTensor`, one registration template binds it for every
  tensor element type in the library (19 types).
- Everything runs in C++: lookups, snapping, domain checks, unit conversion,
  windows, delay embeddings, sampling, and analysis. Python only binds
  arguments and wraps results.
- `series(t)` looks up each channel on its own grid, in its own unit, and
  returns the value for one channel or a `NestedTensor` with one leaf per
  channel for several. `series.window(a, b)` is a window, and `series[i]`
  selects channels by index.
- Every channel is a view: a `shared_ptr` to its storage plus a descriptor
  (domain, lags, unit scale). Windows, delay embeddings, channel selection,
  and unit changes copy no values; a write materializes the whole view into
  the writer's own storage, as subsampled tensors do. A delay
  embedding of a numeric channel of shape `S` has shape `(m, *S)`.
- `RegularSampler` samples each window of each numeric channel into a point
  cloud, flattening each observation into a point of dimension
  `prod(shape)`, and returns a `PointCloudTensor` of shape
  `(n_windows, n_channels)`.
- Analysis functions are operations in the style of NumPy ufuncs: a native
  per-element kernel with bound parameters and `extra_dims`, the dimensions
  one input element adds to the result (for persistence, `(max_dim + 1,)`).
  One native `UfuncTask` applies an operation to any container through a
  small adapter. For a series, each output channel keeps its input's position
  and times, and its shape is the input shape followed by
  `extra_dims`. Tensors use the same task.

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
  `start + (dimension - 1) * delay`) for the lazy embedding (items 40-42).
  Not its windows, which are anchored at the end and keep a partial first
  window, and not its tuples, which it stores only at original observation
  times; the issue rules out both.
- `include/mpcf/io/timeseries_io.hpp`: the outline of the writer and reader
  for items 26-27 and 53, with new payloads.
- `src/python/py_timeseries.cpp`: the array-copy and length-check patterns for
  items 10-15.
- `masspcf/persistence/homology.py` (the `TensorTimeSeries` branch): one
  batched Ripser call over all snapshots. `UfuncTask` (item 28) keeps the
  idea and moves it into native code.
- `docs/ts_*.rst`: background wording for item 59; the API differs.

Not recycled: datetime support, interpolation strategies, `TimeSeriesTensor`,
the scikit-learn transformers, `l2_kernel` cross kernels, and the `tensor_eval`
parallel threshold.

## Decisions

- [x] **1. Domain and evaluation.** (decision: (a), closed domain for series built from observations, half-open for windows; revised during item 26: there is no domain. A series can be evaluated from the first to the last observation of every channel, both included, checked against the timestamps on each query; there is no `domain` property, and a series with an empty channel cannot be evaluated. Windows follow the same rule: there is no open or closed end anywhere, see item 43.) Evaluation
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

- [x] **2. Floating-point timestamps.** (decision: snap within a fraction of the gap, capped at a multiple of machine epsilon times the timestamps' magnitude; both user-settable, defaults `1e-6` and `1024`)
  Delays and window starts are computed in floating point, so a query can
  land just before the observation it should reach: `0.3 - 0.1` is less than
  `0.2`. A query below an observation counts as reaching it if it falls short
  by at most `min(snap_tolerance * gap, snap_eps * eps * max(|prev|,
  |next|))`, where the gap is the time since the previous observation and
  `eps` is float64 machine epsilon. Both are keywords of `TimeSeries`
  (`snap_tolerance` in `[0, 1)`, default `1e-6`; `snap_eps` finite and
  nonnegative, default `1024`; 0 in either disables snapping). They belong
  to the channel: `from_channels` keeps each channel's own, and everything
  that evaluates a channel uses them. The gap limit keeps the rule from
  skipping an observation; the epsilon limit keeps wide gaps (daily data in
  seconds) from capturing queries that are clearly earlier. Both are
  relative, so the rule does not depend on the unit. Domain checks stay
  exact. The first choice, `1e-9 * max(1, |t|)` as on the branch, depended
  on the unit near 0 and crossed whole gaps for small spacings or
  epoch-sized timestamps.

- [x] **3. Time units.** (decision: (a), fixed set `as` through `d` with exact integer conversion factors) `Times(times, unit="s")`; plain timestamp
  arrays mean seconds. Options:
  (a) A fixed set covering physics time scales: `"as"`, `"fs"`, `"ps"`,
  `"ns"`, `"us"`, `"ms"`, `"s"`, `"min"`, `"h"`, and `"d"`. Store each unit as
  a power of ten times an integer (1 or 60, 3600, 86400), and convert by
  multiplying or dividing by an exact integer factor rather than by a
  rounded one such as `1e-15`; dividing by an exact factor rounds once.
  Adding `"zs"` or `"ys"` later is one table entry. Aliases such as
  `"sec"`, `"seconds"`, `"hr"`, `"msec"`, and the micro sign `"\u00b5s"`
  are accepted; `unit` reports the short name. `Times.to(unit)`
  converts, and a series converts each query from its query unit into each
  channel's unit. Delays, window sizes, and query times are in the series'
  query unit.
  (b) Any nonempty string, with no conversion.

- [x] **4. Ownership.** (decision: copy in, views out, materialize on write, as subsampled tensors do)
  - The constructor copies its input, materializing indexed point-cloud and
    distance-matrix tensors, so the caller's data stays independent.
  - Channel storage lives in a `std::shared_ptr` and is shared by every view
    of it (decision 8): channel selection, unit changes, windows, and delay
    embeddings copy no values.
  - Reads return views: `values` is a tensor backed by the series, and
    indexing it gives views as for any tensor, as `PointCloudTensor[i]`
    returns a `PointCloud` with an owner.
  - Writes go through `values[...] = x` or a view taken from `values`.
    Before writing, a series whose storage is shared, or
    whose view is a window or an embedding, materializes the whole view into
    its own storage, as `IndexedElementTensor._ensure_writeable` does for
    subsamples. Other views never see the write.
  - Timestamps cannot change: `np.asarray(times)` and `sb.tensor(times)`
    return copies. `Times` has no `values` attribute, which would be
    confused with a series' values.
  - Public numeric results are stablebear tensors (`FloatTensor`), not NumPy
    arrays.

- [x] **5. Native storage and where the code lives.** (decision: as recommended; all evaluation in C++, flat values plus `shape`, `NestedTensor`-style registration)
  Recommendation, from our discussion:
  - All evaluation runs in C++: the previous-value lookup with snapping,
    domain checks, unit conversion, windows, delay embeddings, and sampling.
    Python wrappers only bind arguments and wrap results.
  - A stored channel holds a shared `Times`, a flat `std::vector<T>` (not a
    `Tensor<T>`, so its layout, gathering, and slicing stay under our
    control), and a `shape`. Each observation is a tensor of shape
    `shape`, stored as `prod(shape)` consecutive values and built as a
    `Tensor<T>` only when accessed. Storing a `Tensor<T>` per observation
    instead would cost 80 bytes and 2 to 4 heap allocations per time point.
    Values are kept apart from times rather than as `(time, value)` pairs,
    because the
    batched analysis needs a contiguous array of values. A `Pcf` cannot hold a
    channel: it is defined on `[0, inf)` and only for arithmetic values.
  - `sb::TimeSeries<T>` holds an ordered list of channels of one element type. Following
    the `NestedTensor` pattern (`include/sbear/nested_tensor.hpp`,
    `register_tensor_element` in `src/python/py_tensor.hpp`,
    `stablebear/nested_tensor.py`), one `register_timeseries_element<T>` call
    per element type binds it for bool, float32/64, int32/64, uint32/64,
    pcf32/64, pcf32i/64i, pcloud32/64, distmat32/64, symmat32/64, and
    barcode32/64. Python has one `TimeSeries` class whose `dtype` is the
    element dtype, with a dtype-keyed table for the native class.
  - Serialization marks the element type's own format with a new
    `TensorFormatFlag::TimeSeries`, so one writer and reader serve all
    element types, as `TensorFormatFlag::Nested` does.

- [x] **6. Analysis adapter.** (decision: operations in the style of ufuncs; `extra_dims` extends the observation shape; tensors move to the same task; no public `apply` yet)
  - An operation is a native per-element kernel with its parameters bound and
    `extra_dims`, the dimensions one input element adds to the result, like
    the core dimensions of a NumPy gufunc signature. Persistence maps
    `PointCloud` or `DistanceMatrix` to `Barcode` with `extra_dims`
    `(max_dim + 1,)`; the barcode summaries map `Barcode` to `Pcf` with
    no extra dimensions. The kernels already exist (`run_euclidean_ripser`,
    `run_ripser`, `barcode_to_stable_rank`, ...).
  - One native `UfuncTask` walks all input elements in parallel on the
    existing executor and writes the output, with progress, cancellation, and
    error capture from `StoppableTask`. It owns its inputs and output until
    completion.
  - A container adapter says where the elements are and how to build the
    output. Tensor: the result's shape is the input shape followed by
    `extra_dims`. Series: every channel is processed in the same task, and each
    output channel keeps its position and times, with shape
    `shape + extra_dims`. Persistence of a channel of shape `()` therefore has
    shape `(max_dim + 1,)`; no channels are added. A future container needs
    only an adapter.
  - `compute_persistent_homology` and the summaries keep their signatures and
    forward to private native entry points; pybind11 overloads select the
    container and element type. A public `apply` stays private for now.
  - A singleton channel keeps its observation and an empty channel gives a
    typed empty result.

- [x] **7. Public names and accepted values.** (decision: as recommended; `NestedTensor` values deferred; revised after item 23: channels have no names and are selected by index, and several channels evaluate to a `NestedTensor`)
  - `stablebear.timeseries` exports `Times(times, unit="s")`,
    `TimeSeries(values, times)` for one channel,
    `TimeSeries.from_channels(channels, unit=None)` for several,
    `TimeDelayEmbedding(dimension, delay)` (without the issue's `unit="time"`
    until a second delay mode exists), `Window(size, step=None, overlap=None)`,
    and `RegularSampler(n_points)` with `.times(series)`.
  - `from_channels` takes a list of one-channel series of one element
    type; channel `i` is the `i`th series. `unit` defaults to the
    channels' shared unit and is required when their units differ.
  - A `TimeSeries` exposes `n_channels`, `dtype`, `unit`, `times`
    (a `Times` for one channel, a `NestedTensor` with one leaf per channel in
    the query unit for several), `to(unit)`, `evaluate(t)`, `series(t)` (the
    same as `evaluate(t)`), `series.window(a, b)` (a window),
    and `series[i]` or `series[[i, j]]` (channel selection by index; negative
    indices count from the end). A
    series also exposes `values`, `shape`, `n_observations`,
    `snap_tolerance`, and `snap_eps`: for one channel a tensor of shape
    `(len(), *shape)`, a shape tuple, an int, and two floats; for several a
    `NestedTensor` with one leaf per channel, a tuple of shape tuples, an
    int64 `IntTensor`, and two float64 `FloatTensor`s, each with one entry
    per channel. `len()` is `n_observations` for one channel and raises
    `ValueError` on several, since it must return an int. There is no
    `observation(i)`: for one channel it would be `values[i]`, and across
    channels index `i` refers to unrelated times. (Revised after item 25.)
  - For one channel, `evaluate` returns the observation: a scalar or element
    (`PointCloud`, `Barcode`, ...) for `shape == ()`, otherwise a tensor of
    that shape. For several channels it returns a `NestedTensor` with one
    leaf per channel, in channel order; each leaf is the channel's value as a
    tensor of the channel's shape (0-d for `shape == ()`). Channels may have
    different shapes.
  - `values` may be a tensor of any element type (`FloatTensor`,
    `IntTensor`, `BoolTensor`, `PcfTensor`, `IntPcfTensor`,
    `PointCloudTensor`, `DistanceMatrixTensor`, `SymmetricMatrixTensor`,
    `BarcodeTensor`) or a numeric array-like of shape `(n, *shape)`, with the
    dtype `sb.tensor()` infers (float32, int32, uint32, uint64, and bool are
    kept; other floats become float64 and other integers int64); axis 0
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

- [x] **8. Views.** (decision: every channel is a view of shared storage; a write materializes the whole view; the domain in the descriptor below predates its removal during item 26 and is revised with windows, see item 43)
  - Like a tensor view, a channel is a `shared_ptr` to its storage plus a
    small descriptor of how a query time maps into it: the domain (start, end,
    and end closedness), `lags` (time offsets that add leading observation
    axes), and `scale` (from the query unit to the storage unit). A freshly
    built channel is the identity view.
  - A window narrows the domain to `[a, b)`; it cuts every channel at the
    same bounds in the query unit and works for any element type. A delay
    embedding of shape `S` adds lags `[-(m - 1) * delay, ..., 0]`, giving shape
    `(m, *S)`, oldest first; it requires numeric values and takes its delay in
    the query unit. `.to(unit)` changes only `scale`. Further indexing modes
    (observation-axis selection, time shifts, strides) extend the descriptor
    later.
  - Views compose into one descriptor rather than nesting: a window of an
    embedding keeps the lags and narrows the domain; an embedding of a window
    adds lags and moves the domain start to `a + (m - 1) * delay`, so its
    lookups never reach before `a`; an embedding of an embedding sums the lag
    arrays pairwise.
  - Every view behaves like any other series. Its `times` are the times in
    its domain where its value can change: the source's timestamps inside the
    domain for windows (original timestamps), and the shifted times
    `s + lag` inside the domain for embeddings. `values[i]` is the view's
    value at `times[i]`, and `len()` counts these times.
  - Materializing a view stores its support: the last change time at or
    before the domain start (for a window, the observation carried in before
    `a`, at its original timestamp) followed by the change times inside the
    domain, each with the view's value there, plus the domain. Under
    previous-value evaluation the result evaluates identically everywhere in
    the domain. An embedding with `m` lags may have up to `m * n` change times;
    that memory is spent only on a write.
  - Internal lookups skip the domain check, because shifted and converted
    times may round just outside it. Shifted times may also land one ulp apart
    (`0.2 + 0.1` and `0.3`); snapping makes their values equal, and merging
    them is a later cleanup.

- [x] **9. Saving series.** (decision, revised in item 26: materialize every
  channel on saving) Decision 8 governs memory during a run; a file only
  needs to reproduce the series' values. Save the query unit and one record
  per channel: its unit, timestamps, values, and snap settings, materializing
  views as decision 8 does on a write. A loaded series
  evaluates as the original and needs no original objects, but shares no
  storage. Saving a series together with views of it is rare, so the
  duplicated values cost little. The rejected first choice wrote distinct
  storages once with one descriptor per channel referring to them.

## Stored channels

- [x] **10. `Times`.** `sb::Times` in `include/sbear/timeseries.hpp` validates
  its timestamps and unit and converts units. The new `stablebear/timeseries/`
  package wraps it with `values` (a float64 `FloatTensor` copy), `unit`,
  `len()`, `==`, `repr`, and `to(unit)`. Tests:
  - `Times([0.0, 1.5, 4.0], unit="ms")` has float64 values
    `[0.0, 1.5, 4.0]` (a `FloatTensor`) and unit `"ms"`. **Catches:** a
    dropped or replaced unit, an ndarray returned.
  - Invalid input raises `ValueError`, parametrized over the timestamps
    `[0.0, 0.0]`, `[1.0, 0.0]`, `[0.0, nan]`, `[0.0, inf]`, and
    `[[0.0, 1.0]]`, and over `unit="yr"` with valid timestamps.
    **Catches:** duplicate, decreasing, nonfinite, or 2-D timestamps
    accepted; unknown units accepted.
  - Parametrized: the aliases `"sec"`, `"seconds"`, `"hr"`, `"msec"`, and
    both micro signs give the short names `"s"`, `"h"`, `"ms"`, and `"us"`.
    **Catches:** an alias rejected or mapped to the wrong unit.
  - `times[0] = 5.0` raises `TypeError` and leaves the timestamps unchanged. **Catches:** timestamps that can be
    reordered after validation.
  - Parametrized: `Times([0.0, 1.5], unit="min").to("s")` gives
    `[0.0, 90.0]`; `Times([0.0, 1.5], unit="fs").to("as")` gives
    `[0.0, 1500.0]`; `Times([0.0, 2.25], unit="as").to("fs")` gives
    `[0.0, 0.00225]`. Each result has the target unit. **Catches:** an
    inverted factor, a rounded factor such as `1e-3` (which gives
    `0.0022500000000000003`), the unit label not updated.

- [x] **11. Scalar series.** The stored channel and `sb::TimeSeries<T>` with
  their validation, `register_timeseries_element` for float32 and float64,
  and Python `TimeSeries(values, times)` with `n_channels`, `times`,
  `unit`, `values`, `dtype`, `shape`, `len()`, and `repr`. This is the largest
  item. Tests:
  - Plain timestamps mean seconds: values `[10.0, 20.0, 30.0]` with
    `times=[0.0, 1.5, 4.0]` give `unit == "s"`, `n_channels == 1`,
    `shape == ()`, `len() == 3`, and the same values. This test checks the
    defaults. **Catches:** a missing default unit.
  - An explicit unit is kept:
    `TimeSeries([1.0, 2.0], times=Times([0.0, 0.5], unit="ms"))` has unit
    `"ms"`. **Catches:** `Times` reduced to a plain array.
  - One timestamp per observation: values of shape `(3,)` with 2 or 4
    timestamps raise `ValueError`. **Catches:** a length mismatch accepted.
  - Value dtype: float32, float64, int32, int64, and bool arrays give
    `sb.float32`, `sb.float64`, `sb.int32`, `sb.int64`, and `sb.boolean`.
    **Catches:** float32 widened, integers converted to floats.

- [x] **12. Observation shapes.** Values of shape `(n, *shape)` form one
  channel whose observations have shape `shape`. Tests:
  - Values `[[1, -1], [2, -2], [3, -3]]` with `times=[0.0, 1.0, 2.0]` give
    `shape == (2,)`, `len() == 3`, and `values` equal to the input.
    **Catches:** points split into channels or counted as observations,
    coordinates transposed.
  - Values `[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]` with `times=[0.0, 1.0]`
    give `shape == (2, 2)` and `values` equal to the input.
    **Catches:** higher-rank observations flattened or transposed.

- [x] **13. Every element type.** Register the remaining 12 element types
  (the 7 numeric ones came with item 11) and dispatch on the tensor type in
  Python. Test, parametrized over the 19
  dtypes: a two-observation 1-D tensor whose elements differ (for objects,
  also in size: clouds of 1 and 2 points, 2x2 and 3x3 matrices, barcodes with
  1 and 2 bars, PCFs with 2 and 3 breakpoints) with `times=[0.0, 1.0]` gives a
  series with the tensor's `dtype`, `len() == 2`, and values equal to the
  input. **Catches:** a type bound to the wrong native class or precision,
  elements of different sizes rejected.

- [x] **14. (optional) Indexed tensors are copied.** (skipped; `to_dense()` is in place) A point-cloud tensor that
  holds a selection (indexed with `sb.indices`) gives a series whose
  `values.indices is None`. **Catches:** an indexed tensor passed to a binding
  that expects ordinary storage.

- [x] **15. Packed tables.** (dropped: `times` is required and never inferred
  from a column; split a table with `TimeSeries(data[:, 1:], times=data[:, 0])`)

- [x] **16. Unsupported values.** Test, parametrized `TypeError`: a list of
  two equal-size `sb.PointCloud`s, `np.array(["a", "b"])`, and a
  `NestedTensor`. **Catches:** object sequences silently converted to numeric
  arrays, strings or nested tensors passed to the native constructor.

- [x] **17. Observations.** (dropped after item 25 with `observation(i)`;
  use `values[i]`) `series.observation(i)` returns observation `i`.
  Test with values `[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]` and
  `times=[0.0, 1.0, 2.0]`: `observation(1)` is `[3.0, 4.0]`,
  `observation(-1)` is `[5.0, 6.0]`, and `observation(3)` raises
  `IndexError`. **Catches:** indexing along the coordinates, negative indices
  counted from the wrong end, no bounds check. Writes come in items 48-50.

- [x] **18. Evaluation.** `series.evaluate(t)` and `series(t)` return the
  previous value (decision 1). Tests with times `[0.0, 1.5, 4.0]` and values
  `[10.0, 20.0, 30.0]`:
  - `series(t)` at 0.0, 1.0, 1.5, 3.75, and 4.0 gives 10, 10, 20, 20, and 30,
    and `series(1)` (an int) gives 10, the value at time 1. **Catches:**
    next-value, nearest, or linear interpolation; an open end; an int read as
    a position.
  - `series(-0.5)` and `series(4.5)` raise `ValueError` ("outside the
    domain"). **Catches:** extrapolation before the first or after
    the last observation, NaN instead of an error.

- [x] **19. Empty and singleton series.** Tests:
  - `TimeSeries(sb.zeros((0,), dtype=sb.pcloud64), times=[])` has
    `len() == 0` and dtype pcloud64, and `series(0.0)` raises `ValueError`
    ("without observations"). **Catches:** an invented range, a crash on
    empty storage.
  - `TimeSeries([7.0], times=[2.0])` has `series(2.0) == 7.0`. **Catches:** a singleton series rejected or
    impossible to evaluate.

- [x] **20. Snapping.** Implement decision 2 in the native lookup. Tests:
  - Times `[0.0, 0.25, 0.5]`, values `[1.0, 2.0, 3.0]`, and
    `snap_tolerance=1e-6`: `series(np.nextafter(0.25, 0.0))` is 2.0 and
    `series(0.24)` is 1.0. **Catches:** no snapping, or a tolerance
    wide enough to change values at ordinary times.
  - Parametrized: spacing `1e-10` s near 0 and spacing `1e-6` s near
    `1.7e9`; a query halfway between two observations gets the earlier one.
    **Catches:** a tolerance that ignores the gap and skips observations.
  - Times `[0.0, 86400.0]`: `nextafter(86400, 0)` snaps and `86400 - 0.05`
    does not. **Catches:** no epsilon cap on wide gaps.
  - Times `[0.0, 1.0]`, query `1 - 4 eps`: snaps with `snap_eps=8`,
    not with 2; the property reads back. Negative, infinite, and NaN
    `snap_eps` raise `ValueError`. **Catches:** the cap ignored or
    unvalidated.
  - `snap_tolerance=0.0` is stored and disables snapping, and -0.1 and 1.0 raise
    `ValueError`. **Catches:** the keyword ignored or unvalidated.

- [x] **21. Query unit.** `series.to(unit)` returns the series with another
  query unit; the stored channels are unchanged. Test: `TimeSeries([10.0,
  20.0], times=Times([0.0, 0.5], unit="s")).to("ms")` has unit `"ms"`, times
  `[0.0, 500.0]`, and values 10 at 250.0 and 20 at 500.0. **Catches:**
  queries or the domain check left in the old unit.

## Several channels

- [x] **22. Building multi-channel series.** (reopened: first done with
  names in `f46cc57e0`; rework to channels by index) `TimeSeries.from_channels`
  (decision 7). Tests:
  - `from_channels([a, b], unit="s")` has `n_channels == 2` and unit `"s"`.
    **Catches:** a channel dropped, the unit ignored. Item 23's `times` test
    checks channel order.
  - Parametrized: channels A and B without `unit=`, an empty list, and a
    multi-channel series as a channel raise `ValueError`; A together with a
    point-cloud channel raises `TypeError`. **Catches:** a silently chosen
    unit, an empty series, nested channel sets, mixed element types.

- [x] **23. Domain and times of several channels.** Test with the series of
  channels A and B: the series evaluates at 0.5 (B's first observation) and
  2.5 (A's last), and `series(0.25)` and `series(2.75)` raise `ValueError`;
  `times` is a `NestedTensor` whose leaves are `[0.0, 1.0, 2.5]` and
  `[0.5, 1.5, 3.0]`. **Catches:** a union of the channels' domains, the ends
  excluded, timestamps reported in each channel's own unit.

- [x] **23a. `NestedTensor` of every element type.** One
  `register_tensor_element<T>` in `src/python/py_tensor.hpp` binds `Tensor<T>`,
  its indexed variant when `T` supports selections, and `NestedTensor<T>`,
  and every element type's registration site calls it, so `NestedTensor`
  gains the 13 element types it lacked (bool, the four PCF types, point
  clouds, distance and symmetric matrices, and barcodes, each in 32 and 64
  bits). Extend the Python leaf wrapping and `dtype` table, so that item 24
  can return one for any series. Leaves of object types hold elements of
  different sizes; indexed leaves are made dense. Saving `NestedTensor`s of
  these types stays out of scope. Test, parametrized over the 19 dtypes with
  the builders shared with item 13 (`test/python/_element_values.py`): a
  `NestedTensor` built from two 1-D tensors of that dtype (of lengths 1 and
  2) has that `dtype`, and its leaves equal the inputs.
  **Catches:** a leaf type bound to the wrong native class or precision.

- [x] **24. Evaluating several channels.** Tests with the series of channels A
  and B:
  - `series(1.25)` is a float64 `NestedTensor` of shape `(2,)` with 0-d
    leaves 20.0 and -1.0, and `series(1.5)` has leaves 20.0 and -2.0.
    **Catches:** channel B read in the wrong unit (it would answer at
    1.25 ms), one grid assumed for all channels, channel order lost.
  - Channel A built with `snap_tolerance=0.0` and channel B with the
    default: at `np.nextafter(1.0, 0.0)`, just below A's 1.0 s, leaf 0 is
    10.0 (not snapped), and at `np.nextafter(1.5, 0.0)`, just below B's
    1500 ms, leaf 1 is -2.0 (snapped). **Catches:** one channel's snap
    settings applied to another.
  - A point-cloud channel and a second point-cloud channel give a pcloud64
    `NestedTensor` whose leaves are the two clouds. **Catches:** object
    types left out of multi-channel evaluation.

- [x] **25. Selecting channels.** Tests with the series of channels A and B:
  - `series[1]` and `series[-1]` are one-channel series with unit `"s"` and
    value -2.0 at 1.5; `series[[1, 0]](1.5)` has leaves -2.0 and 20.0.
    **Catches:** the query unit lost on selection, negative indices counted
    from the wrong end, selection order ignored.
  - `series.values` is a `NestedTensor` whose leaves equal the values of
    channels A and B; with channel A built with `snap_tolerance=0.25`,
    `series.snap_tolerance` is a float64 `FloatTensor` equal to
    `[0.25, 1e-6]` and `series[0].snap_tolerance == 0.25`, and
    `series[1].snap_tolerance` differs. **Catches:** channels' values or
    settings mixed up or lost on selection.
  - Channel A with a channel of shape `(2,)` and two observations has
    `shape == ((), (2,))` and `n_observations` equal to `[3, 2]`, and
    `series[1].n_observations == 2`. `len(series)` raises `ValueError`, and
    `series[2]` raises `IndexError`. **Catches:** shapes or counts mixed up,
    `len` silently picking a channel, no bounds check.

## Serialization of stored series

- [x] **26. Save, load, and pickle one channel.**
  `include/sbear/io/timeseries_io.hpp` writes the layout of decision 9 under
  the element type's format with `TensorFormatFlag::TimeSeries` (decision
  5): the query unit, then one record per channel (its unit, timestamps,
  values as a tensor with time along axis 0 using the existing element
  writers, `snap_tolerance`, and `snap_eps`). Units are stored as their length in seconds (`mult *
  10^exp`), not by name.
  Register it in `StreamableObject` and `read_any_object`; loading goes
  through the validating constructor. No `FormatVersion` bump is needed;
  older readers report an unhandled object type. In Python, `TimeSeries` uses
  `_BinaryIoMixin`, and `save` and `load` dispatch it. Test, parametrized over
  the 19 dtypes of `_element_values.py`, each as two observations `a` and
  `b` of shape `()` and as two observations `[a, b]` and `[b, a]` of shape
  `(2,)`, and over
  `save`/`load` and `pickle`: the
  series loads with the same `dtype`, `shape`,
  times, unit, snap settings, and values. **Catches:** a type flag
  mapped to the wrong type or precision; a dropped unit, shape, or
  snap setting; timestamps narrowed to 32 bits.

- [x] **27. Saving several channels.** The format of decision 9 for several
  channels. Test, parametrized over `save`/`load` and `pickle`: the series of
  channels A and B loads with the same number of channels, query unit,
  times, and values at 0.5 and 1.5. **Catches:** order or the query unit lost; a
  channel's own unit lost.

## Snapshot persistence (workflow 1)

- [x] **28. `UfuncTask` and point-cloud persistence.** Add the operation
  interface, `UfuncTask`, and the series adapter (decision 6) in
  `include/sbear/ufunc.hpp`, the persistence operation for point clouds, and
  a private entry point in `src/python/persistence/py_ripser.cpp` returning
  the task with `result()`. `compute_persistent_homology` forwards a
  `TimeSeries`, keeping the existing complex and distance checks; update its
  signature and docstring. Tests:
  - Parametrized over pcloud32 and pcloud64, for the series of the three
    clouds: the result has `n_channels == 1`, the matching barcode
    dtype, `shape == (2,)`, and the input's times and unit, and
    `result.values[i]` is isomorphic to
    `compute_persistent_homology(clouds[i], max_dim=1)`. **Catches:** lost
    or permuted times, a dropped unit, a missing degree axis, snapshots paired
    with the wrong barcodes.
  - A series of one cloud has one observation of shape `(2,)`. **Catches:**
    the existing singleton squeeze applied to series.

- [x] **29. Several point-cloud channels.** Test: `from_channels` of `cam`
  (the three clouds) and `lidar` (clouds 2 and 1, so that reading cam's
  elements for lidar shows, at
  `Times([0.5, 2.0], unit="ms")`) gives channels 0 and 1, each
  with its source's times and barcodes equal to the persistence of that source
  alone. **Catches:** channels swapped or merged, results written to the wrong
  channel, one channel's times given to another.

- [x] **30. Distance-matrix persistence.** The distance-matrix overload and
  binding. Test, parametrized over distmat32 and distmat64: a series of a 2x2
  and a 3x3 matrix, built with `sb.zeros((2,), dtype=...)` and element
  assignment, keeps its times, and each observation is isomorphic to
  `compute_persistent_homology(matrix, max_dim=1)`.
  **Catches:** distance matrices sent down the point-cloud path or to the
  wrong precision, matrices of different sizes rejected.

- [ ] **31. Typed empty series.** Test: an empty pcloud32 series gives a
  barcode32 series with no observations and `shape == (2,)` for `max_dim=1`,
  and unit `"s"`. **Catches:** an untyped or missing empty
  result, failure on empty input.

- [ ] **32. Keyword data argument and parameter forwarding.** Test:
  `compute_persistent_homology(X=series, max_dim=0, reduced=True)` has
  `shape == (1,)` and no infinite bar. **Catches:** `max_dim` or
  `reduced` dropped on the series path, keyword calls failing.

- [ ] **33. Unsupported series.** Test, parametrized `TypeError`:
  `compute_persistent_homology` of a numeric series and of a barcode series,
  and `barcode_to_stable_rank` of a point-cloud series. **Catches:** a numeric
  series silently read as point clouds, an unsupported series passed to a
  native task.

- [ ] **34. Stable rank of a series.** The stable-rank operation, its private
  entry point in `src/python/persistence/py_barcode_summary.cpp`, and a series
  branch in `_barcode_to_pcf`. Test: a series of barcode
  channels with two distinct barcodes each gives a PCF series
  with the same number of channels and times whose channel values array-equal
  `barcode_to_stable_rank` of each channel's `BarcodeTensor`. **Catches:**
  times dropped, channels mixed up, the input released before the
  task finishes.

- [ ] **35. Betti curves and accumulated persistence.** The other two
  operations and entry points, forwarding `max_death`. Tests:
  - Parametrize item 34's test over the three summary functions.
    **Catches:** a summary bound to another summary's kernel.
  - `barcode_to_accumulated_persistence(series, max_death=2.5)` array-equals
    the function applied to each channel's `BarcodeTensor` with
    `max_death=2.5`, and differs from the result without `max_death`.
    **Catches:** `max_death` dropped on the series path.

- [x] **36. Tensors through `UfuncTask`.** (done early, in `46069ff0e`; single clouds, matrices, and barcodes also run as 0-d inputs) Route the existing tensor paths of
  `compute_persistent_homology` (point clouds and distance matrices, including
  indexed tensors) and the three summaries through `UfuncTask` with a tensor
  adapter, and remove `RipserTaskImpl` and `BarcodeSummaryTask` if nothing
  else uses them. Keep the public results, including the existing squeeze of
  a length-1 input. No new tests: the existing persistence and summary tests
  must pass unchanged. **Catches:** behavior changes in the tensor paths.

- [ ] **37. (optional) Selecting a value index.** `series.select(k)` (name to
  be decided) indexes the observation axes and keeps time, for example to
  take degree 1 of a barcode series. Test: degree 1 of item 28's result is a
  series of `shape == ()` whose values equal `result.values[i][1]`.
  **Catches:** the index applied to the time axis.

- [ ] **38. Progress, errors, and the GIL.** (existing) The series path runs
  through `_run_task` and `StoppableTask`, so the existing progress,
  error-propagation, and GIL-release tests (`test_gil_release.py`,
  `test_async_task.py`) cover the waiting side. Confirm that `UfuncTask`
  captures a worker exception the way the replaced tasks did; add no test.

- [ ] **39. (optional) CPU limits.** Persistence of the three-cloud series
  under `sb.system.limit_cpus(1)` equals the result with all threads; restore
  the limit afterwards. **Catches:** results that depend on scheduling.

## Delay embeddings, windows, and sampling (workflow 2)

Items 43-53 use the source series with times `[0.0, 0.5, 1.5, 2.5, 4.0]` and
values `[1.0, 2.0, 3.0, 4.0, 5.0]`.

- [ ] **40. Delay embedding.** (revise before starting: written with a
  domain, removed during item 26; an embedding's domain starts at
  `t_0 + (m - 1) * delay`) `TimeDelayEmbedding(dimension, delay)` returns
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
  - The embedding has domain `(2.0, 4.0)`, unit `"ms"`, `shape == (3,)`, and
    times `[2.0, 2.5, 3.0, 3.5, 4.0]` (the shifted observation times in the
    domain), and evaluating it at 1.5 raises `ValueError`. **Catches:**
    components that refer to times before the series starts, change times
    missing.

- [ ] **41. Embedding shaped observations, and embedding errors.** A channel
  of shape `S` embeds into shape `(m, *S)`, oldest first. Tests:
  - Times `[0.0, 1.0, 2.0]`, values `[[1, -1], [2, -2], [3, -3]]`,
    `dimension=2`, and `delay=1.0` give `shape == (2, 2)` and the value at 2.0
    equal to `[[2, -2], [3, -3]]`. **Catches:** the delay axis placed last,
    newest-first order.
  - Parametrized: `dimension=0`, `delay=0.0`, and `delay=-1.0` raise
    `ValueError`; `dimension=3` and `delay=1.0` on times `[0.0, 1.0]` raise
    `ValueError` (too short); a barcode series raises `TypeError`.
    **Catches:** invalid parameters accepted, an empty domain created
    silently, object values embedded.

- [ ] **42. Embedding several channels.** Each channel is embedded with the
  delay converted from the query unit into its own unit. Test:
  `TimeDelayEmbedding(dimension=2, delay=0.5)` of the series of channels A and
  B has domain `(1.0, 2.5)` and gives leaves `[20, 20]` and `[-1, -2]` at
  1.5. **Catches:** the delay applied in channel B's milliseconds.

- [ ] **43. Window bounds.** (decided during item 26, to be worked out before
  starting items 43-47: a windowed series is a `TimeSeries<TimeSeries<T>>`,
  a series whose observations are series, and each window is an ordinary
  series, evaluable from its first to its last observation like any other.
  There are no half-open windows and no window-specific domain rules. Still
  to decide: the outer series' timestamps (window starts, presumably), and
  which observations a window holds, for example whether the value carried
  in from before the window start becomes an observation at the start.
  Revise items 43-47, 49, and 51-56 accordingly; the text below predates
  this.) `Window(size, step=None, overlap=None)` returns a
  list of lazy start-aligned windows `[a, a + size)` with
  `a = start + k * step` while `a + size <= end`, where
  `step = size - overlap` and the default step is `size`. Incomplete windows
  are dropped. Test with the source series, parametrized over the arguments
  and the window domains: `size=2.0, step=1.0` and `size=2.0, overlap=1.0`
  give `[(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)]`; `size=2.0` gives
  `[(0.0, 2.0), (2.0, 4.0)]`; `size=5.0` gives `[]`. **Catches:** windows
  anchored at the end, partial windows kept, the overlap used as the step, an
  overlapping default.

- [ ] **44. Window contents and `series.window`.** `series.window(a, b)` is
  the window `[a, b)`, the same object `Window` produces; `[a, b)` must lie in
  the domain. Tests with the source series:
  - `w = series.window(1.0, 3.0)` has times `[1.5, 2.5]` (original timestamps),
    `w(1.0) == 2.0` (the observation at 0.5), and `w(2.75) == 4.0`.
    **Catches:** the value before the first observation in the window missing,
    timestamps moved to the window start.
  - `w(3.0)` raises `ValueError`, and `series.window(3.0, 1.0)` and
    `series.window(-1.0, 1.0)` raise `ValueError`. **Catches:** a closed
    window end, reversed bounds or bounds outside the domain accepted.

- [ ] **45. Window parameter errors.** Test, parametrized `ValueError`:
  `size=0.0`, `size=-1.0`, `step=0.0`, `overlap=2.0` and `overlap=-0.5` with
  `size=2.0`, and `step` together with `overlap`. **Catches:** windows that
  never advance, conflicting parameters accepted.

- [ ] **46. Embedding a window.** Test: `TimeDelayEmbedding(dimension=2,
  delay=1.0)` of `series.window(1.0, 3.0)` has domain `(2.0, 3.0)`; its value
  at 2.0 is `[2, 3]` (with the value at 1.0 from the source) and at 2.75 is
  `[3, 4]`; evaluating it at 1.5 or 3.0 raises `ValueError`. **Catches:**
  history taken from before the window, the half-open end lost.

- [ ] **47. Windows of several channels.** Test:
  `Window(size=1.0, step=1.0)` of the series of channels A and B gives windows
  with domains `(0.5, 1.5)` and `(1.5, 2.5)`; the second gives
  leaves 20.0 and -2.0 at 1.5 and has `times` leaves `[]` and `[1.5]`.
  **Catches:** channels cut at different bounds, channel B cut in
  milliseconds.

## Writing

Items 48-50 use the source series of items 43-46 and the embedding of item
40.

- [ ] **48. Writing a series.** `values[...] = x` (a tensor backed by the
  series) and writes through views taken from `values`, with the
  materialization of decision 4 for shared storage. Tests with values
  `[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]` and `times=[0.0, 1.0, 2.0]`:
  - After `other = series[0]` (sharing storage) and
    `series.values[1] = [7.0, 8.0]`, `series(1.0)` is `[7.0, 8.0]` and
    `other(1.0)` is still `[3.0, 4.0]`. **Catches:** a lost write, a write
    leaking into a view that shares storage.
  - `series.values[0, 1] = 9.0` gives `series.values[0] == [1.0, 9.0]`, and
    after `row = series.values[2]` and `row[0] = 0.0`, `series.values[2]` is
    `[0.0, 6.0]`. **Catches:** `values` or its views returned as detached
    copies.

- [ ] **49. Writing a window.** Test: with `w = series.window(1.0, 3.0)`,
  `w.values[0] = 30.0` gives `w(1.75) == 30.0`, keeps `w(1.0) == 2.0`
  (the carried-in observation) and `w.times == [1.5, 2.5]`, and leaves the
  source (`series(1.75) == 3.0`) and the window `series.window(1.0, 2.0)`
  unchanged. **Catches:** the carried-in observation dropped on materializing,
  the write reaching the source or a sibling window.

- [ ] **50. Writing an embedding.** Test: for item 40's embedding `e`,
  `e.values[1] = [0.0, 0.0, 0.0]` (time 2.5) gives `e(2.75)` equal to
  `[0, 0, 0]`, keeps `e(2.0) == [10, 20, 20]`, `e(3.25) == [20, 20, 30]`, and
  `len(e) == 5`, and leaves the source unchanged. **Catches:** a write into a
  lagged view reaching the source, materialization at observation times only
  (which would change the value at 3.25).

- [ ] **51. Sampler query times.** `RegularSampler(n_points).times(series)`
  returns `Times` with `a + (b - a) * k / n_points` for `k = 0, ..., n_points
  - 1`, in the series' unit. The end is never sampled, even for a closed
  domain. Test: `RegularSampler(n_points=4).times(w)` for `w =
  series.window(1.0, 3.0)` gives `[1.0, 1.5, 2.0, 2.5]` with the window's
  unit. **Catches:** the end point included, spacing `(b - a) / (n - 1)`.

- [ ] **52. Sampling one channel.** `RegularSampler(n_points)(windows)`
  evaluates each window at its query times and returns a `PointCloudTensor`
  of shape `(len(windows), n_channels)` whose clouds have `n_points` points;
  each observation is flattened into a point of dimension `prod(shape)`, and a
  scalar channel gives 1-D points. Test: the three windows of
  `Window(size=2.0, step=1.0)` with `n_points=2` give shape `(3, 1)` and the
  clouds `[[1], [2]]`, `[[2], [3]]`, and `[[3], [4]]`, of dtype pcloud64.
  **Catches:** windows stacked out of order, values at window starts missing,
  the wrong precision.

- [ ] **53. Sampling several channels.** Test: `Window(size=1.0, step=1.0)` of
  the series of channels A and B with `n_points=2` gives shape `(2, 2)` with
  the clouds `[[10], [20]]` and `[[-1], [-1]]` in the first window and
  `[[20], [20]]` and `[[-2], [-2]]` in the second. **Catches:** channels
  swapped, channels evaluated on one grid.

- [ ] **54. Sampler errors.** Test, parametrized: `n_points=0` and an empty
  list raise `ValueError`; a point-cloud series raises `TypeError`.
  **Catches:** an empty result without a type, object values sampled into
  point clouds.

- [ ] **55. The delay workflow end to end.** Test in
  `test/python/test_integration.py`: times `[0.0, 0.5, ..., 3.5]` and values
  `[0, 1, 0, -1, 0, 1, 0, -1]` through
  `TimeDelayEmbedding(dimension=2, delay=0.5)`,
  `Window(size=1.0, step=1.0)` (three windows), `RegularSampler(n_points=4)`,
  and `compute_persistent_homology(..., max_dim=1)` give clouds of shape
  `(3, 1)` with four 2-D points each and barcodes of shape `(3, 1, 2)`.
  **Catches:** steps that do not compose.

- [ ] **56. Saving views.** Windows and embeddings are saved materialized
  (decision 9), with no stored domain. A windowed series is a series of
  series (see item 43), so saving one saves each window as an ordinary
  series; revise this item with item 43. Test, parametrized over `save`/`load`
  and `pickle`: the window `[1.5, 2.5)` of item 42's embedding loads with
  the same number of channels, domain, unit, and values at 1.5 and 2.0, and
  evaluating it at 2.5 still raises `ValueError`. **Catches:** a lag, bound,
  or query unit lost on saving, the half-open end lost.

- [ ] **57. Golden files.** Pin the formats (TESTING.md principle 11) with a
  generator modeled on `test/golden/serialization/generate_point_cloud_v3.py`,
  which records explicit expected values. Write item 28's barcode32 result,
  the series of channels A and B, and item 56's window as `.sb` and
  pickle files, and teach `test_serialization_goldens.py` to compare series
  (number of channels, unit, times, values at listed times). **Catches:** a later
  change to a payload that breaks saved files.

## Documentation and release

Code items update the docstrings they touch (AGENTS.md). These items add the
pages.

- [ ] **58. API reference.** `docs/api_timeseries.rst` is in the toctree in
  `docs/stablebear.rst` (added with item 20). Check that the docstrings state the contracts of
  decisions 1-4 and 6-8: the domain, previous-value
  evaluation and snapping, units, query units and conversion, channels and
  their order, time indexing, observation shapes and flattening into
  points, copies and lazy derived
  channels. Check that `compute_persistent_homology`, the three summaries,
  `save`, and `load` document series input and output.

- [ ] **59. User guide.** A `docs/timeseries.rst` guide linked from
  `docs/userguide.rst`: times, units, the domain, and channels, then two small
  examples. The first follows a point-cloud series through persistence to
  stable ranks; the second takes two numeric channels in different units
  through a delay embedding, windows, and sampling to persistence. Link each
  operation to its API entry on first mention. The page exists since item 20
  with sections on evaluation and snapping, with figures from
  `docs/_static/gen_timeseries_snap_fig.py`; further figures follow the
  dropdown rule in AGENTS.md.

- [ ] **60. Changelog.** A New features entry in `CHANGELOG.md` linking #242,
  under the heading of the release that ships it (to be decided; 0.5.1 is in
  progress).

## Final checks

- [ ] **61. Test run.** After a CMake build and install, run the new test
  files, then the full Python suite and `sb_test` from `test/`.
- [ ] **62. Docs build.** Run `make html` in `docs/` (with `LC_ALL=C.UTF-8` if
  the locale is missing), check for warnings, and read the new pages.
- [ ] **63. Push.** Push `issue-242-timeseries`, open a PR against `main`, and
  remove this plan before merging.

## Coverage of the acceptance criteria

| Criterion in #242 | Items |
| --- | --- |
| Time, domain, evaluation, shape, and ownership contracts; empty and singleton behavior | 1-9, 19, 48-50 |
| `Times`, native storage and access, default seconds, explicit units, one timestamp per observation | 10-13, 17-18, 20-21 |
| Floating-point arrays with explicit times (packed tables dropped) | 11-12, 16 |
| Native operation contract, dispatch, and adapter; calls, parameters, CPU limits | 6, 28-29, 32, 36, 38-39 |
| Persistence and barcode summaries of series | 28-35 |
| Delay embedding and windows | 40-47 |
| Regular sampling into point clouds and through persistence | 51-55 |
| Save, load, and pickle for both workflows | 26-27, 56-57 |
| Tests and documented examples | all test items, 58-59 |
| Channels with their own grids and units (added) | 3, 21-25, 23a, 29, 42, 47, 53 |

## Out of scope

As in the issue: datetime timestamps, sample-count delays and windows, other
samplers, interpolation, partial windows, gap policies, tensors of series,
reductions over time, provenance, and normalization. Also deferred here:
joining channels into one point cloud when sampling, series with channels of
different element types, observations whose shape changes over time
(a later per-observation tensor storage mode), `NestedTensor` values (6
more element types, cheap to add once a use such as spike trains needs
them),
`Tensor<TimeSeries>`, views instead of copies for stored channels,
and the branch's notebooks and scikit-learn transformers.
