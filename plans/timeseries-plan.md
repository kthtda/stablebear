# Time series (#242): implementation plan

This plan implements #242 on `issue-242-timeseries`, branched from `main` at
`05ce5ba4a`: unit-bearing timestamps, time series made of channels with their
own time grids and units, persistence of snapshot series (workflow 1), and
windowed delay embeddings evaluated into point clouds (workflow 2), with
serialization and documentation. Useful parts of the `origin/timeseries`
branch are recycled where they fit (see "Recycling").

Where the plan departs from the issue:

- Channels with their own time grids and units are in this release; the issue
  defers a channel-combination API.
- Every channel holds one observation per time, a tensor of a fixed `shape`
  (often `()`).
- `times` is always given explicitly; a table's column 0 is never read as
  timestamps.
- A delay embedding adds a leading delay axis to each observation. There
  is no sampler: a tensor of windows evaluates at a `FloatTensor` of times,
  as a `PcfTensor` does, and persistence reads the result's last two axes
  as point clouds, so `PointCloudTensor` needs no `FloatTensor`
  constructor (item 51).
- `Window` takes a size and a step, not an overlap: the overlap is
  `size - step`, and a step larger than the size, which leaves gaps, has no
  overlap to give (item 43).
- Windows are a 1-D tensor of series (`Tensor<TimeSeries<T>>`), each window
  an ordinary series of the observations it contains, with their start and
  end times as a float tensor on request (item 43).
- As in Polars `group_by_dynamic`, `closed` chooses which ends of a window
  are included, and with `partial=True` windows continue past the last
  full one as partial windows (item 47a).

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
observable from Python (TESTING.md principle 4). The exception is
`gallop_upper_bound` (`test/test_gallop.cpp`), a general algorithm whose
forward direction no time-series path uses yet.

## Conventions

- **Files.** Source: `include/sbear/timeseries.hpp` (`Times`, channels,
  `TimeSeries<T>`), `include/sbear/timeseries/time_delay_embedding.hpp` and `windowing.hpp` (windows,
  embeddings), `include/sbear/ufunc.hpp` (operations and
  `UfuncTask`), `include/sbear/io/timeseries_io.hpp`,
  `src/python/py_timeseries.{hpp,cpp}`, and the package
  `stablebear/timeseries/` with thin wrappers. Tests:
  `test/python/test_timeseries.py` (storage, access, evaluation, units,
  channels, serialization), `test/python/test_time_delay_embedding.py`,
  `test/python/test_windowing.py` (windows and evaluating them), and
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
- **Parameters.** Spell out `dimension`, `delay`, `size`, `step`, `closed`,
  `partial`, `n_points`, `unit`, `max_dim`, and `reduced` (TESTING.md principle 18) unless the test
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
  windows, delay embeddings, evaluation, and analysis. Python only binds
  arguments and wraps results.
- `series(t)` looks up each channel on its own grid, in its own unit, and
  returns the value for one channel or a `NestedTensor` with one leaf per
  channel for several. `series[i]` selects channels by index.
  `Window(size, step, closed, partial)(series)` is a tensor of windows, and
  `return_bounds=True` adds their start and end times.
- Every channel is a view: a `shared_ptr` to its storage plus a descriptor
  (domain, lags, unit scale). Windows, delay embeddings, channel selection,
  and unit changes copy no values. Series are immutable: `values` is a
  copy (decision 4). A delay
  embedding of a numeric channel of shape `S` has shape `(m, *S)` and equals
  the delayed values at every time of its domain (item 40).
- A tensor of series, such as windows, evaluates at times `t` of shape
  `t_shape` to a tensor of shape `(*tensor_shape, *t_shape, *shape)`, as a
  `PcfTensor` does, or a `NestedTensor` with one such leaf per channel.
  Evaluating windows of an embedding on a grid gives point clouds in the
  last two axes, which persistence reads directly (items 51-53).
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

- [x] **1. Domain and evaluation.** (decision: (a), closed domain for series built from observations, half-open for windows; revised during item 26: there is no domain. A series can be evaluated from the first to the last observation of every channel, both included, checked against the timestamps on each query; there is no `domain` property, and a series with an empty channel cannot be evaluated. Windows follow the same rule: there is no open or closed end anywhere, see item 43. Revised again during item 43: windows are `[a, a + size)`, so a channel may carry an open end after its last timestamp, set by `Channel::until`; it is the only stored bound, and series without one keep the rule above. Revised again in item 47a: a window's channel may also have an open start, or a closed end after its last timestamp; `Domain` records both ends and whether each is closed (`closedStart`, `closedEnd`), as `Channel` does (`start_closed`, `end_closed`), matching `closed` in Python.) Evaluation
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

- [x] **4. Ownership.** (decision: copy in, views out, materialize on write, as subsampled tensors do; revised at item 48: series are immutable, and `values` returns a copy, since nothing else can change a series either, such as adding observations. Writing through `values` with copy-on-write was tried: copying when `values` is read left a tensor kept from before writing into views made later, and copying at write time needs every slice of `values` to reach its tensor through the series, as a proxy. Revisit when series gain operations that change them.)
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
    domain checks, unit conversion, windows, delay embeddings, and
    evaluation.
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
    until a second delay mode exists), `Window(size, step=None)`
    (windows always start at time 0; the `relative` option and `overlap`,
    which is `size - step`, were dropped during item 43; revised with items
    40-56; `RegularSampler` was dropped at item 51).
  - `from_channels` takes a list of one-channel series of one element
    type; channel `i` is the `i`th series. `unit` defaults to the
    channels' shared unit and is required when their units differ.
  - A `TimeSeries` exposes `n_channels`, `dtype`, `unit`, `times`
    (a `Times` for one channel, a `NestedTensor` with one leaf per channel in
    the query unit for several), `to(unit)`, `evaluate(t)`, `series(t)` (the
    same as `evaluate(t)`), and `series[i]` or `series[[i, j]]` (channel selection by index; negative
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
  - `Window(size, step)(series, return_bounds=False)` returns a 1-D
    `TimeSeriesTensor` of dtype `sb.series(series.dtype)`, one window per
    entry; with `return_bounds=True` also a float64 `FloatTensor` of shape
    `(n_windows, 2)` whose row `k` is `(a_k, a_k + size)` in the series'
    unit. (Revised during item 43: windows were first a series of series
    over their start times. Its evaluation by previous value meant nothing
    for windows, `series[i]` selected channels rather than windows, and its
    results could not go to `pdist` or the reductions. Axis or element tags
    carrying the times through every tensor operation were considered and
    rejected as too complex.)
  - `windows(t)` evaluates every window at the times `t` (item 51). (A
    `RegularSampler(n_points)(windows)` returning a `PointCloudTensor` was
    dropped at item 51.) Joining channels into one cloud is a later
    extension.
  - `TimeSeries` is generic (`TimeSeries[PointCloud]`) in Python
    3.10-compatible annotations.
  - An empty list gives an empty float64 channel, so a typed empty channel of
    point clouds is built from `sb.zeros((0,), dtype=sb.pcloud64)`.

- [x] **8. Views.** (decision: every channel is a view of shared storage; a write materializes the whole view. Revised during item 40, first with index tables and then, for memory, as follows: a view made by `stack_at(times, lags)` stacks a source channel's observations at lagged times. It keeps its times, its lags, and per observation a base, the source observation for the smallest lag; each component is found when read by a galloping search back from the base (steps of 1, 2, 4, ... observations, then bisection; `gallop_upper_bound` in `include/sbear/algorithms/gallop.hpp`), with the source's snapping. A view costs two numbers per observation, and views of views nest through their source. Revised again during item 43: every view evaluates as any channel does, by previous value on its own times with its own snapping (its source's settings, applied to its own gaps), and reads values from its source by index; snapping against the source happens only when a view is built, where computed times such as `0.3 - 0.1` must reach the observations they mean. Views that looked up the source at query time showed observations they did not hold, such as the one at a window's open end, and would have shown the dropped ones of a thinning view. `Channel` offers only `stack_at`, `take` (observations by index, no new axis; it replaced `view_at`, one lag), and the cursor lookup `index_at(next, t)`, which snaps. It does not hand out its timestamp array: `time(k)` reads one timestamp to build new times from and `times()` returns a copy, so the only way from a time to an observation is `index_at`, and every lookup outside `Channel` snaps (revised after item 43, when window bounds first searched the raw timestamps); derived series, embeddings and windows alike, are built from them outside `Channel` and `TimeSeries`, in `timeseries/time_delay_embedding.hpp` and `timeseries/windowing.hpp`, and look like any other series to callers, including ufuncs, which read elements through `flat_value(i)`. The lags, domain, and descriptor described below predate this and are kept for history.)
  - Like a tensor view, a channel is a `shared_ptr` to its storage plus a
    small descriptor of how a query time maps into it: the domain (start, end,
    and end closedness), `lags` (time offsets that add leading observation
    axes), and `scale` (from the query unit to the storage unit). A freshly
    built channel is the identity view.
  - A window (item 43) covers the observations in `[a, a + size]` and shifts
    their timestamps; it cuts every channel at the same bounds in the query
    unit and works for any element type. A delay
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
    before the domain start followed by the change times inside the
    domain, each with the view's value there, plus the domain. Under
    previous-value evaluation the result evaluates identically everywhere in
    the domain. An embedding with `m` lags may have up to `m * n` change times;
    that memory is spent only on a write.
  - Internal lookups skip the domain check, because shifted and converted
    times may round just outside it. Shifted times may also land one ulp apart
    (`0.2 + 0.1` and `0.3`); snapping makes their values equal, and item 40
    merges them.

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
  counted from the wrong end, no bounds check.

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

- [x] **24b. Evaluating at many times.** (added after item 35) `series(t)`
  also takes an array-like or `FloatTensor` `t` of shape `t_shape`, as PCF
  tensors do (`tensor_eval.hpp`): one channel gives a tensor of shape
  `(*t_shape, *shape)`, several a `NestedTensor` with one such leaf per
  channel; a scalar `t` behaves as before. Natively, the domain is checked
  once, the queries are sorted once in the query unit, and each channel
  merges them, converted to its own unit, against its timestamps in
  `O(n + m)`, and in `O(m log n)` since the cursor gallops (item 40's
  efficiency pass), with the same snapping as a single query; results come back in
  the caller's order. Items 51-53 evaluate windows through it. Tests with times
  `[0.0, 1.5, 4.0]` and values `[10.0, 20.0, 30.0]`, and with channels A and
  B:
  - `series([4.0, 0.0, 1.0, 1.5])` is `[30, 10, 10, 20]`, and
    `series([[0.0, 4.0], [1.5, 1.0]])` has shape `(2, 2)` with the values in
    place. **Catches:** results in sorted rather than caller order, the
    query shape lost.
  - For channels A and B, `series([1.25, 1.5])` has leaves `[20, 20]` and
    `[-1, -2]`. **Catches:** channel B read in the query unit, one sorted
    grid shared by all channels.
  - `series([0.0, np.nextafter(1.5, 0.0)])` equals the two single
    evaluations, and `series([0.0, 4.5])` raises `ValueError`. **Catches:**
    snapping or the domain check skipped on the batch path.

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

- [x] **28a. Input core dimensions.** (added during item 33) An operation
  may declare a `core_rank`, the trailing input dimensions it consumes per
  call, as a NumPy gufunc's core dimensions; it then receives a
  `CoreBlock` (a contiguous span and its shape). Adapters split each shape
  into batch and core dimensions: a tensor of shape `(..., n, d)` runs once
  per batch index (copied first if not contiguous), and a series channel of
  shape `(..., n, d)` once per observation and batch index. A coordinates
  persistence operation of core rank 2 runs Ripser on an `(n, d)` block, so
  float tensors and float series need no point-cloud conversion, and a
  `FloatTensor` of shape `(b, n, d)` is a batch of `b` clouds. Test,
  parametrized over float32 and float64: a strided view of a rank-3
  `FloatTensor` gives shape `(b, max_dim + 1)` and each cloud's barcodes
  match the point-cloud path. **Catches:** a strided view read as
  contiguous, batch and core axes swapped.

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

- [x] **31. Typed empty series.** Test: an empty pcloud32 series gives a
  barcode32 series with no observations and `shape == (2,)` for `max_dim=1`,
  and unit `"s"`. **Catches:** an untyped or missing empty
  result, failure on empty input.

- [x] **32. Keyword data argument and parameter forwarding.** Test:
  `compute_persistent_homology(X=series, max_dim=0, reduced=True)` has each
  observation isomorphic to the persistence of its cloud alone with the same
  parameters. **Catches:** `max_dim` or
  `reduced` dropped on the series path, keyword calls failing.

- [x] **33. Unsupported series.** A float series whose observations have
  shape `(..., n, d)` is read as point clouds, one point per row, as a
  `FloatTensor` is (done with item 28a, through the coordinates operation;
  tested against the matching point-cloud series in both precisions). Test, parametrized `TypeError`:
  `compute_persistent_homology` of a float series of shape `()` and of a
  barcode series (done), and `barcode_to_stable_rank` of a point-cloud series
  (with item 34). **Catches:** a non-matrix numeric series read as point
  clouds, an unsupported series passed to a native task.

- [x] **34. Stable rank of a series.** The stable-rank operation, its private
  entry point in `src/python/persistence/py_barcode_summary.cpp`, and a series
  branch in `_barcode_to_pcf`. Test: a series of barcode
  channels with two distinct barcodes each gives a PCF series
  with the same number of channels and times whose channel values array-equal
  `barcode_to_stable_rank` of each channel's `BarcodeTensor`. **Catches:**
  times dropped, channels mixed up, the input released before the
  task finishes.

- [x] **35. Betti curves and accumulated persistence.** The other two
  operations and entry points, forwarding `max_death`. Tests:
  - Parametrize item 34's test over the three summary functions.
    **Catches:** a summary bound to another summary's kernel.
  - `barcode_to_accumulated_persistence(series, max_death=1.25)` array-equals
    the function applied to each channel's `BarcodeTensor` with
    `max_death=1.25`, and differs from the result without `max_death` in
    every channel.
    **Catches:** `max_death` dropped on the series path.

- [x] **36. Tensors through `UfuncTask`.** (done early, in `46069ff0e`; single clouds, matrices, and barcodes also run as 0-d inputs) Route the existing tensor paths of
  `compute_persistent_homology` (point clouds and distance matrices, including
  indexed tensors) and the three summaries through `UfuncTask` with a tensor
  adapter, and remove `RipserTaskImpl` and `BarcodeSummaryTask` if nothing
  else uses them. Keep the public results, including the existing squeeze of
  a length-1 input. No new tests: the existing persistence and summary tests
  must pass unchanged. **Catches:** behavior changes in the tensor paths.

- [x] **37. (optional) Selecting a value index.** (skipped) `series.select(k)` (name to
  be decided) indexes the observation axes and keeps time, for example to
  take degree 1 of a barcode series. Test: degree 1 of item 28's result is a
  series of `shape == ()` whose values equal `result.values[i][1]`.
  **Catches:** the index applied to the time axis.

- [x] **38. Progress, errors, and the GIL.** (confirmed, and tested for every task in `test/test_stoppable_task.cpp`: an exception thrown in a parallel worker is rethrown once by `wait()` and by `wait_for()` when finished; `UfuncTask` runs through the same path, as a scratch program confirmed for a tensor and a series; the C++ test replaces the end-to-end coverage `test_async_task.py` cited from the removed `test_bugscan_persistence.py`) The series path runs
  through `_run_task` and `StoppableTask`, so the existing progress,
  error-propagation, and GIL-release tests (`test_gil_release.py`,
  `test_async_task.py`) cover the waiting side. Confirm that `UfuncTask`
  captures a worker exception the way the replaced tasks did; add no test.

- [x] **39. (optional) CPU limits.** Persistence of the three-cloud series
  under `sb.system.limit_cpus(1)` equals the result with all threads; restore
  the limit afterwards. **Catches:** results that depend on scheduling.

## Delay embeddings, windows, and evaluation (workflow 2)

(Revised after item 39 for the decisions of items 26 and 43: no stored
domain, and windows as series of series, with the choices that follow from
them settled item by item; revised again during item 43: windows are a
tensor of series, with their bounds on request.)

Items 43-53 use the source series with times `[0.0, 0.5, 1.5, 2.5, 4.0]` and
values `[0.0, 5.0, 15.0, 25.0, 40.0]`, each `10 * t`, so a value shows which
observation it came from.

- [x] **40. Delay embedding.** (reopened: first done in `1185ecd29` with
  points only at the source's timestamps, which made the embedding at `t`
  the point of the last such timestamp rather than the delayed values at
  `t`) `TimeDelayEmbedding(dimension, delay)` returns a series that
  evaluates at every `t` from `t_0 + (m - 1) * delay` to `t_{n-1}`, both
  included, to the point `(x(t - (m - 1) * delay), ..., x(t - delay), x(t))`,
  oldest first as in giotto-tda, GUDHI, DelayEmbeddings.jl, and Kantz and
  Schreiber, so `shape == (m,)`. Its times are decision 8's change times:
  the times in that domain where a component reaches an observation,
  `t_i + k * delay` for `k = 0, ..., m - 1`, at most `m * n` of them.
  Shifted times that round to either
  side of an observation (`0.2 + 0.1` and `0.3`) give the same point; each
  run of times whose components read the same observations counts once, at
  the time its members snap to: the source's own timestamp if the run has
  one (0.3, not `0.2 + 0.1`), else its latest time. The first point stays at
  the domain's start, and the last is `t_{n-1}`, a timestamp, so the domain
  keeps both ends (a run reaching from the start to `t_{n-1}` gives both).
  On a regular grid with the delay a multiple of the spacing, every point
  after the first is at one of the source's timestamps. It is a view of the source built in
  `timeseries/time_delay_embedding.hpp`: one merged pass walks the shifted copies of the
  timestamps without storing them, one cursor per component finds the runs
  through `Channel::index_at` (now public), and `Channel::stack_at`
  makes the view (decision 8), 16 bytes per point. Each point's components
  are the source evaluated at the point's delayed times, with the source's
  snapping; between points it evaluates by previous value on its own times
  with its own snapping, as every view does (revised during item 43: it
  first looked up the source at the delayed query times, so within the snap
  distance below a point it could differ from a saved copy); `TimeSeries` and its
  binding know nothing of it. Tests with times `Times([0.0, 1.0, 2.5, 4.0], unit="ms")`,
  values `[0.0, 10.0, 25.0, 40.0]` (each `10 * t`, so a carried value stands
  out), `dimension=3`, and `delay=1.0`:
  - The times are `[2.0, 2.5, 3.0, 3.5, 4.0]` and the values
    `[[0, 10, 10], [0, 10, 25], [10, 10, 25], [10, 25, 25], [10, 25, 40]]`,
    and the embedding at 3.25 is `[10, 10, 25]`, that is
    `(x(1.25), x(2.25), x(3.25))`. **Catches:** points only at the source's
    timestamps (3.25 would give `[0, 10, 25]`), newest-first order, the same
    delay for every component.
  - The embedding has unit `"ms"` and `shape == (3,)`, is `[0, 10, 10]` at
    2.0, the domain's start, and evaluating it at 1.75 raises `ValueError`
    ("outside the domain"). **Catches:** a domain starting at the first
    source timestamp after `t_0 + (m - 1) * delay` (2.5), points whose
    components reach before the series starts.

- [x] **40a. Embedding stride.** (skipped: an embedding can be evaluated
  at any time in its domain, so a grid of query times thins its points
  instead of a stride.) An embedding has a
  point at every change time (item 40), a superset of the points at
  observation times that giotto-tda, GUDHI, DelayEmbeddings.jl, and nolds
  give by default. giotto-tda (`stride`) and GUDHI (`skip`) can keep only
  every k-th point, counted in samples. Decide whether to add `stride=1` to
  `TimeDelayEmbedding`. Every `stride`-th change time counts samples only
  when the delay is a multiple of a regular spacing, so a stride needs a
  definition for other series, for example over the source's timestamps.
  In workflow 2 the grid windows are evaluated on (item 51) already sets
  the spacing, so a
  stride mainly serves embeddings used on their own, such as thinning
  points before persistence. If added: one test that `stride=2` keeps every
  other time and point.

- [x] **40b. Embeddings of non-numeric series.** (decided with item 43:
  allowed. `register_derived_element` binds `delay_embed` and `windows` for
  all 19 element types, the Python check is gone, and item 41's barcode case
  became a test of a PCF embedding, whose observations are `m` delayed PCFs.
  Windows of every element type evaluate, item 51.) Before, `TimeDelayEmbedding` accepted only the 7 numeric element types and raised
  `TypeError` otherwise (item 41 tests this for barcodes). The native
  `delay_embed` is built from `index_at` and `stack_at`, which work for every
  element type, so only the binding in `py_timeseries_derived.cpp` and the
  Python check restrict it. Decide whether to allow all 19 types, for
  example an embedding of a point-cloud series whose observations are `m`
  delayed clouds, and what the sampler and persistence would then do with
  such observations. If allowed: bind `delay_embed` for every type, drop the
  check, and turn item 41's barcode case into a test of a point-cloud
  embedding.

- [x] **41. Embedding shaped observations, and embedding errors.** A channel
  of shape `S` embeds into shape `(m, *S)`, oldest first. Tests:
  - Times `[0.0, 1.0, 2.0]`, values `[[0, 0], [10, -10], [20, -20]]`,
    `dimension=2`, and `delay=1.0` give `shape == (2, 2)` and the value at 2.0
    equal to `[[10, -10], [20, -20]]`. **Catches:** the delay axis placed last,
    newest-first order.
  - Parametrized: `dimension=0`, `delay=0.0`, and `delay=-1.0` raise
    `ValueError`; `dimension=3` and `delay=1.0` on times `[0.0, 1.0]` raise
    `ValueError` (no time is far enough from the start). **Catches:** invalid
    parameters accepted, an embedding that can never be evaluated created
    silently.
  - (with item 40b) PCFs of values 0, 10, and 20 at times `[0.0, 1.0, 2.0]`,
    `dimension=2`, `delay=1.0`: a pcf64 series of shape `(2,)` at times
    `[1.0, 2.0]` whose values array-equal `[[p0, p1], [p1, p2]]`.
    **Catches:** object elements rejected or stacked out of order.

- [x] **42. Embedding several channels.** (reopened with item 40) Each
  channel is embedded on its own times, with the delay converted from the
  query unit into its own unit, and the domain is the intersection of the
  channels' domains. Test: `TimeDelayEmbedding(dimension=2, delay=0.5)` of
  the series of channels A and B has times `[0.5, 1.0, 1.5, 2.5]` and
  `[1.0, 1.5, 2.0, 3.0]` (B's 1000, 1500, 2000, and 3000 ms, reported in
  seconds), evaluates at 1.0 and 2.5, raises `ValueError` at 0.75, and gives
  leaves `[20, 20]` and `[-1, -2]` at 1.75. **Catches:** the delay applied
  in channel B's milliseconds (leaf 1 would be `[-2, -2]`), one channel's
  times used for another, the union of the channels' domains.

- [x] **42b. Embedding views.** (added after item 40; reopened with item 40)
  Tests of the paths through the views that items 40-42 do not reach:
  - Parametrized, with `dimension=2` and `delay=0.1`: times
    `[0.1, 0.2, 0.3, 0.4]` with values `[1, 2, 3, 4]` give times
    `[0.2, 0.3, 0.4]` and points `[[1, 2], [2, 3], [3, 4]]`, since
    `0.2 + 0.1` computes just above 0.3 and `0.3 - 0.1` just below 0.2;
    times `[0.0, 0.7, 0.8, 0.9]` with values `[0, 7, 8, 9]` give times
    `[0.1, 0.7, 0.8, 0.9]` and points `[[0, 0], [0, 7], [7, 8], [8, 9]]`,
    since `0.7 + 0.1` computes just below 0.8 and snaps to it; times
    `[0.7, 0.8, 0.9]` with values `[7, 8, 9]` give times `[0.7 + 0.1, 0.9]`
    and points `[[7, 8], [8, 9]]`, the first point staying at the domain's
    start just below 0.8. **Catches:** component lookups that skip snapping,
    one point at two times an ulp apart, a point recorded at a time that
    snaps to an observation instead of at the observation, the domain's
    start lost when merging.
  - Times `[-3.0, -2.5, nextafter(-1.5, 0)]`, values `[-30, -25, -15]`,
    `snap_tolerance=0.0`, `dimension=2`, `delay=1.0`: times
    `[-2.0, -1.5, nextafter(-1.5, 0)]` and points
    `[[-30, -25], [-25, -25], [-25, -15]]`;
    the last two times minus 1.0 both round to -2.5. **Catches:** delayed
    lookup times validated as strictly increasing timestamps, which raised
    `ValueError` for this valid series.
  - Times `[0, 1, ..., 8]` with values `[0, 10, ..., 80]`, `dimension=2`,
    `delay=6.0`: times `[6, 7, 8]` and points `[[0, 60], [10, 70], [20, 80]]`.
    **Catches:** the search back from the base stopping short or overshooting
    when a component is many observations back.
  - An embedding (`dimension=2`, `delay=1.5`) of item 40's embedding has
    `shape == (2, 3)`, times `[3.5, 4.0]`, and the points
    `[[0, 10, 10], [10, 25, 25]]` and `[[0, 10, 25], [10, 25, 40]]`.
    **Catches:** views of views not composed.
  - Item 40's embedding at `[4.0, 3.25]` gives both points in the caller's
    order. **Catches:** the batch lookup bypassing the view.
  - Persistence of an embedding of 1-D points equals persistence of the same
    points stored plainly. **Catches:** the ufunc adapter reading a view
    wrongly.

- [ ] **42c. (later) Per-channel dimensions and delays.** As in Cao, Mees,
  and Judd (Physica D 121, 1998, eq. 1), whose reconstruction concatenates
  each channel's delay vector with its own dimension `d_i` and delay
  `tau_i`: `TimeDelayEmbedding` accepts a list for `dimension` and for
  `delay`, one entry per channel, and a single value still applies to every
  channel. Channel `c` embeds with `d_c` and `tau_c` into shape
  `(d_c, *S_c)` and keeps its own times; the series' domain is the
  intersection, from `max_c(t_{c,0} + (d_c - 1) * tau_c)` to
  `min_c(t_{c,last})`, which is the paper's shared start. A CMJ vector is
  then built downstream by evaluating the series and
  concatenating the leaves. `delay_embed` takes vectors of dimensions and
  delays. Tests: per-channel dimensions give per-channel shapes and the
  intersected domain; a list of the wrong length raises `ValueError`.
  (The paper lists each channel newest first; we keep oldest first, as the
  libraries do.)

- [x] **42a. Tensors of series.** (implemented together with item 43, whose
  `Window` gives the only tensors of series; saving moved to item 56;
  revised during item 43: first done as series of series, which are gone)
  Windows (items 43-47) are a `Tensor<TimeSeries<T>>`, bound through
  `register_tensor_element` for each of the 19 inner element types, with
  the Python wrapper `TimeSeriesTensor` and a dtype for it. Tensor
  elements are default-constructed, so `TimeSeries<T>` needs a default (an
  empty one-channel series), and tensor bindings need `==`, which compares
  units, times, shapes, values, and snap settings. Saving waits for item
  56. The dtype is parametrized by the inner element type,
  `sb.series(sb.float64)`. `sb.series(dtype)` returns the same object for
  the same inner dtype, since dtypes are compared with `is`, and its inner
  dtype reads back. For now users get tensors of series only from
  `Window`. `TimeSeriesTensor` is not exported until a use needs it.
  Windows work for all 19 element types. Test: two windows of a float64
  series are a `TimeSeriesTensor` of shape `(2,)` and
  `dtype is sb.series(sb.float64)`, and `windows[1]` is the second window,
  a float64 series. **Catches:** an inner series bound to the wrong native
  type, windows out of order, a tensor of windows mistaken for a tensor of
  floats.

- [x] **43. Windows.** `Window(size, step=None)(series)`
  returns a 1-D tensor of series with one window per start
  `a_k = t_0 + k * step` while `a_k + size <= t_{n-1}`, where the default
  step is `size`; a smaller step makes windows overlap, a larger one leaves
  gaps. With `return_bounds=True` it also
  returns the bounds, a float64 tensor of shape `(n_windows, 2)` with rows
  `(a_k, a_k + size)` in the series' unit, as computed, not snapped:
  each channel snaps with its own timestamps and settings, so one window's
  channels can snap differently and no single snapped bound exists. Window
  `k` covers `[a_k, a_k + size)`, as the issue asks and as Polars
  `group_by_dynamic`, Spark, and Flink do by default: it is a series with
  the source's channels, each holding the value in effect at `a_k`, at time
  0, then the observations with `a_k < t < a_k + size`, with the open end
  `size` (`Channel::until`). Every window channel can thus be evaluated
  from 0 up to `size`, past its last observation too, but not at `size`; a
  window is never empty, and an observation at `a_k + size` belongs only to
  the next window. An open end is the one place a series keeps an end that
  is not a timestamp: `TimeSeries::domain` takes the earliest end of its
  channels, open if any channel's is, and the error names it, as in
  `[0, 2)`. A delay embedding of a window keeps the open end, and its
  change times run up to it. (Revised during item 43: windows first held
  only the observations in `[a_k, a_k + size]`, and the sampler read values
  outside them from the source. The sampler must not see values the user
  cannot, so the window holds them itself; the closed end then gave way to
  `[)`, which needed views that evaluate on their own points, decision 8.
  A copy of the last observation one float below `size` first stood in for
  the open end, but it showed in `times` as `1.9999999999999998` and would
  have needed writes to keep it equal to the observation it copied.) A
  window's timestamps are always
  shifted so that `a_k` is 0, in each channel's own unit (a `relative=False`
  option keeping the original timestamps was dropped). Window bounds are
  computed in floating point (`0 + k * 0.3` is `0.8999999999999999` for
  `k = 3`), so they are cut with the channel's snapping, as evaluation does:
  a bound that falls just short of an observation reaches it: a window
  whose start reaches an observation is shifted by that observation's time,
  so its first time is exactly 0, and an end that reaches one excludes it.
  A bound that rounds just past an observation still misses it, since
  snapping only moves forward. `t_0` and `t_{n-1}` are the first and last
  times at which every channel has an observation (`TimeSeries::domain`).
  As embeddings are, windows are built in `timeseries/windowing.hpp` from a
  generic `Channel` view, so `Channel` knows nothing of windows: each window
  channel is `Channel::take(times, indices)`, a view whose observation `k`
  is the source observation `indices[k]`, at the window's shifted times. It
  copies no values and stores 16 bytes per observation per window, so
  overlapping windows multiply that by the overlap. Shifting can round two
  close times together, so each time is at least one float above the one
  before. Tests with the source series:
  - Parametrized over the arguments and the starts in the bounds:
    `size=2.0, step=1.0` gives `[0.0, 1.0, 2.0]`; `size=1.0, step=1.5` gives
    `[0.0, 1.5, 3.0]`; `size=2.0` gives `[0.0, 2.0]`; the ends are the
    starts plus `size`. **Catches:** windows
    anchored at the end, partial windows kept, a step larger than the size
    rejected or clamped, an overlapping default, bounds out of step with the
    windows.
  - With `size=2.0, step=1.0`, the window at 1.0 has times `[0.0, 0.5, 1.5]`
    and values `[5.0, 15.0, 25.0]`; it is 5 at 0 and 25 at 1.75, after its
    last observation, and evaluating it at 2.0 raises `ValueError` naming
    the domain `[0, 2)`. **Catches:** no value at the window's start, a
    closed end or one at the last observation, the observation before the
    start kept at its own time, timestamps not shifted or shifted by the
    wrong amount.
  - With `size=1.5, step=1.5`, only the second window holds the
    observation at 1.5 (values `[0, 5]` and `[15, 25]`). **Catches:** an
    observation at the end kept in the earlier window.
  - With `size=0.5, step=0.5`, the window at 3.0, between the observations
    at 2.5 and 4.0, has times `[0.0]`, values `[25.0]`, and the value 25.0
    at 0.25. **Catches:** an empty window, a crash when no observation falls
    in the window, a window that cannot be evaluated past its only
    observation.
  - (added) Times `[0.0, 0.3, 0.6, 0.9, 1.2]`, values `10 * t`, `size=0.3`,
    `step=0.3`: `3 * 0.3` computes just below 0.9, and still window 2 holds
    `[6]`, without 0.9, and window 3 holds `[9]` at time exactly 0.
    **Catches:** a start that rounds just short of an observation carrying
    the value before it in (window 3 would be `[6, 9]` with 9 at
    `1.1e-16`), an end just short of one keeping it.
  - (added) An embedding (`dimension=2`, `delay=0.4`) of the window at 1.0
    has times `[0.4, 0.5, 0.9, 1.5, 1.9]`, is `[25, 25]` at 1.95, and raises
    at 2.0 naming `[0.4, 2)`. **Catches:** an embedding of a window stopping
    at the window's last observation (1.9, where the newer component
    reaches 1.5, would be missing) or losing its open end.
  - (dropped with `take`: a window from -1.0 over times
    `[-1.0, -0.3, 0.2, 0.5]` read the observation before one of its own
    when `(t - a_k) + a_k` rounded below `t`. Windows now read their
    observations by index, so no time is added back.)

- [x] **44. `series.window(a, b)`.** (skipped: `Window` covers it; later items take one window as `Window(size=2.0, step=1.0)(series)[1]`, the window at 1.0) One window, with the same rules as item
  43 and `size = b - a`; `t_0 <= a <= b <= t_{n-1}` is required. Tests with
  the source series:
  - `w = series.window(1.0, 3.0)` has times `[0.0, 0.5, 1.5]` and values
    `[5.0, 15.0, 25.0]`; `w(0.25) == 5.0`, `w(1.0) == 15.0`, and `w(-0.25)` and `w(2.0)` raise
    `ValueError`. **Catches:** the window differing from the one
    `Window` produces, evaluation reaching before the window.
  - `series.window(3.0, 1.0)` and `series.window(-1.0, 1.0)` raise
    `ValueError`. **Catches:** reversed bounds or bounds outside the domain
    accepted.

- [x] **45. Window parameter errors.** Test, parametrized `ValueError`:
  `size` and `step` each `0.0`, `-1.0`, `inf`, `-inf`, and `nan`.
  **Catches:** windows that never advance or never repeat.
  (Done with item 43: windowing raises `ValueError` when no window fits,
  for `size=5.0` on the source series and for a series without
  observations, as an embedding that can never be evaluated does in item
  41, rather than returning an empty tensor. **Catches:** an empty result
  that fails later, far from its cause.)

- [x] **46. Embedding a window, and windowing an embedding.** Tests:
  - `TimeDelayEmbedding(dimension=2, delay=1.0)` of the window at 1.0 of
    `Window(size=2.0, step=1.0)`
    (times `[0.0, 0.5, 1.5]`, open end 2) has times `[1.0, 1.5]` and points
    `[[5, 15], [15, 25]]`; evaluating it at 0.75 and at 2.0 raises
    `ValueError`. **Catches:** history taken from before the window, the
    window's open end lost.
  - `Window(size=2.0, step=1.0)` of `TimeDelayEmbedding(dimension=2,
    delay=1.0)` of the source (times `[1.0, 1.5, 2.5, 3.5, 4.0]`) has
    windows starting at 1.0 and 2.0, and the first has times
    `[0.0, 0.5, 1.5]` and values `[[0, 5], [5, 15], [15, 25]]`. **Catches:** the embedding's points
    lost when windowing.

- [x] **47. Windows of several channels.** One cut, `[a, a + size)` in the
  query unit, applies to every channel. Test: `Window(size=1.0, step=1.0)` of
  the series of channels A and B has starts `[0.5, 1.5]` in seconds; the
  first window's `times` leaves are `[0.0, 0.5]` and `[0.0]`, without B's
  observation at 1500 ms, and the second's are `[0.0]` and `[0.0]`; the
  first window evaluates at 0.5 to leaves 10 and -5 and raises at 1.0.
  (Revised: the test gives A the values `10 * t` and B `-10 * t`, with `t`
  in seconds, so the values show where they were observed.)
  **Catches:** channels cut at different
  bounds, channel B cut in milliseconds.

- [x] **47a. (decision) Closed ends and partial windows.** (decision:
  as in Polars `group_by_dynamic`, a `closed` argument and partial
  windows, on by default; an open start keeps the value carried in; bounds
  are nominal; revised in item 47c: partial windows are opt-in, so every
  window is `size` long unless asked otherwise, and a partial window's
  bounds end where it does)
  `Window(size, step=None, closed="left", partial=False)`.
  `closed` is `"left"` (`[a, a + size)`, the windows of items 43-47),
  `"right"` (`(a, a + size]`), `"both"`, or `"none"`, as in Polars. A
  window always holds the value in effect at `a_k` at time 0, since that
  value is also in effect just after `a_k` and evaluation on `(0, size]`
  needs it; an open start only makes 0 itself outside the domain. An
  observation at `a_k` is that value, so with an open start it still shows
  at time 0. A closed end includes an observation at `a_k + size`, or one
  the end snaps onto, so with `"both"` it is in two windows and with
  `"none"` only as the next window's value at 0. With `partial=True`,
  windows continue while their start is in the domain (`a_k <= t_{n-1}`,
  or `a_k < t_{n-1}` with an open start or an open end of the series), and
  a window whose nominal end passes `t_{n-1}` ends where the series does,
  closed at its last observation or open at its open end: a partial
  window. Polars likewise ends with windows that hold only the last rows.
  `partial=False` keeps only windows that end in the domain and raises when
  none fits, as item 43 does. The bounds are `(a_k, a_k + size)`, and for a
  partial window `(a_k, t_{n-1})`, where it ends (first nominal, as Polars'
  `include_boundaries` reports them; revised in item 47c).
  Considered: closing only the last window, which keeps the last
  observation only when the windows land on it and needs a closed end
  past a channel's last observation anyway; and partial windows off by
  default, which keeps every window `size` long for evaluating all of
  them on one grid (item 51).

- [x] **47b. `closed`.** A channel's bounds become an open start and an
  end after its last timestamp that is open or closed, replacing
  `Channel::until`; `TimeSeries::Domain` gains `closedStart` (and `openEnd` becomes `closedEnd`), and the error
  names the domain as in `(0, 1]`. An embedding keeps both: its start is
  open when its source's is, and it can be evaluated at a closed end. An
  invalid `closed` raises `ValueError`. With `partial=False`, a closed
  window end must also be inside an open end of the series. Tests with the
  source series:
  - One test per value of `closed`, on the window at 1.0 of times
    `[0.0, 0.5, 1.0, 1.5, 2.0]`, values `10 * t`, and `size=1.0, step=1.0,
    partial=False`, evaluated at its edges: `"left"` is 10 at 0 and raises
    at 1 naming `[0, 1)`; `"right"` raises at 0 naming `(0, 1]`, is 10 at
    0.25 and 20 at 1; `"both"` is 10 at 0 and 20 at 1; `"none"` raises at 0
    and at 1 naming `(0, 1)` and is 10 at 0.25.
    **Catches:** an observation at a closed end left out, one at an open
    end kept, the value carried in dropped for an open start, the domain's
    ends shown with the wrong brackets.
  - `TimeDelayEmbedding(dimension=2, delay=1.0)` of the window at 1.0 of
    `Window(size=2.0, step=1.0, closed="right", partial=False)` over times
    `[0.0, 1.0, 2.0, 4.0]` with values `10 * t` (values `[10, 20]` at
    `[0.0, 1.0]`, end 2 closed past its last observation) has times
    `[1.0, 2.0]`, is `[20, 20]` at 2.0, and raises at 1.0 naming `(1, 2]`. **Catches:** a closed end
    past the last observation evaluated as open, an embedding losing the
    window's open start or closed end.
  - `closed="middle"` raises `ValueError`.
    **Catches:** an unknown `closed` read as one of the four.

- [x] **47c. Partial windows.** (revised: opt-in, `partial=False` by
  default) Tests with times `[0.0, 1.0, 2.0, 3.0, 4.0]` and values
  `10 * t`:
  - Parametrized over `partial`, `size=3.0, step=3.0, closed="left"`: the
    bounds are `[[0, 3]]` without and `[[0, 3], [3, 4]]` with partial
    windows. **Catches:** `partial` ignored, a partial window's bound
    reported past the end of the series.
  - With `partial=True`, the window at 3.0 has times `[0.0, 1.0]` and
    values `[30, 40]`, is 40 at 1.0, and raises at 1.5 naming `[0, 1]`.
    **Catches:** a partial window's end taken from the nominal bound.
  - `size=2.0, step=2.0, closed="left", partial=True` gives starts
    `[0.0, 2.0, 4.0]`, the window at 4.0 holding only 40, at 0.
    **Catches:** a window of one instant at the last observation missing.
  - `size=2.0, step=2.0, closed="right", partial=True` gives starts
    `[0.0, 2.0]`, and the window at 2.0 is 40 at 2.0. **Catches:** a
    window kept at the last observation with an open start, where nothing
    in it can be evaluated.
  Existing window tests spell out `closed` and `partial` (Conventions).

## Writing

(Skipped: series are immutable, decision 4. Items 48-50 are kept for when
writes return.) Items 48-50 use the source series of items 43-46 and the
embedding of item 40.

- [x] **48. Writing a series.** (skipped: series are immutable, decision 4) `values[...] = x` (a tensor backed by the
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

- [x] **49. Writing a window.** (skipped: series are immutable, decision 4) A write into a window first materializes
  it into storage of its own (decision 4), its observations at its times,
  including the one at 0, and keeps its open end; a write changes nothing
  but the written values. Neither the source nor other windows change.
  Test: with `windows = Window(size=2.0, step=1.0)(series)` and
  `w = windows[1]`, `w.values[1] = 30.0` gives
  `w(0.5) == 30.0`, `w.times == [0.0, 0.5, 1.5]`, `len(w) == 3`, and
  `w(2.0)` still raises;
  `RegularSampler(n_points=2)(w)` is `[[5], [30]]`; and
  `series(1.5) == 15.0` and `windows[0]` are unchanged.
  **Catches:** the write reaching the source or a sibling window, the value
  at 0 lost on materializing.

- [x] **50. Writing an embedding.** (skipped: series are immutable, decision 4) Test: for item 40's embedding `e`,
  `e.values[1] = [0.0, 0.0, 0.0]` (time 2.5) gives `e(2.75)` equal to
  `[0, 0, 0]`, keeps `e(3.0) == [10, 10, 25]`, which reads two of the
  written point's stored observations, and `len(e) == 5`, and leaves the
  source unchanged. **Catches:** a write into a view reaching the source or
  another point that shares its stored observations.

- [x] **51. Evaluating a tensor of series.** (revised: there is no
  `RegularSampler`, nor items 51-54 as first planned, a sampler's query
  times, sampling windows into a `PointCloudTensor`, sampling several
  channels, and sampler errors. Its open questions, where `size` comes from
  for a partial or right-closed window and which grid suits each `closed`,
  go away when the user chooses the times.) A tensor of series evaluates
  at times `t`, a number, an array, or a `FloatTensor` of shape `t_shape`,
  in each series' unit: `windows(t)` and `windows.evaluate(t)` give, for
  one-channel series, a tensor of shape `(*windows.shape, *t_shape,
  *shape)` of the element dtype, as `PcfTensor(t)` gives
  `(*shape, *t_shape)`; a number gives `(*windows.shape, *shape)`. Each
  element evaluates through item 24b in one native call. Evaluating
  windows of an embedding on a grid, such as
  `windows(np.arange(n) * size / n)`, gives clouds of `n` points in the
  last two axes, which `compute_persistent_homology` reads as point clouds
  (item 55). Tests with times `[0.0, 0.5, ..., 4.0]` and values `10 * t`,
  and `Window(size=2.0, step=1.0, closed="left", partial=False)` (windows
  at 0, 1, and 2):
  - `windows([0.0, 1.0])` is `[[0, 10], [10, 20], [20, 30]]`, and
    `windows(1.5)` is `[15, 25, 35]`. **Catches:** windows out of order,
    windows and times transposed, a number given a time axis.
  - `windows([[0.0, 1.0], [1.5, 0.5]])` is
    `[[[0, 10], [15, 5]], [[10, 20], [25, 15]], [[20, 30], [35, 25]]]`.
    **Catches:** the query shape lost, results in sorted rather than query
    order.
  Partial windows, which a grid on `[0, size)` cannot cover, are tested in
  item 53.

- [x] **52. Evaluating several channels.** A tensor of series with several
  channels evaluates to a `NestedTensor` with one leaf per channel, each of
  shape `(*windows.shape, *t_shape, *shape)` of its channel, as
  `series(t)` gives one leaf per channel. Test: channel A with `10 * t` at
  `[0.0, 0.5, ..., 2.5]` s and channel B with `-10 * t` at
  `[500, 1000, ..., 3000]` ms; `Window(size=1.0, step=1.0, closed="left",
  partial=False)` has windows at 0.5 and 1.5, and `windows([0.0, 0.5])`
  has leaves `[[5, 10], [15, 20]]` and `[[-5, -10], [-15, -20]]`, every
  value from a different observation. (Added) Channel B of points
  `(-10 * t, -20 * t)` and A of scalars, at `[0.0, 0.5, ..., 2.5]` s, give
  leaves of shapes `(2, 2)` and `(2, 2, 2)`. **Catches:** channels swapped, windows and
  channels transposed, channel B read in the query unit.

- [x] **53. Evaluation errors.** A time outside a window's domain raises
  `ValueError` naming the window's index and its domain, as in
  `TimeSeries at position (1): Time 1.5 is outside the domain [0, 1]`, as for
  any tensor of series. Every domain is
  checked in order before the parallel evaluation, so the error names the
  first such window whichever worker would reach it, using
  `Domain::contains` and `Domain::to_string`. Test: times
  `[0.0, 1.0, 2.0, 3.0]` with values `10 * t` and `Window(size=2.0,
  step=2.0, closed="left", partial=True)` give `[0, 2)` and the partial
  window `[2, 3]`, which can be evaluated from 0 to 1; `windows([0.0, 1.5])`
  raises naming window 1 and `[0, 1]`. **Catches:** a partial window
  evaluated past its end, an error that does not say which window.

- [x] **54. Sampler errors.** (dropped with the sampler, item 51)

- [x] **55. The delay workflow end to end.** Test in
  `test/python/test_integration.py`: times `[0.0, 0.5, ..., 3.5]` and values
  `[0, 1, 0, -1, 0, 1, 0, -1]` through
  `TimeDelayEmbedding(dimension=2, delay=0.5)` (domain `[0.5, 3.5]`),
  `Window(size=1.0, step=1.0, closed="left", partial=False)` (windows at
  0.5, 1.5, and 2.5), evaluation at `[0.0, 0.25, 0.5, 0.75]`, and
  `compute_persistent_homology(..., max_dim=1)` give a float tensor of
  shape `(3, 4, 2)`, three clouds of four 2-D points, and a
  `BarcodeTensor` of shape `(3, 2)`. Where the windows of an embedding
  start is tested in `test_windowing.py`. **Catches:** steps that do not compose.

- [ ] **56. Saving embeddings and windows.** An embedding is saved
  materialized (decision 9): its support and values, as an ordinary series.
  A tensor of windows is saved as a tensor of series, each window
  materialized: its observations at its times, with its value at 0 and
  its ends, so
  the format records whether a channel's start is open and its end when it
  has one (items 43 and 47b). A loaded
  one therefore evaluates exactly as the original, with no source in the file; with
  overlapping windows an observation is written once per window holding
  it. Tests, parametrized
  over `save`/`load` and `pickle`:
  - Item 40's embedding loads with the same times and the same values at 2.5
    and 3.25. **Catches:** the embedded points lost on saving.
  A loaded embedding evaluates exactly as the view, since views evaluate by
  previous value on their own points with their own snapping (decision 8).
  - Item 51's windows load in order, and evaluating them at
    `[0.0, 1.0]` gives item 51's values. **Catches:** windows reordered or their values at the
    bounds lost on saving.

- [ ] **57. Golden files.** Pin the formats (TESTING.md principle 11) with a
  generator modeled on `test/golden/serialization/generate_point_cloud_v3.py`,
  which records explicit expected values. Write item 28's barcode32 result,
  the series of channels A and B, and item 56's tensor of windows as `.sb` and
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
  channels. Operations do not describe containers: `compute_persistent_homology`
  and the three summaries say only that the result is the same kind of
  container as the input, and the `TimeSeries` docstring states once how
  element-wise operations treat a series. Check that `save` and `load` list
  series.

- [ ] **59. User guide.** A `docs/timeseries.rst` guide linked from
  `docs/userguide.rst`: times, units, the domain, and channels, then two small
  examples. The first follows a point-cloud series through persistence to
  stable ranks; the second takes two numeric channels in different units
  through a delay embedding and windows, evaluated on a grid, to
  persistence. Link each
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
| Evaluating windows into point clouds and through persistence (no sampler) | 51-53, 55 |
| Save, load, and pickle for both workflows | 26-27, 56-57 |
| Tests and documented examples | all test items, 58-59 |
| Channels with their own grids and units (added) | 3, 21-25, 23a, 29, 42, 47, 52 |

## Out of scope

As in the issue: datetime timestamps, sample-count delays and windows, other
samplers, interpolation, gap policies, reductions over
time, provenance, and normalization. Also deferred here:
joining channels into one point cloud when evaluating windows, series with channels of
different element types, observations whose shape changes over time
(a later per-observation tensor storage mode), `NestedTensor` values (6
more element types, cheap to add once a use such as spike trains needs
them),
views instead of copies for stored channels, tensors carrying time labels
through operations,
and the branch's notebooks and scikit-learn transformers.
