# Time series (#242): implementation plan

This plan implements #242 on `issue-242-timeseries`, branched from `main` at
`05ce5ba4a`: unit-bearing timestamps, native time-series storage, persistence
of snapshot series (workflow 1), channels with their own time grids and units,
and windowed delay embeddings sampled into point clouds (workflow 2), with
serialization and documentation. Useful parts of the `origin/timeseries`
branch are recycled where they fit (see "Recycling").

Channels with independent time grids go beyond the issue, which defers a
channel-combination API; we added them to this release.

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

**Test budget.** The items add about 50 test functions, many of them
parametrized, plus three optional ones. Keep that number from growing. Before
adding a test that is not listed, check whether a listed test's
parametrization covers it, and prefer leaving a guarantee to the design and
its docstring over testing every combination. Do not test what the public API
cannot observe. No C++ unit tests are planned: every guarantee here is
observable from Python (TESTING.md principle 4).

## Conventions

- **Files.** Source: `include/sbear/timeseries.hpp` (storage),
  `include/sbear/timeseries/series_task.hpp` (analysis adapter),
  `include/sbear/io/timeseries_io.hpp`, `src/python/py_timeseries.{hpp,cpp}`,
  and the package `stablebear/timeseries/` with `series.py` (`Times`,
  `TimeSeries`) and `derived.py` (`combine_channels`, `TimeDelayEmbedding`,
  `Window`, `RegularSampler`). Tests: `test/python/test_timeseries.py`
  (storage, access, evaluation, units, serialization),
  `test/python/test_timeseries_derived.py` (channels, embedding, windows,
  sampling), and `test/python/persistence/test_timeseries_persistence.py`
  (workflow 1). Items that touch another file name it.
- **Data.** Use timestamps, delays, and window sizes that are exact in binary
  floating point (multiples of 0.25), so expected values do not depend on
  rounding; item 21 tests rounding on purpose. Give observations distinct
  values (10, 20, 30, ...) so that mix-ups show.
- **Precisions.** Parametrize over 32 and 64 bits only where the code
  dispatches on precision: construction per value family, serialization, and
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
- `sb::TimeSeries<T>` stores recorded observations: `Times`, a `Tensor<T>` of
  values, and a domain. Axis 0 of the values is time and the remaining axes
  are value axes, so a persistence result of shape `(n_times, max_dim + 1)`
  keeps time and homology degree apart. Point counts, matrix sizes, and bar
  counts are internal to each element and may change over time. The template
  is instantiated for numeric values and for `PointCloud`, `DistanceMatrix`,
  `Barcode`, and `Pcf`, each at both precisions. Channels that share one time
  grid are a value axis of one stored series.
- `sb::SeriesTask<TaskT, InT, OutT>` runs an existing batch task (`RipserTask`,
  `RipserDistMatTask`, `BarcodeSummaryTask`) on the values of a stored series
  and returns a `TimeSeries<OutT>` with the input's times and domain.
- Derived numeric series are lazy and store no values: combined channels,
  delay embeddings, and windows hold their sources and parameters, and
  `evaluate(t)` asks each source for the value it needs. Combined channels
  look up each channel on its own grid, in its own unit.
- `stablebear.timeseries` wraps the native types. Evaluation and the derived
  series use NumPy on the stored arrays.
- `compute_persistent_homology` and the barcode summaries forward a stored
  `TimeSeries` to native `spawn_*_series_task` entry points.

## Recycling

The `origin/timeseries` branch predates the rename (`mpcf`/`masspcf` became
`sb`/`stablebear`). Read its files with `git show origin/timeseries:<path>`
and copy ideas and small blocks, not whole files.

- `include/mpcf/timeseries.hpp`: `snap_in_place` and `default_snap_tol` give
  the snapping rule of item 21. The rest (offsets from `start_time` and
  `time_step`, channels in one value array, interpolation variants) does not
  fit the new storage.
- `include/mpcf/algorithms/embed_time_delay.hpp`: the oldest-to-newest
  component order and the valid start `start + (dimension - 1) * delay`
  (items 37-38). Not its windows, which are anchored at the end and keep a
  partial first window, and not its tuples, which it stores only at original
  observation times; the issue rules out both.
- `include/mpcf/io/timeseries_io.hpp`: the outline of the object writer and
  reader for items 23 and 48, with new payloads.
- `src/python/py_timeseries.cpp`: the array-copy and length-check patterns for
  items 10-14.
- `masspcf/persistence/homology.py` (the `TensorTimeSeries` branch): one
  batched Ripser call over all snapshots. Item 24 keeps the idea and moves it
  into native code.
- `masspcf/timeseries/timeseries.py`: unit-factor handling in
  `_dt64_to_float_seconds` may inform item 22; datetime support itself stays
  out.
- `docs/ts_*.rst`: background wording for item 51; the API differs.

Not recycled: datetime support, interpolation strategies, `TimeSeriesTensor`,
the scikit-learn transformers, `l2_kernel` cross kernels, and the `tensor_eval`
parallel threshold.

## Decisions

- [x] **1. Domain and evaluation.** (decision: (a), closed domain for series built from observations, half-open for windows) Evaluation
  uses the previous value: at time `t`, the observation with the largest
  timestamp `<= t`. Queries outside the domain raise `ValueError`. Options for
  the domain:
  (a) A series built from observations has the closed domain
  `[t_0, t_{n-1}]`, so every observation can be evaluated. Windows have
  half-open domains `[a, b)`, and series derived from a window keep its
  half-open end. The series stores one flag for this.
  (b) Every domain is half-open: `[t_0, end)` with an optional `end=` that
  defaults to `t_{n-1}`. One rule and no flag, but by default the last
  observation cannot be evaluated, and a singleton series cannot be evaluated
  at all.
  Recommendation: (a). It also matches the issue, which defines sampling for
  half-open windows and a separate endpoint rule for whole series. With either
  option, `domain` is a `(start, end)` tuple, and an empty series has no
  domain (`domain is None`) and cannot be evaluated.

- [x] **2. Floating-point timestamps.** (decision: (a), snap within `1e-9 * max(1, |t|)`, no user parameter) Delays
  and window starts are computed in floating point, so a query can land just
  before the observation it should reach: `0.3 - 0.1` is less than `0.2`.
  Options:
  (a) Snap a query to an observation that lies above it by at most
  `1e-9 * max(1, |t|)`, as `snap_in_place` does on the branch. No user
  parameter; domain checks stay exact.
  (b) No snapping; document that times and delays should be exactly
  representable.
  Recommendation: (a). With (b), delay embeddings of data sampled every 0.1 s
  pick the previous observation at many grid times.

- [x] **3. Time units.** (decision: (a), fixed set `as` through `d` with exact integer conversion factors) `Times(values, unit="s")`; plain timestamp
  arrays mean seconds. Options:
  (a) A fixed set covering physics time scales: `"as"`, `"fs"`, `"ps"`,
  `"ns"`, `"us"`, `"ms"`, `"s"`, `"min"`, `"h"`, and `"d"`. Store each unit as
  a power of ten times an integer (1 or 60, 3600, 86400), and convert by
  multiplying or dividing by an exact integer factor rather than by a
  rounded one such as `1e-15`; dividing by an exact factor rounds once.
  Adding `"zs"` or `"ys"` later is one table entry. `Times.to(unit)` and
  `TimeSeries.to(unit)` convert, and combined channels convert each query to
  each channel's unit. Single-series parameters (delays, window sizes, query
  times) are in that series' unit.
  (b) Any nonempty string, with no conversion.
  Recommendation: (a). Channels in different units must be combinable, which
  needs known factors, and a typo such as `"sec"` raises.

- [ ] **4. Ownership.** (decision) Options:
  (a) Copy in, copy out for stored series. The constructor copies its values,
  materializing indexed point-cloud and distance-matrix tensors; `values`,
  `series[i]`, and `evaluate` return copies; `Times.values` is a read-only
  array. Stored series are then immutable, so derived series can hold
  references to their sources without copying them and without seeing later
  changes.
  (b) Views, with documented write-through.
  Recommendation: (a). Mark the copies with a `ponytail:` comment; views can
  come later if profiling shows a cost.

- [ ] **5. Where the code lives.** (decision) Options:
  (a) C++ holds storage and validation (`sb::Times`, `sb::TimeSeries<T>`),
  serialization, and the analysis adapter. Python with NumPy implements
  evaluation and the derived series on the stored arrays, through one shared
  lookup (`searchsorted` plus snapping).
  (b) Also port evaluation and the derived series to C++, adapting the
  branch's `embed_time_delay.hpp`.
  Recommendation: (a). The derived series are numeric, and NumPy vectorizes
  their lookups; one evaluator keeps channels, embedding, windows, and
  sampling consistent. The analysis path has no per-observation Python loop
  either way.

- [ ] **6. Analysis adapter.** (decision) Options:
  (a) `sb::SeriesTask<TaskT, InT, OutT>` derives from an existing batch task.
  A base declared before the task (base-from-member) holds the input series
  and the output tensor; the task is constructed on the series' values, and
  `result()` returns `TimeSeries<OutT>` with the input's times and domain. The
  task supplies scheduling, progress, cancellation, and error capture. The
  base keeps the input and output alive until completion, which matters
  because `BarcodeSummaryTask` stores only a reference to its input. Because
  the values tensor is the batch, a singleton series keeps its time axis and
  an empty series gives a typed empty result. Each operation declares its
  contract with a `make_*_series_task` overload, and pybind11 overloads on the
  native series type select the implementation.
  (b) Python spawns the existing tasks on `series.values` and rebuilds the
  series. No new C++, but dispatch and result construction stay in Python,
  against the issue.
  Recommendation: (a).

- [ ] **7. Public names and accepted values.** (decision) Recommendation: the
  issue's names, except that `TimeDelayEmbedding` drops `unit="time"` until a
  second delay mode exists, plus `combine_channels`.
  - `stablebear.timeseries` exports `Times(values, unit="s")`,
    `TimeSeries(values, times=None)`, `combine_channels(channels, unit=None)`,
    `TimeDelayEmbedding(dimension, delay)`,
    `Window(size, step=None, overlap=None)`, and `RegularSampler(n_points)`
    with `.times(series)`. A `RegularSampler` takes a sequence of series and
    returns a `FloatTensor` of shape `(n_series, n_points, *value_shape)`.
  - A stored `TimeSeries` exposes `times`, `unit`, `values`, `dtype`, `shape`,
    `value_shape`, `domain`, `len()`, `series[i]`, `series[a:b]`, `to(unit)`,
    and `evaluate(t)` for a scalar `t`. It is generic (`TimeSeries[PointCloud]`)
    in Python 3.10-compatible annotations.
  - `values` may be numeric (NumPy arrays, nested lists, `FloatTensor`, or
    `IntTensor`; float32 stays float32 and everything else becomes float64),
    or a `PointCloudTensor`, `DistanceMatrixTensor`, `BarcodeTensor`, or
    `PcfTensor`. Axis 0 is time. Anything else raises `TypeError`: booleans,
    strings, other tensor types, and sequences containing stablebear objects,
    which `np.asarray` would silently turn into numeric arrays.
  - An empty list gives an empty float64 series, so a typed empty series of
    point clouds is built from `sb.zeros((0,), dtype=sb.pcloud64)`.

- [ ] **8. Derived series are lazy.** (decision) Options:
  (a) Combined channels, delay embeddings, and windows store no values. Each
  holds its sources and parameters and exposes `unit`, `dtype`,
  `value_shape`, `domain`, and `evaluate(t)`, which asks the sources for the
  values it needs; the sampler uses the same interface. Combined channels
  look up each channel on its own grid in its own unit. A window evaluates its
  source, so the value carried in at its start needs no stored copy.
  `times` lists recorded observation times: a `Times` for a stored series, a
  `NestedTensor` of `FloatTensor`s (one per channel, in the query unit) for
  combined channels, and the source's times inside `[a, b)` for a window of
  either. A delay embedding has no `times`. Internal lookups skip the domain
  check, because shifted and converted times may round just outside it.
  (b) Materialize each derived series as a stored series at the union of the
  times where any component changes.
  Recommendation: (a). It needs no union of time grids, keeps original
  timestamps, and costs no memory per derived series.

- [ ] **9. Saving derived series.** (decision) Options:
  (a) Keep the representation: save a derived series as a tree whose leaves
  are stored numeric series and whose nodes record their kind (combined,
  embedding, window), parameters, and query unit. A loaded tree evaluates as
  the original without the original objects.
  (b) Materialize on saving, which brings back the union of time grids that
  option 8(a) avoids.
  Recommendation: (a).

## Storage and access

- [ ] **10. `Times`.** `sb::Times` in `include/sbear/timeseries.hpp` validates
  its timestamps and unit. The new `stablebear/timeseries/` package wraps it
  with `values` (a read-only float64 array), `unit`, `len()`, `==`, and
  `repr`. Tests:
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

- [ ] **11. Numeric series with explicit times.** `sb::TimeSeries<T>` with its
  validation, the float32 and float64 bindings, and Python
  `TimeSeries(values, times)` with `times`, `unit`, `values`, `dtype`,
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

- [ ] **12. Point clouds, distance matrices, barcodes, and PCFs.** Bind
  `TimeSeries<T>` for the four object families at both precisions, and
  dispatch on the tensor type in Python. Test, parametrized over the eight
  dtypes: a two-observation tensor whose elements differ in size (clouds of
  1 and 2 points, 2x2 and 3x3 matrices, barcodes with 1 and 2 bars, PCFs with
  2 and 3 breakpoints) with `times=[0.0, 1.0]` gives a series with the
  tensor's `dtype`, `shape == (2,)`, and values equal to the input.
  **Catches:** a family bound to the wrong native type or precision, elements
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
  two equal-size `sb.PointCloud`s, `np.array(["a", "b"])`,
  `np.array([True, False])`, and a `SymmetricMatrixTensor`. **Catches:**
  object sequences silently converted to numeric arrays, strings or booleans
  passed to the native constructor.

- [ ] **16. Positional access and ownership.** `series[i]` returns a copy of
  observation `i`: a float for scalar values, otherwise the element or tensor
  type of the values (`PointCloud`, `BarcodeTensor`, ...). Tests with values
  `[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]` and `times=[0.0, 1.0, 2.0]`:
  - `series[1]` is `[3.0, 4.0]`, `series[-1]` is `[5.0, 6.0]`, and
    `series[3]` raises `IndexError`. **Catches:** indexing along a value axis,
    negative indices counted from the wrong end, no bounds check.
  - After `obs = series[0]` and `obs[0] = 99.0`, `obs[0] == 99.0` and
    `series[0]` is still `[1.0, 2.0]`. **Catches:** observations returned as
    views into the series (decision 4).

- [ ] **17. Time slicing.** `series[a:b]`, with an optional step, returns a
  stored series of the selected observations with the default closed domain.
  Test: for item 16's series, `series[1:]` has times `[1.0, 2.0]`, values
  `[[3.0, 4.0], [5.0, 6.0]]`, and domain `(1.0, 2.0)`. **Catches:** times and
  values sliced differently, the source's domain kept.

- [ ] **18. (optional) Selecting value axes.** `series[:, j]` indexes the
  value axes and keeps the time axis, for example to pick one homology degree
  of a barcode series. Test: a barcode series of shape `(2, 2)` with four
  distinct barcodes gives `series[:, 1]` of shape `(2,)` with the two
  degree-1 barcodes and the same times. **Catches:** the index applied to the
  time axis.

- [ ] **19. Evaluation.** `series.evaluate(t)` returns the previous value
  (decision 1), as `series[i]` would. Tests with times `[0.0, 1.5, 4.0]` and
  values `[10.0, 20.0, 30.0]`:
  - `evaluate` at 0.0, 1.0, 1.5, 3.75, and 4.0 gives 10, 10, 20, 20, and 30.
    **Catches:** next-value, nearest, or linear interpolation; an open end.
  - `domain == (0.0, 4.0)`, and `evaluate(-0.5)` and `evaluate(4.5)` raise
    `ValueError`. **Catches:** extrapolation outside the domain, NaN instead
    of an error.

- [ ] **20. Empty and singleton series.** Tests:
  - `TimeSeries(sb.zeros((0,), dtype=sb.pcloud64), times=[])` has
    `len() == 0`, dtype pcloud64, and `domain is None`, and `evaluate(0.0)`
    raises `ValueError`. **Catches:** an invented domain, a crash on empty
    storage.
  - `TimeSeries([7.0], times=[2.0])` has domain `(2.0, 2.0)` and
    `evaluate(2.0) == 7.0`. **Catches:** a singleton series rejected or
    impossible to evaluate.

- [ ] **21. Snapping.** Implement decision 2 in the shared lookup. Test with
  times `[0.0, 0.25, 0.5]` and values `[1.0, 2.0, 3.0]`:
  `evaluate(np.nextafter(0.25, 0.0))` is 2.0 and `evaluate(0.25 - 1e-6)` is
  1.0. **Catches:** no snapping, or a tolerance wide enough to change values
  at ordinary times.

- [ ] **22. Unit conversion.** `Times.to(unit)` and `TimeSeries.to(unit)`
  rescale timestamps (and the domain) by the exact factors of decision 3,
  keeping values. Tests:
  - Parametrized: `Times([0.0, 1.5], unit="min").to("s")` gives
    `[0.0, 90.0]`; `Times([0.0, 1.5], unit="fs").to("as")` gives
    `[0.0, 1500.0]`; `Times([0.0, 2.25], unit="as").to("fs")` gives
    `[0.0, 0.00225]`. Each result has the target unit. **Catches:** an
    inverted factor, a rounded factor such as `1e-3` (which gives
    `0.0022500000000000003`), the unit label not updated.
  - `TimeSeries([10.0, 20.0], times=Times([0.0, 0.5], unit="s")).to("ms")`
    has times `[0.0, 500.0]`, domain `(0.0, 500.0)`, unit `"ms"`, and the
    same values. **Catches:** the domain left in the old unit.

## Serialization of stored series

- [ ] **23. Save, load, and pickle.** `include/sbear/io/timeseries_io.hpp`
  writes a stored series as a standalone object: unit, timestamps, domain
  (with its closedness, or none), then the values tensor with its own format.
  Give each value type a format code (for example 20000 plus the value type's
  base format, with the same subformat) and register them in
  `StreamableObject` and `read_any_object`. Loading goes through the
  validating constructor. No `FormatVersion` bump is needed; older readers
  report an unhandled object type. In Python, `TimeSeries` uses
  `_BinaryIoMixin`, and `save` and `load` dispatch it. Test, parametrized over
  the ten dtypes and over `save`/`load` and `pickle`: a two-observation
  series (item 12's tensors, and numeric values) with
  `Times(..., unit="ms")` loads with the same type, `dtype`, times, unit,
  domain, and values. **Catches:** a format code mapped to the wrong family or
  precision, a dropped unit or domain, timestamps narrowed to 32 bits in
  32-bit series.

## Snapshot persistence (workflow 1)

The series in this section hold three clouds with different point counts and
different barcodes: `[[0, 0], [1, 0]]`, the unit square
`[[0, 0], [1, 0], [1, 1], [0, 1]]` (one degree-1 bar), and
`[[0, 0], [2, 0], [5, 0]]`, with `times=Times([0.0, 1.3, 4.0], unit="ms")`.

- [ ] **24. `SeriesTask` and point-cloud persistence.** Add `sb::SeriesTask`
  (decision 6) in `include/sbear/timeseries/series_task.hpp`, a
  `make_ripser_series_task` overload for point-cloud series, and bindings in
  `src/python/persistence/py_ripser.cpp`, with each task class exposing
  `result()`. The registered task base is not the first base, so the
  pybind11 classes need `py::multiple_inheritance()`.
  `compute_persistent_homology` forwards a `TimeSeries`, keeping the existing
  complex and distance checks; update its signature and docstring. Tests:
  - Parametrized over pcloud32 and pcloud64: the result has the input's times
    and unit, `shape == (3, 2)`, and the matching barcode dtype, and each
    `result[i]` is isomorphic to
    `compute_persistent_homology(clouds[i], max_dim=1)`. **Catches:** lost or
    permuted times, a dropped unit, a flattened homology axis, snapshots
    paired with the wrong barcodes.
  - A series of one cloud gives `shape == (1, 2)`. **Catches:** the existing
    singleton squeeze applied to series.

- [ ] **25. Distance-matrix persistence.** The distance-matrix overload and
  binding. Test, parametrized over distmat32 and distmat64: a series of a 2x2
  and a 3x3 matrix, built with `sb.zeros((2,), dtype=...)` and element
  assignment, keeps its times, and each observation is isomorphic to
  `compute_persistent_homology(matrix, max_dim=1)`. **Catches:** distance
  matrices sent down the point-cloud path or to the wrong precision, matrices
  of different sizes rejected.

- [ ] **26. Typed empty series.** Test: an empty pcloud32 series gives a
  barcode32 series of shape `(0, 2)` for `max_dim=1`, with unit `"s"`.
  **Catches:** an untyped or wrongly shaped empty result, failure on empty
  input.

- [ ] **27. Keyword data argument and parameter forwarding.** Test:
  `compute_persistent_homology(X=series, max_dim=0, reduced=True)` has shape
  `(3, 1)` and no infinite bar. **Catches:** `max_dim` or `reduced` dropped on
  the series path, keyword calls failing.

- [ ] **28. Stable rank of a series.** A `make_stable_rank_series_task`
  overload, its binding in `src/python/persistence/py_barcode_summary.cpp`,
  and a series branch in `_barcode_to_pcf`. Test: a barcode series of shape
  `(2, 2)` with four distinct barcodes gives a PCF series with the same times
  and shape whose values array-equal `barcode_to_stable_rank` of the input
  `BarcodeTensor`. **Catches:** times dropped, cells mixed up, the input
  released before the task finishes.

- [ ] **29. Betti curves and accumulated persistence.** The other two
  overloads and bindings, forwarding `max_death`. Tests:
  - Parametrize item 28's test over the three summary functions.
    **Catches:** a summary bound to another summary's task.
  - `barcode_to_accumulated_persistence(series, max_death=2.5)` array-equals
    the function applied to the `BarcodeTensor` with `max_death=2.5`, and
    differs from the result without `max_death`. **Catches:** `max_death`
    dropped on the series path.

- [ ] **30. Unsupported series.** Test, parametrized `TypeError`:
  `compute_persistent_homology` of a numeric series and of a barcode series,
  and `barcode_to_stable_rank` of a point-cloud series. **Catches:** a numeric
  series silently read as point clouds, an unsupported family passed to a
  native task.

- [ ] **31. Progress, errors, and the GIL.** (existing) The series path runs
  through `_run_task`, so the existing progress, error-propagation, and
  GIL-release tests (`test_gil_release.py`, `test_async_task.py`) cover it,
  and the existing persistence tests cover calls without series. Confirm; add
  no test.

- [ ] **32. (optional) CPU limits.** Persistence of the three-cloud series
  under `sb.system.limit_cpus(1)` equals the result with all threads; restore
  the limit afterwards. **Catches:** results that depend on scheduling.

## Channels with their own time grids

Items 33-35 combine channel A, with times `Times([0.0, 1.0, 2.5], unit="s")`
and values `[10.0, 20.0, 30.0]`, and channel B, with times
`Times([500.0, 1500.0, 3000.0], unit="ms")` and values `[-1.0, -2.0, -3.0]`.

- [ ] **33. Combined channels.** `combine_channels(channels, unit=None)`
  (decision 8) holds numeric channels with equal `value_shape` and `dtype`.
  `evaluate(t)` converts `t` from the query unit into each channel's unit,
  evaluates each channel, and stacks the results into value shape
  `(n_channels, *value_shape)`. `unit` defaults to the channels' shared unit.
  Test: `combine_channels([a, b], unit="s")` gives `[10, -1]` at 0.5,
  `[20, -1]` at 1.25, `[20, -2]` at 1.5, and `[30, -2]` at 2.5, with
  `value_shape == (2,)` and unit `"s"`. **Catches:** a channel read in the
  wrong unit (B would answer at 1.5 ms), channels stacked in the wrong order,
  a shared grid assumed.

- [ ] **34. Domain and times of combined channels.** The domain is the
  intersection of the channels' domains in the query unit; its end is closed
  only if every channel ending there is closed. Test: the combined series of
  item 33 has domain `(0.5, 2.5)`, `evaluate(0.25)` raises `ValueError`, and
  `times` is a `NestedTensor` whose leaves are `[0.0, 1.0, 2.5]` and
  `[0.5, 1.5, 3.0]`. **Catches:** the union of domains, timestamps reported
  in the channels' own units.

- [ ] **35. Combination errors.** Test, parametrized: channels in different
  units without `unit=`, channels with different value shapes, channels with
  disjoint domains, and an empty list raise `ValueError`; a point-cloud
  channel raises `TypeError`. **Catches:** a silently chosen unit, ragged
  stacking, an empty domain, object values stacked as numbers.

## Delay embeddings, windows, and sampling (workflow 2)

Items 39-48 use the source series with times `[0.0, 0.5, 1.5, 2.5, 4.0]` and
values `[1.0, 2.0, 3.0, 4.0, 5.0]`. The derived series here accept any numeric
series: stored, combined, or derived.

- [ ] **36. `PointCloudTensor` from a `FloatTensor`.** Accept a `FloatTensor`
  whose last two axes are points and coordinates, as an ndarray already is.
  Test in `test/python/test_point_cloud_tensors.py`, parametrized over float32
  and float64: a `FloatTensor` of shape `(2, 3, 2)` gives a tensor of shape
  `(2,)` with the matching pcloud dtype, and `np.asarray(clouds[1])` equals
  the second `(3, 2)` block. **Catches:** the current `TypeError`, the wrong
  precision.

- [ ] **37. Delay embedding.** `TimeDelayEmbedding(dimension, delay)` returns
  a lazy series (decision 8) whose value at `t` is
  `(x(t - (m - 1) * delay), ..., x(t))`, so the value shape becomes
  `(m,) + value_shape`. The domain starts at `start + (m - 1) * delay` and
  keeps the source's end and its closedness. Tests with times
  `Times([0.0, 1.0, 2.5, 4.0], unit="ms")`, values
  `[10.0, 20.0, 30.0, 40.0]`, `dimension=3`, and `delay=1.0`:
  - `evaluate` at 2.0, 2.75, 3.25, 3.75, and 4.0 gives `[10, 20, 20]`,
    `[10, 20, 30]`, `[20, 20, 30]`, `[20, 30, 30]`, and `[20, 30, 40]`. The
    values at 3.25 and 3.75 differ from the value at the last observation
    time, 2.5. **Catches:** newest-first order, the same delay for every
    component, values held from the last observation time.
  - The embedded series has domain `(2.0, 4.0)`, unit `"ms"`, and
    `value_shape == (3,)`, and `evaluate(1.5)` raises `ValueError`.
    **Catches:** components that refer to times before the series starts.

- [ ] **38. Vector values and embedding errors.** Tests:
  - Times `[0.0, 1.0, 2.0]`, values `[[1, -1], [2, -2], [3, -3]]`,
    `dimension=2`, and `delay=1.0` give `value_shape == (2, 2)` and
    `evaluate(2.0) == [[2, -2], [3, -3]]`. **Catches:** value axes flattened
    or transposed.
  - Parametrized: `dimension=0`, `delay=0.0`, and `delay=-1.0` raise
    `ValueError`; `dimension=3` and `delay=1.0` on times `[0.0, 1.0]` raise
    `ValueError` (too short); a barcode series raises `TypeError`.
    **Catches:** invalid parameters accepted, an empty domain created
    silently, object values embedded.

- [ ] **39. Window bounds.** `Window(size, step=None, overlap=None)` returns a
  list of lazy start-aligned windows `[a, a + size)` with
  `a = start + k * step` while `a + size <= end`, where
  `step = size - overlap` and the default step is `size`. Incomplete windows
  are dropped. Test with the source series, parametrized over the arguments
  and the window domains: `size=2.0, step=1.0` and `size=2.0, overlap=1.0`
  give `[(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)]`; `size=2.0` gives
  `[(0.0, 2.0), (2.0, 4.0)]`; `size=5.0` gives `[]`. **Catches:** windows
  anchored at the end, partial windows kept, the overlap used as the step, an
  overlapping default.

- [ ] **40. Window contents.** A window evaluates its source inside
  `[a, b)`, so the value at its start comes from the source even when the
  last observation lies before the window. Tests with the window `[1.0, 3.0)`
  from `Window(size=2.0, step=1.0)`:
  - Its times are `[1.5, 2.5]` (original timestamps), `evaluate(1.0) == 2.0`
    (the observation at 0.5), and `evaluate(2.75) == 4.0`. **Catches:** the
    value before the first observation in the window missing, timestamps
    moved to the window start.
  - `evaluate(3.0)` raises `ValueError`. **Catches:** a closed window end.

- [ ] **41. Window parameter errors.** Test, parametrized `ValueError`:
  `size=0.0`, `size=-1.0`, `step=0.0`, `overlap=2.0` and `overlap=-0.5` with
  `size=2.0`, and `step` together with `overlap`. **Catches:** windows that
  never advance, conflicting parameters accepted.

- [ ] **42. Embedding a window.** Test: `TimeDelayEmbedding(dimension=2,
  delay=1.0)` of the window `[1.0, 3.0)` has domain `(2.0, 3.0)`;
  `evaluate(2.0)` is `[2, 3]` (with the value at 1.0 from the source) and
  `evaluate(2.75)` is `[3, 4]`; `evaluate(1.5)` and `evaluate(3.0)` raise
  `ValueError`. **Catches:** history taken from before the window, the
  half-open end lost.

- [ ] **43. Sampler query times.** `RegularSampler(n_points).times(series)`
  returns `Times` with `a + (b - a) * k / n_points` for
  `k = 0, ..., n_points - 1`, in the series' unit. The end is never sampled,
  even for a closed domain. Test: `RegularSampler(n_points=4).times(window)`
  for the window `[1.0, 3.0)` gives `[1.0, 1.5, 2.0, 2.5]` with the window's
  unit. **Catches:** the end point included, spacing `(b - a) / (n - 1)`.

- [ ] **44. Sampling.** `RegularSampler(n_points)(windows)` evaluates each
  series at its query times and returns a `FloatTensor` of shape
  `(len(windows), n_points, *value_shape)`. Test: the three windows of
  `Window(size=2.0, step=1.0)` with `n_points=2` give
  `[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]`. **Catches:** windows stacked out of
  order, values at window starts missing.

- [ ] **45. Sampler errors.** Test, parametrized: `n_points=0` and an empty
  list raise `ValueError`; a list holding a point-cloud series raises
  `TypeError`. **Catches:** an empty result without a value shape, object
  values sampled into a `FloatTensor`.

- [ ] **46. Windows of combined channels.** Test: sampling
  `Window(size=1.0, step=1.0)` of item 33's combined series with
  `RegularSampler(n_points=2)` gives `[[[10, -1], [20, -1]],
  [[20, -2], [20, -2]]]` for the windows `[0.5, 1.5)` and `[1.5, 2.5)`.
  **Catches:** derived series that accept only stored series.

- [ ] **47. The delay workflow end to end.** Test in
  `test/python/test_integration.py`: times `[0.0, 0.5, ..., 3.5]` and values
  `[0, 1, 0, -1, 0, 1, 0, -1]` through
  `TimeDelayEmbedding(dimension=2, delay=0.5)`,
  `Window(size=1.0, step=1.0)` (three windows),
  `RegularSampler(n_points=4)`, `PointCloudTensor`, and
  `compute_persistent_homology(..., max_dim=1)` give samples of shape
  `(3, 4, 2)`, three clouds, and barcodes of shape `(3, 2)`. **Catches:**
  steps that do not compose.

- [ ] **48. Saving derived series.** Implement decision 9: a tree format in
  `include/sbear/io/timeseries_io.hpp` whose leaves reuse item 23's stored
  numeric series, registered as another object type, and Python
  reconstruction of the lazy objects. Test, parametrized over `save`/`load`
  and `pickle`: the window `[1.5, 2.5)` of a
  `TimeDelayEmbedding(dimension=2, delay=0.5)` of item 33's combined series
  loads as the same kind of object with the same domain, unit, and values at
  1.5 and 2.25, and `evaluate(2.5)` still raises `ValueError`. **Catches:** a
  node kind, parameter, or query unit lost on saving, the half-open end lost.

- [ ] **49. Golden files.** Pin both formats (TESTING.md principle 11) with a
  generator modeled on `test/golden/serialization/generate_point_cloud_v3.py`,
  which records explicit expected values. Write a barcode32 series of shape
  `(2, 2)` and item 48's derived series as `.sb` and pickle files, and teach
  `test_serialization_goldens.py` to compare stored series (times, unit,
  domain, values) and derived series (domain, unit, values at listed times).
  **Catches:** a later change to either payload that breaks saved files.

## Documentation and release

Code items update the docstrings they touch (AGENTS.md). These items add the
pages.

- [ ] **50. API reference.** Add `docs/api_timeseries.rst` to the toctree in
  `docs/stablebear.rst`. Check that the docstrings state the contracts of
  decisions 1-4, 7, and 8: domains and their closedness, previous-value
  evaluation and snapping, units and conversion, copies and lazy derived
  series, and the meaning of each axis. Check that
  `compute_persistent_homology`, the three summaries, `save`, and `load`
  document series input and output.

- [ ] **51. User guide.** A `docs/timeseries.rst` guide linked from
  `docs/userguide.rst`: times, units, and domains, then two small examples.
  The first follows a point-cloud series through persistence to stable
  ranks; the second combines two channels in different units and takes them
  through a delay embedding, windows, and sampling to persistence. Link each
  operation to its API entry on first mention. Figures are not needed; if one
  is added, follow the dropdown rule in AGENTS.md.

- [ ] **52. Changelog.** A New features entry in `CHANGELOG.md` linking #242,
  under the heading of the release that ships it (to be decided; 0.5.1 is in
  progress).

## Final checks

- [ ] **53. Test run.** After a CMake build and install, run the new test
  files, then the full Python suite and `sb_test` from `test/`.
- [ ] **54. Docs build.** Run `make html` in `docs/` (with `LC_ALL=C.UTF-8` if
  the locale is missing), check for warnings, and read the new pages.
- [ ] **55. Push.** Push `issue-242-timeseries`, open a PR against `main`, and
  remove this plan before merging.

## Coverage of the acceptance criteria

| Criterion in #242 | Items |
| --- | --- |
| Time, domain, evaluation, shape, and ownership contracts; empty and singleton behavior | 1-9, 20 |
| `Times`, native storage and access, default seconds, explicit units, one timestamp per observation | 10-12, 16-19, 21-22 |
| Floating-point arrays with explicit times or a packed table | 11, 14-15 |
| Native operation contract, dispatch, and adapter; calls, parameters, CPU limits | 24, 27, 31-32 |
| Persistence and barcode summaries of series | 24-30 |
| Delay embedding and windows | 37-42 |
| Regular sampling into `PointCloudTensor` and through persistence | 36, 43-47 |
| Save, load, and pickle for both workflows | 23, 48-49 |
| Tests and documented examples | all test items, 50-51 |
| Channels with their own time grids (added) | 3, 22, 33-35, 46 |

## Out of scope

As in the issue: datetime timestamps, sample-count delays and windows, other
samplers, interpolation, partial windows, gap policies, tensors of series,
operations that combine series other than `combine_channels`, reductions over
time, other value families, provenance, and normalization. Also deferred
here: views instead of copies for stored series, evaluation in C++, and the
branch's notebooks and scikit-learn transformers.
