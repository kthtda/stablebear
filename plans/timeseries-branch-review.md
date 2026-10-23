# Time-series branch review against main

Reviewed `issue-242-timeseries` at `e81fd9cfef628b2dfb95995f99dc509039c419c7`
against `main` at `fc53e0f09e0d6c79a44a2e4e504ab3d7a1008837` on
2026-10-22. The latter is also the merge base. This is a review and work queue;
the implementation and tests have not been changed.

The important gaps are at boundaries: mutable elements leaving immutable
series, domain metadata passing through operations, composed views being
serialized, and new leaf types entering existing container APIs. Most basic
evaluation, snapping, dtype, window, and serialization behavior already has
substantial coverage.

## Execution order and triage

Work down the numbered items, with one focused change per item. Each fix
includes its regression coverage; do not make a separate broad testing pass.
P1 means high-priority correctness work, P2 means a concrete defect or material
follow-up, and P3 means a performance investigation that can be deferred.
Items 1–7 are confirmed defects or inconsistent behavior; item 8 is a coverage
gap without a demonstrated failure. Items 10–12 address specific costs in the
implementation. Item 13 needs release-build profiling before a rewrite.

| Order | Priority | Review item |
| --- | --- | --- |
| 1 | P1 | Isolate mutable observations returned by a series |
| 2 | P1 | Reject NaN queries before batch sorting |
| 3 | P2 | Preserve channel domains through persistence and summaries |
| 4 | P2 | Make closed-window embeddings loadable after saving |
| 5 | P2 | Complete serialization for newly supported nested leaves |
| 6 | P2 | Preserve staged devcontainer state when changing staging mode |
| 7 | P2 | Accept the new leaf types in direct NestedTensor construction |
| 8 | P2 | Cover per-observation batches in numeric series persistence |
| 9 | P2 | Correct and expose the evaluation contracts in the API reference |
| 10 | P2 | Release the GIL during expensive native time-series operations |
| 11 | P2 | Remove the extra coordinate buffer on the existing point-cloud path |
| 12 | P2 | Copy observations into channel storage only once |
| 13 | P3 | Profile the embedding merge's quadratic dimension cost |

### 1. Isolate mutable observations returned by a series

- [x] **P1 — confirmed through public APIs.** In
  [timeseries.tpp](../include/sbear/timeseries.tpp),
  `Channel::copy_observation` (line 253) assigns elements by ordinary C++ copy.
  For point clouds and compressed matrices this shares their coordinate/value
  buffers. `series.values[0][0, 0] = 99` changes the original point-cloud
  series. Mutating a point cloud returned by `series(t)` does the same, as
  does mutating an element of an embedding's `values`. Distance-matrix and
  symmetric-matrix observations also reproduce the problem.

  **Change:** Use the established value-copy boundary, such as
  `detail::store_copy` in [tensor.hpp](../include/sbear/tensor.hpp), when
  observations leave channel storage. Preserve sharing between immutable
  channels and derived views. Update the stale `Channel` comment promising
  materialization on write; that is no longer the intended ownership model.

  **Minimum validation:** Extend the numeric-only isolation coverage with a
  mutable point-cloud observation and representative compressed-matrix
  observations. Check `values` and evaluation results, verify the requested
  mutation actually occurred, then verify the source and an existing derived
  view retain their original coordinates. This targets shared element buffers;
  another matrix of all 19 dtypes is unnecessary.

### 2. Reject NaN queries before batch sorting

- [x] **P1 — confirmed through public APIs.** In
  [timeseries.tpp](../include/sbear/timeseries.tpp), the array overload of
  `TimeSeries::evaluate` (line 415) checks only the minimum and maximum.
  The tensor-of-series overload (line 497) sorts first and checks only its
  endpoint queries. A NaN in the middle can escape both checks and also
  violates the ordering assumptions of the sort comparator.

  For `TimeSeries([10., 20., 30.], times=[0., 1., 2.])`, evaluating
  `[0., np.nan, 2.]` returned `[10., 30., 30.]`; scalar NaN correctly raised.
  Evaluating a tensor of windows at `[0., np.nan, 1.]` also silently returned
  values.

  **Change:** Validate queries before extrema searches and sorting in both
  entry points. Keep the existing domain checks and deterministic series-position
  errors. A small shared query-preparation helper would prevent these two
  implementations from drifting; no general evaluation framework is needed.

  **Minimum validation:** Add a mixed finite/NaN query to the existing
  scalar-versus-batch domain test, and one tensor-of-windows case. Put the NaN
  between valid times so a test cannot pass merely because an endpoint is NaN.

### 3. Preserve channel domains through persistence and summaries

- [x] **P2 — confirmed for both operation families.**
  [ufunc.hpp](../include/sbear/ufunc.hpp),
  `UfuncAdapter<TimeSeries<...>>::finish` (line 211), reconstructs channels with
  their times, shapes, and snap settings, but drops `start_closed()`, `end()`,
  and `end_closed()`.

  A window with domain `(0, 2)` and held observations at 0 and 1 becomes a
  result with domain `[0, 1]`. Both `barcode_to_stable_rank(window)` and
  `compute_persistent_homology(window)` incorrectly accept time 0 and reject
  time 1.5. Checking only result timestamps and values cannot detect this.

  **Change:** Transfer the complete domain metadata when constructing each
  output channel. Account for an explicit closed end coinciding with the last
  timestamp, as in item 4; do not blindly call a setter that rejects equality.

  **Minimum validation:** Use one window with an open start and an end beyond
  its final observation. Verify that a summary and persistence preserve
  acceptance/rejection at the start, in the final interval, and at the end.
  One representative summary exercises the shared adapter; all three summaries
  already have value-level coverage.

### 4. Make closed-window embeddings loadable after saving

- [x] **P2 — confirmed for binary and pickle round trips.**
  `Channel::stack_at` in [timeseries.tpp](../include/sbear/timeseries.tpp)
  (line 113) retains an explicit end while replacing the timestamps. A delayed
  component can introduce a timestamp exactly at that closed end.
  [timeseries_io.hpp](../include/sbear/io/timeseries_io.hpp), line 95,
  reconstructs it with `until`, which requires the end to be strictly later
  than the final timestamp.

  Reproduction:

  ```python
  source = TimeSeries([0., 10., 20., 40.], times=[0., 1., 2., 4.])
  window = Window(size=2., step=1., closed="right")(source)[1]
  embedding = TimeDelayEmbedding(dimension=2, delay=1.)(window)
  # embedding.times is [1., 2.], and its retained closed end is also 2.
  pickle.loads(pickle.dumps(embedding))
  # RuntimeError: Invalid time series in saved data:
  # The end of a channel must come after its last timestamp
  ```

  **Change:** Define one consistent invariant for a closed end at the final
  timestamp and enforce it in view creation and decoding. Continue to read
  already-written valid payloads, including redundant explicit closed ends;
  do not regenerate the compatibility goldens to hide a reader problem.

  **Minimum validation:** Add this composition to the existing `ROUND_TRIPS`
  cases. Check values and the open-start/closed-end evaluation behavior after
  loading. Existing tests cover embeddings and windows separately, but not
  this interaction.

### 5. Complete serialization for newly supported nested leaves

- [x] **P2 — confirmed through public APIs.**
  [nested_tensor.py](../stablebear/nested_tensor.py) now supports boolean,
  PCF, point-cloud, matrix, barcode, and series leaves. The nested save
  dispatch in [io.py](../stablebear/io.py), native bindings in
  [py_io.cpp](../src/python/py_io.cpp), and reader alternatives in
  [tensor_io.hpp](../include/sbear/io/tensor_io.hpp) and
  [io.hpp](../include/sbear/io.hpp) still cover only the six original numeric
  leaf types.

  `sb.save(sb.NestedTensor([sb.BoolTensor([True, False])]), file)` raises
  `TypeError: Unsupported tensor type ...NestedBool`. Point-cloud leaves fail
  similarly. This also affects ordinary results of evaluating multi-channel
  nonnumeric series, and pickling uses the same serialization path.

  **Change:** Connect the new leaf families to both writer and reader dispatch,
  using the existing nested format where possible. Keep this change confined
  to nested serialization; factory support for constructing arbitrary tensors
  of series is explicitly deferred in the implementation plan.

  **Minimum validation:** Reuse the round-trip helper for a boolean leaf, a
  mutable object leaf, and a series leaf. Reuse existing leaf-payload and
  numeric-nesting tests instead of crossing every dtype, depth, and format.
  Add a fixed golden only if this changes or introduces a payload contract.

### 6. Preserve staged devcontainer state when changing staging mode

- [x] **P2 — confirmed from the shell control flow; not run against real
  credentials.** [prepare-home.sh](../.devcontainer/prepare-home.sh),
  lines 62–66, recursively deletes a real staged directory before replacing
  it with a live-host symlink. A normal filesystem started once with
  `SB_DEVCONTAINER_COPY_HOME=1`, then started without it, takes this path.
  Container-created session history and configuration may exist only in that
  staged directory, as the new README explicitly explains.

  **Change:** Preserve or archive an existing copied destination before
  switching to a symlink, with a clear message identifying the retained copy.
  Do not silently merge private state back into the host files.

  **Minimum validation:** Exercise copy-to-live staging with disposable source
  and destination directories containing a destination-only sentinel. Verify
  the sentinel remains recoverable and the new mount source is correct. No
  Docker startup or real authentication files are needed for this check.

### 7. Accept the new leaf types in direct NestedTensor construction

- [x] **P2 — confirmed API inconsistency.**
  [nested_tensor.py](../stablebear/nested_tensor.py), line 195, still accepts
  only `FloatTensor`, `IntTensor`, and NumPy arrays in the direct-leaf branch,
  although `_leaf_tensor` now validates every supported tensor element type.
  `NestedTensor(bool_tensor)` and `NestedTensor(point_cloud_tensor)` raise;
  wrapping the same tensor in a list succeeds. Numeric direct leaves already
  have the documented scalar-outer-tensor behavior.

  **Change:** Route supported direct tensor leaves through the shared leaf
  validation, keeping the preceding `NestedTensor` handling and the existing
  scalar outer shape/depth semantics.

  **Minimum validation:** Extend the existing leaf-construction coverage with
  direct boolean and point-cloud inputs. Verify the scalar outer shape and
  read the leaf with `[()]`; do not add another all-dtype suite.

### 8. Cover per-observation batches in numeric series persistence

- [x] **P2 — important coverage gap; the probe currently passes.** The new
  core-dimension adapter in [ufunc.hpp](../include/sbear/ufunc.hpp) splits each
  channel's observation shape into batch axes and trailing `(n, d)` axes.
  [test_timeseries_persistence.py](../test/python/persistence/test_timeseries_persistence.py)
  checks numeric observations of shape `(n, d)`, while
  [test_ripser.py](../test/python/persistence/test_ripser.py) checks a strided
  tensor batch. Neither exercises a series observation that itself contains
  several point clouds, combined with a channel-offset transition.

  **Change:** Add one small public-API case with two observations, two clouds
  per observation, and a second channel with one observation. Two-point
  one-dimensional clouds at distances 1, 2, 3, 4, 5, and 6 give explicit,
  distinguishable expectations. With `max_dim=0, reduced=True`, the six
  expected barcodes are simply `[[0, distance]]`.

  **Minimum validation:** Check values, retained times, and output shapes
  `(2, 2, 1)` and `(1, 2, 1)` for the two channels' `values`. Parametrizing
  float32/float64 is meaningful here. No combinatorial suite over ranks,
  empty axes, window closures, and summary functions is warranted.

### 9. Correct and expose the evaluation contracts in the API reference

- [x] **P2 — confirmed documentation mismatch.**
  [series.py](../stablebear/timeseries/series.py), `TimeSeries.evaluate`
  (line 410), says every domain is the closed interval between first and last
  observations. Windows can exclude their first timestamp and extend beyond
  their last observation. `TimeSeriesTensor.evaluate` also omits its
  multi-channel return contract. That class is not exported from
  `stablebear.timeseries`, so the module-only
  [API page](../docs/api_timeseries.rst) does not expose its evaluation method.

  **Change:** Describe ordinary and window-derived domains accurately,
  document the per-channel `NestedTensor` result for tensors of series, and
  expose the relevant method in the reference without requiring a new public
  constructor. Include the domain-preservation guarantee fixed in item 3.
  Fold this into existing plan item 58 rather than opening a second broad
  documentation project. Guide and changelog work are already tracked in
  items 59–60 of [timeseries-plan.md](timeseries-plan.md).

  **Minimum validation:** Build the Sphinx docs, inspect the rendered API
  entries and links, and check the examples against actual window evaluation.
  No new Python tests are needed for this prose change.

### 10. Release the GIL during expensive native time-series operations

- [x] **P2 — confirmed binding behavior.**
  [py_timeseries_derived.cpp](../src/python/py_timeseries_derived.cpp),
  lines 19–29, calls evaluation, embedding, and window construction while
  retaining the GIL. The evaluation bindings in
  [py_timeseries.cpp](../src/python/py_timeseries.cpp) do likewise. Large
  computations therefore prevent other Python threads from running even
  when native evaluation uses worker threads. In a local probe, a Python
  timer due after 20 ms ran only when an embedding call returned.

  **Change:** Release the GIL around native computation that does not use
  Python objects, following existing bindings such as
  [py_subsample.cpp](../src/python/point_process/py_subsample.cpp). Keep
  argument conversion, return conversion, and Python-backed file I/O under
  the appropriate GIL ownership. Complete item 1 first so immutable series
  do not expose an obvious competing mutation path.

  **Minimum validation:** Reuse the functional suites and perform a small
  manual threaded responsiveness check. Avoid a wall-clock assertion in CI
  or a custom concurrency test framework.

### 11. Remove the extra coordinate buffer on the point-cloud path

- [x] **P2 — confirmed extra allocation relative to main.**
  [compute_persistence.hpp](../include/sbear/persistence/compute_persistence.hpp),
  lines 117–131, copies every `PointCloud` into a flat `coords` vector, then
  immediately copies it again into Ripser's `rpoints` storage. Main populated
  `rpoints` directly. This adds `n_points * dim * sizeof(T)` bytes and another
  traversal per simultaneous point-cloud task, including indexed clouds.

  **Change:** Share the coordinate-to-Ripser population logic without requiring
  an intermediate coordinate vector, for example through a coordinate accessor.
  Preserve logical indexed-coordinate access. Keep the new contiguous
  numeric-tensor path working.

  **Minimum validation:** Run the existing dense/indexed persistence tests,
  including `test_persistence_accepts_whole_indexed_tensors_and_outer_views`.
  Inspect allocation lifetime or profile a release build to verify removal
  of the buffer. Do not add duplicate numerical tests or allocation hooks.

### 12. Copy observations into channel storage only once

- [x] **P2 — confirmed redundant copy.** The tensor-taking `Channel`
  constructor in [timeseries.tpp](../include/sbear/timeseries.tpp),
  lines 37–46, makes `values.copy()` and then copies every element of that
  tensor into a second container. Numeric data needs two complete buffers;
  value-owned objects such as PCFs are also copied twice. Loading a series
  passes through this constructor after already reading an owning tensor.

  **Change:** Walk the original logical tensor into reserved storage using
  the existing deep element-copy helper, eliminating the intermediate tensor.
  Keep copy-in isolation and support for strided inputs. Do not adopt mutable
  caller-owned storage as the current comment suggests. An owning decode
  fast path can remain separate unless profiling justifies it.

  **Minimum validation:** Reuse the construction/round-trip suites and the
  isolation checks from item 1. Include a small input-mutation check if no
  existing assertion protects copy-in ownership. No allocation-count test is
  necessary.

### 13. Profile the embedding merge's quadratic dimension cost

- [x] **P3 — structural cost confirmed; release impact needs measurement.**
  [time_delay_embedding.hpp](../include/sbear/timeseries/time_delay_embedding.hpp),
  lines 87–120, scans all `m` shifted streams to choose each next time, scans
  them again to advance, then searches all `m` components to detect changes.
  There can be `O(n*m)` candidate times, so construction performs
  `O(n*m*m)` component work before accounting for lookup costs, despite
  storing only a compact view.

  **Change:** First profile construction in a release build over representative
  observation counts and dimensions, separately from materializing `values`.
  If this is material, investigate an ordered pending-event structure and
  incremental change tracking. Optimizing only the minimum-selection loop
  will leave the all-component lookup loop. Preserve rounding coalescence and
  compact storage; a dense table of all component indices defeats the existing
  design's memory goal.

  **Minimum validation:** Reuse the existing rounding, nested-embedding, and
  window-composition tests. Record the profile and decision; no timing-based
  unit tests, benchmark framework, or speculative rewrite is required to
  close this investigation.

  **Outcome (2026-10-22):** At `-O2 -DNDEBUG`, construction cost about 6–7 ns
  per time per component, growing as `n*m*m` (2.4 s for `n=100000, m=64`).
  Adopted a priority-queue merge that checks only components reaching `s` or
  within snapping distance after it (commit `e14d9a316`): 1.1x at `m=2`, 2x at
  `m=8`, 3x at `m=16`, 7.9x at `m=64`, with identical times on irregular
  series and on rounding-coalescing grids.

## Verification performed

- Built and installed with `cmake --build cmake-build-debug
  -j$(nproc --ignore=4)` followed by `cmake --install cmake-build-debug`.
  CMake 4.4.4 and cached GCC 13.4.0 satisfy the repository requirements.
  The build was already current.
- Installed the `test` dependency group because pytest was absent. Every
  pytest command ran from `test/` against the installed package.
- **847 passed** across the three new time-series test files, all Python
  persistence tests, nested-tensor tests, serialization goldens, and async
  task tests. The selected backend was CUDA 12; these time-series operations
  themselves run on the CPU.
- **43 passed** in integration and tensor-comparison tests.
- **362 passed** with `SB_FORCE_CPU=1`, checking the separately built CPU
  extension's time-series and relevant persistence bindings.
- **19 passed** in targeted C++ tests for galloping, stoppable tasks, barcode
  summaries, and related persistence behavior.
- Public-API probes reproduced items 1–5 and 7, and confirmed the currently
  correct uncovered case in item 8. Native-binding inspection and a threaded
  probe support item 10. Copy costs and the devcontainer issue were established
  from source. Debug-build scaling probes informed item 13 but are not release
  performance measurements.
- `git diff --check main...HEAD` passed. The full Python/C++ suites, statistical
  suite, Docker startup, and Sphinx build were not run for this review.

## Coverage to retain without expanding

Keep the fixed serialization goldens, dtype dispatch checks, mixed-unit
channels, snapping-boundary examples, window closure/partial-window cases,
strided numeric persistence, and indexed point-cloud/distance-matrix tests.
The generic task exception tests already protect the waiting mechanism.
There is no reason to duplicate those guarantees per dtype or per operation.

Do not add unsupported series factories, blanket formatting changes, or a
repository-wide dtype registry refactor to this queue. The useful consistency
changes are concrete: reuse the existing deep-copy helper, the common leaf
validator, and the established GIL-release pattern at the affected boundaries.
