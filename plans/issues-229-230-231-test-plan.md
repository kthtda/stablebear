# Test plan: finish PR #233

Initial review on 2026-09-27 used branch HEAD `27106ca6a674347c7defa11a4e467ee5b103dd70`
and refreshed main `4a9c284f4f48d6711083d99ee529819d4c546384` (201 changed files).
The [review checklist](issues-229-230-231-code-review-checklist.md) records the
live PR/issue audit, implementation status, known bug, and historical results.
E3/T1 is now committed in `5c9ae885f`, including both focused regressions.
T2–T4, documentation D4, and final V1 remain open. Remote status has not been
refreshed since the initial review.

## Scope and stopping rule

Protect the user-visible contracts of [#229](https://github.com/kthtda/stablebear/issues/229),
[#230](https://github.com/kthtda/stablebear/issues/230),
[#231](https://github.com/kthtda/stablebear/issues/231),
[#232](https://github.com/kthtda/stablebear/issues/232), and
[#236](https://github.com/kthtda/stablebear/issues/236). Follow the
[testing manifesto](../test/TESTING.md): small distinguishable inputs, obvious
expectations, public APIs, one behavior per test, and meaningful parameterization.

**T1 is complete (`5c9ae885f`). Finish T2–T4, reconcile documentation in
review D4, then perform V1 once on the settled revision. That is sufficient
acceptance for this PR.** Existing coverage counts. Do not reopen completed features just because more combinations can be
tested. Add work only for a demonstrated failure, a specific uncovered issue
requirement, or code changed while completing these items.

Prefer extending an existing test over adding a parallel suite. Preserve both
precisions where bindings or numeric behavior differ; exercise shared sampling
rules across point-cloud and distance-matrix inputs without multiplying each
case by every view, dtype, backend, and generator mode. Use a small C++ test only
for guarantees that the public API cannot observe.

## Existing coverage to retain

“Covered” means committed tests exercise the contract, not that this review
reran them. These areas do not need another acceptance matrix.

| Contract | Current committed evidence |
| --- | --- |
| Point-cloud rank, coordinate views/copies, write-through behavior and rejected writes | `test/python/test_point_cloud_tensors.py`; `test/test_point_cloud.cpp` |
| Runtime nesting, depth-3/scalar/empty values, numeric leaf dtypes, recursive copies, shared views, incompatible bulk assignment | `test/python/test_nested_tensor.py`, `test_tensor_create.py`, `test_tensor_join.py`; nested cases in `test/test_tensor.cpp` |
| Exact-prefix ragged indexing; ordered/repeated/empty selections; validation; owned selections and source-view behavior | `test/python/test_point_cloud_tensors.py`, especially `test_point_selection_*` |
| Generic indexed interface, comparisons/joins/masks/splits, public copied indices, shared view transition and mutation isolation | `test/python/test_indexed_tensor_interface.py`, `test_indexed_tensor_io.py`; indexed cases in `test/test_tensor.cpp` |
| Sampling shapes/dtypes, common sizes, ragged mapping, resampling logical clouds/matrices, retained input indexing, subsequent RNG parity, result mutation independence | `test/python/test_subsample.py`; `0f6d616e2` separates the independent behaviors |
| Fixed-rank coordinate/compressed-matrix storage and ownership | `test/test_fixed_rank_tensor.cpp`, `test/test_point_cloud.cpp`, matrix tests; adaptations committed in `1f88c02eb` and later |
| Indexed persistence/kernel results and retained indexed state, both families and precisions | `test/python/persistence/test_ripser.py`, `test_homological_kernel.py`; these now assert `has_indices()` |
| Barcode tensor isomorphism, aligned shapes/dtypes, tolerances, incompatible inputs | `test/python/persistence/test_barcode_isomorphism.py` |
| Binary/pickle round trips, nested/scalar values, indexed clouds/matrices, loaded mutation and shared storage | `test/python/test_io_roundtrip.py`, `test_indexed_tensor_io.py`, `test_pickle.py`; `test/test_io_readwrite.cpp` |
| Data families use shared pickle reconstruction; metadata/unsupported-state policy | `_pickle_roundtrip` assertions and `test_pickle_inventory_non_data_objects` in `test/python/test_pickle.py`; shared `_BinaryIoMixin` |
| Fixed legacy bytes and pinned V3 cloud bytes with independent values/checksums | `test/python/test_serialization_goldens.py`; 93 artifacts under `test/golden/serialization/` |
| Bad magic/version/truncation/strides, invalid indexed source references/selections, V3 cloud overflow | `test/python/test_serialization_robustness.py`, `test_pickle.py`; `test/test_tensor_io_core.cpp`, `test_io_readwrite.cpp` |

Earlier claims that the fixed-rank test was untracked, matrix IO/consumer tests
were pending, and no current-format corpus existed are obsolete. The `0.5-pre`
corpus contains 12 standalone-cloud artifacts with pinned provenance; it does
not claim to cover every new tensor representation. The additional 0.4.7 3×3
matrix files independently protect compressed payload ordering.

The pickle helper already catches a bespoke reducer introduced for tested data
families. Do not require reflective discovery/instantiation of every future
public class. Preserve old reconstruction symbols and immutable golden bytes;
do not regenerate files to make a failing compatibility test pass.

## T1. Overlapping indexed assignment — completed

- [x] Fix review E3 and add focused public and C++ regressions (`5c9ae885f`).

The Python test in `test/python/test_indexed_tensor_interface.py` explicitly
lists four 2×2 matrices and their vertex selections. It compares the complete
reversed matrices and unchanged source matrices. This protects the assignment
path where Python has already materialized the shared backing.

The C++ test `TensorProperties.OverlappingAssignmentSnapshotsBeforeMaterialization`
in `test/test_tensor.cpp` uses real distance matrices and starts with active
selections. It checks that `assign_from()` itself materializes the result and
preserves RHS values and source independence. These two tests cover distinct
entry states without adding a matrix of view/dtype combinations.

During implementation, after installing the GCC 13 CPU build, these commands
passed from `test/`:

```bash
# 49 passed on the installed stablebear._sb_cpu module.
SB_FORCE_CPU=1 python -m pytest python/test_indexed_tensor_interface.py python/test_bugscan_setitem.py python/test_bugscan_element_assign.py -q
# 171 passed in the CMake GCC 13 CPU build.
../cmake-build-e3/sb_test --gtest_filter='TensorProperties.*:TensorTpp*'
```

The Python regression was subsequently rerun successfully after switching to
full-matrix assertions. The final simplified C++ regression was rebuilt with
`cmake --build cmake-build-e3 --target sb_test -j$(nproc)` and passed with
`--gtest_filter=TensorProperties.OverlappingAssignmentSnapshotsBeforeMaterialization`.
`make html` also succeeded during implementation. No extra public documentation
is needed for restoring expected assignment behavior.

The initial Python build used pip under the old guidance; `AGENTS.md` now
requires direct CMake builds. The results above are development evidence, not
a claim that the full suites ran on `5c9ae885f`. Full-suite and CUDA validation
remain V1 work. T1 needs no further tests unless a new failure or code change
warrants them.

## T2. Sampling and duplicate-removal semantics

- [ ] Extend an existing size/value test to assert default sampling without
  replacement selects each logical row at most once. Drawing all four distinct
  row IDs in each of two samples checks uniqueness and a fresh population per
  sample; the expected unordered IDs are `[0, 1, 2, 3]`.
- [ ] Add one nonempty insufficient-population case with `allow_partial=True`:
  requesting more than `N` returns `N` rows/vertices without replacement.
  Existing empty and with-replacement size tests cover those branches.
- [ ] Add a point-cloud duplicate case with distinct source indices holding
  equal coordinates. From equal generator seeds, compare unfiltered draws
  with `discard_duplicates=True`: keep first occurrences in draw order and
  do not redraw. Check the subsequent sampling call agrees between generators.
- [ ] Add matrix-specific duplicate cases: repeated sampled indices are
  removed, while distinct zero-distance vertices survive. Compare logical
  results with the small principal submatrix for the retained indices.

**Bugs protected:** replacement accidentally enabled by default; populations
exhausted across samples; wrong partial counts; filtering before drawing or
redrawing; unstable filtering order; coordinate-based deduplication applied to
matrices.

Keep these in `test/python/test_subsample.py`. Use public `indices` for draw
identity and coordinates/matrix entries for logical results. Do not pin the
exact random sequence across standard-library implementations. The two duplicate
rules warrant separate readable tests rather than a type-dependent reference
algorithm.

The coordinate filter has explicit signed-zero/NaN handling. A small table for
these two cases is useful if needed to exercise those branches while adding the
duplicate test; an exhaustive floating-value/dimension/precision matrix is not
required. Keep normal finite-coordinate behavior the main example.

## T3. Validation and deterministic random streams

- [ ] Cover a representative invalid count type and nonpositive count with
  the documented `TypeError`/`ValueError`. Parameterize `n_points` and
  `n_samples` if clearer than separate tests; no catalogue of exotic coercion
  objects or every invalid flag is required.
- [ ] Put an undersized input last in a small ragged input. A non-partial call
  must raise before advancing its generator: the next valid draw must match a
  same-seed control that never made the failing call. This checks validation
  after valid earlier cells. Do not repeat the RNG assertion for every error.
- [ ] Check row-major stream allocation with a small seeded example: compare
  batched samples with the same logical sequence of individual calls.
- [ ] Separately compare same-seed results with one worker and an available
  multi-worker run. Restore CPU limits afterward. Use existing RNG helpers;
  do not introduce scheduling instrumentation.

**Bugs protected:** invalid calls consuming random state, worker scheduling
choosing seeds, or wrong output-to-stream mapping. Also inspect that the shared
sampler reserves one stream per output and zero for an empty outer output.
Existing resampling tests already check equivalent indexed/dense calls leave
the generator at the same subsequent state.

One representative family/precision can establish the shared stream algorithm;
the existing family/precision cases exercise bindings. No required matrix of
explicit/global generators × all worker counts × all outer layouts × both
extension modules. Do not add statistical uniformity thresholds or benchmarks
as merge gates: review the uniform draw algorithm and test its exact contracts.

## T4. Sampling owns its input snapshot

- [ ] Add one direct input-to-output independence test: retain an input view,
  sample from that input, save expected sampled values, successfully mutate
  through the retained input view, and verify sampled values remain unchanged.

**Bug protected:** retaining the caller's coordinate/matrix storage instead of
the fresh source copy required by #229/#236. Current mutation tests mostly
check the opposite direction (result writes do not change input).

Use the existing point-cloud/distance-matrix fixture where it keeps the test
simple. This belongs with sampling tests. Parent/sibling view visibility,
repeated-row write independence, copied indices, and resampling isolation
already have tensor/sampling coverage; do not reproduce them here or require
allocation counters to prove the same ownership behavior.

## V1. Final verification

- [ ] Complete review S1 and documentation D4 before the final full run; E3 is done.
- [ ] Build/install settled code and run existing Python and C++ suites.
- [ ] Run Sphinx after documentation changes.
- [ ] Inspect CI for the same revision and record the tested commit, actual
  modules, results/skips, and failures still needing resolution.

Follow [AGENTS.md](../AGENTS.md): configure, build, and install directly with
CMake, using GCC 13 or newer on Linux. Do not build this project through pip.
Always run pytest and the C++ executable from `test/`. Match build jobs to
available CPUs.

```bash
# From the repository root:
cmake -B cmake-build-debug
cmake --build cmake-build-debug -j$(nproc)
cmake --install cmake-build-debug
cmake --build cmake-build-debug --target sb_test -j$(nproc)

# Run from test/:
cd test
python -m pytest python
../cmake-build-debug/sb_test
```

Use `SB_FORCE_CPU=1 python -m pytest python` for a separate `_sb_cpu` run when
the default run loads a CUDA module; no duplicate run is needed if the default
already loads `_sb_cpu`. Record module identity. Test an actual available CUDA
module locally or use CI evidence for the same code revision. Unavailable CUDA
12/13 variants are limitations to report, not passes or a reason to invent
another backend matrix. Sampling/IO themselves execute on CPU.

Run `make html` from `docs/`. Existing CI handles supported platform wheels,
source distribution, coverage and memory checks. Inspect relevant failures;
do not require a separate coverage-instrumented local run or percentage target
on top of passing regression/CI checks.

The prior 2026-09-26 result (2,427 Python CUDA-module tests; 2,397 CPU-module tests
with 30 skips; 491 C++ tests; successful HTML build) was for `991d8e5ce` plus
then-pending tests. It cannot serve as the final result for current HEAD. CI for `27106ca6a` was
pending, with the CUDA job skipped, at the initial review; it has not been
refreshed for this progress update.

After V1 passes, stop. Repeat focused checks for a subsequent code fix, and
repeat broader validation only when that change or a failure justifies it.

## Explicitly outside the completion gate

These were in the previous plan or are possible extensions. They are not
unfinished acceptance requirements for this PR:

- **Individual matrix selector APIs (old A5).** Standalone `DistanceMatrix`
  and `SymmetricMatrix` currently provide scalar `(i, j)` access. Adding slice,
  boolean-mask, and integer-tensor principal-submatrix APIs is separate feature
  work; #236 requests sampling of matrix tensors. Point-cloud coordinate
  indexing already has tests. Do not claim the matrix APIs exist or implement
  them merely to satisfy a test-plan expansion.
- **Every operation at every nesting depth/shape/dtype.** Representative deep,
  scalar, empty, view, copy, and invalid-assignment cases cover the new recursive
  behavior. Add cases when a specific operation changes or fails.
- **A second complete golden corpus.** Keep the existing 93 fixed files and
  round trips. New nested/indexed current-format goldens can be added when
  future format evolution needs them. No all-types/all-protocols corpus,
  manifest-discovery framework, or arbitrary corpus-size budget is needed now.
- **All-direction CPU/CUDA12/CUDA13 serialization exchange.** Shared IO tests
  on actual available modules and historical backend-produced files provide
  meaningful evidence without a cross-product subprocess harness.
- **Broad fuzzing, exhaustive malformed metadata, task cancellation/lifetime
  matrices, allocation instrumentation, and stochastic uniformity thresholds.**
  Keep targeted safety/ownership regressions; investigate a concrete concern
  if source review or CI reveals one.
- **Barcode scalar/empty/view permutations and whole-repository test rewrites.**
  The #232 core contract is covered. The manifesto guides tests changed while
  finishing this PR; it does not require rewriting every existing test.

Compatibility exclusions and API differences stay visible in review D4.
Reducing test scope must not hide a known data-corruption bug or change an
expected result to match a failure.
