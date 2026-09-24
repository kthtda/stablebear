# Code review checklist: PR #233

This document records review findings and implementation evidence for
[PR #233](https://github.com/kthtda/stablebear/pull/233), covering issues
#229–#232. The 2026-09-22 update is review only: no fixes or tests were added.
Previous completed items remain as historical evidence; new findings below
identify gaps beyond the cases those fixes covered.

## Current review: 2026-09-22

Reviewed local branch `issue-229-uniform-subsampling` at
`cced5d9b78c56f97cee4495505df5c0560438fcb`, against PR base/main
`4a9c284f4f48d6711083d99ee529819d4c546384`. GitHub PR #233 currently points
to `b0b39ee1530230d0ac021276b5f5c3e770fe948f`: the local D2 implementation
and its checklist commit are two commits ahead of that remote head. The D3
work in `stash@{0}` was not applied and is not counted as branch coverage.
The local net diff contains 148 changed files.

**Assessment:** The major features exist, but the branch is not ready for
acceptance. Four additional correctness findings are recorded in section E;
D3's extra resampling copy and D4's completion audit remain open. Passing
regression tests do not cover the failing cases below.

| PR scope | Current coverage and remaining work |
| --- | --- |
| #229 uniform point-cloud subsampling | API, indexed backing, owned selections, shared materialization, and indexed persistence integration are implemented. D3 still copies indexed input twice; E3/E4 expose C++ indexed-storage correctness gaps. Committed sampling tests do not yet establish the full RNG/deduplication/copy-count contract. |
| #230 binary-backed pickle and compatibility | Shared reducers, standalone object IO, and released 0.4.7 fixtures are present. E1 demonstrates scalar data loss, including inside the new nested type; E4 corrupts legitimate C++ coordinate views. The PR's checked issue-tracking box is not sufficient acceptance evidence. |
| #231 runtime nesting and ragged indexing | Recursive types, exact-prefix selection validation, ordinary views, and D2 copy boundaries are implemented. E2 allows bulk writes to violate homogeneous nesting depth; scalar IO also fails. |
| #232 barcode tensor isomorphism | Implemented with aligned, same-dtype tensor semantics; existing tests and additional scalar/empty/transposed probes passed. No substantive issue found in the reviewed scope. |
| #236 distance-matrix subsampling | Not implemented: the sampler accepts only `PointCloudTensor`. The issue and PR explicitly identify this as a follow-up blocked by #229; do not count it as delivered or as an unexpected defect in this PR. |

**Verification:** Rebuilt and installed the committed source after stashing D3.
From `test/`, the full Python suite passed **2,358 tests with `_sb_cuda12`**;
`SB_FORCE_CPU=1` passed **2,328 tests with `_sb_cpu`, with 30 CUDA tests
skipped**. The full C++ executable passed **486 tests**. `make html` in `docs/`
succeeded. These are local builds, not clean-wheel/platform-matrix validation.
Sampling and serialization run on CPU even when the CUDA module is loaded.
CUDA 13 and bidirectional cross-module file writing/reading were not exercised.

Focused, temporary reproductions confirmed E1/E2 on both extension modules,
and E3/E4 with standalone C++ programs compiled against the reviewed headers.
Additional sampler probes passed validation, no RNG advancement on failure,
row-major stream allocation, empty cases, stable duplicate filtering, and
resampling value/ownership checks. These probes are review evidence, not
committed regression coverage. See the updated [test plan](issues-229-230-231-test-plan.md)
for the coverage inventory and missing cases.

The original review covered commit `6d3e00e36f7114dcffe54a675262e49b7f44e9e7`
against the same base. Sections A–D retain its findings and subsequent fixing
commits; its original focused reproduction results should not be confused with
the current full-suite run.

P1 denotes a correctness or compatibility blocker for the stated issue scope.
P2 denotes an API/validation gap that should also be addressed before acceptance.
Prioritize the correctness findings in section E and the remaining D3 work
before D4 acceptance. Each item describes its own completion criteria; related
fixes may be implemented together. Record the fixing commit
and verification evidence before checking an item off.

References such as “test plan E3” refer to the separate
[test plan](issues-229-230-231-test-plan.md). Checklist IDs in this document
identify review work, not test-plan items. The broader test plan still applies.

## A. Storage ownership and view semantics

- [x] **A1. [P1] Materialize the shared indexed backing state.**

  **Finding:** `PointCloudTensor._ensure_writeable()` replaces only the calling
  wrapper's `_data`. Given `view = samples[:1]`, writing
  `view[0][0, 0] = 999` changes the view while the parent keeps its old value.
  Existing views silently cease to describe the same tensor state.

  **Change:** Separate shared backing state from outer-view metadata. A write
  through any view must materialize the complete shared backing tensor once,
  then apply the mutation through that view's mapping. Existing parent,
  sibling, and cell views must resolve the updated state.

  **Complete when:** All aliases of the same logical cell observe the write;
  other sampled cells, repeated selected rows, the input, and separate sampling
  results remain independent. Later writes do not repeat materialization, and
  no mixed locally indexed/materialized state remains.

  Source: [point_cloud.py](../stablebear/point_cloud.py), `_ensure_writeable()`
  and `astype()`. Verification: test plan A4, E2–E3.

  **Completed:** Fixed by `33acda7fc`. Verified with the full Python suite
  (2,219 passed), the full C++ suite (472 passed), and focused point-cloud and
  subsampling tests on both the default and forced-CPU backends.

- [x] **A2. [P1] Preserve nested outer views instead of deep-copying them.**

  **Finding:** Python wraps outer views through a C++ constructor that
  recursively copies children. Modifying `nested[:1][0][0]` leaves the parent
  unchanged. Replacing a child through the slice also leaves the parent
  unchanged. This breaks normal view semantics and adds unexpected copying.

  **Change:** Distinguish construction that copies caller-owned values from
  internal construction that wraps an existing outer view. Preserve the stored
  depth descriptor for empty views, while keeping `copy()` and `deepcopy()`
  recursively independent.

  **Complete when:** Slicing and other ordinary outer views share the intended
  storage and propagate writes; value construction/assignment and explicit
  deep copies retain their documented independence. Metadata-only views do
  not recursively copy child coordinates or selections.

  Sources: [nested_tensor.py](../stablebear/nested_tensor.py), `_to_py_tensor()`;
  [nested_tensor.hpp](../include/sbear/nested_tensor.hpp), the const-reference
  constructor and `copy_nested()`. Verification: test plan B3–B4.

  **Completed:** Fixed by `ef3d1620f`. Verified with the full Python suite
  (2,224 passed), the full C++ suite (473 passed), focused nested-tensor tests
  on both the default and forced-CPU backends, and explicit construction,
  assignment, outer-view, empty-view, `copy()`, and `deepcopy()` coverage.

- [x] **A3. [P1] Own point-cloud selections independently of the caller.**

  **Finding:** After `selected = points[selections]`, changing
  `selections[0][0]` changes the selected coordinates. Replacing the child
  selection can change the output cloud's length. The adapter retains mutable
  caller-owned selection storage.

  **Change:** Copy selections at the public indexing boundary, then share that
  owned storage among result views. Keep direct indexing's aligned source-view
  semantics separate from subsampling's independent coordinate snapshot.

  **Complete when:** Editing, replacing, or deleting caller-owned selections
  cannot change or invalidate an existing result. Views of the result still
  share its owned selections without unnecessary additional copies.

  Source: [tensor.tpp](../include/sbear/tensor.tpp), `make_indexed_tensor()`.
  Verification: test plan C2, E1–E2.

  **Completed:** Fixed by `9b31341a4`. Verified with the full Python suite
  (2,226 passed), the full C++ suite (474 passed), focused point-cloud selection
  tests on both the default and forced-CPU backends, and explicit coverage for
  caller mutation, child replacement, caller deletion, shared result views,
  retained source-view behavior, coordinate-rank validation, and IO round trips.

## B. Indexed tensor API and downstream algorithms

- [x] **B1. [P1] Accept whole indexed tensors in persistence algorithms.**

  **Finding:** Persistent homology and homological kernels reject the output of
  `subsample` with an incompatible-arguments `TypeError`. The bindings accept
  ordinary point-cloud tensors or individual clouds, but not indexed tensors.
  Individual sampled clouds work, so single-cloud coverage misses the failure.

  **Change:** Let the algorithms and bindings consume const logical point-cloud
  access for both indexed and materialized tensors. Preserve task input
  lifetime and avoid materializing the indexed source as a workaround.

  **Complete when:** Whole sampled/selected tensors and their outer views work
  at both precisions, agree with dense equivalents, and retain indexed storage
  through const computations.

  Sources: [py_ripser.cpp](../src/python/persistence/py_ripser.cpp),
  [py_homological_kernel.cpp](../src/python/persistence/py_homological_kernel.cpp),
  and their C++ task implementations. Verification: test plan F1–F2.

  **Completed:** Fixed by `2edada84e`. Verified with the full Python suite
  (2,232 passed), the full C++ suite (474 passed), and focused public-API tests
  on both the default and forced-CPU backends. Coverage includes both
  precisions, whole indexed tensors, multidimensional outer views, reordered
  and repeated rows, reduced and unreduced persistence, mixed indexed/dense
  kernel inputs, dense-equivalent results, and retained indexed storage. A
  public-API smoke check also covered whole `subsample` outputs at both
  precisions.

- [x] **B2. [P1] Complete the ordinary tensor interface for indexed results.**

  **Finding:** Equality, `array_equal`, stacking, concatenation, masked
  selection, and splitting fail on indexed results. The indexed binding
  registration provides only part of the interface assumed by the Python
  `PointCloudTensor` wrapper.

  **Change:** Audit the inherited public operations. Provide indexed
  implementations where appropriate and explicit, semantically correct
  materialization paths for operations that require ordinary storage. Reuse
  common bindings where possible without losing ownership or view behavior.

  **Complete when:** Promised operations accept indexed inputs, including
  combinations with materialized tensors where valid, and produce correct
  shapes, values, comparison results, and mutation behavior. No inherited
  operation fails merely because an expected backend method is missing.

  Sources: [py_tensor.hpp](../src/python/py_tensor.hpp),
  `register_indexed_tensor_bindings()`;
  [tensor_create.py](../stablebear/tensor_create.py);
  [_tensor_base.py](../stablebear/_tensor_base.py).
  Verification: test plan E5, I1.

  **Completed:** Fixed by `9c0ea99c1`. Tensor algorithms and bindings now
  dispatch generically from the complete C++ property bitmask, with one
  registration path for ordinary and indexed tensors. Selection and join
  operations materialize self-contained logical values, splits retain indexed
  view semantics, and assignments update shared indexed state without mutating
  the original source. Verified for both point-cloud precisions and mixed
  indexed/materialized inputs with the full Python suite (2,248 passed) and
  full C++ suite (475 passed).

- [x] **B3. [P2] Validate the complete public point-cloud indexing contract.**

  **Finding:** A source with shape `(2, 1)` accepts selections with shape
  `(2, 3)`, although #231 requires an exact prefix match. Out-of-bounds
  selections succeed at construction and fail only when a cloud is accessed.

  **Change:** Validate prefix compatibility, nesting depth, child dtype/rank,
  and row bounds before returning a result. Generic C++ singleton broadcasting
  may remain available independently of the stricter public point-cloud API.

  **Complete when:** Invalid selections fail at the indexing call with the
  required exception categories, including invalid late children. Valid scalar,
  empty, and extra-trailing-axis selections continue to work. Validation does
  not materialize coordinates or mutate either input.

  Sources: [point_cloud.py](../stablebear/point_cloud.py), `__getitem__()`;
  [tensor.tpp](../include/sbear/tensor.tpp), `make_indexed_tensor()`;
  [point_cloud.hpp](../include/sbear/point_cloud.hpp), `index_into()`.
  Verification: test plan C1–C2.

  **Completed:** Fixed by `b0c9791a0`. Indexed tensor construction now
  validates every aligned element/index pair generically in C++, while the
  public point-cloud API requires the source shape to exactly match the
  selection's leading dimensions. Invalid nesting, leaf dtypes/ranks, and
  point indices fail before a result is returned; access revalidates indices
  after shared source replacement. Added the `sb.indices()` convenience
  constructor and documented the selection contract. Verified with the full
  Python suite (2,262 passed) and full C++ suite (476 passed).

## C. Serialization and compatibility

- [x] **C1. [P1] Preserve indexed representation through binary IO and pickle.**

  **Finding:** Python `_save()` materializes indexed tensors before dispatch.
  Save/load returns an ordinary point-cloud tensor, losing the adapter and
  source sharing. Tensor pickling uses the same path. Large sample collections
  can consequently produce much larger files.

  **Change:** Implement a distinct tensor-level indexed encoding containing
  the materialized source, aligned source-view metadata, and owned nested
  selections. Keep existing subtype meanings and compatibility readers intact.

  **Complete when:** Binary and pickle round trips retain indexed behavior,
  ordered/repeated/empty selections, ambient dimensions, source sharing, and
  safe mutation semantics. Empty and non-owning views produce self-contained
  payloads. Saving does not expand every output cloud into dense coordinates.

  Sources: [io.py](../stablebear/io.py), `_save()`;
  [tensor_io.hpp](../include/sbear/io/tensor_io.hpp);
  [io.hpp](../include/sbear/io.hpp). Verification: test plan G1–G3, H2, J1.

  **Completed:** Fixed by `0e437023b`. Binary and pickle round trips retain
  indexed point-cloud storage, shared coordinate sources, owned selections,
  and safe materialization on mutation. Coverage includes ordered, repeated,
  empty, scalar, and sliced selections at both precisions, plus a payload-size
  check confirming that saving does not materialize every selected cloud.

- [x] **C2. [P1] Restore the legacy nested `(5, 64)` decoder.**

  **Finding:** The review initially treated `(5, 64)` as a supported historical
  format because commit `0c2cb6efc` wrote one-level `uint64` index tensors with
  that identifier.

  **Decision:** No decoder or fixture is required. An ancestry audit showed
  that `0c2cb6efc` exists only on this feature branch: it is not an ancestor of
  `main` or `origin/main`, and no release tag contains it. Main therefore never
  produced `(5, 64)` files.

  **Complete when:** Repository history confirms that the format was never on
  main or in a release, and compatibility plans no longer require it.

  Sources: historical writer at commit `0c2cb6efc`; ancestry of `main`,
  `origin/main`, and release tags. Verification: test plan G2.

  **Resolved without a fix:** Compatibility with an unreleased intermediate
  branch format is intentionally not retained.

- [x] **C3. [P2] Route standalone `PointCloud` IO and pickle through binary IO.**

  **Finding:** `sb.save(cloud, ...)` raises `AttributeError` because `PointCloud`
  has no `_data`. Pickle instead uses default Python object state, including
  `_owner`, `_outer_index`, and `_detached`. An owner-backed cloud can therefore
  serialize its parent tensor rather than only the logical standalone value.

  **Change:** Add standalone cloud support to public binary IO and an explicit
  binary-backed pickle reducer. Audit compatibility needs for any replaced
  Python-state representation; preserve required historical reconstruction
  paths without continuing to write that representation.

  **Complete when:** Standalone and owner-backed clouds at both precisions
  round-trip through public IO and pickle with correct logical values and
  types. Pickles do not capture unrelated parent cells or depend on the original
  parent. Required legacy bytes remain readable.

  Sources: [point_cloud.py](../stablebear/point_cloud.py), `PointCloud`;
  [io.py](../stablebear/io.py), object dispatch and `save()`.
  Verification: test plan H1–H4.

  **Completed:** Fixed by `654ba0b9c`. Standalone and owner-backed point
  clouds now serialize their logical value through binary object IO at both
  precisions; pickle uses the same path without retaining the parent tensor.

- [x] **C4. [P1] Use binary-first pickle loading with explicit legacy fallback.**

  **Finding:** Issue #230 requires all newly written pickles to use Stablebear
  binary IO while retaining compatibility with previously written pickle
  representations. The current review items require both halves independently
  but do not define one coherent decoding order or failure contract.

  **Change:** Route every new pickle reducer through the same binary bytes
  written by public `save()`, and use shared reconstruction entry points that
  first attempt public binary `load()`. If the payload is not a supported
  Stablebear binary representation, dispatch to the retained legacy decoder
  for that specific type and historical representation. Keep historical
  callable/module paths that existing pickle opcodes reference. Do not resume
  writing legacy representations.

  **Complete when:** Every supported public pickleable type writes binary IO,
  binary payloads are always attempted first, and fixed historical pickles for
  every replaced representation load through an explicit type-specific
  fallback. Corrupt current binary data is not silently accepted as legacy
  state. If both paths fail, the raised error identifies the binary failure and
  the attempted legacy path instead of hiding the original cause. Unsupported
  types and unrecognized legacy state fail clearly.

  Sources: [_tensor_base.py](../stablebear/_tensor_base.py), tensor reducers;
  [io.py](../stablebear/io.py), load and pickle reconstruction dispatch;
  standalone-object reducers in `stablebear/`; historical implementations and
  their retained reconstruction symbols. Verification: test plan H1–H4.

  **Completed:** Fixed by `654ba0b9c`. Supported objects share one binary IO
  reducer and loader. Recognized current binary payloads never fall back after
  corruption; non-binary payloads use the retained type-specific legacy
  decoders. Stablebear 0.4.7 binary and pickle goldens cover the historical
  formats that existed on `main`.

## D. Simplification and completion review

- [x] **D1. Consolidate the indexed storage model.**

  **Recommendation:** The branch combines tensor-level adapters, per-element
  selection/local-materialization state, and Python wrapper-level detachment.
  Consolidate around materialized `PointCloud` values, a read-only logical
  accessor, and one shared tensor-level materialization transition. The
  `PointCloud` index member may implement that transient read-only accessor,
  but copies and ordinary tensor serialization must materialize it; no released
  format requires a second persistent indexed-cloud ownership model.

  **Complete when:** Ownership and mutation paths are explicit, obsolete local
  indexing/materialization paths are removed or confined to transient logical
  access, and the fixes in section A do not require duplicated mechanisms.
  Verify the property-bitmask design and const algorithm access remain intact.

  Sources: [point_cloud.hpp](../include/sbear/point_cloud.hpp),
  [tensor.hpp](../include/sbear/tensor.hpp),
  [point_cloud.py](../stablebear/point_cloud.py).
  Verification: test plan B5, E1–E3, F1.

  **Completed:** Fixed by `5cf5411b0`. Persistent selections and the one-time
  materialization transition live in shared tensor-level state. Locally
  indexed `PointCloud` values are confined to transient read-only access;
  copies, casts, and ordinary serialization materialize their logical values.
  Indexed serialization is property-driven through an extensible codec, and
  the Python façade delegates writes through the shared tensor transition.

- [x] **D2. Make copy and view construction explicit.**

  **Recommendation:** Distinguish “copy supplied values” from “wrap shared
  storage” at nested and indexed construction boundaries. Apply the same
  distinction in bindings so the ownership contract does not depend on an
  incidental const-reference overload or an implicit handle copy.

  **Complete when:** Public value construction and assignment isolate caller
  data; internal view construction shares backing state; explicit deep copies
  remain independent. Document these boundaries and eliminate redundant copies
  introduced while wrapping values across Python/C++.

  Sources: [nested_tensor.hpp](../include/sbear/nested_tensor.hpp),
  [nested_tensor.py](../stablebear/nested_tensor.py),
  [py_tensor.cpp](../src/python/py_tensor.cpp), and indexed construction.
  Verification: test plan B3–B4, C2, E1–E2.

  **Completed:** Fixed by `44924b7e0`. Nested value
  construction now recursively copies regardless of C++ value category;
  explicit leaf/outer-view factories and Python internal wrapping preserve
  shared storage. Python list construction no longer recursively copies its
  already-copied children again, and construction from an existing Python
  nested tensor isolates caller data. Indexed construction always copies
  caller selections; a separately named owned-indices factory adopts fresh
  sampler selections. Binary readers explicitly wrap their fresh storage.
  Public ownership behavior is documented in `docs/tensors.rst`.

  Verified with 2,358 Python tests using `_sb_cuda12`, 2,328 passed and 30 CUDA
  tests skipped using the separate `_sb_cpu` extension (`SB_FORCE_CPU=1`),
  all 486 C++ tests, and a successful Sphinx HTML build. New regressions cover
  all six numeric leaf dtypes, deep/scalar/empty and noncontiguous values,
  repeated children, assignment independence, temporary C++ views, and buffer
  identity for shared views and adopted selections. The CUDA suite includes
  device-parametrized tests; the ownership checks themselves do not require GPU
  execution. The documented depth-3 example also checks differently shaped
  tensors and demonstrates indexing at each level.

- [ ] **D3. Copy logical coordinates only once when resampling indexed input.**

  **2026-09-22 status:** Still present at the reviewed commit. The proposed
  implementation and additional tests are stashed, not part of this branch.
  No stashed verification is credited to this review.

  **Finding from source inspection:** `subsample.py` first materializes indexed
  input, then the C++ sampler copies those coordinates again into its fresh
  source. This adds an unnecessary full coordinate allocation/copy and violates
  the intended one-copy sampling design.

  **Change:** Accept const logical input directly and populate the fresh
  materialized source once. Preserve complete validation before random-stream
  reservation and coordinate-copy completion before returning.

  **Complete when:** Indexed and ordinary inputs each incur one logical source
  copy per input cloud, regardless of sample count. Resampling creates no
  adapter chain, keeps input/result independence, and leaves random-stream
  behavior unchanged. Verify the direct indexed dispatch and resulting values.

  Sources: [subsample.py](../stablebear/point_process/subsample.py),
  [subsample.hpp](../include/sbear/point_process/subsample.hpp).
  Verification: test plan D2–D4, E1, E4.

  **Implemented and verified in the working tree (2026-09-22):** The C++
  sampler accepts const tensors with arbitrary property bitmasks; Python binds
  ordinary and indexed inputs at both precisions and no longer materializes
  indexed input before sampling. The existing logical coordinate copier now
  writes directly into each fresh source cloud. Validation still completes
  before reserving random streams, and the parallel walk completes before
  returning. Resampling retains input storage and produces dense source cells
  shared only among that input cell's new samples, without an adapter chain.

  The focused Python regression rejects intermediate `materialize()` calls,
  compares indexed resampling with a dense equivalent, and checks subsequent
  generator state. Subsampling itself executes on CPU in both extension
  modules.

  Remains unchecked until the fixing commit is recorded.

- [ ] **D4. Reconcile documentation and verify the settled implementation.**

  Update public documentation, docstrings, and stale storage descriptions to
  match the accepted ownership, indexing, serialization, and `PointCloud`
  behavior. Follow [CLAUDE.md](../CLAUDE.md) for build/install and verification.
  Record fixing commits and evidence for each finding, including actual backend
  coverage and remaining limitations. Do not treat forced CPU execution inside
  a CUDA module as separate CPU-extension coverage.

  **Complete when:** Review findings have concrete resolution evidence and the
  relevant test-plan acceptance checks pass. Generate current-version golden
  files only after the intended serialized representation has settled; retain
  actual 0.4.7 and other historical bytes unchanged. Any unresolved finding
  remains unchecked rather than being hidden by a changed expected result.

  Verification: test plan H4, I3, J1, K1.


## E. Additional correctness findings from the 2026-09-22 review

- [ ] **E1. [P1] Serialize the value in every scalar tensor, including nested numeric leaves.**

  **Finding:** `serialized_tensor_size()` treats shape `()` as one element
  only for tensors of nested nodes. Numeric scalar leaves are serialized with
  zero elements and reload as zero. A scalar ordinary `PointCloudTensor` loses
  its entire cloud. Both public binary IO and pickle exhibit this data loss.

  ```python
  import pickle
  import numpy as np
  import stablebear as sb

  x = sb.NestedTensor([np.array(42, dtype=np.int64)])
  y = pickle.loads(pickle.dumps(x))
  print(x[0][()], y[0][()])  # 42, 0
  ```

  A `PointCloudTensor(np.array([[1., 2.], [3., 4.]]))` has outer shape `()`;
  its restored cloud has shape `(0, 0)` instead of `(2, 2)`. Reproduced on both
  modules, for all six nested numeric dtypes and both point-cloud precisions.
  Ordinary scalar IO has an inherited size-convention problem; the new nested
  scalar-leaf feature makes this directly relevant to #231 as well as #230.

  **Complete when:** Scalar tensors serialize exactly one logical value at
  every nesting level, while true zero-extent tensors serialize none. Public
  binary and pickle tests compare nonzero scalar values and cloud contents to
  independent expectations, including scalar leaves inside the documented
  depth-3 example. Preserve supported historical decoding deliberately.

  Source: [tensor_io.hpp](../include/sbear/io/tensor_io.hpp),
  `serialized_tensor_size()` at lines 456–468 and ordinary tensor payload IO.
  Verification: test plan G1, H2, J1. Existing round-trip/golden tests use
  non-scalar numeric leaves and do not catch this case.

- [ ] **E2. [P1] Preserve the nested depth invariant across bulk assignment and descriptor boundaries.**

  **Finding:** Scalar element assignment checks depth through `_decay_value`,
  but slice, boolean-mask, and integer-index assignment of a tensor RHS bypass
  that check. An incompatible assignment succeeds and leaves an invalid
  tensor: stored root depth disagrees with its children, and later slicing or
  serialization fails.

  ```python
  leaf = lambda values: sb.tensor(values, dtype=sb.int64)
  target = sb.NestedTensor([leaf([1]), leaf([2])])
  rhs = sb.NestedTensor([sb.NestedTensor([leaf([3])])])
  target[:1] = rhs  # Incorrectly succeeds; target.depth stays 2
  target[:]        # Now raises: children have different nesting depths
  ```

  Mask `[True, False]` and integer selector `[0]` reproduce the same problem.
  Full-slice assignment can leave `target.depth == 2` but
  `target[:].depth == 3`. Related descriptor-validation gaps: the scalar-outer
  constructor silently ignores `depth=3` or even `depth="bad"`; stacking or
  concatenating empty nested tensors of depths 2 and 3 silently adopts the
  first operand's depth. Empty data must not erase incompatible type metadata.

  **Complete when:** Every construction/join/assignment route validates the
  declared recursive type, including empty operands, before copying or writing
  values. Wrong-depth bulk writes reject without changing the destination or
  any aliases. Compatible writes still propagate through views, and valid
  independently shaped children remain supported.

  Sources: [nested_tensor.py](../stablebear/nested_tensor.py), scalar
  construction at lines 107–113 and `_decay_value()` at 219–225;
  [_tensor_base.py](../stablebear/_tensor_base.py), tensor-RHS assignment paths
  at 529–531, 559–572, and 607–608;
  [tensor_create.py](../stablebear/tensor_create.py), join dispatch.
  Verification: test plan B2–B4. D2's positive ownership tests pass but do not
  cover incompatible descriptors or bulk assignment failure atomicity.

- [ ] **E3. [P1] Snapshot overlapping indexed assignment before mutating shared backing.**

  **Finding:** The C++ overlap guard in `Tensor::assign_from()` explicitly
  excludes indexed tensors. Assigning a reversed view back into the same
  indexed state overwrites values that later RHS reads still need. With four
  logical values `1, 2, 3, 4`:

  ```cpp
  auto reversed = indexed[std::vector<sb::Slice>{
    sb::range(std::nullopt, std::nullopt, -1)}];
  indexed.assign_from(reversed);
  // Observed: 4, 3, 3, 4. Required: 4, 3, 2, 1.
  ```

  Reproduced using a simple indexable C++ value and the generic indexed tensor
  interface. This is a C++ storage bug, not a demonstrated public Python
  `PointCloudTensor` assignment regression: Python currently rejects a
  point-cloud tensor RHS before reaching this route.

  **Complete when:** Overlapping RHS values are preserved across the shared
  materialization transition and subsequent writes, including reversed and
  shifted views, already-materialized indexed backing, and combinations with
  ordinary backing where aliasing is possible. Nonoverlapping assignment
  retains value independence and ordinary view behavior.

  Source: [tensor.tpp](../include/sbear/tensor.tpp), `assign_from()` overlap
  guard at lines 304–305 and indexed writes immediately below it.
  Verification: test plan B5, E3, E5. Existing ordinary-overlap tests and
  constant-value indexed assignment tests do not exercise this case.

- [ ] **E4. [P1] Preserve coordinate-view identity when sharing serialized or cast sources.**

  **Finding:** Indexed serialization and `pcloud_cast()` deduplicate coordinate
  tensors by allocation owner alone. Different slices of one allocation have
  the same owner but can have different offsets, shapes, strides, and values.
  The second cloud is incorrectly replaced with the first cloud's coordinates.

  Reproduced through valid C++ construction: create one `(2, 2, 1)` numeric
  tensor containing clouds `[[1], [2]]` and `[[10], [20]]`, then construct two
  `PointCloud<double>` cells from its axis-0 views. Select row 0 from both.
  The indexed input returns `1, 10`; its binary round trip returns `1, 1`.
  Casting the ordinary source with `pcloud_cast<float>()` likewise returns
  `1, 1` instead of `1, 10`. Python's usual value constructors copy coordinates,
  so this reproduction targets the supported C++ view-construction boundary.

  **Complete when:** Shared-source reuse distinguishes the complete logical
  coordinate view or explicitly records/reconstructs its mapping. Distinct
  slices, transposes, and strided views survive IO and precision casts with
  correct shapes and values, while identical views still share when intended.

  Sources: [tensor_io.hpp](../include/sbear/io/tensor_io.hpp), indexed source
  table at lines 422–448; [point_cloud.hpp](../include/sbear/point_cloud.hpp),
  `pcloud_cast()` source map at lines 327–361.
  Verification: test plan E5, G1, J1; require C++ storage/view fixtures rather
  than Python constructors that remove the alias before the operation.
