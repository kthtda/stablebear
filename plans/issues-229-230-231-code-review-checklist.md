# Code review checklist: issues #229, #230, and #231

This document records the review findings and proposed implementation work.
Creating this checklist does not implement fixes or tests. All items remain open.

Reviewed branch: `issue-229-uniform-subsampling`, commit
`6d3e00e36f7114dcffe54a675262e49b7f44e9e7`, against `origin/main` at
`4a9c284f4f48d6711083d99ee529819d4c546384`.

The review checked the existing CMake build/install and used small in-memory
reproductions with the installed CUDA extension and CPU execution forced.
It did not run the full regression suite, exercise GPU execution, or create
test files. Findings A1–A3, B1–B3, and C1/C3 were reproduced; C2 and the
simplification recommendations also rely on source/history inspection.

P1 denotes a correctness or compatibility blocker for the stated issue scope.
P2 denotes an API/validation gap that should also be addressed before acceptance.
Work through sections A–D in order. Each item describes its own completion
criteria; related fixes may be implemented together. Record the fixing commit
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

- [ ] **B2. [P1] Complete the ordinary tensor interface for indexed results.**

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

- [ ] **B3. [P2] Validate the complete public point-cloud indexing contract.**

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

## C. Serialization and compatibility

- [ ] **C1. [P1] Preserve indexed representation through binary IO and pickle.**

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

- [ ] **C2. [P1] Restore the legacy nested `(5, 64)` decoder.**

  **Finding:** Source/history inspection confirms that earlier branch code
  wrote the one-level nested `uint64` format `(5, 64)`. Current dispatch only
  recognizes the new nested formats. The required compatibility reader is
  missing. This was not validated with a historical fixture during the review.

  **Change:** Retain a decoder for the old payload and normalize it into the
  general `NestedTensor` representation. New writes should continue using the
  current representation.

  **Complete when:** Fixed bytes produced by the actual historical writer load
  with correct dtype, shape, selections, and depth. Include empty and scalar
  cases. Do not manufacture historical fixtures with the current writer.

  Sources: [io.hpp](../include/sbear/io.hpp), `read_any_tensor()`;
  historical writer at commit `0c2cb6efc`. Verification: test plan G2.

- [ ] **C3. [P2] Route standalone `PointCloud` IO and pickle through binary IO.**

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

## D. Simplification and completion review

- [ ] **D1. Consolidate the indexed storage model.**

  **Recommendation:** The branch combines tensor-level adapters, per-element
  selection/local-materialization state, and Python wrapper-level detachment.
  Consolidate around materialized `PointCloud` values, a read-only logical
  accessor, and one shared tensor-level materialization transition. Keep legacy
  decoding at the serialization boundary rather than preserving a second
  runtime ownership model solely to read old files.

  **Complete when:** Ownership and mutation paths are explicit, obsolete local
  indexing/materialization paths are removed or confined to compatibility
  conversion, and the fixes in section A do not require duplicated mechanisms.
  Verify the property-bitmask design and const algorithm access remain intact.

  Sources: [point_cloud.hpp](../include/sbear/point_cloud.hpp),
  [tensor.hpp](../include/sbear/tensor.hpp),
  [point_cloud.py](../stablebear/point_cloud.py).
  Verification: test plan B5, E1–E3, F1.

- [ ] **D2. Make copy and view construction explicit.**

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

- [ ] **D3. Copy logical coordinates only once when resampling indexed input.**

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
  behavior unchanged. Use copy/allocation evidence in addition to value checks.

  Sources: [subsample.py](../stablebear/point_process/subsample.py),
  [subsample.hpp](../include/sbear/point_process/subsample.hpp).
  Verification: test plan D2–D4, E1, E4.

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
