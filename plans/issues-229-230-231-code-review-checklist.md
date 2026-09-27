# Code review checklist: PR #233

Initial review on 2026-09-27 covered the live
[PR #233](https://github.com/kthtda/stablebear/pull/233), its five linked issues,
the plans, and the branch diff against refreshed `origin/main`.
Progress updated through local commit `5e612d55f` (T2 sampling tests); remote state has not been refreshed since the initial review.

## Review baseline and conclusion

- Initial review: `issue-229-uniform-subsampling` at
  `27106ca6a674347c7defa11a4e467ee5b103dd70`, matching the remote PR head at that time.
- `main`, refreshed `origin/main`, and merge base:
  `4a9c284f4f48d6711083d99ee529819d4c546384`.
- Net change: 201 files. The working tree was clean before the initial plan update.
- The PR has no conversation comments, submitted reviews, or inline review
  comments at the initial review. Its body marks #230/#232 complete and #229/#231/#236
  incomplete. All five issues remain open.

**The implementation is substantially complete. E3's overlapping indexed
assignment fix and both regressions are committed in `5c9ae885f`.**
T2 sampling semantics are verified and committed in `5e612d55f`. The remaining testing
work is T4 (T3 is complete using existing randomness coverage), followed by documentation cleanup and final regression validation.
Use the bounded
[test plan](issues-229-230-231-test-plan.md);
the previous exhaustive A–K acceptance matrix is superseded.

| Issue | Current branch evidence | Remaining acceptance work |
| --- | --- | --- |
| [#229 uniform subsampling](https://github.com/kthtda/stablebear/issues/229) | Sampling, indexed storage, shared materialization, independent resampling, public selection inspection, and const consumers are implemented. Ownership/view regressions are committed. | E3 is complete (`5c9ae885f`); T2 is committed (`5e612d55f`); T3 is complete; finish T4 and reconcile documented API choices. |
| [#230 binary-backed pickle](https://github.com/kthtda/stablebear/issues/230) | Shared binary reducer, standalone cloud IO, retained legacy reconstruction functions, public-type round trips, and fixed compatibility files are committed. | Documentation and final validation. No new reducer-discovery framework or all-type golden corpus is required. |
| [#231 nesting and ragged indexing](https://github.com/kthtda/stablebear/issues/231) | Runtime recursion, depth validation, copying/views, exact-prefix selection, scalar/empty metadata, and binary/pickle coverage are committed. | E3 is complete (`5c9ae885f`); documentation reconciliation and final validation remain. No new depth-by-operation test matrix. |
| [#232 barcode tensor isomorphism](https://github.com/kthtda/stablebear/issues/232) | Aligned same-dtype comparison, tolerance forwarding, and error cases are implemented and tested at both precisions. | Add the missing `.rst` description/example and include existing tests in final validation. |
| [#236 distance-matrix subsampling](https://github.com/kthtda/stablebear/issues/236) | Implemented in `991d8e5ce`; committed tests cover principal submatrices, repeated vertices, resampling, views/mutation, IO, and persistence/kernel consumers. | E3 is complete (`5c9ae885f`); T2, including matrix-specific duplicate semantics, is committed (`5e612d55f`); T3 is complete; T4 remains. Acceptance still depends on completing #229. |

## Corrections to the previous plan

- The FixedRankTensor test and API adaptations are committed (`1f88c02eb`,
  `d5165c928`, `b81b8f310`). The previous missing/untracked-test build blocker
  no longer applies. This is source evidence, not a fresh clean-build result.
- Persistence tests now inspect actual indexed state with `has_indices()`;
  the warning about checking only the backend wrapper's class is obsolete.
- `e971d2b49` and `52fe35043` cover public copied selections and indexed
  view/mutation behavior. `0f6d616e2` separates resampling value, storage,
  generator, and mutation checks into readable tests.
- The serialization corpus contains 93 artifacts: 63 original 0.4.7 files,
  16 additional 0.4.7 matrix files (`b81b8f310`), two V1 binaries, and 12 pinned
  pre-release V3 standalone-cloud files (`946df8486`, directory `0.5-pre`).
  The V3 corpus is not a released 0.5 corpus or a complete nested/indexed corpus.
- At the initial review, `test_subsample.py` lacked duplicate-filtering and
  explicit without-replacement uniqueness assertions. T2 now covers these in
  `5e612d55f`. T3 count rejection is committed in `c0d3af44a`;
  failure/no-advance RNG coverage is committed in `efd53e46f`.

## Completion status

### E3. Fix overlapping indexed assignment — completed

- [x] Preserve RHS logical values before writing into overlapping indexed
  backing, with Python and C++ regressions. Committed in `5c9ae885f`.

Before the fix, `Tensor::assign_from()` excluded indexed operands from its
alias snapshot. Reversing four indexed distance matrices corrupted the result:
their off-diagonal distances became `[4, 3, 3, 4]` instead of `[4, 3, 2, 1]`.

**Resolution:** compare active backing storage for ordinary and indexed
operands and snapshot the logical RHS before writing when they alias. Indexed
destinations use `writable_at()` to preserve shared materialization. The
snapshot retains the tensor's other property bits.

**Committed regressions:**

- `test_overlapping_indexed_matrix_assignment_preserves_rhs_values` in
  `test/python/test_indexed_tensor_interface.py` uses four explicit 2×2
  matrices and selections, compares complete expected matrices, and checks
  source independence. Python materializes the destination before C++ assignment.
- `TensorProperties.OverlappingAssignmentSnapshotsBeforeMaterialization` in
  `test/test_tensor.cpp` uses real two-point distance matrices and calls
  `assign_from()` with selections still active. It checks reversal,
  materialization, and unchanged source matrices.

**Verification:** 49 focused Python tests on `_sb_cpu` and 171 C++ tensor/property
tests passed during implementation with GCC 13. The revised Python regression
and final simplified C++ regression also passed individually; the latter was
rebuilt directly with CMake. `make html` succeeded during implementation.
These are focused development results, not a full-suite run of the committed
revision. See test-plan T1 for commands and scope.

This restores expected assignment behavior; no extra user-documentation
paragraph or additional overlap test matrix is needed. Full PR and CUDA
validation remain V1 work. All subsequent builds use direct CMake as required
by the corrected `AGENTS.md`.

### S1. Finish meaningful sampling coverage — T2 committed (`5e612d55f`)

- [x] T2: no-replacement full samples, nonempty partial sampling, stable cloud
  and matrix duplicate filtering, preservation of distinct zero-distance
  vertices, and unchanged generator advancement when filtering is enabled.
- [x] T3: count validation (`c0d3af44a`) and failure without RNG advancement
  (`efd53e46f`) and row-major stream allocation (`337f840e2`) are committed
  (**114 sampling tests passed**). Scheduling independence is accepted using
  existing tensor/randomness tests and index-based seed allocation review; no
  separate worker-count sampling test is required.
- [ ] T4: mutating the original input after sampling leaves its snapshot intact.

**T2 verification:** direct CMake build/install with GCC 13, followed by
`SB_FORCE_CPU=1 python -m pytest python/test_subsample.py -q` from `test/`:
**106 passed** on the installed `_sb_cpu` module. The explicit C++ filter-order
tests also passed (**3 tests**: float/double coordinates and indices). These
results cover the tests committed in `5e612d55f`; no sampler implementation
change was needed. The
[test plan](issues-229-230-231-test-plan.md) records the bounded scope and build
commands. Existing ownership, view, and consumer coverage remains sufficient;
no additional sampling matrix or statistical certification is required.

### F1. Accept implemented distance-matrix integration

- [x] Implement generic sampling/indexed matrix support (`991d8e5ce`) and
  commit its main integration tests (`d5165c928`, `b81b8f310`, and subsequent
  indexed-view/resampling test commits).
- [ ] E3 is complete; close acceptance after shared sampling tests and final validation.

For selections `S`, results preserve the compressed logical principal submatrix
`M[S, S]`, including repeated vertices. Const persistence/kernel consumers and
binary/pickle round trips have committed coverage. Duplicate removal is by
source index; distinct vertices at zero distance must survive. Test this
family-specific difference without duplicating the whole point-cloud test plan.

### D4. Reconcile public documentation and issue descriptions

- [ ] Update `CHANGELOG.md`: replace the removed public `IndexTensor` with
  `NestedTensor` and describe the final distance-matrix, pickle, and barcode
  features. Remove obsolete intermediate storage-subtype descriptions.
- [ ] Correct `docs/indexing.rst` to say the result owns copied selections;
  “one view of selections” currently suggests caller mutation can change the
  result. Distinguish direct indexing's source view from subsampling's snapshot.
- [ ] Update `docs/saving.rst` to name nested and distance-matrix tensors,
  explain standalone data-object save/load and shared binary-backed pickling,
  and state retained legacy-reader support. Keep dtype/enum metadata and
  non-pickleable execution objects outside the data-object promise.
- [ ] Add a short `.rst` example of `BarcodeTensor.is_isomorphic_to`; its
  docstring and tests already cover the behavior.
- [ ] Reconcile PR/issue descriptions with the settled choices below and
  replace old test counts with final evidence. GitHub descriptions and issue
  state have not been changed by this work.

Settled choices that should remain explicit:

1. Full point-cloud element access returns the write-aware `PointCloud` façade,
   rather than the `FloatTensor` described in #229. Reads retain indexed state;
   successful writes transition shared backing. This is documented and tested
   behavior, not a reason to reinstate eager materialization.
2. Point-cloud layout selections are transient logical access; persistent
   selections and materialization belong to the tensor's shared state.
3. The old nested `(5, 64)` encoding was produced only by intermediate commits
   on this branch, never main or a release. Keep the recorded C2 compatibility
   decision and reconcile #231's stale requirement; do not manufacture fixtures
   or promise a decoder for that intermediate format.
4. Public `indices` returns an independent selection copy, and returns `None`
   after shared materialization. Existing tests cover this contract.
5. Tensor-valued `PointCloudTensor` RHS assignment remains a pre-existing
   limitation. Do not expand this PR into unrelated assignment API work; E3 is
   the separate, demonstrated distance-matrix assignment corruption.

### V1. Validate once and record the result

- [ ] After fixes/docs settle, follow test-plan V1: build/install, run existing
  Python and C++ suites, build Sphinx, and review CI for that revision.
- [ ] Record the tested commit, actual extension modules, results/skips, and
  limitations. Accept #229/#231/#236 after their shared open items resolve;
  preserve #230/#232's completed implementation status.

Existing CI covers platform wheels, source distribution, coverage and memory
checks. Use it; a separate local platform/CUDA-version/serialization-exchange
matrix is not required. State unavailable CUDA coverage accurately.

**E3 is complete. Stop when S1, D4, and V1 are complete.** F1 then has the same evidence it
needs. Reopen work for a concrete failure, an uncovered issue requirement, or a
new implementation change, not merely another possible test permutation.

## Completed review findings retained for traceability

The old review IDs remain useful history. These items are complete; the smaller
test plan does not reopen them. Detailed reproductions and historical run logs
remain in Git history before this update.

| ID | Resolution | Implementation commit(s) |
| --- | --- | --- |
| A1 | Writes materialize shared indexed backing so existing aliases see them. | `33acda7fc` |
| A2 | Nested outer views retain shared storage. | `ef3d1620f` |
| A3 | Indexed results own selections independently of callers. | `9b31341a4` |
| B1 | Whole indexed tensors work with persistence/kernel consumers. | `2edada84e` |
| B2 | Common tensor bindings support indexed operations. | `9c0ea99c1` |
| B3 | Public selection validates exact prefixes, rank/dtype, and bounds. | `b0c9791a0` |
| C1 | Binary IO/pickle preserve indexed tensor representation. | `0e437023b` |
| C2 | No compatibility decoder required for unreleased `(5, 64)`. | Historical ancestry audit of `0c2cb6efc` |
| C3 | Standalone/owner-backed clouds serialize logical values via binary IO. | `654ba0b9c` |
| C4 | Data pickles share binary reconstruction; historical symbols remain readable. Recognized corrupt binary payloads do not enter legacy fallback. | `654ba0b9c` |
| D1 | Shared tensor state owns persistent indexing/materialization. | `5cf5411b0` |
| D2 | Value construction/copy and view construction have explicit ownership. | `44924b7e0` |
| D3 | Indexed resampling copies logical input directly once. | `8134f40c4` |
| E1 | Scalar tensors and nested scalar leaves retain serialized values. | `b70dbec19` |
| E2 | Depth/dtype checks guard nested bulk assignment and empty joins. | `339252aae` |
| E3 | Overlapping indexed assignment snapshots logical RHS values before writes/materialization. | `5c9ae885f` |
| E4 | Coordinate construction copies external views, removing ambiguous source identity. | `1570fdafd` |

## Verification evidence and its limits

The initial plan review inspected source, committed tests, live issue/PR text,
and CI status without rebuilding. E3's subsequent build and focused verification
are recorded above; full suites have not been rerun. At the initial inspection,
platform build/test and analysis jobs for `27106ca6a` were still pending; CUDA build/test/coverage was skipped. No final CI pass is claimed.

The previous plan recorded these **historical** local results:

| Revision/worktree reviewed | Python, CUDA 12 module | Python, separate CPU module | C++ | Docs |
| --- | --- | --- | --- | --- |
| `cced5d9b7`, committed source (2026-09-22) | 2,358 passed | 2,328 passed, 30 skipped | 486 passed | HTML succeeded |
| `991d8e5ce` plus then-pending tests (2026-09-26) | 2,427 passed | 2,397 passed, 30 skipped | 491 passed | HTML succeeded |

These are prior evidence, not results for `27106ca6a`. Sampling and serialization
execute on CPU even when a CUDA extension is loaded. A final run must identify
the module actually loaded; `force_cpu()` within a CUDA module is not a separate
`_sb_cpu` test.
