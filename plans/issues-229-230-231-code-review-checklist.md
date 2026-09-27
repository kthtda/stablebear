# Code review checklist: PR #233

Updated 2026-09-27 after reviewing the live
[PR #233](https://github.com/kthtda/stablebear/pull/233), its five linked issues,
the plans, and the branch diff against refreshed `origin/main`.

## Review baseline and conclusion

- Branch: `issue-229-uniform-subsampling`, HEAD
  `27106ca6a674347c7defa11a4e467ee5b103dd70`, matching the remote PR head.
- `main`, refreshed `origin/main`, and merge base:
  `4a9c284f4f48d6711083d99ee529819d4c546384`.
- Net change: 201 files. The working tree was clean before this plan update.
- The PR has no conversation comments, submitted reviews, or inline review
  comments at this review. Its body marks #230/#232 complete and #229/#231/#236
  incomplete. All five issues remain open.

**The implementation is substantially complete. The remaining correctness
blocker is E3: overlapping indexed assignment corrupts values, including through
the public distance-matrix API.** The remaining testing work is a small set of
sampling contract checks, followed by documentation cleanup and final regression
validation. Use the bounded [test plan](issues-229-230-231-test-plan.md);
the previous exhaustive A–K acceptance matrix is superseded.

| Issue | Current branch evidence | Remaining acceptance work |
| --- | --- | --- |
| [#229 uniform subsampling](https://github.com/kthtda/stablebear/issues/229) | Sampling, indexed storage, shared materialization, independent resampling, public selection inspection, and const consumers are implemented. Ownership/view regressions are committed. | Fix E3; finish focused sampling tests T2–T4; reconcile documented API choices. |
| [#230 binary-backed pickle](https://github.com/kthtda/stablebear/issues/230) | Shared binary reducer, standalone cloud IO, retained legacy reconstruction functions, public-type round trips, and fixed compatibility files are committed. | Documentation and final validation. No new reducer-discovery framework or all-type golden corpus is required. |
| [#231 nesting and ragged indexing](https://github.com/kthtda/stablebear/issues/231) | Runtime recursion, depth validation, copying/views, exact-prefix selection, scalar/empty metadata, and binary/pickle coverage are committed. | Shared indexed overlap fix and documentation reconciliation; no new depth-by-operation test matrix. |
| [#232 barcode tensor isomorphism](https://github.com/kthtda/stablebear/issues/232) | Aligned same-dtype comparison, tolerance forwarding, and error cases are implemented and tested at both precisions. | Add the missing `.rst` description/example and include existing tests in final validation. |
| [#236 distance-matrix subsampling](https://github.com/kthtda/stablebear/issues/236) | Implemented in `991d8e5ce`; committed tests cover principal submatrices, repeated vertices, resampling, views/mutation, IO, and persistence/kernel consumers. | E3 and shared sampling tests, including matrix-specific duplicate semantics. Implementation is already present; acceptance still depends on completing #229. |

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
- Conversely, current `test_subsample.py` has no `discard_duplicates` checks,
  argument-rejection tests, failure/no-advance RNG test, or explicit
  without-replacement uniqueness assertion. Earlier descriptions of pending
  tests are not evidence that these guarantees are covered at HEAD.

## Remaining work

### E3. Fix overlapping indexed assignment — correctness blocker

- [ ] Preserve RHS logical values before writing into overlapping indexed
  backing, and add the focused regression in test-plan T1.

`Tensor::assign_from()` in [tensor.tpp](../include/sbear/tensor.tpp) still
restricts its alias snapshot to two non-indexed operands. A write can overwrite
an element needed by a later RHS read. This also affects indexed tensors whose
backing has already been materialized, since the C++ property remains indexed.

The earlier review described only a C++ reproduction because point-cloud tensor
RHS assignment was rejected. The distance-matrix route now exposes this publicly:

```python
import numpy as np
import stablebear as sb

source = sb.DistanceMatrixTensor(np.array([
    [[0., 1.], [1., 0.]],
    [[0., 2.], [2., 0.]],
    [[0., 3.], [3., 0.]],
    [[0., 4.], [4., 0.]],
]))
selected = source[sb.NestedTensor([
    sb.indices([0, 1]) for _ in range(4)
])]
selected[:] = selected[::-1]
# Required off-diagonal values: [4, 3, 2, 1]
# Observed with the installed backend: [4, 3, 3, 4]
```

Current source inspection confirms the excluded overlap path. The Python probe
used the pre-existing installed `_sb_cuda12` module, both before and after an
initial materializing write; it is corroborating evidence, not a fresh HEAD
build. Python makes the destination writable before `assign_from()`; a C++
caller can also trigger materialization within assignment itself.

**Done when:** a pre-write value snapshot protects aliased reads, the public
reversal regression passes, and existing ordinary assignment tests still pass.
Cover materialization within C++ assignment separately only if that transition
needs a different path in the fix. Add shifted or mixed-storage cases only for a
distinct alias path affected by the fix. Do not require every combination of
view, dtype, element family, and property bits.

### S1. Finish meaningful sampling coverage

- [ ] Complete test-plan T2–T4 using small, explicit public-API examples.

The additions protect documented behavior: sampling without replacement,
nonempty partial samples, stable duplicate filtering, validation before RNG
reservation, deterministic streams, and input-to-result snapshot independence.
Reuse existing coverage for shapes, sizes, dtype, resampling, result-to-input
isolation, shared views, and consumers.

This is missing regression coverage, not a finding that the sampler currently
produces incorrect draws. Source review of the uniform draw algorithm
complements these tests; statistical certification is not a merge requirement.

### F1. Accept implemented distance-matrix integration

- [x] Implement generic sampling/indexed matrix support (`991d8e5ce`) and
  commit its main integration tests (`d5165c928`, `b81b8f310`, and subsequent
  indexed-view/resampling test commits).
- [ ] Close acceptance after E3, shared sampling tests, and final validation.

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
  replace old test counts with final evidence. This review edits local plans
  only; it does not change GitHub descriptions or issue state.

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

**Stop when E3, S1, D4, and V1 are complete.** F1 then has the same evidence it
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
| E4 | Coordinate construction copies external views, removing ambiguous source identity. | `1570fdafd` |

## Verification evidence and its limits

This update reviewed source, committed tests, live issue/PR text, and CI status.
It did not rebuild or run full suites. The E3 probe is described separately
above. At inspection, platform build/test and analysis jobs for HEAD were still
pending; CUDA build/test/coverage was skipped. No final CI pass is claimed.

The previous plan recorded these **historical** local results:

| Revision/worktree reviewed | Python, CUDA 12 module | Python, separate CPU module | C++ | Docs |
| --- | --- | --- | --- | --- |
| `cced5d9b7`, committed source (2026-09-22) | 2,358 passed | 2,328 passed, 30 skipped | 486 passed | HTML succeeded |
| `991d8e5ce` plus then-pending tests (2026-09-26) | 2,427 passed | 2,397 passed, 30 skipped | 491 passed | HTML succeeded |

These are prior evidence, not results for `27106ca6a`. Sampling and serialization
execute on CPU even when a CUDA extension is loaded. A final run must identify
the module actually loaded; `force_cpu()` within a CUDA module is not a separate
`_sb_cpu` test.
