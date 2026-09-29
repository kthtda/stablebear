# Relative subsampling: test plan for #243

Status: proposed for joint review. The initial new tests have been removed.
G4 is reviewed and complete; other items are unstarted.

Read [TESTING.md](../test/TESTING.md) before writing any tests. Work through the items
below together, using their identifiers in discussion. Each item is a review
unit, not a requirement to create one large test or a matrix of test cases.
Before implementing an item, agree on the smallest useful example and check
whether existing tests already cover it. Keep new work uncommitted for review.

Use small, explicit inputs and direct expected values. Prefer public Python
behavior; propose a C++ test only when the relevant guarantee cannot be checked
clearly through the public API. Keep each eventual test about one behavior.
The review question is: **What bug would make this test fail, and can I see
that immediately?**

## A. Queries and output axes

- [ ] **A1 — Coordinate queries.** Use a few distinguishable reference points
  and two external queries whose eligible regions select different points.
  Check the selected coordinates directly.
- [ ] **A2 — Reference-index queries.** Show that query order and repetitions
  are respected. Review the minimum cases needed for `query=None`, signed
  negative indices, and native unsigned tensors, including one sliced tensor.
  Use uint32/uint64 only where their conversion paths need coverage.
- [ ] **A3 — Axis conventions.** Distinguish a single distribution from a list,
  including a one-element list. Check query/distribution/sample axis order
  using distributions with visibly different eligible regions. Include the
  expected shape for an empty query collection.

## B. Distribution weights

- [ ] **B1 — Uniform intervals.** Put points exactly at `start`, inside the
  interval, and exactly at `end`; only the first two are eligible.
- [ ] **B2 — Gaussian weights.** Agree on a clear check of the documented
  Gaussian weight ratios. Separately review the numerical cases worth keeping:
  a far-away query must not become an empty region, and tiny positive weights
  must remain eligible without replacement.
- [ ] **B3 — Mixtures.** Use components with simple, different supports and
  known coefficients. Check addition of native amplitudes before normalization.
  Review a small nested example, zero coefficients, and common scaling; prefer
  direct weight expectations over identical seeded draws. A mixture in a list
  occupies one distribution entry.

## C. Drawing and support policies

- [ ] **C1 — Weighted selection.** Choose the smallest fixed-seed probability
  check that distinguishes weighted selection from uniform selection. Give the
  expected probability and justify the tolerance. Review coverage of both
  replacement modes, including the successive-draw law without replacement.
  Use an obvious one-eligible-point example to show replacement permits repeats.
- [ ] **C2 — Insufficient and empty support.** Use an interval with fewer
  eligible points than requested. Strict mode raises and identifies the failing
  query/distribution pair; partial mode returns the eligible points. Review zero
  support separately: strict mode raises, partial mode gives empty samples.
- [ ] **C3 — Duplicate removal.** Check that the relative sampler applies the
  existing post-draw rule without redraws. Coordinate duplicates can shorten a
  cloud sample; matrix samples remove repeated indices, not distinct vertices
  at distance zero. Reuse existing duplicate-helper coverage.

## D. Reference representations

- [ ] **D1 — Distance matrices.** Use a small matrix with distinguishable
  distances. Weights must use the chosen query row, and each output must equal
  the principal submatrix selected by its returned indices, in drawn order.
  Include repeated indices in a replacement example.
- [ ] **D2 — Logical input values and precision.** Review one sliced or indexed
  reference with an obvious reordered selection. Check against explicit values,
  not a copy made by the implementation. Exercise float32/float64 on one basic
  cloud and matrix case rather than parametrizing every test.

## E. Validation at the new API boundary

- [ ] **E1 — Distribution validation.** Review a short table of invalid
  Gaussian/Uniform parameters and mixture inputs: empty or mismatched lists,
  negative/nonfinite coefficients, and an all-zero mixture. Include an empty
  distribution list and an unsupported distribution object at the sampler API.
- [ ] **E2 — Query and reference validation.** Review representative wrong-rank
  and out-of-range index cases, especially a large unsigned index that must not
  wrap. Include coordinate queries for a distance matrix and unsupported batches.
- [ ] **E3 — Numeric data validation.** Check rejection of nonfinite coordinates
  or distances at the appropriate boundary. Check negative-distance rejection
  where such data can enter; reuse matrix-constructor coverage when it already
  prevents the invalid input. Avoid repeating shared argument-validation tests.

## F. Randomness and reporting

- [ ] **F1 — Generator integration.** Check that reseeding replays a sequence
  of calls and consecutive calls advance the generator. Distinct output cells,
  including repeated distribution entries, must receive distinct streams.
  Review worker-count replay as a manual check before adding executor machinery.
- [ ] **F2 — Verbosity.** Reporting must preserve seeded results and generator
  advancement. Review one empty-region diagnostic that identifies its
  query/distribution pair, and the corresponding quiet behavior. Do not assert
  progress-bar formatting or timing.

## G. Composition and existing coverage

- [ ] **G1 — Persistence pipeline.** One tiny example should run through
  subsampling → persistent homology → stable rank → mean over the sample axis.
  Use a known expected rank and retain query/distribution axes. Review the
  minimum integration coverage needed for clouds and distance matrices.
- [ ] **G2 — Indexed storage and serialization.** First review existing
  `test_indexed_tensor_interface.py` and `test_indexed_tensor_io.py` coverage.
  Add only a missing relative-output integration check, if needed. Keep
  mutation, copy-on-write, and serialization tests in those owning files;
  avoid duplicating their storage tests in the sampling suite.
- [ ] **G3 — Final verification.** Run the agreed focused tests and relevant
  existing uniform-sampling regressions on the default and forced-CPU backends.
  Build/install with CMake using **`-j10`**, and run pytest from `test/`.
  Build the documentation and execute its example. Record the checks actually
  run; the removed suite's earlier results are not replacement coverage.
- [x] **G4 — Walk on indexed tensors.** Added focused tensor-iteration coverage
  for a small indexed tensor whose logical outer shape differs from its source
  shape, checking each logical index exactly once in sequential order and the
  explicit selected values. A sliced outer view checks traversal of the view
  rather than its backing source. Reviewed and committed in `e9b1c8b09`.

  Reviewed tests in `test/test_tensor_iteration.cpp`:

  - `IndexedTensorVisitsLogicalIndicesAndSelectedValues`: one source cloud,
    a 2×2 indexed outer shape, and explicit row-major indices and selected values.
  - `IndexedOuterSliceVisitsOnlyItsLogicalElements`: an outer `[1::2]` slice,
    with explicit view-relative indices and selected values.

  Overload audit: sequential walks share `detail::walk_impl`; parallel walks
  share `detail::parallel_walk_impl`. Both use the tensor's logical shape, with
  no indexed-specific branch. The generator overloads also use `tensor.size()`
  to reserve streams. Existing parallel/random tests exercise ordinary tensors,
  with direct indexed coverage of those overloads deferred beyond this item.

  Validation: built `sb_test` with `-j10` and installed via CMake. From `test/`,
  `../cmake-build-debug/sb_test --gtest_filter='TensorWalk.*:WalkRandom.*'`
  passed all 16 tests, including the two new cases.

## How we will use this plan

G4 is complete; return to A1 next. Agree on each item's input and
expected result before expanding its tests.
Review each small addition before moving on. Mark an item complete only once
we have reviewed the coverage, including a decision that existing tests suffice.
Expand a topic only when a concrete uncovered behavior justifies it.
