# Autodiff prototype: implementation plan

Progress: 28 of 30 items, 66 of 72 points (92%)

Spec: [autodiff-prototype-spec.md](autodiff-prototype-spec.md)

## Phase A: Gradients through norms, the kernel, and stable ranks

- [x] **A1.** [3] (ca4dd9a3b) `_autodiff.differentiable` with framework
  adapters; `lp_norm` gradients with respect to `p` and PCF values; PCF
  arithmetic in framework operations.
- [x] **A2.** [1] (950a5dd48) Recognize Keras 3 variables.
- [x] **A3.** [2] (6f52154da) `Pcf(values, times=times)` from framework
  arrays.
- [x] **A4.** [1] (4d4439161) Shared `fw` fixture: each autodiff test runs
  once per installed framework.
- [x] **A5.** [3] (fe530e3b4) Homological kernel gradients with respect to
  point coordinates, through its critical edges.
- [x] **A6.** [3] (996ada010) `lp_norm` and arithmetic gradients with
  respect to PCF breakpoint times; `_tracked_parts`.
- [x] **A7.** [2] (67f21b327) Stable rank of tracked barcodes in framework
  operations.

## Phase B: Transforming bars by a PLF

- [x] **B1.** [3] (48db3b4cd) `transform_bars(bc, f)`: C++ task, removal of
  bars with f(b) >= f(d), gradients with respect to the bars; raises under
  JAX and `tf.function`.
- [x] **B2.** [2] (0411be474) `Plf(values, times=times)` and gradients of
  `transform_bars` with respect to the PLF.
- [x] **B3.** [3] (afa4adba9) Batches of tracked barcodes: C++ serial
  conversion for barcode tensors, `_TrackedBarcodeTensor` as the single
  source of truth, batched `transform_bars` and stable rank.
- [x] **B4.** [2] (386e324b2) This plan and its spec.

## Phase C: The cost rule for what exists

- [x] **C1.** [1] (0e6eefcb6) PLF vjp of `transform_bars` without a
  dense Jacobian: scatter the gradients of the breakpoint times and values.
  Tests: the existing gradient tests pass; backward on 10^6 bars runs.
  Note: the backward pass on 10^6 bars takes 0.42 s in a Debug build,
  checked by hand; the suite has no test of that size.
- [x] **C2.** [2] (7afe77bac) Parallel C++ conversions to and from
  serial content for PCF and barcode tensors, without temporaries per
  element.
  Tests: the round-trip tests pass; note the timings under this item.
  Note: for 10^6 rows in 2000 elements, to serial content takes 1 to 2 ms
  and from it 6 to 7 ms, down from 60 to 70 ms each. The tracked pipeline of
  D13 takes 0.22 s, down from 0.58 s; 0.16 s of it is the single-threaded
  sort of the stable rank. These are timings of a Debug build; in a Release
  build each conversion takes under 1 ms.
- [x] **C3.** [2] (a2e8a6f0d) The kernel's backward pass reuses the
  critical edges of its forward pass.
  Tests: the kernel gradient tests pass; the kernel runs once per
  forward and backward.
- [x] **C4.** [3] (0e517a1a9) PCF arithmetic with a tracked operand
  without a Python loop over elements: the union of the operands' times comes from
  `iterate_rectangles_with_rows` in C++, which reports the breakpoints of
  each rectangle, in parallel over the elements.
  Tests: C++ tests of `iterate_rectangles_with_rows` (rows of each
  rectangle; one function ending first; same rectangles as
  `iterate_rectangles`); arithmetic on a stable rank with equal
  lifetimes; the arithmetic gradient tests pass; note the timings.
  Note: in a Release build, `x * F - G` on 10^5 PCFs of 20 points takes
  116 ms tracked, against 21 ms for `F - G` untracked; the merge is no
  longer the main cost. `pdist` on 1500 PCFs of 50 points is unchanged: 29
  ms for p=1 and 95 ms for p=2.
- [x] **C5.** [2] (cf5ad84f1) The sorting order of the tracked stable
  rank computed in C++, per barcode in parallel, in place of one `lexsort` over all bars.
  Tests: the stable rank tests pass; equal lifetimes and an empty barcode
  keep their order; note the timings.
  Note: in a Release build on 10^6 bars in 2000 barcodes, the tracked
  pipeline of D13 runs forward in 50 ms, down from 190 ms, against 14 ms
  untracked; forward and backward take 233 ms, down from 370 ms.

## Phase D: One generic tracked tensor

- [x] **D1.** [3] (06f36d271) The tracked mixin and type descriptions (D5a), with
  `tracked_class(BarcodeTensor)` replacing `_TrackedBarcodeTensor`; no
  change in behavior.
  Tests: the tracked barcode tests pass unchanged.
- [x] **D2.** [2] (706c2fd3a) Cache the derived C++ tensor (D5); operations hand their
  C++ result to the tracked object.
  Tests: a tracked operation converts once; an in-place PyTorch change
  of the framework array is seen by the next read.
  Note: only `copy` hands over a C++ tensor so far; D3 and D6 are the first
  operations that have one.
- [x] **D3.** [3] (065d39a3f) `PcfTensor` on the mixin, with times and values as
  separate column groups; arithmetic, norms, and stable ranks return it.
  Tests: indexing matches untracked tensors; gradient through indexing;
  copying into a `PcfTensor` raises; the tests of PCF operations (norms,
  distances, arithmetic) run on tracked PCFs too, as in D9.
  Note: the tests of PCF arithmetic are not parametrized: they run over
  integer PCFs and in-place operations, which cannot be tracked;
  `test_autodiff.py` checks tracked arithmetic. `pdist` of tracked PCFs
  raises until E4. A tracked element is built from serial content, since
  `Pcf(rows)` rejects the repeated times of a stable rank.
- [x] **D4.** [3] (1a25847da) `Pcf`, `Plf`, and `Barcode` as the 0-d case.
  Tests: `Pcf(values, times=times)` and `Plf(values, times=times)` derive
  their C++ objects; the existing autodiff tests pass unchanged.
  Note: `_data` of a tracked `Pcf` or `Plf` gives the derived C++ object
  and does not raise as for a tensor (D8): their methods read it
  throughout. There is no tracked `Barcode`, which no operation gives.
  The vjp's rebuild must not read the framework arrays: TensorFlow forbids
  reading a variable inside its operations.
- [x] **D5.** [3] (1c9cf6528) Deferred writes (D5b) and building a tracked tensor from
  a list of elements in one pass.
  Tests: a write is seen by reads and by the next operation; gradients
  flow to a written element and not to the one it replaced; n writes
  make one pass.
  Note: a written value's dtype is not checked against the tensor's: the
  tensor's dtype follows its arrays after the pass. Only `PcfTensor` builds
  from a list, since there is no tracked `Barcode` (D4).
- [x] **D6.** [2] (173fe4f36) `differentiable` wraps a declared output type: the
  operation returns a C++ tensor and the decorator returns its tracked
  form.
  Tests: the kernel and `transform_bars` use it and pass their tests.
  Note: the tracked result keeps the C++ tensor only when the framework has
  run the operation by the time the decorator returns, as PyTorch and
  eager JAX and TensorFlow do. The kernel gets its bars as an array, so it
  builds a C++ tensor to return: one conversion, in place of one in the
  next operation.
- [x] **D7.** Type descriptions for `PointCloudTensor`,
  `DistanceMatrixTensor`, and `SymmetricMatrixTensor`.
  - [x] **D7a.** [2] (bd364ef48) `PointCloudTensor`: C++ conversions to and from
    serial content, with the points as rows of as many columns as the
    clouds have dimensions; a group of columns may leave its width open.
    Tests: serial round trip; clouds of different dimensions raise;
    indexing a tracked tensor.
  - [x] **D7b.** [2] (054196cfc) `DistanceMatrixTensor` and `SymmetricMatrixTensor`: C++
    conversions with the entries as rows of one column, and the size of
    each matrix kept beside the enumeration (D5a).
    Tests: serial round trip, with matrices of no points and of one;
    sizes that do not match raise; a negative distance raises; indexing a
    tracked tensor of each type.
    Note: writes to a tracked tensor of matrices raise: a write would
    need the size of the value written.
- [x] **D8.** [3] (b70327319) Reading `_data` of a tracked tensor raises (D5), so an
  operation without gradients cannot drop them; `detach()` gives the
  untracked tensor; comparisons, saving, and pickling use the values. Done
  before D3, as is D9: they guard the tensors that D3 to D5 rework.
  Tests: a Betti curve, a `TimeSeries`, and a `NestedTensor` of tracked
  barcodes raise; `detach()` equals the untracked tensor; `==` and
  `array_equal` compare values; save and pickle give the untracked tensor.
- [x] **D9.** [3] (cb57cd98f) The tests of barcode operations run on tracked barcodes of
  each framework as well as on untracked ones: the stable rank and
  `transform_bars` give the same values, and the operations without
  gradients raise. D3 adds tracked PCFs to the tests of PCF operations.
  Tests: the framework helpers of `test_autodiff.py` move to a shared
  module; a parameter of the barcode tests makes their input tracked.
  Note: the tests found that a tracked tensor without bars could not be
  read. A tracked stable rank has one breakpoint per bar, so the tests of
  equal and infinite lifetimes compare values. The tests of errors are not
  parametrized: JAX wraps an error of the operation in its own.

## Phase E: Operation coverage

- [x] **E1.** [3] (0720fbe24) Betti curves of tracked barcodes.
  Tests: matches the untracked curve; norm gradient against finite
  differences; a batch.
  Note: one commit holds E1 and E2, which share their helpers. A tracked
  curve has one breakpoint per birth and death, so equal times repeat.
- [x] **E2.** [2] (0720fbe24) Accumulated persistence functions of tracked barcodes.
  Tests: matches the untracked function; norm gradient against finite
  differences.
- [ ] **E3.** [3] Persistent homology of tracked point clouds, through the
  critical simplices of each bar.
  Tests: bars match the untracked ones; gradient against finite
  differences in dimensions 0 and 1.
- [ ] **E4.** [3] `pdist` and the L2 kernel of tracked PCFs, with the
  distances and their gradients accumulated by rectangle iteration in C++,
  without building the difference of each pair.
  Tests: match the untracked matrices; gradient with respect to values
  and times against finite differences.
