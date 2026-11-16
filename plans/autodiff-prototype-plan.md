# Autodiff prototype: implementation plan

Progress: 13 of 26 items, 28 of 62 points (45%)

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
  Note: the backward pass on 10^6 bars takes 0.42 s, checked by hand; the
  suite has no test of that size.
- [x] **C2.** [2] (7afe77bac) Parallel C++ conversions to and from
  serial content for PCF and barcode tensors, without temporaries per
  element.
  Tests: the round-trip tests pass; note the timings under this item.
  Note: for 10^6 rows in 2000 elements, to serial content takes 1 to 2 ms
  and from it 6 to 7 ms, down from 60 to 70 ms each. The tracked pipeline of
  D13 takes 0.22 s, down from 0.58 s; 0.16 s of it is the single-threaded
  sort of the stable rank.
- [ ] **C3.** [2] The kernel's backward pass reuses the critical edges of
  its forward pass.
  Tests: the kernel gradient tests pass; the kernel runs once per
  forward and backward.
- [ ] **C4.** [3] PCF arithmetic with a tracked operand without a Python
  loop over elements.
  Tests: the arithmetic gradient tests pass; note the timings.

## Phase D: One generic tracked tensor

- [ ] **D1.** [3] The tracked mixin and type descriptions (D5a), with
  `tracked_class(BarcodeTensor)` replacing `_TrackedBarcodeTensor`; no
  change in behavior.
  Tests: the tracked barcode tests pass unchanged.
- [ ] **D2.** [2] Cache the derived C++ tensor (D5); operations hand their
  C++ result to the tracked object.
  Tests: a tracked operation converts once; an in-place PyTorch change
  of the framework array is seen by the next read.
- [ ] **D3.** [3] `PcfTensor` on the mixin, with times and values as
  separate column groups; arithmetic, norms, and stable ranks return it.
  Tests: indexing matches untracked tensors; gradient through indexing;
  copying into a `PcfTensor` raises.
- [ ] **D4.** [3] `Pcf`, `Plf`, and `Barcode` as the 0-d case.
  Tests: `Pcf(values, times=times)` and `Plf(values, times=times)` derive
  their C++ objects; the existing autodiff tests pass unchanged.
- [ ] **D5.** [3] Deferred writes (D5b) and building a tracked tensor from
  a list of elements in one pass.
  Tests: a write is seen by reads and by the next operation; gradients
  flow to a written element and not to the one it replaced; n writes
  make one pass.
- [ ] **D6.** [2] `differentiable` wraps a declared output type: the
  operation returns a C++ tensor and the decorator returns its tracked
  form.
  Tests: the kernel and `transform_bars` use it and pass their tests.
- [ ] **D7.** [2] Type descriptions for `PointCloudTensor`,
  `DistanceMatrixTensor`, and `SymmetricMatrixTensor`.
  Tests: serial round trip; indexing a tracked tensor of each type.

## Phase E: Operation coverage

- [ ] **E1.** [3] Betti curves of tracked barcodes.
  Tests: matches the untracked curve; norm gradient against finite
  differences; a batch.
- [ ] **E2.** [2] Accumulated persistence functions of tracked barcodes.
  Tests: matches the untracked function; norm gradient against finite
  differences.
- [ ] **E3.** [3] Persistent homology of tracked point clouds, through the
  critical simplices of each bar.
  Tests: bars match the untracked ones; gradient against finite
  differences in dimensions 0 and 1.
- [ ] **E4.** [3] `pdist` and the L2 kernel of tracked PCFs.
  Tests: match the untracked matrices; gradient with respect to values
  and times against finite differences.
