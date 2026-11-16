# Autodiff prototype: implementation plan

Progress: 11 of 15 items, 25 of 34 points (74%)

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

## Phase C: One source of truth for PCFs and PLFs

- [ ] **C1.** [3] `_TrackedPcfTensor`: framework times and values plus the
  enumeration as its only state; `_data` derived; no in-place changes;
  indexing, reshaping, and copying keep gradients; arithmetic, norms, and
  stable ranks return it.
  Tests: indexing matches untracked tensors; gradient through indexing;
  `__setitem__` and in-place arithmetic raise; copying into a `PcfTensor`
  raises.
- [ ] **C2.** [3] `_TrackedPcf`: the same for a single PCF, built by
  `Pcf(values, times=times)` and by arithmetic on PCFs.
  Tests: arithmetic results are tracked; in-place arithmetic raises; the
  existing autodiff tests pass unchanged.
- [ ] **C3.** [2] `_TrackedPlf`: the same for `Plf(values, times=times)`.
  Tests: values derived from the framework arrays; `transform_bars`
  gradients unchanged.
- [ ] **C4.** [1] Mark D5 as done for PCFs and PLFs in the spec.
