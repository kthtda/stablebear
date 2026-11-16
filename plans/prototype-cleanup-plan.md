# Prototype cleanup: implementation plan

Progress: 1 of 8 items, 3 of 17 points (18%)

Specs: [autodiff-prototype-spec.md](autodiff-prototype-spec.md) and
[linear-subspaces-spec.md](linear-subspaces-spec.md), which each item
updates where it changes a decision.

Gradients are supported for PyTorch only, for now. The APIs released on
`main` keep their behavior; what the two plans above added may change.

## Phase A: PyTorch only

- [x] **A1.** [3] (d6fba7c74) Remove the JAX and TensorFlow adapters, the Keras
  unwrapping, the declared shapes of results (`out=`, and `residuals=` with
  its shapes), and every branch for values that a framework traces
  abstractly. The tests run on PyTorch.
  Tests: the gradient tests pass on PyTorch; the tests of compiled
  gradients and of abstract values go.
  Note: one commit holds A1 and A2. The one adapter module stays behind
  `_framework`, so that the package does not import PyTorch. The test of
  a kernel handing over its C++ barcodes goes: its bars now come from
  framework operations.
- [ ] **A2.** [2] The homological kernel has one path for gradients, through
  its critical edges: the vjp of a single pair, which existed for
  `jax.jit`, goes, with its binding.
  Tests: the kernel gradient tests pass.

## Phase B: The public surface of tracked objects

- [ ] **B1.** [2] A tracked `Pcf` or `Plf` raises where an operation would
  drop its gradients, as a tracked tensor does: reading `_data` raises, and
  the members that only read values (`to_numpy`, `size`, `==`, printing)
  use the detached object.
  Tests: evaluating, integrating, and copying a tracked `Pcf` raise;
  `to_numpy` and `==` give values; arithmetic and norms still carry
  gradients.
- [ ] **B2.** [3] `to_serial_content()` on `PcfTensor`, `BarcodeTensor`,
  `PointCloudTensor`, `DistanceMatrixTensor`, and `SymmetricMatrixTensor`:
  the public way to the rows of a tensor, as NumPy arrays, or with the
  content as a PyTorch tensor that carries gradients for a tracked tensor.
  The tests and the guide use it in place of `_tvalues` and `_ttimes`.
  Tests: round trip with `from_serial_content` for PCFs; the content of a
  tracked tensor is a PyTorch tensor with the values of the untracked
  content; a loss on it gives gradients.
- [ ] **B3.** [2] One rule for results with gradients: a dense numeric
  result is a PyTorch tensor, and a tensor of elements of different sizes
  is a tracked stablebear tensor. `project` of a PyTorch tensor of any
  shape (..., n, d) gives a PyTorch tensor of that shape, and a space keeps
  such a tensor as it is.
  Tests: `project` of a PyTorch tensor of shape (b, n, d) gives one of
  that shape; persistent homology of it in a space gives a tracked
  `BarcodeTensor`.

## Phase C: Names, measurements, and documents

- [ ] **C1.** [2] `stablebear.linear` becomes `stablebear.geometry`, and
  `InnerProductDistance` is exported from `stablebear`.
  Tests: the tests import from the new module; the docs build.
- [ ] **C2.** [2] Timings in a Release build of the tracked and untracked
  pipelines (summaries, persistent homology of a batch, the kernel of a
  batch and its projection), recorded in the specs.
- [ ] **C3.** [1] The specs, the guide, and the changelog say PyTorch only
  and describe the rules of B1 to B3.
