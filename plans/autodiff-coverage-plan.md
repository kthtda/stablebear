# Autodiff coverage: implementation plan

Progress: 10 of 11 items, 24 of 25 points (96%)

Spec: [autodiff-prototype-spec.md](autodiff-prototype-spec.md), which each
item updates where it changes a decision.

An audit probed the public API with inputs that carry gradients. This plan
closes what it found: results that lost gradients without an error, errors
that did not say why, and common operations without gradients.

## Phase A: No silent loss of gradients

- [x] **A1.** [2] (9d60838d3) A PyTorch tensor given where a plain number is required
  raises `TypeError`, in place of being read as a number: `p` of `pdist`
  and `cdist`, `LpDistance(p)`, `max_death` of
  `barcode_to_accumulated_persistence`, and `delay` of `delay_embed`.
  Tests: each raises for a PyTorch scalar and still takes a number.
- [x] **A2.** [3] (0711696af) Evaluating a `Pcf`, `Plf`, or `PcfTensor` gives a PyTorch
  tensor when the times or the function are PyTorch tensors or carry
  gradients, with gradients with respect to the values of the function
  and, for a `Plf`, to the times and to where it is evaluated.
  Tests: values match the untracked evaluation; gradients against finite
  differences with respect to PCF values, PLF values, PLF times, and the
  times of evaluation of a PLF; a tracked `PcfTensor`.
- [x] **A3.** [3] (95458978d) An integer index into a tracked tensor gives an element
  that carries gradients: a tracked `Pcf` or `Barcode`, and for a point
  cloud the PyTorch tensor of its points. The barcode summaries and
  `transform_bars` take a tracked `Barcode`, and `BarcodeTensor` of a list
  with one builds a tracked tensor.
  Tests: gradient through `out[1]` of persistent homology matches that
  through `out[1:2]`; a summary of a tracked `Barcode`; iteration gives
  tracked elements; `to_numpy` and `len` of a tracked `Barcode`.

## Phase B: Errors that say why, and missing inputs

- [x] **B1.** [2] (addadb8dc) PCF arithmetic with one number per PCF: a PyTorch tensor,
  NumPy array, or `FloatTensor` of the shape of the PCF tensor scales each
  PCF, where it now combines with each breakpoint and fails.
  Tests: `F * w` for a PyTorch vector `w` matches scaling each PCF;
  gradient with respect to `w`; a shape that does not broadcast raises.
- [x] **B2.** [2] (ea35c0905) Where a dense array is required (`FloatTensor`,
  `PointCloud`, `DistanceMatrix`, `SymmetricMatrix`,
  `from_serial_content`, `linkage_to_barcode`, writes into numeric
  tensors), a PyTorch tensor without gradients is converted, and one with
  gradients raises a `TypeError` that says so. `allclose` of a tracked
  tensor compares its values.
  Tests: both cases for each; `allclose` of a tracked and a plain tensor.
  Note: `allclose` does not take PCF tensors, tracked or not, so the
  tracked case of the audit was not about gradients.
- [x] **B3.** [3] (e0c76c443) Distance matrices with gradients: `DistanceMatrixTensor`
  of a PyTorch tensor of shape (..., n, n) is tracked, and
  `compute_persistent_homology` and `compute_homological_kernel` of
  tracked distance matrices give bars that carry gradients with respect to
  the distances.
  Tests: bars match those of the untracked matrices; gradients against
  finite differences for both operations; a batch.

## Phase C: Operations without gradients

- [x] **C1.** [3] (8b24d5256) `sum` and `mean` of tracked PCFs along an axis, as one
  sort and one running sum of the jumps of all members.
  Tests: match the untracked reductions on a grid of times; gradient of a
  norm of the mean against finite differences, with respect to values and
  to times.
The last value of each sum is taken directly, not from the running sum, so that sums of PCFs that end at 0 end at exactly 0.
- [x] **C2.** [2] (153283c27) Powers of PCFs with gradients (`tracked ** 2`, and a
  PyTorch exponent), and `lp_distance` of tracked PCFs or with a PyTorch
  `p`.
  Tests: values match; gradients against finite differences.
- [x] **C3.** [2] (6e84cd0ca) `concatenate`, `stack`, `split`, and `array_split` of
  tracked tensors, in one pass of the pending writes.
  Tests: match the untracked results for PCFs, barcodes, and point
  clouds; gradient through a concatenation.
Joins take one concatenation and one gather of the rows, without pending writes, which would take a step per element.
- [x] **C4.** [2] (d514227bd) `cdist` of tracked PCFs.
  Tests: matches the untracked distances; gradient against finite
  differences with respect to each side.
The distances are those across in the pairwise distances of both tensors together, so the pairs within each are computed too.
- [ ] **C5.** [1] Plotting a tracked tensor plots its values, and
  `is_isomorphic_to` of a tracked `BarcodeTensor` compares them.
  Tests: both run and agree with the detached tensor.

## Not planned

Left raising, as the audit found them: `integrate` and `max_time` of
tracked PCFs, `subsample` of tracked point clouds, `astype`, and time
series, nested tensors, and PLF tensors of tracked values.
