# Linear subspaces and inner products: implementation plan

Progress: 3 of 10 items, 8 of 23 points (35%)

Spec: [linear-subspaces-spec.md](linear-subspaces-spec.md)

## Phase A: Projecting a single cloud

- [x] **A1.** [2] (120c476f7) `InnerProduct(gram)` in `stablebear/linear.py`, exported
  from `stablebear`: `gram`, equality, and `ip(vectors)` returning an
  `InnerProductSpace` with `vectors`, `inner_product`, and `dtype`; NumPy
  only.
  Tests: `gram` gives the matrix back; equal matrices give equal inner
  products; a matrix that is not square, not symmetric, or not positive
  definite raises; the space gives its vectors and inner product back, with
  the dtype of what holds them; points of another dimension raise.
- [x] **A2.** [3] (3185c4e5f) `LinearSubspace(vectors)` with `vectors`, `dim`,
  `ambient_dim`, and `project` of a single cloud given as a NumPy array,
  `FloatTensor`, or `PointCloud`, plain or in an `InnerProductSpace` (D4,
  D5).
  Tests: projection onto a coordinate axis; projecting twice equals
  projecting once; vectors that are not orthonormal give the projection of
  an orthonormal basis of their span; under a Gram matrix, the difference
  between a point and its projection is orthogonal to the vectors in that
  inner product; an `InnerProductSpace` gives one back and a plain cloud a
  `PointCloud`; dependent vectors raise; points of another dimension raise.
  Note: the projection solves a system with V G V^T in place of a Cholesky
  factorization, which each framework also provides.
- [x] **A3.** [3] (3d8e92adc) Framework arrays for the points, the vectors, and the Gram
  matrix (D7): the array operations (product, transpose, solving a
  linear system) in NumPy and in each adapter, and the linear algebra
  written against them.
  Tests: the projection of framework points matches NumPy; gradients with
  respect to the points, to the vectors, and to the Gram matrix against
  finite differences; the compiled gradient matches the eager one.
  Note: the Cholesky factor is not needed until B1, which adds it to the
  adapters. Two inner products of matrices that a framework traces are
  equal only if the matrices are the same object.

## Phase B: Distances under an inner product

- [ ] **B1.** [2] `compute_persistent_homology` of an `InnerProductSpace` of
  a single cloud, by the transformed points X L^T (D6), with and without gradients.
  Tests: equals the barcodes of the transformed points; the identity matrix
  gives the barcodes of the plain cloud; gradient with respect to the Gram
  matrix against finite differences.
- [ ] **B2.** [3] `compute_homological_kernel` of two `InnerProductSpace`s of
  single clouds with equal inner products, with and without gradients.
  Tests: equals the kernel of the transformed points; the kernel of a cloud
  and its projection under the same inner product is valid; one
  `InnerProductSpace` and one plain cloud raise, and so do different inner
  products; gradient
  of the kernel of a cloud and its projection with respect to the vectors
  of the subspace against finite differences.

## Phase C: Batches

- [ ] **C1.** [2] C++: multiply every point of every cloud of a tensor of
  point clouds by a (d, d) matrix, in parallel over the clouds (D8), with
  its binding.
  Tests (C++): one cloud; clouds of different sizes; an empty cloud; an
  indexed cloud; a matrix of another dimension raises.
- [ ] **C2.** [2] `project` of an untracked `PointCloudTensor`, plain or in
  an `InnerProductSpace`, and persistent homology and the kernel of
  untracked batches in one.
  Tests: each projected cloud equals the projection of that cloud alone;
  the barcodes and kernels of a batch in an `InnerProductSpace` equal those
  of each of its clouds alone.
- [ ] **C3.** [2] `project` of a tracked `PointCloudTensor` as one
  framework product (D8).
  Tests: matches the untracked projection; gradient with respect to the
  vectors against finite differences; persistent homology of a tracked
  batch in an `InnerProductSpace` raises.

## Phase D: Documentation

- [ ] **D1.** [2] API reference entries for `InnerProduct`,
  `InnerProductSpace`, and `LinearSubspace`, and the `InnerProductSpace`
  inputs in
  the docstrings of `compute_persistent_homology` and
  `compute_homological_kernel`. (Each class gets its docstring in the item
  that adds it.)
- [ ] **D2.** [2] A user guide page on projecting point clouds: what a
  projection under an inner product is, the kernel of a cloud and its
  projection, and optimizing a subspace with a framework, with one figure.
