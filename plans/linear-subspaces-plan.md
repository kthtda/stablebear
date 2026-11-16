# Linear subspaces, inner products, and distances: implementation plan

Progress: 19 of 19 items, 43 of 43 points (100%)

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

## Phase B: Distance spaces

- [x] **B1.** [2] (bd36e573d) `Distance` and `DistanceSpace(points, distance)` with
  `points`, `distance`, and `dtype`; `InnerProductSpace` becomes its
  subclass, with `points` in place of `vectors` (D2, D2a); `ip.distance`,
  and calling a distance on points.
  Tests: a distance space gives its points and distance back; an
  `InnerProductSpace` is a `DistanceSpace` with the distance of its inner
  product; distances of equal inner products are equal; projecting a
  `DistanceSpace` that is not an `InnerProductSpace` raises.
- [x] **B2.** [2] (9974eb30c) `compute_persistent_homology` of a `DistanceSpace` of a
  single cloud with the distance of an inner product, by the transformed
  points X L^T (D6), with and without gradients; the Cholesky factor in
  each adapter.
  Tests: equals the barcodes of the transformed points; the identity matrix
  gives the barcodes of the plain cloud; gradient with respect to the Gram
  matrix against finite differences.
- [x] **B3.** [1] (2e4802e25) Deprecate `distance_type` (D6b).
  Tests: passing it warns; passing it with a `DistanceSpace` raises; the
  default does not warn.
- [x] **B4.** [3] (6eca2162e) `compute_homological_kernel` of two `DistanceSpace`s of
  single clouds with equal distances of inner products, with and without
  gradients.
  Tests: equals the kernel of the transformed points; the kernel of a cloud
  and its projection under the same inner product is valid; one
  `DistanceSpace` and one plain cloud raise, and so do different distances;
  gradient of the kernel of a cloud and its projection with respect to the
  vectors of the subspace against finite differences.
- [x] **B5.** [2] (795a53ab4) C++: the pairwise Lp distances of every cloud of a tensor
  of point clouds as a tensor of distance matrices, for p >= 1 and
  infinity, in parallel over the clouds, with its binding.
  Tests (C++): p = 1, 2, 3, and infinity on one cloud; clouds of different
  sizes; an empty cloud; p < 1 raises.
- [x] **B6.** [2] (04a5e563a) `LpDistance(p)`, and persistent homology and the kernel of
  an untracked single cloud in its `DistanceSpace`, through the distance
  matrix (D6).
  Tests: p = 2 gives the Euclidean barcodes; barcodes for p = 1 and
  infinity of a small cloud with known distances; equal p give equal
  distances; p < 1 raises; the kernel under p = 1 equals the kernel of the
  two distance matrices.
- [x] **B7.** [3] (54d523661) Gradients under `LpDistance(p)` (D6a): the critical edges
  of persistent homology and of the kernel from distance matrices, and the
  bars as the distances of those pairs in framework operations.
  Tests: the bars of framework points match NumPy for p = 1 and 3;
  gradients of persistent homology and of the kernel against finite
  differences for p = 1 and 3; points that JAX traces raise.
  Note: gradients at equal distances are a kink; taxicab and Chebyshev
  distances tie easily on points with round coordinates, so the gradient
  tests use a cloud whose distances all differ.

## Phase C: Batches

- [x] **C1.** [2] (27022cae2) C++: multiply every point of every cloud of a tensor of
  point clouds by a (d, d) matrix, in parallel over the clouds (D8), with
  its binding.
  Tests (C++): one cloud; clouds of different sizes; an empty cloud; an
  indexed cloud; a matrix of another dimension raises.
- [x] **C2.** [2] (3d182707a) `project` of an untracked `PointCloudTensor`, plain or in
  an `InnerProductSpace`, and persistent homology and the kernel of
  untracked batches in a `DistanceSpace`, for both kinds of distance.
  Tests: each projected cloud equals the projection of that cloud alone;
  the barcodes and kernels of a batch in a `DistanceSpace` equal those of
  each of its clouds alone.
  Note: one commit holds C2 and C3, which share the product of a batch
  with a matrix. The dimension of an untracked batch is checked when it
  is multiplied, not when it is put in a space.
- [x] **C3.** [2] (3d182707a) `project` of a tracked `PointCloudTensor` as one
  framework product (D8).
  Tests: matches the untracked projection; gradient with respect to the
  vectors against finite differences; persistent homology of a tracked
  batch in a `DistanceSpace` raises.

## Phase D: Documentation

- [x] **D1.** [2] (19a719510) API reference entries for the new classes, and the
  `DistanceSpace` inputs in the docstrings of `compute_persistent_homology`
  and `compute_homological_kernel`; `docs/persistence.rst` without
  `distance_type`. (Each class gets its docstring in the item that adds
  it.)
- [x] **D2.** [2] (ab9ee25b1) A user guide page on projecting point clouds and choosing
  distances: what a projection under an inner product is, the kernel of a
  cloud and its projection, the distance of an Lp norm, and optimizing a
  subspace with a framework, with one figure.

## Phase E: Batches with gradients

- [x] **E1.** [2] (5471c39c5) C++: the bars and critical edges of persistent homology
  and of the homological kernel for every element of a tensor of point
  clouds or of distance matrices, in parallel over the elements, with their
  bindings.
  Tests: the bars of a batch equal those of each element alone, for both
  operations and both kinds of element; the edges of each element are
  pairs of its own points; an empty cloud gives no bars.
- [x] **E2.** [2] (5b6ddea64) A batch of point clouds that carries gradients from
  framework arrays: `PointCloudTensor` of a list with framework arrays, and
  framework arrays of shape (..., n, d) where a point cloud is taken.
  Tests: the batch holds each array as a cloud, of different sizes;
  gradients flow through its projection; a framework array of shape
  (b, n, d) is a batch of b clouds.
- [x] **E3.** [3] (6285fe38b) `compute_persistent_homology` with gradients of a batch,
  plain or in a `DistanceSpace` of either kind of distance (D8).
  Tests: the bars of a batch match the untracked ones; gradients with
  respect to the points, to the vectors of a subspace, and to a Gram
  matrix against finite differences; under `LpDistance(1)`; points that
  JAX traces raise.
  Note: this commit also annotates `PointCloudTensor.__init__`, which E2
  left with a pyright error.
- [x] **E4.** [3] (f6fede085) `compute_homological_kernel` with gradients of two batches,
  plain or in `DistanceSpace`s of either kind of distance.
  Tests: the bars of a batch match the untracked ones; gradient of the
  kernels of a batch and its projection with respect to the vectors of
  the subspace against finite differences; under `LpDistance(1)`; batches
  of different shapes raise.
