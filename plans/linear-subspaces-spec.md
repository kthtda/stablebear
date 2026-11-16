# Linear subspaces and inner products: design and decisions

Plan: [linear-subspaces-plan.md](linear-subspaces-plan.md)

Point clouds can be projected onto a linear subspace, orthogonally with
respect to a chosen inner product, and the distances that persistent
homology and the homological kernel use can come from that inner product.
All of it carries gradients of PyTorch, JAX, and TensorFlow, so that a
subspace or an inner product can be optimized. This file records the design
and the decisions behind it, with the alternatives that were rejected.
Update it when a decision changes.

It builds on the autodiff prototype
([autodiff-prototype-spec.md](autodiff-prototype-spec.md)), whose decisions
are cited as "autodiff D5" and so on.

## Goals

- Project a point cloud onto the span of a set of vectors, orthogonally
  with respect to the Euclidean inner product or one the user gives.
- Use the same inner product for the distances of persistent homology and
  of the homological kernel, so that a cloud and its projection can be
  compared by the kernel.
- Gradients flow to the points, to the vectors that span the subspace, and
  to the matrix of the inner product.
- An inner product is never silently ignored: an operation either uses the
  inner product of a cloud or raises.

## Example

```python
ip = sb.InnerProduct(gram)        # (d, d), symmetric positive definite
V = sb.LinearSubspace(vectors)    # (k, d), the rows span the subspace

X_ip = ip(X)                      # an InnerProductSpace holding X
Y_ip = V.project(X_ip)            # orthogonal with respect to gram
Y = V.project(X)                  # Euclidean

kernel = sb.persistence.compute_homological_kernel(X_ip, Y_ip)
```

## Objects

**D1. An inner product is its Gram matrix.** `InnerProduct(gram)` takes a
symmetric positive definite (d, d) matrix G, with the inner product of x and
y being x^T G y. `gram` gives the matrix back. Two inner products are equal
when their matrices are. A matrix that is not square, not symmetric, or not
positive definite raises `ValueError`.

Rejected: a callable (x, y) -> number, which cannot be turned into the
matrices that the projection and the distances need, and diagonal weights
only, which a full matrix covers.

**D2. Vectors carry their inner product in an `InnerProductSpace`.**
`ip(vectors)` returns an `InnerProductSpace`, which holds the vectors and
the inner product and nothing else: `vectors` and `inner_product` give them
back, and `dtype` is the dtype of what holds the vectors, such as
`pcloud64`. For now that is a point cloud or a batch of them (see D7): the
points are the vectors, and their dimension must be that of the matrix. A
cloud that is not in an `InnerProductSpace` has the Euclidean inner
product.

The class is not tied to point clouds: it says only that its vectors live
in a space with the given inner product, so that other kinds of vectors,
with inner products of their own, can use it later. Which kinds an inner
product accepts is for the inner product to say: `InnerProduct(gram)`
accepts point clouds.

An `InnerProductSpace` is not a tensor: it has no indexing or arithmetic.
An operation that knows inner products takes the vectors out; any other
operation does not accept it, and so raises instead of ignoring the inner
product.

Rejected: a class for point clouds only (`InnerProductPointCloud`), which
would need a sibling for every other kind of vector; an `inner_product`
argument on each operation, which the user would have to repeat at every
step from projection to kernel; and an attribute on `PointCloudTensor`,
which every existing operation would then silently ignore.

**D3. A subspace is the span of the rows of a matrix.** `LinearSubspace(
vectors)` takes a (k, d) array whose k rows are vectors of dimension d.
They need not be orthogonal or of unit length, but must be linearly
independent with k >= 1; otherwise it raises `ValueError`. `vectors` gives
the array back, `dim` is k, and `ambient_dim` is d. The subspace has no
inner product of its own: `project` takes it from the cloud.

Rejected: requiring orthonormal vectors, which depends on the inner product
and which an optimizer would not preserve.

## Projection

**D4. `project` returns the projected points in ambient coordinates.** For
points X of shape (n, d), vectors V, and Gram matrix G, the result is the
(n, d) array

    Y = X G V^T (V G V^T)^{-1} V,

each row the point of the subspace closest to the row of X in the distance
of G. With the Euclidean inner product, G is the identity. The result of
projecting an `InnerProductSpace` is one with the same inner product; the
result of projecting a plain cloud is plain, of the kind of the input (see
D7).

Ambient coordinates are what `compute_homological_kernel` takes for its
second cloud, and an orthogonal projection does not increase distances, so
a cloud and its projection, under the same inner product, satisfy the
kernel's condition that the first distances dominate the second.

Rejected: returning the (n, k) coefficients in the given vectors, which
depend on the choice of vectors and cannot stand next to the original cloud
in the kernel.

**D5. The small linear algebra runs once, the points take one product.**
The (d, d) matrix P = G V^T (V G V^T)^{-1} V is computed once per call,
by solving a system with the (k, k) matrix V G V^T, and the points are
multiplied by it.
The cost for the points is one matrix product, for a batch as well (D7).

## Distances

**D6. Distances under an inner product are Euclidean distances of
transformed points.** With the Cholesky factorization G = L^T L, the
distance of G between x and y is the Euclidean distance between L x and
L y. `compute_persistent_homology` and `compute_homological_kernel` of an
`InnerProductSpace` therefore run their existing Euclidean computation on the
points X L^T. No distance code changes, and the gradients of both
operations with respect to the points (autodiff D16 and the kernel's vjp)
extend to G through this product.

The kernel takes two `InnerProductSpace`s with equal inner products, or two
plain clouds. One of each, or different inner products, raise `TypeError`. The barcodes that come back are plain: bars are numbers and
have no inner product.

Rejected: passing the Gram matrix to Ripser and to the kernel as a distance
function, which changes both C++ paths and their gradients for the same
result.

## Inputs and gradients

**D7. One implementation over NumPy and the frameworks.** `project` and
`InnerProduct(gram)` take, as points:

- a single cloud of shape (n, d): a NumPy array, `FloatTensor`,
  `PointCloud`, or a PyTorch, JAX, or TensorFlow array;
- a batch: a `PointCloudTensor`, with or without gradients (autodiff D5a).

The vectors and the Gram matrix are NumPy arrays or framework arrays. If
any of the points, the vectors, and the matrix is a framework array, the
computation runs in that framework and the result carries gradients with
respect to all that are: a framework array for a single cloud (as in
autodiff D14), and a tracked `PointCloudTensor` for a batch. Otherwise a
single cloud gives a `PointCloud` and a batch a `PointCloudTensor`.

The linear algebra of D5 and D6 is written once against a small set of
array operations (product, transpose, solving a linear system, Cholesky
factor) that NumPy and each framework adapter provide. No operation has a shape
that depends on values, so projection works while JAX or `tf.function`
traces, unlike persistent homology.

**D8. A batch is multiplied in C++, or as one framework product.** An
untracked `PointCloudTensor` is multiplied by a (d, d) matrix in C++, in
parallel over the clouds, by one new primitive that serves both the
projection (P) and the distances (L^T). A tracked `PointCloudTensor` holds
the points of all its clouds as one (m, d) framework array, so its
projection is one framework product with the same enumeration.

Persistent homology and the kernel with gradients take a single cloud only
(autodiff D16), and that does not change: a tracked batch, in an
`InnerProductSpace` or not, raises there.

**D9. Checks need concrete values.** That the Gram matrix is symmetric
positive definite and that the vectors are independent is checked when the
values are concrete, and raises `ValueError`. While a framework traces the
matrix or the vectors abstractly, they are not checked, and invalid values
give NaN.

## Names and modules

**D10. `stablebear.linear` holds the new classes.** `InnerProduct`,
`InnerProductSpace`, and `LinearSubspace` live in
`stablebear/linear.py` and are exported from `stablebear`. The existing
`stablebear/inner_product.py` holds `l2_kernel`, the inner products of
PCFs, and is unrelated.

## Open questions

- Should a subspace also give the (n, k) coefficients of the projected
  points, for use as lower-dimensional coordinates?
- Should affine subspaces (a subspace through a point other than the
  origin) be supported?
- Should persistent homology and the kernel take batches of tracked clouds?
