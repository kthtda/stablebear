# Linear subspaces, inner products, and distances: design and decisions

Plan: [linear-subspaces-plan.md](linear-subspaces-plan.md)

Point clouds can be projected onto a linear subspace, orthogonally with
respect to a chosen inner product, and persistent homology and the
homological kernel can use a chosen distance: the one an inner product
induces, or that of an Lp norm. All of it carries gradients of PyTorch,
JAX, and TensorFlow, so that a subspace or an inner product can be
optimized. This file records the design and the decisions behind it, with
the alternatives that were rejected. Update it when a decision changes.

It builds on the autodiff prototype
([autodiff-prototype-spec.md](autodiff-prototype-spec.md)), whose decisions
are cited as "autodiff D5" and so on.

## Goals

- Project a point cloud onto the span of a set of vectors, orthogonally
  with respect to the Euclidean inner product or one the user gives.
- Choose the distance that persistent homology and the homological kernel
  use, with the distance of an inner product as one choice, so that a cloud
  and its projection can be compared by the kernel.
- Gradients flow to the points, to the vectors that span the subspace, and
  to the matrix of the inner product.
- A distance or an inner product is never silently ignored: an operation
  either uses that of its input or raises.

## Example

```python
ip = sb.InnerProduct(gram)        # (d, d), symmetric positive definite
V = sb.LinearSubspace(vectors)    # (k, d), the rows span the subspace

X_ip = ip(X)                      # an InnerProductSpace holding X
Y_ip = V.project(X_ip)            # orthogonal with respect to gram
Y = V.project(X)                  # Euclidean

# An InnerProductSpace is a DistanceSpace, with the distance of gram.
kernel = sb.persistence.compute_homological_kernel(X_ip, Y_ip)

taxicab = sb.LpDistance(1)
barcodes = sb.persistence.compute_persistent_homology(taxicab(X))
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

**D2. Points carry their distance in a `DistanceSpace`.** A
`DistanceSpace` holds points and a distance and nothing else: `points` and
`distance` give them back, and `dtype` is the dtype of what holds the
points, such as `pcloud64`. For now the points are a point cloud or a batch
of them (see D7). A cloud that is not in a `DistanceSpace` has the
Euclidean distance.

The class is not tied to point clouds or to distances that come from
norms: it says only that its points live in a space with the given
distance, so that other kinds of points and distances can use it later.
Which points a distance accepts is for the distance to say.

A `DistanceSpace` is not a tensor: it has no indexing or arithmetic. An
operation that knows distances takes the points out; any other operation
does not accept it, and so raises instead of ignoring the distance.

Rejected: a class for point clouds only, which would need a sibling for
every other kind of point; a `distance` argument on each operation, which
the user would have to repeat at every step from projection to kernel; and
an attribute on `PointCloudTensor`, which every existing operation would
then silently ignore.

**D2a. An `InnerProductSpace` is a `DistanceSpace`.** `ip(points)` returns
an `InnerProductSpace`, a subclass of `DistanceSpace` that also gives its
`inner_product` back. Its distance is the one the inner product induces,
`ip.distance`. The result of a projection can therefore go straight into
persistent homology or the kernel. Both classes call what they hold
`points`.

Rejected: two unrelated classes with a conversion between them, which adds
a step to every pipeline, and `vectors` as the name in `InnerProductSpace`,
which gives one object two names.

**D2b. A distance is an object.** `Distance` is the base class of all
distances. Calling a distance on points gives a `DistanceSpace`. Two
distances are equal when they are of the same kind with equal parameters.
There are two kinds:

- the distance induced by an inner product, `ip.distance`, with
  d(x, y) = sqrt((x - y)^T G (x - y));
- `LpDistance(p)`, the distance of the Lp norm, for p >= 1 or infinity,
  with d(x, y) = (sum_i |x_i - y_i|^p)^(1/p), and the largest
  |x_i - y_i| for p = infinity. `LpDistance(2)` is the Euclidean distance.

Users do not subclass `Distance` yet: how a distance is computed (D6) is
internal.

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
D7). A `DistanceSpace` that is not an `InnerProductSpace` has no orthogonal
projection, and raises `TypeError`.

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

**D6. A distance is computed in one of two ways.** `compute_persistent_
homology` and `compute_homological_kernel` take a `DistanceSpace` in place
of a point cloud and use its distance.

- *As Euclidean distances of transformed points.* With the Cholesky
  factorization G = L^T L, the distance of an inner product between x and y
  is the Euclidean distance between L x and L y. The existing Euclidean
  computation runs on the points X L^T, and `LpDistance(2)` runs it on the
  points as they are. No distance code changes, and the gradients of both
  operations with respect to the points (autodiff D16 and the kernel's vjp)
  extend to G through this product.
- *Through a matrix of pairwise distances.* Any other distance, for now
  `LpDistance(p)` with p other than 2, gives the pairwise distances of each
  cloud as a distance matrix, computed in C++, and the existing computation
  on distance matrices runs on it.

The kernel takes two `DistanceSpace`s with equal distances, or two plain
clouds. One of each, or different distances, raise `TypeError`. That the
first distances dominate the second is the user's to ensure, as for plain
clouds. The barcodes that come back are plain: bars are numbers and have no
distance.

Rejected: passing the Gram matrix to Ripser and to the kernel as a distance
function, which changes both C++ paths and their gradients for the same
result.

**D6a. Gradients through a matrix of distances follow the critical
edges.** Each endpoint of a bar is the distance between two points
(autodiff D16), whatever the distance. For a distance computed through a
matrix, persistent homology and the kernel report the critical edges of
the bars computed from that matrix, and the bars are the distances of those
pairs of points again, as framework operations with the formula of the
distance. This needs concrete points, so it raises while JAX or
`tf.function` traces them, for the kernel as well, which with Euclidean
and inner product distances keeps its vjp and works while they trace.

**D6b. `distance_type` is deprecated.** `compute_persistent_homology` has a
`distance_type` argument with the single value `DistanceType.Euclidean`.
A `DistanceSpace` replaces it. Passing `distance_type` gives a
`DeprecationWarning`, and passing it with a `DistanceSpace` raises
`TypeError`.

## Inputs and gradients

**D7. One implementation over NumPy and the frameworks.** `project` and
the distances take, as points:

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

**D8a. Persistent homology and the kernel differentiate a batch through
its critical edges.** A batch that carries gradients is a tracked
`PointCloudTensor`: the result of a projection or a product, a
`PointCloudTensor` built from a list that holds framework arrays, or a
framework array of shape (..., n, d), which is a batch of clouds of n
points as a NumPy array of that shape is. C++ computes the bars and the
critical edges of every element, in parallel over the elements, from the
point clouds or, for a distance computed through matrices, from the
distance matrices (D6a). The bars are then the distances of those pairs
of points as framework operations on the one array that holds the points
of all clouds: one gather and one product for the whole batch. The result
is a tracked `BarcodeTensor` of the shape of the untracked result.

This needs concrete points, so a batch raises while JAX or `tf.function`
traces it. The kernel of a single pair of clouds with Euclidean or inner
product distances keeps its vjp, and still works while they trace.

Rejected: a Python loop over the clouds of a batch, which costs a
framework operation per cloud (autodiff D13).

**D9. Checks need concrete values.** That the Gram matrix is symmetric
positive definite and that the vectors are independent is checked when the
values are concrete, and raises `ValueError`. While a framework traces the
matrix or the vectors abstractly, they are not checked, and invalid values
give NaN.

## Names and modules

**D10. `stablebear.linear` holds the new classes.** `InnerProduct`,
`InnerProductSpace`, `LinearSubspace`, `Distance`, `DistanceSpace`, and
`LpDistance` live in
`stablebear/linear.py` and are exported from `stablebear`. The existing
`stablebear/inner_product.py` holds `l2_kernel`, the inner products of
PCFs, and is unrelated.

## Open questions

- Should a subspace also give the (n, k) coefficients of the projected
  points, for use as lower-dimensional coordinates?
- Should affine subspaces (a subspace through a point other than the
  origin) be supported?
- Should users be able to define a distance of their own, by a function
  of two points or by subclassing `Distance`?
- Should norms be objects of their own, between inner products and
  distances?
