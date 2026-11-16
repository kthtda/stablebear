# Autodiff prototype: design and decisions

Plan: [autodiff-prototype-plan.md](autodiff-prototype-plan.md)

The `autodiff-prototype` branch lets gradients of PyTorch flow through
stablebear operations, so that a loss built from PCFs, PLFs, and barcodes
can be optimized with respect to point clouds, breakpoints, or the
parameters of a transformation. This file records the
design and the decisions behind it, with the alternatives that were
rejected. Update it when a decision changes.

## Goals

- A user writes ordinary stablebear code with PyTorch tensors as inputs and
  gets a result that carries gradients.
- Code that does not use PyTorch runs exactly as before, with no extra
  cost, and does not import it.
- Values and gradients are never silently wrong: an operation either
  handles gradients correctly or raises.
- Most of the library can be differentiated, with a bounded overhead over
  the untracked path (see D13).

## Architecture

**D1. The core stays free of PyTorch.** An operation opts in with the
`_autodiff.differentiable` decorator, naming the parameters that can carry
gradients and a vector-Jacobian product (vjp) written in NumPy. One adapter
module, `_autodiff_torch`, runs the operation as a
`torch.autograd.Function`, and holds every other use of PyTorch. The rest
of the package reaches it through `_autodiff._framework`, which imports it
only for a PyTorch tensor. This spec and the code call such tensors
"framework arrays", to tell them from stablebear tensors.

**D1a. PyTorch is the only framework, for now.** The first version also had
adapters for JAX and TensorFlow. They were removed: every operation whose
result depends on values (which bars exist, in which order breakpoints
fall) needs concrete values, and `jax.grad` and `tf.function` trace with
abstract ones, so persistent homology, the barcode summaries,
`transform_bars`, and all batches raised there. Supporting them cost a
declared shape for every result, a branch for abstract values in every
operation, and a second path for the kernel that existed only for
`jax.jit`. The seam of D1 stays, so that an adapter can come back for the
operations that it can support.

An operation whose result is a tensor declares its type (`returns=`) and
returns the untracked C++ tensor; the decorator makes the framework array
of its serial content and returns the tracked tensor, which keeps the C++
tensor (D5).

**D2. Tracked arguments describe themselves.** An object that carries
gradients reports its framework arrays, how to rebuild the object from
NumPy values of those arrays, and how to split the vjp's gradient into one
per array (`_tracked_parts`). The decorator uses this to convert tracked
arguments to NumPy, call the operation, and route gradients back.

**D3. Some operations are written in framework operations instead of a
vjp.** PCF arithmetic, the summaries of tracked barcodes (stable rank,
Betti curve, accumulated persistence function), and persistent homology
compute their
results with framework operations (gather, concat, sort), so the framework
differentiates them itself. PCF arithmetic takes the union of the operands'
breakpoint times, and each time of the result comes from the first operand
that has it.

## Representation of tracked objects

**D4. Ragged data is a flat array plus an enumeration.** A tensor of PCFs
or of barcodes holds elements of different lengths, and frameworks
differentiate only dense arrays. A tracked tensor therefore holds all rows
(breakpoints, or bars) as one framework array of shape (m, 2), in C order of
the elements, with an enumeration of shape `shape + (2,)` giving each
element's [start, stop) rows. This is the "serial content" layout of
`from_serial_content`, and the same layout as TensorFlow's `RaggedTensor`
(values plus row splits). C++ converts tensors of PCFs and of barcodes to and
from it.

Rejected: one framework array per element, which needs Python loops over a
batch and gives no vectorized gradients.

**D5. One source of truth.** The state of a tracked object is its
framework arrays and its enumeration. Its C++ object (`_data`) is derived
from them, so the two cannot disagree. The derived C++ object is cached:
a tracked object is immutable, and PyTorch's version counter of the array
tells when it was changed in place, in which case the cache is rebuilt. An operation that already has the C++
result hands it to the tracked object as the initial cache. Indexing,
reshaping, and copying keep the gradients; any other operation that would
build a new object without them raises, instead of dropping them silently.

The raise is the default, not a check written per operation: reading `_data`
of a tracked tensor raises, so a function that was not written for gradients
(a Betti curve, a `TimeSeries` or `NestedTensor` of the tensor) cannot
return a result without them. Code that handles gradients, or needs only
the values, asks for the C++ tensor by another name. `detach()` gives the
untracked tensor of the current values, for a user who wants them without
gradients. Comparisons, saving, and pickling use the values: their results
could not carry gradients.

The untracked classes are unchanged: `_data` stays a plain attribute, and
each element lives in its own heap allocation.

Rejected: storing a C++ object and framework arrays as independent state,
which the first versions did. Assigning to an element of a tracked
`BarcodeTensor` changed the C++ bars but not the framework bars, and later
results used the stale ones. Also rejected: one class whose `_data` property
checks whether the object is tracked, which adds a Python call to every
`_data` access on hot paths.

An operation's forward and backward passes get the cached C++ tensor of a
tracked argument when the values the framework passes them equal the
argument's current ones, which one comparison pass tells.

A single `Pcf` or `Plf` that carries gradients follows the same rule:
reading its `_data` raises, so evaluating, integrating, or copying it
raises, while `to_numpy`, `size`, `==`, and printing give its values. There
is no tracked `Barcode`: no operation gives one (D10).

**D5c. `to_serial_content` is the public way to the rows.** A tensor of
PCFs, barcodes, point clouds, or matrices gives its rows with
`to_serial_content()`: the content and the enumeration, and for matrices
the sizes, as NumPy arrays. For a tracked tensor the content is the PyTorch
tensor that carries its gradients, so a loss can be built from it. It is the
inverse of the released `from_serial_content` for PCFs. The attributes that
hold the framework arrays of a tracked object stay private.

**D5a. One generic tracked tensor.** The tracking logic is a mixin shared by
all tensor types, not a class written per type. Every type is a tensor
whose elements are blocks of rows of a fixed width: bars (birth, death),
breakpoints (time, value), points (d coordinates), matrix entries. A type
describes itself with its C++ conversions to and from serial content, its
groups of columns that can be tracked separately (times and values of a
PCF), and its element class. The mixin provides the state, the derived
`_data`, `shape`, indexing, reshaping, copying, writes (D5b), and
`_tracked_parts`. `tracked_class(T)` builds and caches the private subclass
of a tensor type `T`, so a tracked object is still an instance of `T`.
Single objects (`Pcf`, `Plf`) are the 0-d case: `Pcf(values, times=times)`
returns the tracked subclass when an argument is a framework array. A dense numeric
tensor needs no enumeration: see D14.

The points of a `PointCloudTensor` are one group of d columns, where d is
the dimension of its clouds, which must agree.

The entries of a distance or symmetric matrix are rows of one column: those
below the diagonal of a distance matrix, and those on and below it of a
symmetric matrix, by rows. A distance matrix has no entries both when it has
no points and when it has one, so the enumeration does not give its size. A
tracked tensor of matrices therefore also keeps the size of each matrix, an
integer array of the tensor's shape, and its serial content is (content,
enumeration, sizes). Writes to such a tensor are not supported yet.

Rejected: storing the zero diagonal of a distance matrix, which makes the
number of entries tell the size but adds n entries per matrix whose
gradients must be ignored.

**D5b. Writes are deferred.** `x[i] = v` on a tracked tensor records the
replacement in a table of pending writes, at no cost per write, and the next
operation that needs the framework arrays applies all of them in one pass.
Reading an element looks in the table first. A loop of n writes therefore
costs one pass, not n. Basic indexing of a tracked tensor gives a copy, not
a view as for untracked tensors: writing to the result does not change the
tensor it came from.

A constructor given a list of elements, such as `PcfTensor([f, g])`, returns
a tracked tensor when an element carries gradients: every element is written
to an empty tensor, which one pass then fills. A list without such an element
is scanned once more than before to tell.

Rejected: applying each write at once, which rebuilds the array per write,
and one framework array per element, which makes writes cheap but needs a
Python loop for every batch operation.

**D6. Indexing reuses the tensor indexing rules.** A tracked tensor indexes,
reshapes, or broadcasts an `IntTensor` of its elements' flat indices with
the same arguments, then gathers the rows of the selected elements from the
framework array. The rules are therefore exactly those of untracked
tensors, and repeated elements (as from `broadcast_to`) accumulate
gradients correctly. An integer index that selects one element returns an element that carries
the gradients of its rows: a tracked `Pcf` or `Barcode`, whose
`to_serial_content` gives its rows as a framework array, or, for a point
cloud, the framework array of its points (linear D7a). The barcode summaries
and `transform_bars` take a tracked `Barcode`, and a list with tracked
elements builds a tracked tensor. A matrix is still an untracked element of
the current values.

Changed: the first version returned an untracked element, "like reading a
number out of an array". That made `barcodes[1]`, the natural way to take
the barcode of dimension 1, lose the gradients that `barcodes[1:2]` kept.

**D7. Native ragged storage in C++ is deferred.** Storing tensors of
barcodes (and PCFs) in C++ as one buffer plus offsets, instead of one heap
vector per element, would make the serial conversion free and let a framework
array share memory with the C++ tensor through DLPack. It would not remove
the need for D5: gradients still live on the framework array, and C++
indexing does not record framework operations. Sharing memory would help
on the CPU only, where the barcode code runs. It is a
storage refactor of several days, to revisit if profiling shows the copies
or the per-element allocations matter. The `_serial` interface would stay.

## Operations

**D8. Results may depend on values.** Which bars `transform_bars`
removes, the order of the breakpoints of a sum of PCFs, and the number of
bars of persistent homology depend on the values of their inputs. PyTorch
runs eagerly, so the operations read the values and build the result for
them. (`transform_bars` removes bars rather than keeping them with zero
length, which would give the same stable ranks and Betti curves but
different barcodes.)

**D9. Derivatives at kinks are one-sided.** At a breakpoint of a PLF, the
derivatives are those of the segment after it. A bar with an infinite
endpoint maps to the limit of the PLF, whose derivative is 1 with respect to
the last value where the PLF ends flat, and 0 otherwise. For the
homological kernel, a zero-length critical edge has gradient 0. For a
distance between equal PCFs, the gradient is 0. Where breakpoints of the two
PCFs of a pair share a time, the derivative of their distance or product
with respect to that time is that of moving both. For persistent homology,
an endpoint of a bar at 0 or at infinity has gradient 0, and where a
critical simplex has several longest edges, the gradient is that of the
first one found.

**D17. Evaluation stays in PyTorch.** Evaluating a `Pcf`, `Plf`, or
`PcfTensor` gives a framework array when the function carries gradients or
the times are a framework array. For PCFs, C++ evaluates PCFs with the same
times whose values number their breakpoints, and the result gathers the
tracked values of those breakpoints: there is no loop over the PCFs, and no
gradient with respect to times, since a PCF is constant between its
breakpoints. For a PLF, the interpolation runs in the framework, so the
result has gradients with respect to the times and values of the PLF and to
where it is evaluated.

**D10. Gradient paths require tensors.** With gradients, `transform_bars`
takes a `BarcodeTensor` only, since a `Barcode` cannot carry gradients and
its typing overload promises a `Barcode` result. Wrap a single barcode as
`BarcodeTensor([bc])`.

**D11. The first time of a PLF gets a gradient.** A PLF's first breakpoint
time must be 0, but its gradient is reported anyway; a user optimizing the
times masks it.

**D16. Persistent homology differentiates the critical edges.** In a
Vietoris-Rips complex a simplex appears at the length of its longest edge,
so each endpoint of a bar is the distance between two points. Ripser
records, on request, the longest edges of the birth and death simplices of
each bar; the barcodes of a framework array of shape (n, d) are computed
with them, and the bars are the lengths of those edges again as framework
operations on the points. Without the request, Ripser runs as before.
Batches of point clouds, for persistent homology and for the kernel, are
described in D8a of [linear-subspaces-spec.md](linear-subspaces-spec.md).

A `DistanceMatrixTensor` of a PyTorch tensor of shape (..., n, n) that
carries gradients is tracked: its content is the entries below the diagonal
of every matrix, gathered from the tensor, so the entries on and above the
diagonal get gradient 0. Persistent homology and the kernel of such tensors
take each endpoint of a bar from the entry of its critical edge. A matrix
of shape (n, n) is a tensor of one matrix, as a tracked tensor has at least
one axis. A PyTorch tensor without gradients is read as an array, as
before.

Rejected: finding the pair of points of each endpoint by searching for its
value among the pairwise distances, which needs no change to Ripser but
takes the first pair at that distance when distances are equal, which need
not be an edge of the critical simplex, and sorts all pairwise distances.

## Cost

**D13. The cost rule.** A library call on a tracked tensor may make a fixed
number of passes over its data and does no work per element in Python.
Reading, writing, or indexing costs in proportion to what it touches, not to
the whole tensor. In particular:

- A vjp never builds a dense Jacobian; it scatters into its gradients.
- A backward pass reuses what the forward pass computed instead of
  recomputing it.
- Conversions between C++ tensors and serial content run in parallel in
  C++, without temporaries per element, and a tracked object keeps its C++
  tensor (D5), so an operation converts once, for its result.

Measured on 2000 barcodes of 500 bars with a PLF of 50 breakpoints
(PyTorch, CPU, a Debug build, before these changes): `transform_bars`,
stable rank, and `lp_norm` took 0.04 s untracked and 0.58 s tracked. About 60% of the
tracked time was serial conversion, 65 to 70 ms each and single-threaded,
and the sort of the tracked stable rank took 0.14 s. The backward pass was
not run: the PLF vjp built a dense bars-by-breakpoints matrix, which would
need 1.6 GB. PCF arithmetic with a tracked operand ran a Python loop at
about 13 microseconds per PCF, 20 times slower than untracked.

These numbers are from an unoptimized build and overstate the C++ costs.
In a Release build after items C1 to C4 of the plan, the same pipeline
takes 13 to 15 ms untracked, 180 to 200 ms tracked forward, and 360 to 380
ms forward and backward; a conversion to or from serial content takes
under 1 ms; and `x * F - G` on 10^5 PCFs of 20 points takes 116 ms tracked
against 21 ms for `F - G` untracked. Take timings in a Release build
(`cmake-build-release`).

After the cleanup for PyTorch only, in a Release build on the CPU with the
default number of threads (8 performance and 8 efficiency cores), best of
three runs:

| Pipeline | Untracked | Tracked forward | Forward and backward |
| --- | --- | --- | --- |
| `transform_bars`, stable rank, `lp_norm`; 2000 barcodes of 500 bars, a PLF of 50 breakpoints | 15 ms | 45 ms | 239 ms |
| Persistent homology, dimensions 0 and 1; 200 clouds of 100 points | 12 ms | 20 ms | 22 ms |
| Kernel of a batch and its projection; 200 clouds of 200 points | 8 ms | 15 ms | 18 ms |
| Projection; 2000 clouds of 500 points | 1.5 ms | 5.4 ms | 13.5 ms |

A tracked forward pass costs 2 to 4 times the untracked one. The backward
pass of the first pipeline stands out: the vjp of `transform_bars` and the
sort of the stable rank handle 10^6 bars in NumPy and PyTorch on one thread.

Known limits: every operation passes through host memory, since the C++
code takes host arrays, which costs a transfer per operation for arrays on
a GPU.

**D14. Numeric results with gradients are framework arrays.** `lp_norm` of
tracked PCFs returns a framework array, so `loss.backward()` works
directly, and operations that take a `FloatTensor` also take a framework
array. A dense framework array already is the tracked form
of a `FloatTensor`; wrapping it would add a step for the user without
adding information. This is an exception to returning stablebear tensors
from public functions.

Arithmetic and comparisons of a `FloatTensor` or `IntTensor` with a
framework array, in either order, give a framework array as well: the
tensor's values enter the operation as an array of the dtype and on the
device of the other operand. So the norms of plain PCFs can be divided by a
tracked scale. In-place arithmetic with a framework array raises, because a
`FloatTensor` cannot hold the gradients, and so does arithmetic of a tensor
without a dense numeric form, such as a power of a `PcfTensor`. Before
this, a framework array of one element was read as a number, which dropped
its gradient without a word, and a larger one raised.

`pdist` and `l2_kernel` of tracked PCFs likewise return the (n, n) matrix
as a framework array, not a `DistanceMatrix` or `SymmetricMatrix`. Their
vjp accumulates, in C++ and in parallel over the PCFs, the gradient of each
PCF over the rectangles of its pair with every other, without building the
difference of a pair. Each pair is therefore iterated twice, once per
function. `cdist` has no gradients yet and raises.

**D15. One framework operation per stablebear operation.** Each tracked
operation is its own node in the framework's graph, so the framework
schedules the backward pass and accumulates gradients of shared
intermediate results once.

Rejected: keeping all values in the C++ tensors, recording our own list of
operations, and giving the framework one operation for a whole pipeline.
It removes every conversion and makes writes cheap, but a tracked tensor
that feeds many separate results (a loop of 1000 norms) is differentiated
once per result, and all operations so far would be rewritten.

## Open questions

- Should `Barcode`, `Pcf`, and `Plf` element objects returned by indexing a
  tracked tensor carry gradients too, as 0-d tracked tensors?
- Should `compute_homological_kernel` take batches of tracked point
  clouds?
