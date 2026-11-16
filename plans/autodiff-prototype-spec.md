# Autodiff prototype: design and decisions

Plan: [autodiff-prototype-plan.md](autodiff-prototype-plan.md)

The `autodiff-prototype` branch lets gradients from PyTorch, JAX, and
TensorFlow flow through stablebear operations, so that a loss built from
PCFs, PLFs, and barcodes can be optimized with respect to point clouds,
breakpoints, or the parameters of a transformation. This file records the
design and the decisions behind it, with the alternatives that were
rejected. Update it when a decision changes.

## Goals

- A user writes ordinary stablebear code with framework arrays as inputs
  and gets a framework result that carries gradients.
- Code that does not use a framework runs exactly as before, with no extra
  cost.
- Values and gradients are never silently wrong: an operation either
  handles gradients correctly or raises.
- Most of the library can be differentiated, with a bounded overhead over
  the untracked path (see D13).

## Architecture

**D1. The core stays framework-free.** An operation opts in with the
`_autodiff.differentiable` decorator, naming the parameters that can carry
gradients and a vector-Jacobian product (vjp) written in NumPy. A small
adapter per framework runs the operation as a custom op of that framework:
`torch.autograd.Function` for PyTorch, `custom_vjp` with `pure_callback`
for JAX (which also works under `jax.jit`), and `custom_gradient` with
`numpy_function` for TensorFlow. Keras 3 variables are unwrapped to the
variable or array of their backend.

**D2. Tracked arguments describe themselves.** An object that carries
gradients reports its framework arrays, how to rebuild the object from
NumPy values of those arrays, and how to split the vjp's gradient into one
per array (`_tracked_parts`). The decorator uses this to convert tracked
arguments to NumPy, call the operation, and route gradients back.

**D3. Some operations are written in framework operations instead of a
vjp.** PCF arithmetic and the stable rank of tracked barcodes compute their
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
a tracked object is immutable, JAX and TensorFlow arrays cannot change, and
PyTorch's version counter of the array tells when it was changed in place,
in which case the cache is rebuilt. An operation that already has the C++
result hands it to the tracked object as the initial cache. Indexing,
reshaping, and copying keep the gradients; any other operation that would
build a new object without them raises, instead of dropping them silently.

The untracked classes are unchanged: `_data` stays a plain attribute, and
each element lives in its own heap allocation.

Rejected: storing a C++ object and framework arrays as independent state,
which the first versions did. Assigning to an element of a tracked
`BarcodeTensor` changed the C++ bars but not the framework bars, and later
results used the stale ones. Also rejected: one class whose `_data` property
checks whether the object is tracked, which adds a Python call to every
`_data` access on hot paths.

Status: `_TrackedBarcodeTensor` derives `_data` without a cache; PCFs and
PLFs still store both (see the plan).

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
Single objects (`Pcf`, `Plf`, `Barcode`) are the 0-d case. A dense numeric
tensor needs no enumeration: see D14.

**D5b. Writes are deferred.** `x[i] = v` on a tracked tensor records the
replacement in a table of pending writes, at no cost per write, and the next
operation that needs the framework arrays applies all of them in one pass.
Reading an element looks in the table first. A loop of n writes therefore
costs one pass, not n. Basic indexing of a tracked tensor gives a copy, not
a view as for untracked tensors: writing to the result does not change the
tensor it came from.

Rejected: applying each write at once, which copies the whole array per
write in JAX and TensorFlow, and one framework array per element, which
makes writes cheap but needs a Python loop for every batch operation.

**D6. Indexing reuses the tensor indexing rules.** A tracked tensor indexes,
reshapes, or broadcasts an `IntTensor` of its elements' flat indices with
the same arguments, then gathers the rows of the selected elements from the
framework array. The rules are therefore exactly those of untracked
tensors, and repeated elements (as from `broadcast_to`) accumulate
gradients correctly. An integer index that selects one element returns an
untracked element object (`Barcode`, `Pcf`) of its current values, like
reading a number out of an array.

**D7. Native ragged storage in C++ is deferred.** Storing tensors of
barcodes (and PCFs) in C++ as one buffer plus offsets, instead of one heap
vector per element, would make the serial conversion free and let a framework
array share memory with the C++ tensor through DLPack. It would not remove
the need for D5: gradients still live on the framework array, and C++
indexing does not record framework operations. Sharing memory would help
mainly PyTorch on CPU: JAX has no buffers while it traces, `pure_callback`
and `numpy_function` pass copies, and the barcode code is CPU only. It is a
storage refactor of several days, to revisit if profiling shows the copies
or the per-element allocations matter. The `_serial` interface would stay.

## Operations

**D8. Data-dependent shapes raise under abstract tracing.** JAX traces with
abstract values even without `jax.jit` (in `jax.grad`), and so does
`tf.function`. Where the shape of a result depends on values, the operation
raises `NotImplementedError` while tracing:

- PCF arithmetic with tracked times, since the union of times depends on
  their order.
- `transform_bars`, since which bars it removes depends on their values.
  (Chosen over keeping removed bars as zero-length bars, which would give
  the same stable ranks and Betti curves but different barcodes.)
- The stable rank of several tracked barcodes, which sorts with a concrete
  `lexsort`. A single barcode sorts with the framework, so the kernel's
  stable rank works in JAX.

**D9. Derivatives at kinks are one-sided.** At a breakpoint of a PLF, the
derivatives are those of the segment after it. A bar with an infinite
endpoint maps to the limit of the PLF, whose derivative is 1 with respect to
the last value where the PLF ends flat, and 0 otherwise. For the
homological kernel, a zero-length critical edge has gradient 0.

**D10. Gradient paths require tensors.** With gradients, `transform_bars`
takes a `BarcodeTensor` only, since a `Barcode` cannot carry gradients and
its typing overload promises a `Barcode` result. Wrap a single barcode as
`BarcodeTensor([bc])`.

**D11. The first time of a PLF gets a gradient.** A PLF's first breakpoint
time must be 0, but its gradient is reported anyway; a user optimizing the
times masks it.

**D12. TensorFlow gradients are made dense.** A gather after a custom op
gives `IndexedSlices` gradients, which the TensorFlow adapter converts to a
dense tensor before calling the vjp.

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
(PyTorch, CPU, before these changes): `transform_bars`, stable rank, and
`lp_norm` took 0.04 s untracked and 0.58 s tracked. About 60% of the
tracked time was serial conversion, 65 to 70 ms each and single-threaded,
and the sort of the tracked stable rank took 0.14 s. The backward pass was
not run: the PLF vjp built a dense bars-by-breakpoints matrix, which would
need 1.6 GB. PCF arithmetic with a tracked operand ran a Python loop at
about 13 microseconds per PCF, 20 times slower than untracked.

Known limits: every operation passes through host memory, since the C++
code takes host arrays, which costs a transfer per operation for arrays on
a GPU.

**D14. Numeric results with gradients are framework arrays.** `lp_norm` of
tracked PCFs returns a framework array, so `loss.backward()` and
`jax.grad` work directly, and operations that take a `FloatTensor` also
take a framework array. A dense framework array already is the tracked form
of a `FloatTensor`; wrapping it would add a step for the user without
adding information. This is an exception to returning stablebear tensors
from public functions.

**D15. One framework operation per stablebear operation.** Each tracked
operation is its own node in the framework's graph, so the framework
schedules the backward pass and accumulates gradients of shared
intermediate results once.

Rejected: keeping all values in the C++ tensors, recording our own list of
operations, and giving the framework one operation for a whole pipeline.
It removes every conversion and makes writes cheap, but a tracked tensor
that feeds many separate results (a loop of 1000 norms) is differentiated
once per result, JAX needs a deferred-execution layer because it has no
values while tracing, and all operations so far would be rewritten.

## Open questions

- Should `Barcode`, `Pcf`, and `Plf` element objects returned by indexing a
  tracked tensor carry gradients too, as 0-d tracked tensors?
- Which other barcode summaries need gradients: Betti curves and
  accumulated persistence functions have none yet.
- Should `compute_homological_kernel` take batches of tracked point
  clouds?
