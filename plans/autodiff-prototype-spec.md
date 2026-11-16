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

**D5. One source of truth.** A tracked object is an instance of a private
subclass (such as `_TrackedBarcodeTensor`) whose only state is the
framework arrays and the enumeration. Its C++ object (`_data`) is built from
them on each use and never stored, so the two cannot disagree. There is no
cache, since an in-place framework operation on the arrays would make it
stale; the conversion is O(m). A tracked object cannot be changed in place.
Indexing, reshaping, and copying keep the gradients; any other operation
that would build a new object without them raises, instead of dropping them
silently.

The untracked classes are unchanged: `_data` stays a plain attribute.

Rejected: storing both a C++ object and framework arrays on one object,
which the first versions did. Assigning to an element of a tracked
`BarcodeTensor` changed the C++ bars but not the framework bars, and later
results used the stale ones. Also rejected: one class whose `_data` property
checks whether the object is tracked, which adds a Python call to every
`_data` access on hot paths.

Status: done for barcodes; PCFs and PLFs still store both (see the plan).

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

## Open questions

- Should `Barcode`, `Pcf`, and `Plf` element objects returned by indexing a
  tracked tensor carry gradients too, as 0-d tracked tensors?
- Which other barcode summaries need gradients: Betti curves and
  accumulated persistence functions have none yet.
- Should `compute_homological_kernel` take batches of tracked point
  clouds?
