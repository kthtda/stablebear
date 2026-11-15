# Piecewise linear functions with evaluation: implementation plan

Progress: 15 of 17 items, 34 of 38 points (89%)

Spec: [plf-spec.md](plf-spec.md) (#253)

## Phase A: C++ `Plf`

- [x] **A1.** Add `Plf<Tt, Tv>`.
  - [x] **A1a.** [2] (1b161138e) Make `Pcf`'s sorted-times `evaluate` take
    `std::span<const time_type>` and `std::span<value_type>` instead of
    two indexable objects and a count, and pass vectors to it from
    `tensor_eval` and `py_pcf.cpp`. The existing Python evaluation tests
    cover it.
  - [x] **A1b.** [3] (7318d3057) Add `Plf<Tt, Tv>` in
    `include/sbear/functional/plf.hpp`: validating constructor, zero
    default, both `evaluate` overloads, `==`. Add it to `SB_HEADER_FILES`.
    Tests: no breakpoints, nonzero first time, repeated time, decreasing
    time, and infinite time throw; value at each breakpoint; value halfway
    between; after the last breakpoint it continues the last segment; a
    final flat segment keeps it constant; a single breakpoint is constant;
    negative time throws; sorted-times `evaluate` matches scalar
    `evaluate`.

## Phase B: Python `Plf`

- [x] **B1.** [2] (2089e44fc) Bind `Plf_f32_f32` and `Plf_f64_f64` in
  `src/python/functional/py_plf.cpp` (constructor from an array, `size`,
  array view), and add `stablebear/functional/plf.py` with the `Plf`
  class and its docstring, without `__call__`.
  Tests: array and list give the expected breakpoints; `to_numpy` round
  trip; dtype inference from `float32`, `float64`, and an integer array;
  `dtype=` overrides; copy; invalid breakpoints raise `ValueError`;
  `astype`; `==`.
- [x] **B2.** Evaluate a `Plf`.
  - [x] **B2a.** [2] (3d3cbdd90) Move `__call__`, `__array__`, and `__eq__` from
    `Pcf` into `_PiecewiseFunctionMixin` in
    `stablebear/functional/_piecewise.py`, and the array evaluation in
    `py_pcf.cpp` into a template in `pypcf_support.hpp`. `__call__`
    accepts any `NumericTensor`, and returns a tensor of the function's
    value type.
    Tests: `IntTensor` times on a float `Pcf` give a `FloatTensor`;
    `FloatTensor` times on an integer `Pcf` give an `IntTensor`.
    Note: `size` and `to_numpy` stay in each class, since the docs list
    inherited public methods only in a dropdown.
  - [x] **B2b.** [3] (b91cb9888) `Plf.__call__` at scalars, arrays, lists, and
    `FloatTensor`s: rename the mixin to the base class
    `_PiecewiseFunction` and derive `Plf` from it, and bind what `Pcf`
    and `Plf` share, evaluation among it, in `bind_piecewise_function`.
    Evaluation reads arrays through their strides.
    Tests: values at and between breakpoints, and after the last with a
    sloped and a flat last segment; 2-d and unsorted arrays keep their
    shape and order; transposed and reversed arrays; list; `FloatTensor`
    of each precision; negative time raises.
- [x] **B3.** Integrate a `Pcf` into a `Plf`.
  - [x] **B3a.** [2] (5f0f7b9b6) Constructing a `Pcf` requires strictly increasing
    times, in `construct_pcf` in `pypcf_support.hpp`, with the `Pcf`
    docstring, `docs/concepts.rst`, and a changelog entry.
    Tests: a repeated time raises `ValueError` for a float and an integer
    `Pcf`; existing tests that built `Pcf`s with repeated times use
    distinct ones.
  - [x] **B3b.** [3] (f0bde11aa) Add `integrate(const Pcf&)` returning a `Plf` in
    `include/sbear/functional/plf.hpp`, and bind it as `Pcf.integrate()`.
    It is tested from Python, where the breakpoints are easier to read.
    Tests: breakpoints hold the running integral; a last value different
    from the one before adds a breakpoint at `t_n + max(1, t_n)`, both
    below and above `t_n = 1`; an equal one adds none; a
    single-breakpoint `Pcf` gives a line through 0; `pcf32` gives a
    `float32` `Plf`; an integer `Pcf` raises `TypeError`.

## Phase C: `PlfTensor`

- [x] **C1.** [3] (dc0022579) Bind `Plf32Tensor` and `Plf64Tensor` in
  `src/python/py_tensor_plf.cpp`; add the `plf32` and `plf64` dtypes,
  `PlfTensor`, and the `zeros` and
  `tensor` dispatch. Constrain the scalar operators in
  `include/sbear/functional/piecewise_function.hpp` to types with the
  compound operator each one calls: `CanAddTo` and its siblings only check
  the signature, so without this the tensor bindings would instantiate
  arithmetic for `Plf` and fail to compile. The build checks it.
  Tests: `zeros` holds zero functions; set and get an element; `tensor`
  of `Plf`s infers the dtype; `PlfTensor` from a nested list; `copy` keeps
  the dtype.
- [x] **C2.** [2] (702027839) Evaluate `PlfTensor`s: bind `__call__` for every
  evaluable element type instead of only `PcfLike` ones in
  `src/python/py_tensor.hpp`.
  Tests: scalar on 1-d and 2-d tensors; array of times on a 1-d and 2-d
  tensor gives shape `T.shape + t.shape`; list; `FloatTensor`; the
  parallel path agrees with the serial one.
- [x] **C3.** [2] (4ae017c99) `PcfTensor.integrate()`, elementwise and parallel,
  returning a `PlfTensor`.
  Tests: each element equals that `Pcf`'s `integrate()`; shape and
  precision are kept; an empty tensor gives an empty `PlfTensor`.

## Phase D: Documentation

- [x] **D1.** [1] (e856f4a17) API reference: `stablebear.functional.plf` in
  `docs/api_base.rst`, `Plf` and `PlfTensor` aliases in `docs/conf.py`,
  and `assert_type`s for `Plf.__call__` and `Pcf.integrate` in
  `test/typing/check_types.py`.
- [x] **D2.** [2] (985baf176) User guide: a PLF section in `docs/concepts.rst`
  covering how a PLF continues its last segment and integrating a `Pcf`,
  the dtype tables there and in `docs/tensors.rst`, PLF evaluation in
  `docs/evaluation.rst`, and a changelog entry.

## Phase E: Saving, loading, and pickling

- [x] **E1.** [3] (3554ddfc1) Save, load, and pickle a `Plf`: `write_element` and
  `read_element` in `include/sbear/io/plf_io.hpp`, base format 110 in
  `tensorFormatV3`, `Plf_f32` and `Plf_f64` in `StreamableObject`,
  `save_plf32_object` and `save_plf64_object` in `py_io.cpp`, the maps in
  `stablebear/io.py`, and `_BinaryIoMixin` on `Plf`. Add `Plf` to the
  types listed in `docs/saving.rst`.
  Tests: `save` and `load` give back an equal `Plf` of each precision;
  pickle gives back an equal `Plf`.
- [x] **E2.** [2] (f9faaeb9f) Save, load, and pickle a `PlfTensor`: `Tensor<Plf_f32>`
  and `Tensor<Plf_f64>` in `StreamableTensor`, `save_plf32_tensor` and
  `save_plf64_tensor`, the maps in `stablebear/io.py`, and
  `_BinaryIoMixin` on `PlfTensor`. Add `PlfTensor` to `docs/saving.rst`.
  Tests: `save` and `load` give back an equal 2-d tensor of each
  precision; an empty tensor; pickle.
- [x] **E3.** [2] (42dcb86ec) Pin a V3 PLF corpus with
  `test/golden/serialization/generate_plf_v3.py`, through
  `pinned_corpus.py`: a `Plf` and a 2-d `PlfTensor` at each precision,
  binary and pickle, with explicit expected breakpoints. Describe it in
  the serialization golden README.
  Tests: every file loads to its expected breakpoints.

## Phase F: Plotting

- [ ] **F1.** [2] `plotting.plot` draws a `Plf`, a list or tuple of
  `Plf`s, and a 1-d `PlfTensor` as lines through their breakpoints, and
  `max_time` extends them along their last segment; update its docstring.
  Tests: a single `Plf` draws its breakpoints; `max_time` adds the point
  at `max_time` for a sloped and a flat last segment; a tensor extends
  every PLF to the latest breakpoint; a list gives the same lines as the
  `PlfTensor`; a list mixing `Pcf`s and `Plf`s raises `TypeError`.
- [ ] **F2.** [2] Plotting guide: a PLF section in `docs/plotting.rst`
  with a figure of a `Pcf` and its integral, from
  `docs/_static/gen_plotting_gallery.py` with a code dropdown, and the
  changelog entry mentions plotting.
