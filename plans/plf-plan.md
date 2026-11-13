# Piecewise linear functions with evaluation: implementation plan

Progress: 3 of 13 items, 7 of 29 points (24%)

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
- [ ] **B2.** [2] `Plf.__call__` at scalars, arrays, lists, and
  `FloatTensor`s, with the scalar and sorted-array bindings.
  Tests: values at and between breakpoints, and after the last with a
  sloped and a flat last segment; 2-d and unsorted arrays keep their
  shape and order; list; `FloatTensor` of each precision; negative time raises.
- [ ] **B3.** [3] Add `integrate(const Pcf&)` returning a `Plf` in
  `include/sbear/functional/plf.hpp`, and bind it as `Pcf.integrate()`.
  It is tested from Python, where the breakpoints are easier to read.
  Tests: breakpoints hold the running integral; a last value different
  from the one before adds a breakpoint at `t_n + max(1, t_n)`, both
  below and above `t_n = 1`; an equal one adds none; a single-breakpoint
  `Pcf` gives a line through 0; a repeated `Pcf` time gives one
  breakpoint; `pcf32` gives a `float32` `Plf`; an integer `Pcf` raises
  `TypeError`.

## Phase C: `PlfTensor`

- [ ] **C1.** [3] Bind `Plf32Tensor` and `Plf64Tensor` in
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
- [ ] **C2.** [2] Evaluate `PlfTensor`s: bind `__call__` for every
  evaluable element type instead of only `PcfLike` ones in
  `src/python/py_tensor.hpp`.
  Tests: scalar on 1-d and 2-d tensors; array of times on a 1-d and 2-d
  tensor gives shape `T.shape + t.shape`; list; `FloatTensor`; the
  parallel path agrees with the serial one.
- [ ] **C3.** [2] `PcfTensor.integrate()`, elementwise and parallel,
  returning a `PlfTensor`.
  Tests: each element equals that `Pcf`'s `integrate()`; shape and
  precision are kept; an empty tensor gives an empty `PlfTensor`.

## Phase D: Documentation

- [ ] **D1.** [1] API reference: `stablebear.functional.plf` in
  `docs/api_base.rst`, `Plf` and `PlfTensor` aliases in `docs/conf.py`,
  and `assert_type`s for `Plf.__call__` and `Pcf.integrate` in
  `test/typing/check_types.py`.
- [ ] **D2.** [2] User guide: a PLF section in `docs/concepts.rst`
  covering how a PLF continues its last segment and integrating a `Pcf`,
  the dtype tables there and in `docs/tensors.rst`, PLF evaluation in
  `docs/evaluation.rst`, and a changelog entry.

## Phase E: Saving, loading, and pickling

- [ ] **E1.** [3] Save, load, and pickle a `Plf`: `write_element` and
  `read_element` in `include/sbear/io/plf_io.hpp`, base format 110 in
  `tensorFormatV3`, `Plf_f32` and `Plf_f64` in `StreamableObject`,
  `save_plf32_object` and `save_plf64_object` in `py_io.cpp`, the maps in
  `stablebear/io.py`, and `_BinaryIoMixin` on `Plf`. Add `Plf` to the
  types listed in `docs/saving.rst`.
  Tests: `save` and `load` give back an equal `Plf` of each precision;
  pickle gives back an equal `Plf`.
- [ ] **E2.** [2] Save, load, and pickle a `PlfTensor`: `Tensor<Plf_f32>`
  and `Tensor<Plf_f64>` in `StreamableTensor`, `save_plf32_tensor` and
  `save_plf64_tensor`, the maps in `stablebear/io.py`, and
  `_BinaryIoMixin` on `PlfTensor`. Add `PlfTensor` to `docs/saving.rst`.
  Tests: `save` and `load` give back an equal 2-d tensor of each
  precision; an empty tensor; pickle.
- [ ] **E3.** [2] Pin a V3 PLF corpus with
  `test/golden/serialization/generate_plf_v3.py`, through
  `pinned_corpus.py`: a `Plf` and a 2-d `PlfTensor` at each precision,
  binary and pickle, with explicit expected breakpoints. Describe it in
  the serialization golden README.
  Tests: every file loads to its expected breakpoints.
