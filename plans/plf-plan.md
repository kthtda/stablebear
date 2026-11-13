# Piecewise linear functions with evaluation: implementation plan

Progress: 2 of 11 items, 5 of 22 points (23%)

Spec: [plf-spec.md](plf-spec.md) (#253)

## Phase A: C++ `Plf` and integration

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
- [ ] **A2.** [2] Add `integrate(const Pcf&)` returning a `Plf`.
  Tests: breakpoints hold the running integral; a last value different
  from the one before adds a breakpoint at `t_n + max(1, t_n)`, both
  below and above `t_n = 1`; an equal one adds none; a single-breakpoint
  `Pcf` gives a line through 0; a repeated `Pcf` time gives one
  breakpoint.

## Phase B: Python `Plf`

- [ ] **B1.** [2] Bind `Plf_f32_f32` and `Plf_f64_f64` in
  `src/python/functional/py_plf.cpp` (constructor from an array, `size`,
  array view, `get_time_type`), and add `plf32`/`plf64` and
  `stablebear/functional/plf.py` with the `Plf` class and its docstring,
  without `__call__`.
  Tests: array and list give the expected breakpoints; `to_numpy` round
  trip; dtype inference from `float32`, `float64`, and an integer array;
  `dtype=` overrides; copy; invalid breakpoints raise `ValueError`;
  `astype`; `==`.
- [ ] **B2.** [2] `Plf.__call__` at scalars, arrays, lists, and
  `FloatTensor`s, with the scalar and sorted-array bindings.
  Tests: values at and between breakpoints, and after the last with a
  sloped and a flat last segment; 2-d and unsorted arrays keep their
  shape and order; list; `FloatTensor` of each precision; negative time raises.
- [ ] **B3.** [1] `Pcf.integrate()`, binding A2.
  Tests: result equals the `Plf` built from the expected breakpoints;
  `pcf32` gives `plf32`; an integer `Pcf` raises `TypeError`.

## Phase C: `PlfTensor`

- [ ] **C1.** [3] Bind `Plf32Tensor` and `Plf64Tensor` in
  `src/python/py_tensor_plf.cpp`; add `PlfTensor` and the `zeros` and
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
