# Piecewise linear functions with evaluation: spec

Issue: [#253](https://github.com/kthtda/stablebear/issues/253)

Plan: [plf-plan.md](plf-plan.md)

## Goal

Stablebear represents functions only as piecewise constant functions
(`Pcf`). Many objects in TDA are continuous, and a step function only
approximates them. Add a type for piecewise linear functions (PLFs) that
represents them exactly, evaluate it at points, singly and in tensors, and
form one as the integral of a `Pcf`. Arithmetic, reductions, distances,
and the CUDA backend come later.

## The function

A PLF is given by breakpoints
$(t_0, v_0), (t_1, v_1), \ldots, (t_n, v_n)$ with $t_0 = 0$ and
$t_0 < t_1 < \cdots < t_n$. It is defined on $[0, \infty)$, as a `Pcf`
is:

- on $[t_i, t_{i+1}]$ it interpolates linearly between $v_i$ and
  $v_{i+1}$;
- on $[t_n, \infty)$ it continues its last segment, with slope
  $(v_n - v_{n-1}) / (t_n - t_{n-1})$.

The breakpoints are the whole function: no other value is stored. In a
`Pcf` the last value extends to infinity; in a PLF the last segment does.
A PLF that should stay constant after its data ends needs a final flat
segment, such as a repeated last value at a later time. A single
breakpoint $(0, v_0)$ gives the constant function $v_0$.

A PLF is continuous: times are strictly increasing, so it cannot jump.

## C++: `Plf<Tt, Tv>`

- `include/sbear/functional/plf.hpp`, with aliases `Plf_f32` and `Plf_f64`.
  Only floating-point types are supported.
- It stores a `std::vector<TimePoint<Tt, Tv>>`, as `Pcf` does, and has
  `points()`, `size()`, `==`, and `!=`.
- The default constructor gives the zero function, the single breakpoint
  $(0, 0)$, so tensors of PLFs can be zero-filled.
- Constructing from breakpoints validates them and throws
  `std::invalid_argument` (a `ValueError` in Python) if there are none,
  if the first time is not 0, if the times are not strictly increasing, or
  if a time is not finite. Values are not checked.
- `evaluate(t)` returns the value at `t`, and throws `std::domain_error`
  for a negative `t`. `evaluate(sorted_times, out, n)` evaluates at sorted
  times in one pass, with the same signature as for `Pcf`, so that
  `tensor_eval` works for both.
- Between breakpoints it computes
  $v_i + (v_{i+1} - v_i) \cdot \frac{t - t_i}{t_{i+1} - t_i}$,
  which gives exactly $v_i$ at $t = t_i$. After $t_n$ it uses the same
  formula with $i = n - 1$.
- `integrate(const Pcf<Tt, Tv>&)` returns the `Plf<Tt, Tv>`
  $F(t) = \int_0^t f(u)\,du$, for float `Pcf`s. Its breakpoints are those of
  the `Pcf`, with $F(t_0) = 0$ and
  $F(t_{i+1}) = F(t_i) + v_i \cdot (t_{i+1} - t_i)$. After $t_n$, $F$
  grows with the `Pcf`'s last value $v_n$, but its last segment has slope
  $v_{n-1}$ (0 for a single breakpoint). When the two differ, the `Plf`
  gets one more breakpoint, at time $t_n + \max(1, t_n)$ with value
  $F(t_n) + v_n \cdot \max(1, t_n)$, so that its last segment has slope
  $v_n$. Stepping by $\max(1, t_n)$ rather than 1 keeps the new time
  distinct from $t_n$ in float32 at large times. A `Pcf` may repeat a
  time, which gives an interval of length zero; the `Plf` keeps one
  breakpoint for it.
- `Plf` has no arithmetic. The scalar operators in
  `piecewise_function.hpp` are constrained to types that have the compound
  operator they call, so that a `Plf` does not appear to support them.

## Python: `sb.Plf`

`Plf(arr, dtype=None)` in `stablebear/functional/plf.py`, exported as
`sb.Plf`, mirrors `Pcf`:

- `arr` is an `(n, 2)` ndarray or a list of `(time, value)` pairs, or a
  `Plf` to copy. `Plf(f.to_numpy())` gives back `f`.
- `dtype` is `plf32` or `plf64`. Without it, a `float32` array gives
  `plf32`, any other real array gives `plf64`, and a list gives `plf64`.
- Invalid breakpoints raise `ValueError`, as above.
- `size`, `to_numpy()`, `__array__`, `__eq__`, `astype(dtype)`, and `str`
  behave as for `Pcf`.
- `f(t)` follows `Pcf.__call__`: a scalar gives a Python float, a list or
  ndarray gives an ndarray of the same shape, and a `FloatTensor` gives a
  `FloatTensor` of the same type and shape. A negative time raises
  `ValueError`.
- The docstring states prominently that the function extrapolates its
  last segment, and how to make it constant instead.

```python
f = sb.Plf(np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]]))
f(0.5)          # 1.0
f([0.5, 2.0])   # array([1. , 1.5])
f(5.0)          # 0.0, continuing the slope -0.5 of the last segment

g = sb.Plf([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 1.0]])
g(5.0)          # 1.0, after a final flat segment
```

## Integration: `Pcf.integrate()` and `PcfTensor.integrate()`

- `Pcf.integrate()` returns the `Plf` $t \mapsto \int_0^t f(u)\,du$, of
  the same precision: `pcf32` gives `plf32`, and `pcf64` gives `plf64`.
  An integer `Pcf` raises `TypeError`.

  ```python
  f = sb.Pcf([[0.0, 2.0], [1.0, -1.0], [3.0, 0.5]])
  F = f.integrate()
  F.to_numpy()    # [[0, 0], [1, 2], [3, 0], [6, 1.5]]
  ```

  The last breakpoint, at $3 + \max(1, 3) = 6$, gives the tail its slope
  0.5; without it, the segment from 1 to 3 would continue with slope -1.

- `PcfTensor.integrate()` integrates every element and returns a
  `PlfTensor` of the same shape and precision, computed in parallel on the
  CPU.

## Python: `sb.PlfTensor`

- Dtypes `plf32` and `plf64` in `stablebear/typing.py`.
- `PlfTensor` in `stablebear/base_tensor.py` is a `FunctionTensor`: it has
  the generic tensor operations (indexing, assignment, views, copying,
  stacking, equality) but no arithmetic.
- `sb.zeros(shape, dtype=plf32)` fills it with the zero function.
  `sb.tensor([...])` of `Plf`s infers its dtype, and `PlfTensor([...])`
  builds one from a nested list of `Plf`s.
- Evaluation follows `PcfTensor`: `T(t)` has shape `T.shape + t.shape`,
  for a scalar, an ndarray, a list, or a `FloatTensor`.

## Out of scope

Tracked in follow-up issues:

- arithmetic and composition of PLFs;
- strictly increasing PLFs: monotonicity and inverses;
- reductions, Lp distances, and L2 kernels;
- CUDA support (PLFs work on the CPU in both backends' modules);
- saving and loading, pickling, casting tensors between `plf32` and
  `plf64`, and time series of PLFs;
- integrating integer `Pcf`s, and integrating from a time other than 0.
