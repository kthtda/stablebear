# Backlog crash and wrong-result bugs: implementation plan

Progress: 6 of 7 items, 10 of 11 points (91%)

Spec: the GitHub issues linked from each item. Branch:
`fix/backlog-crash-bugs`.

#173 (`_gpu_detect` import path) is left out. On Linux the C++ detector reads
only `/sys/class/drm`, while the Python fallback tries `lspci` first, so
enabling it could stop GPU detection on hosts without NVIDIA DRM nodes.

## Phase A: No crashes on empty input or a zero Poisson mean

- [x] **A1.** [1] (9d7a01d0f) #186: `subdivide()` in
  `include/sbear/algorithms/subdivide.hpp` returns no blocks when
  `nItems == 0`, instead of one block ending at `SIZE_MAX`.
  Tests (`test/test_subdivide.cpp`): zero items gives no blocks, for block
  sizes 1 and 5.
- [x] **A2.** [2] (27ef00f70) #186: `parallel_reduce()` in
  `include/sbear/algorithms/functional/reduce.hpp` returns the zero function
  for an empty range, and `average()` in `include/sbear/functional/pcf.hpp`
  returns it for an empty vector instead of dividing by zero.
  Tests (`test/test_parallel_reduce.cpp`): `parallel_reduce` over an empty
  vector gives the zero function; `average` of an empty vector gives the
  zero function.
- [x] **A3.** [2] (a4e12cccd) #192: evaluating a tensor at an empty time
  array returns an empty result of shape `elems.shape() + (0,)` instead of
  reading `domain_indices[0]` in `include/sbear/algorithms/tensor_eval.hpp`.
  Tests (`test/python/test_pcf_tensor_eval.py`): a `(2,)` PcfTensor at
  `np.array([])` gives shape `(2, 0)`, for pcf32 and pcf64; a `(2, 2)`
  PcfTensor gives shape `(2, 2, 0)`.
- [x] **A4.** [1] (db98ad791) #36: `sample_poisson` in
  `include/sbear/point_process/poisson.hpp` requires a positive rate,
  `lo < hi` in every dimension, and a positive `rate * volume`, so
  `std::poisson_distribution` never gets a mean of 0, which aborts in builds
  with libstdc++ assertions. This replaces the `rate >= 0` and `lo <= hi`
  rules from #118.
  Tests (`test/python/test_bugscan_pointprocess.py`): `rate=0` raises;
  `lo == hi` and `lo > hi` in one dimension raise; a box whose volume
  underflows to 0 raises.

## Phase B: NumPy-consistent truth values and comparisons

- [x] **B1.** [2] (c742398bd) #45: `FloatTensor` and `IntTensor` get the
  `__bool__` that `BoolTensor` has in `stablebear/base_tensor.py`: a single
  element gives its truth value, and any other size raises `ValueError`.
  Share one function between `NumericTensor` and `BoolTensor`.
  Tests (`test/python/test_tensor.py`): a one-element `0` is `False` and `2`
  is `True`, for FloatTensor and IntTensor; a 0-d tensor gives its value; two
  elements raise `ValueError`; zero elements raise `ValueError`.
- [x] **B2.** [2] (1b0002ad2) #51: both `allclose` overloads in
  `include/sbear/tensor.tpp` use one element check, `detail::is_close`, that
  compares infinities with `==`, so equal infinities are close and opposite
  infinities are not. The matrix overload also stops treating NaN as close.
  Tests (`test/python/test_allclose.py`): `[1, inf, -inf]` is close to a
  copy; `[inf]` is not close to `[-inf]`; `[inf]` is not close to `[1e300]`;
  the same for `DistanceMatrix` (no negative entries) and `SymmetricMatrix`;
  a `SymmetricMatrix` with NaN is not close to itself.
- [ ] **B3.** [1] #183: `Barcode()` in `stablebear/persistence/barcode.py`
  accepts any real-valued array-like input: a `float32` array gives a
  `barcode32`, and anything else, including lists and integer arrays, is
  converted to a `barcode64`. Input that is not real-valued raises
  `TypeError` instead of the `AttributeError` from reading `self._data`
  before it was set.
  Tests (`test/python/persistence/test_persistence_edge_cases.py`): a list
  gives a `barcode64` with its bars; an `int64` array gives a `barcode64`;
  `[]` gives an empty barcode; a `float32` array gives a `barcode32`; strings
  raise `TypeError`.
