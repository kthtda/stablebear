## 0.4.4

### Breaking changes

* **`Pcf` no longer sorts breakpoints; unsorted, negative, and offset times are rejected** — `Pcf` construction used to silently sort the supplied `(time, value)` rows and checked the `t=0` requirement on the unsorted input row 0, so out-of-order breakpoints were accepted and a negative time on a later row could be reordered to the front into a malformed PCF that then evaluated at negative times. Breakpoints must now be supplied in non-decreasing time order (out-of-order input raises `ValueError` instead of being silently sorted), and the first time must be 0 — so any input whose first time is not 0, in particular any negative time, raises `ValueError`. `Pcf` evaluation also rejects `t < 0` directly (it previously compared against the first breakpoint rather than 0). ([#11](https://github.com/kthtda/stablebear/issues/11))

### Bug fixes

* **`astype` supports the full reasonable cast matrix** — `astype` now covers the float/signed-int/unsigned-int/bool numeric grid (e.g. `float ↔ uint`, `bool ↔ any numeric` with `True`/`False` ↔ `1`/`0`), which also fixes cross-dtype assignment such as `float_tensor[:] = uint_tensor`. The non-numeric families gain same-family 32↔64 precision casts for `DistanceMatrixTensor`, `SymmetricMatrixTensor`, and `BarcodeTensor` (point-cloud and PCF precision casts already existed). Genuinely undefined casts still raise a clear `TypeError`. ([#50](https://github.com/kthtda/stablebear/issues/50), [#98](https://github.com/kthtda/stablebear/issues/98), [#99](https://github.com/kthtda/stablebear/issues/99), [#95](https://github.com/kthtda/stablebear/issues/95), [#114](https://github.com/kthtda/stablebear/issues/114))
* **A single point cloud is now subscriptable** — a 0-d `PointCloudTensor` (one cloud, e.g. `sb.PointCloudTensor(arr)` from an `(n_points, dim)` array) can be indexed as its underlying array, so the natural plotting idiom `pc[:, 0]` / `pc[:, 1]` works directly instead of raising `IndexError`. Indexing tensors of rank ≥ 1 still selects clouds as before. ([#133](https://github.com/kthtda/stablebear/issues/133))
* **Storing a matrix or point cloud in a tensor cell now copies it (no aliasing)** — assigning a `DistanceMatrix`, `SymmetricMatrix`, or point cloud into a tensor cell (`t[i] = m`) or slice (`t[0:2] = src`) now copies the element instead of sharing its buffer, so later mutation of the source (or assigning one object into several cells) no longer silently corrupts the stored cells. Matches NumPy object-array assignment. ([#39](https://github.com/kthtda/stablebear/issues/39))
* **Self-aliasing slice assignment no longer corrupts data** — assigning an overlapping view of a tensor into itself (`a[:] = a[::-1]`, `a[1:] = a[:-1]`) silently produced wrong results, because the element-wise copy read positions it had already overwritten. The right-hand side is now materialized first when it overlaps the destination (matching NumPy), so `a[:] = a[::-1]` reverses correctly; non-overlapping assignments are unaffected. ([#6](https://github.com/kthtda/stablebear/issues/6))
* **Self-aliasing slice assignment of an object-element tensor keeps cells independent** — extending the overlapping-assignment fix to point-cloud, `DistanceMatrix`, and `SymmetricMatrix` tensors: `t[1:] = t[:-1]` (or `t[:] = t[::-1]`) duplicated only the outer buffer, so the matrix/cloud elements stayed shared with the source cells. The overlap copy now copies each stored element too, so every cell is independent. ([#129](https://github.com/kthtda/stablebear/issues/129))
* **Assigning an array into a `PointCloudTensor` slice distributes one cloud per cell** — `t[:] = arr` (and `t[...]` / partial-index `t[i] = arr`) on a `PointCloudTensor` previously stored the entire array as one cloud broadcast into every selected cell. An ndarray or `FloatTensor` right-hand side is now distributed across the selected cells (leading axes index cells, trailing `(n_points, dim)` axes form each cloud, like `PointCloudTensor(arr)`); a shape that does not match the selected region raises `ValueError`. ([#40](https://github.com/kthtda/stablebear/issues/40))
* **Sampling functions reject degenerate inputs instead of segfaulting** — `point_process.sample_poisson` crashed the interpreter (SIGSEGV) when the Poisson mean `rate * volume` was non-finite (a non-finite or negative `rate`, or non-finite `lo`/`hi`). It now validates that `rate` is finite and non-negative and that `lo`/`hi` are finite, raising a clean `ValueError`. Likewise `random.noisy_sin` / `random.noisy_cos` segfaulted on `n_points=0` and now raise `ValueError`. ([#31](https://github.com/kthtda/stablebear/issues/31), [#32](https://github.com/kthtda/stablebear/issues/32), [#33](https://github.com/kthtda/stablebear/issues/33))
* **`split(t, 0)` raises instead of crashing the interpreter** — `split` with `n_sections=0` performed an integer modulo-by-zero and aborted the process with SIGFPE. It now raises a clean `ValueError`, matching the guard `array_split` already had. ([#19](https://github.com/kthtda/stablebear/issues/19))
* **A broadcast view is now read-only, matching NumPy** — `broadcast_to` returned a writable stride-0 view, so writing through it (`view[0, 0] = x` or in-place `view += 1`) silently scattered the write across the broadcast axis and corrupted the shared source. As in `numpy.broadcast_to`, writing through a broadcast view now raises `ValueError`; the view stays readable and `.copy()` yields a writeable tensor. Size-1 axes from `newaxis`/`expand_dims` are unaffected. ([#16](https://github.com/kthtda/stablebear/issues/16), [#9](https://github.com/kthtda/stablebear/issues/9))
* **Ragged nested lists are rejected instead of silently misplacing elements** — building a tensor from a nested list validated only the first sub-list at each depth, so ragged input whose total element count happened to match a rectangular product was accepted and its elements scattered into the wrong positions. Every branch is now validated against the inferred shape, so any ragged structure raises `ValueError`. ([#4](https://github.com/kthtda/stablebear/issues/4))
* **`==` / `!=` against a scalar return a `BoolTensor`** — comparing a numeric tensor to a scalar (`t == 2.0`) returned a plain Python `bool` via the identity fallback, silently breaking the mask-select idiom `t[t == 2.0]`. `==` and `!=` now broadcast a scalar or `ndarray` right-hand side and return an elementwise `BoolTensor`, consistent with the ordering operators ([#57](https://github.com/kthtda/stablebear/issues/57)) and NumPy. ([#47](https://github.com/kthtda/stablebear/issues/47))
* **Rank-0 (scalar) tensors are now computed instead of silently zeroed** — constructing a numeric or bool tensor from a 0-d NumPy scalar (`FloatTensor(np.array(3.5))`) dropped the value and returned the dtype's zero, and arithmetic whose broadcast result was rank-0 (`zeros(()) + 1.0`, a squeezed size-1 tensor, …) returned `0`. The element walk now visits the single rank-0 element, so 0-d construction preserves the scalar and 0-d arithmetic matches NumPy; genuinely empty tensors (a size-0 dimension) are still skipped. ([#3](https://github.com/kthtda/stablebear/issues/3), [#8](https://github.com/kthtda/stablebear/issues/8))

## 0.4.3

### New features

* **NumPy/list constructors for the non-numeric tensor families** — `PointCloudTensor`, `DistanceMatrixTensor`, and `SymmetricMatrixTensor` can now be built directly from a NumPy array or nested list instead of `zeros()`-and-fill. `PointCloudTensor(arr)` reads a trailing `(n_points, dim)` block per cell (ragged lists accepted); `DistanceMatrixTensor`/`SymmetricMatrixTensor` accept a stack of square matrices. Dtype is inferred (overridable with `dtype=`), and a generic `sb.tensor(data, dtype=...)` factory dispatches to the right type. ([#53](https://github.com/kthtda/stablebear/issues/53), [#92](https://github.com/kthtda/stablebear/issues/92), [#84](https://github.com/kthtda/stablebear/issues/84))

### Breaking changes

* **The `stablebear.tensor` submodule is now `stablebear.base_tensor`** — `stablebear.tensor` is now the `tensor()` factory function (above), so the tensor classes moved. Update module-path imports (`from stablebear.tensor import FloatTensor` → `from stablebear.base_tensor import FloatTensor`); the top-level `sb.FloatTensor` form is unaffected.

### Bug fixes

* **`compute_persistent_homology` reports malformed point clouds** — a point cloud with per-cloud rank ≠ 2 now raises a `RuntimeError` instead of silently returning empty barcodes. ([#27](https://github.com/kthtda/stablebear/issues/27))
* **Reductions accept a negative `dim`** — `mean`/`max_time` now take `dim=-1` (counting from the last axis), matching NumPy; out-of-range raises `IndexError`. ([#26](https://github.com/kthtda/stablebear/issues/26))
* **A `Generator` now advances between draws** — consecutive draws (or draws after a single `random.seed(s)`) are now independent rather than identical, while staying reproducible for a given seed. ([#86](https://github.com/kthtda/stablebear/issues/86))
* **`max_time` and `plotting.plot` handle empty tensors** — `max_time` on a zero-size dimension now raises `ValueError` instead of crashing, and `plotting.plot` becomes a no-op. ([#46](https://github.com/kthtda/stablebear/issues/46))
* **`tensor + ndarray` works** — arithmetic (`+`, `-`, `*`, `/` and in-place variants) with an ndarray/list/tuple on the right now works, matching the already-working reflected form. ([#62](https://github.com/kthtda/stablebear/issues/62))
* **`np.asarray` on a `DistanceMatrix`/`SymmetricMatrix` returns the dense matrix** — `np.asarray`/`np.array` now yield the dense `(n, n)` matrix (plus a `to_numpy()` alias) instead of a 0-d object array. ([#75](https://github.com/kthtda/stablebear/issues/75))
* **Comparing a tensor to a scalar or array works** — `t > 3` (and `<`, `<=`, `>=`) now returns a `BoolTensor`, so `t[t > 3]` works; non-numeric tensors raise a clear `TypeError`. ([#57](https://github.com/kthtda/stablebear/issues/57))
* **`transpose` and `squeeze` accept negative axes** — `transpose(..., -1)` and `squeeze(axis=-1)` now resolve axes against the tensor's rank instead of raising a `TypeError`, matching NumPy and the existing `swapaxes` behavior. ([#17](https://github.com/kthtda/stablebear/issues/17), [#18](https://github.com/kthtda/stablebear/issues/18))

## 0.4.2

* masspcf is now stablebear.

### Bug fixes

* **`flatten()` now exposes correct data through NumPy and `print`** — `np.asarray(t.flatten())`, `print(t.flatten())`/`repr`, and operations chained after `flatten` (e.g. `expand_dims`, `transpose`) returned the first element repeated, because the flattened axis was given a stride of `0`. The flattened axis now has stride `1`, so the NumPy/buffer view reads the correct row-major data. Element indexing (`f[i]`) was already correct; the two paths now agree. ([#15](https://github.com/kthtda/stablebear/issues/15))
* **Out-of-range reduction axis raises instead of corrupting memory** — `mean` and `max_time` with a `dim` outside the tensor's rank silently returned a wrong-shaped result (`dim == ndim`) or corrupted the heap and aborted the interpreter (`dim > ndim`). The dimension is now validated at the C++ root cause, raising a catchable `IndexError`. ([#24](https://github.com/kthtda/stablebear/issues/24))
* **`l2_kernel` and `pdist` no longer crash on negative-step views** — passing a PCF view with a negative step (e.g. `X[::-1]`) segfaulted, because the internal 1-D value iterator stored its stride as an unsigned `size_t` and a negative stride wrapped to a huge value. The iterator now tracks a signed logical position, so reversed and strided views integrate correctly. ([#23](https://github.com/kthtda/stablebear/issues/23))
* **`from_serial_content` validates enumeration bounds** — an enumeration entry whose `stop` exceeded the content length, or whose `start` was negative, read out of bounds: small overflows returned uninitialized heap memory and large ones segfaulted, both silently. Each entry is now validated as `0 <= start < stop <= len(content)` and raises `ValueError` otherwise, matching the existing `start >= stop` check. ([#37](https://github.com/kthtda/stablebear/issues/37))

### Performance

* **Reductions read input slices in place** — `mean` and `max_time` no longer deep-copy every PCF into a temporary vector before reducing; they iterate each slice in place via the value iterator, cutting allocation on large reductions.

### Persistence

* `barcode_to_accumulated_persistence` no longer prints progress by default (`verbose` default changed from `True` to `False`). ([#30](https://github.com/kthtda/stablebear/issues/30))

## 0.4.1

### Persistence

* **`Barcode.is_isomorphic_to` tolerates floating-point noise** — bar endpoints are now compared with a numerical tolerance (`atol=1e-8`, `rtol=1e-5` by default; `abs(a - b) <= atol + rtol * abs(b)`, with infinite endpoints matched exactly). This makes barcodes computed via different but mathematically equivalent routes — for example, from a point cloud versus a precomputed distance matrix — compare equal despite low-order rounding differences. Pass `atol=0, rtol=0` for the previous bitwise comparison.

### Performance

* **x86-64 baseline raised to v3** — distribution wheels now target the x86-64-v3 microarchitecture level (Haswell / Excavator / Zen 1 and newer, 2013+). On Linux this is `-march=x86-64-v3` (AVX2, FMA, BMI1/2, F16C, LZCNT, MOVBE); on Windows it is the equivalent `/arch:AVX2` baseline. This covers essentially all x86-64 laptops and workstations from the last decade. The main exceptions are pre-2022 Atom-derived chips such as Celeron N, Pentium Silver, and Jasper Lake. macOS x86_64 wheels are the exception: they use the SSE4.2 (`x86-64-v2`) baseline instead of AVX2, so they run under Rosetta 2 and on every Intel Mac (Apple Silicon wheels are native `arm64` and unaffected).
* **Parallel tensor evaluation** — `tensor_eval` now dispatches across threads when the total work reaches a tunable threshold (500 by default). "Total work" is the element count for scalar evaluation, and the element count times the number of query points for the batch (array-of-times) overload. Set the threshold with `masspcf.system.set_parallel_eval_threshold(n)` and read it back with `masspcf.system.get_parallel_eval_threshold()`. On multi-core CPUs this gives a several-fold speedup on large tensors.

#### Runtime CPU check

masspcf now verifies at import time that the CPU supports the instruction set the wheel was built against, and raises a clear `ImportError` with rebuild instructions if it does not — instead of letting the extension crash with an illegal-instruction signal. Set `SB_SKIP_CPU_CHECK` to any value other than `0` to bypass the check.

#### Building from source

If the distribution wheel does not run on your CPU — or you simply want a build tuned to your exact hardware — install from source:

```bash
pip install --no-binary=masspcf masspcf
```

Source builds default to `-march=native` on Linux and macOS, and to a CPUID-probed `/arch:` flag on MSVC, so the resulting extension targets whatever features your CPU actually has (including AVX-512 where available). The baseline can also be pinned explicitly at configure time with `-DSB_X86_64_LEVEL=v1|v2|v3|v4|native`.

The plain-cmake developer build (without `SKBUILD`) still defaults to `v3`.

### Indexing

Tensor indexing is now much closer to NumPy. These changes apply to every tensor type, since they share one indexing implementation.

* **Negative indices and slice bounds** — `t[-1]`, `t[-1, -1]`, `t[-3:-1]`, `t[:, -2:]`, and negative bounds with a negative step now resolve against the axis size like NumPy (previously they crashed, returned garbage, or gave the wrong shape). An out-of-range integer index raises `IndexError`, and `t[::0]` raises `ValueError`.
* **Ellipsis and newaxis** — `t[...]`, `t[..., 0]`, `t[None]`, and `t[:, None]` / `t[:, np.newaxis]` are now supported.
* **More index objects** — Python lists (`t[[0, 2]]`), NumPy integer scalars (`t[np.int64(1)]`), scalar booleans (`t[True]` / `t[False]`), multi-dimensional integer index arrays, the empty tuple `t[()]`, and partial integer tuples (fewer indices than the rank) all behave as in NumPy. A multi-dimensional integer index array adopts its own shape into the result (e.g. a `(4, 6)` tensor indexed by a `(2, 2)` array yields `(2, 2, 6)`). Invalid indices (e.g. floats) now raise `IndexError`.
* **0-d tensors** — iterating over or calling `len()` on a 0-d tensor now raises `TypeError`, matching NumPy.
* **Leading-axes boolean masks** — a boolean mask matching the leading axes of a tensor (e.g. a row mask `t[mask]`) selects along those axes instead of raising.
* **Assignment** — assigning a scalar into a slice (`t[1:3] = 5.0`), assigning into a single row of an N-D tensor (`t[1] = row`), and assigning a cross-dtype tensor (int → float) now work.
* **`DistanceMatrix` / `SymmetricMatrix`** — negative `(i, j)` indices are resolved, and out-of-range access raises `IndexError`.
* **Memory safety** — out-of-shape values in multi-axis (outer / `np.ix_`-style) assignment now raise instead of writing out of bounds, and out-of-bounds selectors in a multi-axis read now raise `IndexError`.

Multiple advanced indices keep their outer (`np.ix_`-style) semantics, as documented in [Indexing and Masking](https://github.com/kthtda/stablebear/blob/main/docs/indexing.rst).

### Packaging

* **Self-contained Windows wheels** — binary wheels for Windows now bundle the Microsoft Visual C++ runtime (e.g. `msvcp140.dll`, `msvcp140_atomic_wait.dll`) via [delvewheel](https://github.com/adang1345/delvewheel), so they import on machines without a separately installed Visual C++ Redistributable. This fixes the `DLL load failed` / `ImportError` on `_sb_cpu` that affected stock Windows installs in prior releases.

## 0.4.0

Major rewrite of the core data structures and significant expansion of the API.

### New features

* **Tensor type system** — `NdArray` replaced with a family of purpose-built tensor classes: `PcfTensor`, `IntPcfTensor`, `FloatTensor`, `IntTensor`, `BoolTensor`, `PointCloudTensor`, `DistanceMatrixTensor`, `SymmetricMatrixTensor`, `BarcodeTensor`. Each has a corresponding `dtype` sentinel.
* **NumPy-like tensor operations** — `reshape`, `transpose`/`.T`, `squeeze`, `expand_dims`, `swapaxes`, `concatenate`, `stack`, `split`, `array_split`, `astype`, iteration, `ndim`, `size`, `len()`.
* **Advanced indexing** — slicing with negative strides, multi-axis boolean masking, mixed integer+boolean indexing, broadcasting assignment.
* **Arithmetic** — element-wise `+`, `-`, `*`, `/`, `//`, `**`, unary `-` on numeric tensors; broadcasting support.
* **Persistence module** — `compute_persistent_homology` (Ripser), `Barcode`, `BarcodeTensor`, and barcode summaries: `barcode_to_stable_rank`, `barcode_to_betti_curve`, `barcode_to_accumulated_persistence`.
* **`DistanceMatrix` and `SymmetricMatrix`** — dedicated types with I/O support, `from_dense` and `to_dense` for NumPy interop.
* **Tensor I/O** — `save`/`load` for all tensor types; `from_serial_content` for in-memory deserialization.
* **PCF evaluation** — evaluate PCFs at given time points.
* **Plotting** — built-in plotting for PCFs and barcodes (matplotlib).
* **`lp_distance`** — scalar Lp distance between two individual PCFs.
* **`cdist`** — cross-distance matrices between two collections of PCFs.
* **`lp_norm`** — Lp norms for collections of PCFs.
* **`allclose`** — free function for element-wise approximate equality (FloatTensor, DistanceMatrix, SymmetricMatrix).
* **`array_equal`** — exact element-wise equality check for tensors.
* **`iterate_rectangles`** — iterate over the rectangle decomposition of two PCFs.
* **`flatten`, `copy`** — tensor methods for flattening and deep-copying.
* **`pickle` support** — all tensor types can be pickled and unpickled.
* **Deterministic random generation** — seedable `Generator` for reproducible output across threads.
* **`point_process` submodule** — `sample_poisson` for sampling spatial Poisson point processes.

### Breaking changes

* `pdist` returns a `DistanceMatrix` instead of a NumPy array. Call `.to_dense()` to get the previous behavior.
* `l2_kernel` returns a `SymmetricMatrix` instead of a NumPy array. Call `.to_dense()` to get the previous behavior.
* `NdArray` replaced with `Tensor` classes (dropped **xtensor**/**xtl** dependencies).
* `Pcf` construction now only accepts `n x 2` arrays (not `2 x n`) to avoid ambiguity with `2x2` arrays.
* I/O format bumped to version 2.
* Requires C++20 (was C++17). Minimum Python version is 3.10.

### Infrastructure

* CUDA backend reworked: auto-detection, pip-installed CUDA toolkit support (`nvidia.cu12`/`nvidia.cu13`), version-specific modules.
* Wheels built for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x86_64).
* CUDA matrix integration refactored into modular components.
* GPU occupancy floor added to block scheduler.
