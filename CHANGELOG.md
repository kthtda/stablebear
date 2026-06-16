## 0.4.3

### New features

* **NumPy/list constructors for the non-numeric tensor families** — `PointCloudTensor`, `DistanceMatrixTensor`, and `SymmetricMatrixTensor` can now be built directly from a NumPy array or nested list, instead of allocating with `zeros()` and filling element by element. `PointCloudTensor(arr)` reads a trailing `(n_points, dim)` block per cell (with ragged lists of per-cloud arrays also accepted); `DistanceMatrixTensor` and `SymmetricMatrixTensor` accept a stack of square matrices via `from_numpy(...)` (or the constructor). Dtype is inferred from the array (`float32` → `pcloud32`/`distmat32`/…, `float64` → the 64-bit dtype) and can be overridden with `dtype=`. A generic `sb.tensor(data, dtype=...)` factory dispatches to the right tensor type from the dtype sentinel. ([#53](https://github.com/kthtda/stablebear/issues/53), [#92](https://github.com/kthtda/stablebear/issues/92), [#84](https://github.com/kthtda/stablebear/issues/84))

### Breaking changes

* **The `stablebear.tensor` submodule has been renamed to `stablebear.base_tensor`** — the name `stablebear.tensor` now refers to the new `tensor()` factory function (above), so the module that holds the tensor classes (`PcfTensor`, `FloatTensor`, `PointCloudTensor`, …) moved to `stablebear.base_tensor`. Code that imported from the module path must update its imports (`from stablebear.tensor import FloatTensor` → `from stablebear.base_tensor import FloatTensor`). The classes remain re-exported at the top level, so the recommended `import stablebear as sb; sb.FloatTensor` form is unaffected.

### Bug fixes

* **A `Generator` now advances between draws** — two consecutive draws from the same generator (or from the global generator after a single `random.seed(s)`) returned byte-for-byte identical tensors, silently breaking any repeated-sampling workflow (bootstrap, Monte Carlo) and contradicting the `numpy.random.Generator` / `torch.Generator` convention. Each sampling call now reserves a fresh, contiguous block of seed slots and advances the generator past it, so successive draws are independent yet fully reproducible for a given seed. The first draw after seeding is unchanged, so existing seeds reproduce their historical first result. ([#86](https://github.com/kthtda/stablebear/issues/86))
* **`max_time` and `plotting.plot` no longer segfault on an empty tensor** — reducing an empty PcfTensor (or any tensor whose reduction dimension has size 0) with `max_time` crashed the interpreter, because `max_element` seeded the reduction from the first element of an empty range and reduced past-the-end iterators. `max` has no identity over an empty range, so `max_time` now raises a catchable `ValueError` (mirroring NumPy's zero-size reduction error), and `plotting.plot` — which calls `max_time` internally — degrades to a graceful no-op when there is nothing to draw. ([#46](https://github.com/kthtda/stablebear/issues/46))
* **`tensor + ndarray` no longer raises a raw `TypeError`** — the reflected form `ndarray + tensor` worked (via `__radd__`), but the forward `tensor + ndarray` passed the array straight through to C++ and raised a bare pybind `TypeError`, a surprising asymmetry. The forward operators now wrap an ndarray (or list/tuple) right-hand side as a same-type tensor, casting to the tensor's dtype, so `+`, `-`, `*`, `/` and their in-place variants are symmetric. ([#62](https://github.com/kthtda/stablebear/issues/62))
* **Comparing a tensor to a scalar or array no longer raises `AttributeError`** — the ordering operators (`<`, `<=`, `>`, `>=`) accessed `rhs._data` unconditionally, so a scalar right-hand side raised `AttributeError: 'int' object has no attribute '_data'` and the canonical NumPy mask-select `t[t > 3]` was impossible without round-tripping through NumPy. A scalar or `ndarray` RHS is now broadcast against the tensor (NumPy semantics) and returns a `BoolTensor`, so `t[t > 3]` works directly; a non-numeric tensor (e.g. `PcfTensor`) raises a clear `TypeError` instead of the internal `AttributeError`. ([#57](https://github.com/kthtda/stablebear/issues/57))

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
