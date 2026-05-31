# Tensor Indexing Review — NumPy Parity Gaps

## 1. Executive Summary

masspcf tensor indexing is **partially NumPy-compatible**: the positive-only "happy path" matches NumPy closely, but the package is systematically non-compatible for anything involving negative basic indices, several whole index-object categories, and multi-axis advanced indexing. Concretely, what already works is the positive basic-slicing path (positive ints, positive/empty slices, positive-bound steps including negative-step reversal with omitted/positive bounds), single 1-D integer-array fancy indexing (ndarray or `IntTensor`, including negative entries and OOB which correctly raise `IndexError`), full-shape boolean masks, per-axis bool masks via `x[:, mask]`, view/write-through semantics, and Tensor-RHS broadcast assignment.

Because all eight wrapper types (`FloatTensor`, `IntTensor`, `BoolTensor`, `PcfTensor`, `IntPcfTensor`, `PointCloudTensor`, `DistanceMatrixTensor`, `SymmetricMatrixTensor`) share one `__getitem__`/`__setitem__` in `masspcf/_tensor_base.py`, every gap and every working pattern is **uniform across types** — so fixing the shared layer fixes all types at once.

Nearly every failure traces back to a small number of root causes:

1. **`extract()` in `include/mpcf/tensor.tpp` does no negative-index resolution and no bounds checking.** A negative `SliceIndex` produces a negative memory offset (silent garbage on read, deterministic **SIGSEGV** on write — verified exit code 139); a too-large index reads out of bounds; negative `SliceRange` bounds are clamped to 0 instead of resolved (so `a[-3:-1]`, `a[-2:]`, `a[:-1]`, `a[-1:-4:-1]` all return wrong shapes/values); `step == 0` silently returns empty instead of raising.
2. **`_pyslice_to_slice` only understands `int` and `slice`** — so `Ellipsis`, `None`/`np.newaxis`, `np.integer` scalars, `np.bool_`, Python lists, and floats all raise `"Unhandled slice type"`.
3. **`_coerce_index_arrays` only coerces `np.ndarray`** — Python-list fancy indices fall through.
4. **The advanced-index loop applies indices sequentially (outer / `np.ix_` product)** instead of NumPy's vectorized zip-and-broadcast, silently producing wrong shapes/values and never raising on mismatched index lengths.
5. **The C++ `outer_select`/`multi_axis_assign`/`outer_assign` paths skip bounds and values-shape validation**, creating additional OOB read/write hazards.

The single highest-priority defect is the silently-wrong / memory-unsafe negative-index handling in `extract()`: it both crashes (negative `SliceIndex` write → SIGSEGV) and returns silently-wrong data (negative slice bounds), and it is reachable from ordinary Python expressions like `t[-1]`, `t[-1, 2:4]`, `t[-3:-1]`, and `t[-1] = v`. A central fix that mirrors Python's `slice.indices(n)` plus single-int negative resolution in `extract()` (and resolving negative ints in the Python int / `_get_element` paths) repairs the majority of correctness and crash findings at once.

## 2. Methodology

**Oracle.** NumPy was used as the behavioral oracle. For each index expression we compared `np.asarray(masspcf_result)` (via a `FloatTensor` round-trip from a known `np.arange(...)` array) against the corresponding NumPy result, checking shape, values, and the exception type/message where NumPy raises.

**Source audit.** Findings were traced to source in three layers:
- `masspcf/_tensor_base.py` — the shared `__getitem__`/`__setitem__`, `_pyslice_to_slice`, `_coerce_index_arrays`, `_resolve_negative_indices`, `_getitem_slices`, `_setitem_slices`, `__len__`/`__iter__`.
- `src/python/py_tensor.{hpp,cpp}` — the pybind11 bindings (`_get_element`/`_set_element`, `__setitem__`, `slice_index`/`slice_range`/`slice_all`, `index_select`, `outer_select`/`outer_assign`, `multi_axis_assign`).
- `include/mpcf/tensor.tpp` — `extract()` (SliceIndex / SliceRange branches), `masked_select`, `index_select`, `validate_axis_indices`, `resolve_selectors`, `outer_select`/`outer_assign`, `multi_axis_assign`, `assign_from`.

**Cross-type battery.** Each gap was exercised across all eight wrapper types to confirm the shared-path hypothesis; gaps and working patterns reproduce identically on every type (verified e.g. `x[-1]` and `x[[0,1]]` error uniformly across `FloatTensor`/`IntTensor`/`BoolTensor`/`DistanceMatrixTensor`/`SymmetricMatrixTensor`). Memory-safety findings were verified at runtime (negative `SliceIndex` write → process exit code 139; OOB reads return populated-but-garbage tensors with no exception).

**How to reproduce.** Build/install first, then run the harness from the `test/` directory (running from the repo root shadows the installed package):

```bash
cmake --build cmake-build-debug -j$(nproc) && cmake --install cmake-build-debug
cd test && micromamba run -n py313 python
```

Then paste any repro block from Section 4 (or the full script in the Appendix). For the crash repros, run each in a fresh interpreter, as a SIGSEGV terminates the process.

## 3. Parity Matrix

Legend: ✅ works · ⚠️ partial · ❌ wrong (silently incorrect shape/values) · 🛑 errors (raises where NumPy succeeds, or wrong exception) · — missing/N-A

| Capability | getitem | setitem | Notes |
|---|:---:|:---:|---|
| basic positive int (axis 0) | ✅ | ⚠️ | getitem matches (sub-tensor on N-D, scalar on 1-D). setitem: full-ndim element assign works; single int on N-D (`x[1]=row`) is broken (treated as full multi-index → IndexError/TypeError). |
| negative int (single) | 🛑 | 🛑 | N-D `x[-1]` → squeeze ValueError; 1-D `x[-1]` → `_get_element` TypeError; setitem TypeError. Never resolved against shape. Uniform across all 8 types. |
| negative int inside all-int tuple | 🛑 | 🛑 | `x[1,-1]`/`x[-1,-1]` → `_get_element`/`_set_element` TypeError; negatives not resolved before the `size_t`-typed C++ call. |
| positive slice (start/stop) | ✅ | ⚠️ | getitem fully matches incl. empty results. setitem Tensor-RHS broadcast works; scalar RHS into a slice → AttributeError. |
| slice step (positive) | ✅ | ✅ | `x[::2]`, `x[1:4:2]` match. |
| negative step (omitted/positive bounds) | ✅ | ✅ | `x[::-1]`, `x[::-2]`, `x[3:0:-1]`, `x[3::-1]` match. |
| negative slice bounds (positive step) | ❌ | ❌ | `x[-3:-1]` → empty; `x[-2:]` → full from front; `x[:,-2:]` → full. Negative start clamped to 0, negative stop not resolved. Silently wrong. setitem target region wrong → ValueError or silent. |
| negative slice bounds (negative step) | ❌ | ❌ | `x[-1:-4:-1]` → (1,6) row 0; `x[-1:0:-1]` → empty. Negatives not resolved before clamp. |
| slice step == 0 | ❌ | ❌ | Returns empty axis instead of raising `ValueError('slice step cannot be zero')`. |
| ellipsis (`...`) | 🛑 | 🛑 | All forms (`x[...]`, `x[...,0]`, `x[0,...]`) → TypeError "Unhandled slice type". No expansion pass. |
| newaxis / `None` | 🛑 | 🛑 | `x[None]`, `x[:,None]` → TypeError "Unhandled slice type". No axis-insertion handling; C++ `Slice` variant has no newaxis member. |
| integer-array fancy (ndarray, single axis) | ✅ | ✅ | `x[np.array([0,2])]` matches incl. duplicates, empty, negative entries; OOB raises IndexError. Copy-on-read, write-through-on-assign. |
| integer-array fancy (IntTensor, single axis) | ✅ | ✅ | `IntTensor` passed directly is recognized as advanced; matches. |
| integer-array fancy (Python list) | 🛑 | 🛑 | `x[[0,2]]` → TypeError "Unhandled slice type" (only ndarray coerced). Inconsistent with the working ndarray form. |
| negative fancy entries | ✅ | ✅ | `x[np.array([-1,-2])]`, `x[:,np.array([-1])]` resolved + bounds-checked. |
| out-of-bounds fancy entries | ✅ | ✅ | `x[np.array([0,10])]` raises IndexError (message differs slightly from NumPy but type/semantics correct). |
| N-D / 0-d integer index array | 🛑 | 🛑 | `x[np.array([[0,1],[2,3]])]` and `x[np.array(2)]` → ValueError "indices must be 1D". NumPy adopts `idx.shape` (N-D) or drops axis (0-d). |
| multi-axis vectorized fancy (zip) | ❌ | ❌ | `x[[0,2],[1,3]]` → (2,2) outer product instead of (2,) zip. Mismatched lengths silently cross-product instead of raising. Largest divergence; equals `np.ix_`. |
| broadcast advanced indices (mismatched ndim) | 🛑 | 🛑 | `x[[[0],[1]],[0,2]]` → ValueError "indices must be 1D". Needs N-D index + vectorized broadcast. |
| combined basic + single advanced | ✅ | ✅ | `x[1:3,np.array([0,2])]`, `x[:,np.array([0,2])]`, `x[np.array([0,2]),1]` match (single advanced → outer == vectorized). |
| combined basic + advanced (advanced separated by slice) | ❌ | ❌ | `x[idx,:,idx2]` does not move advanced block to front and uses outer → wrong/transposed shape. |
| boolean mask (full-shape, flat) | ✅ | ✅ | `x[mask]` full-shape → 1-D C-order select; `masked_assign`/`masked_fill` write through; correct on non-contiguous views; wrong-length RHS raises ValueError. |
| boolean mask (per non-leading axis via `x[:,mask]`) | ✅ | ✅ | 1-D bool mask along a non-leading axis works via `axis_select`. |
| boolean mask (1-D along axis 0 / sub-shape) | 🛑 | 🛑 | `x[rowmask]` and a k-D mask matching leading axes → ValueError "mask shape does not match tensor shape". Always routed to flat `masked_select`. |
| multi-axis boolean masks | ❌ | ❌ | `x[rm,cm]` → (2,3) outer product; NumPy treats as vectorized (errors on unequal true counts). |
| scalar boolean (`True`/`False`, `np.bool_`) | ❌ | — | Python `True`/`False` silently used as int 1/0 (wrong shape); `np.bool_` scalar → TypeError "Unhandled slice type". NumPy: `True`→leading len-1 axis, `False`→len-0 axis. |
| empty tuple `x[()]` | 🛑 | 🛑 | → IndexError "Index out of range"; NumPy returns the full array. |
| partial multi-int tuple (< ndim) | 🛑 | 🛑 | 3-D `x[1,2]` → IndexError; NumPy returns the remaining-axis sub-tensor. |
| numpy scalar int (`np.int64`) index | 🛑 | 🛑 | → TypeError "Unhandled slice type"; NumPy treats like Python int. |
| float index | 🛑 | 🛑 | Correctly errors but wrong type/message (TypeError "Unhandled slice type" vs NumPy IndexError). |
| view semantics (basic slicing aliases parent) | ✅ | ✅ | Slice and single-int views share storage (write-through); fancy/mask reads copy. Matches NumPy. (Untested.) |
| bounds error (single int OOB) | 🛑 | 🛑 | Raises ValueError from squeeze (positive) / TypeError (negative 1-D) instead of IndexError; inconsistent with multi-int OOB which raises IndexError. |
| too many indices | 🛑 | 🛑 | Raises IndexError "Index out of range" (correct type) but non-NumPy message; no arity check. |
| cross-dtype Tensor RHS assignment | — | 🛑 | `x[1:3]=IntTensor` → C++ `__setitem__` TypeError; NumPy casts int→float. No `astype` coercion of Tensor RHS. |
| chained indexing `x[i][j]` | ✅ | ⚠️ | Positive chaining matches `x[i,j]`; inherits negative-int and single-int-setitem bugs when any index is negative or assigns through a single-int view. |

## 4. Findings

Severity legend: 🟥 **crash** (memory-unsafe / SIGSEGV) · 🟧 **wrong-result** (silently incorrect) · 🟨 **error** (raises where NumPy succeeds) · 🟦 **missing** (unsupported feature) · ⬜ **minor/inconsistent** (wrong exception type/message, API divergence) · 🟩 **ok** (matches NumPy).

Categories are ordered worst-severity first.

### Category: Crashes / Memory Safety (🟥)

#### `neg-sliceindex-oob-write-segv` — Negative single-int combined with a slice yields a negative memory offset: OOB read (garbage) and SIGSEGV on write 🟥 crash

```python
import numpy as np; from masspcf.tensor import FloatTensor; import masspcf._mpcf_cpp as cpp
a=np.arange(24.).reshape(4,6)
# READ: numpy a[-1,2:4]=[20,21]; masspcf returns garbage (offset -4)
r=FloatTensor(a)[-1,2:4]; print(np.asarray(r))
# WRITE: deterministic crash
d=FloatTensor(a.copy())._data; d[[cpp.slice_index(-1)]]=FloatTensor(np.zeros(6))._data  # SIGSEGV (exit 139)
```

**NumPy:** `a[-1,2:4]` → `[20., 21.]`; `a[-4,:]` → first row; `a[-1]=0` zeros the last row.
**masspcf:** READ: no exception; `t[-1,2:4]` has offset −4 and returns uninitialized memory (verified `[2.18e-314, 0.0]`); `t[-4,:]` offset −24. WRITE via the negative-offset view dereferences before the buffer and crashes with SIGSEGV (verified process exit code 139). No exception is ever raised.
**Root cause:** `include/mpcf/tensor.tpp:1457` — the `SliceIndex` branch does `ret.m_offset += arg.index * ret.m_strides[i]` with no `if (arg.index < 0) arg.index += dim_size;` resolution and no bounds check. The negative offset is computed and dereferenced.
**Proposed fix:** In the `SliceIndex` branch capture the dim size before overwriting `m_shape[i]=1`, then `auto ix = arg.index; if (ix < 0) ix += n; if (ix < 0 || ix >= n) throw pybind11::index_error(...); ret.m_offset += ix * ret.m_strides[i];`. Mirrors NumPy single-int semantics and removes the OOB hazard.

#### `sliceindex-positive-oob-no-check` — Large positive single-int index has no bounds check (OOB read; OOB write hazard) 🟥 crash

```python
import numpy as np; from masspcf.tensor import FloatTensor; import masspcf._mpcf_cpp as cpp
d=FloatTensor(np.arange(24.).reshape(4,6))._data
print(list(d[[cpp.slice_index(4)]].shape))     # numpy a[4] -> IndexError; masspcf returns (6,) of OOB garbage
print(list(d[[cpp.slice_index(1000)]].shape))  # reads ~6000 elements past buffer
```

**NumPy:** `a[4]` and `a[1000]` both raise `IndexError: index out of bounds for axis 0 with size 4`.
**masspcf:** No exception: `slice_index(4)` returns shape (6,) reading out-of-bounds memory; `slice_index(1000)` reads far past the buffer. Verified both return a (6,)-shaped tensor with no error. The write path would corrupt memory / crash like the negative case.
**Root cause:** `include/mpcf/tensor.tpp:1456-1457` — the `SliceIndex` branch sets `m_shape[i]=1` and adds `arg.index*stride` with no comparison of `arg.index` against the original dim size.
**Proposed fix:** Same fix as `neg-sliceindex-oob-write-segv`: after resolving negatives, validate `ix` against `[0,n)` and throw `pybind11::index_error` otherwise.

#### `multi-axis-assign-no-shape-validation` — `multi_axis_assign` does not validate values shape → OOB write into destination 🟥 crash

```python
import numpy as np; from masspcf.tensor import FloatTensor, BoolTensor
d=FloatTensor(np.arange(24.).reshape(4,6).copy())._data
m0=BoolTensor(np.array([True,False,True,False])); m1=BoolTensor(np.array([True,False,True,False,False,False]))
d.multi_axis_assign([(0,m0._data),(1,m1._data)], FloatTensor(np.full((3,3),-9.0))._data)  # expected (2,2); silently accepted
```

**NumPy:** Broadcasting a (3,3) value into a (2,2) selection raises ValueError.
**masspcf:** No exception (verified): walks over (3,3) values and indexes `true_indices[val_idx]` with `val_idx` up to 2 while `true_indices` has length 2 → OOB vector access / undefined `dst` write. `index_assign`/`axis_assign`/`masked_assign` all validate; this path does not.
**Root cause:** `include/mpcf/tensor.tpp:731-743` `multi_axis_assign` — never compares `values.shape()` against the expected selected shape; the walk indexes `true_indices[...]` with no bounds check.
**Proposed fix:** Build `expected_shape = dst.shape()` with each selected axis set to `true_indices.size()`; throw `std::invalid_argument` if `values.shape() != expected_shape` before the walk (mirror `axis_assign` at `tensor.tpp:643-647`).

#### `outer-assign-no-shape-validation` — `outer_assign` does not validate values shape → OOB write into destination 🟥 crash

```python
import numpy as np; from masspcf.tensor import FloatTensor, IntTensor
d=FloatTensor(np.arange(24.).reshape(4,6).copy())._data
i0=IntTensor(np.array([0,2],dtype=np.int64)); i1=IntTensor(np.array([1,3],dtype=np.int64))
d.outer_assign([(0,i0._data),(1,i1._data)], FloatTensor(np.full((3,3),-7.0))._data)  # expected (2,2); silently accepted
```

**NumPy:** Assigning a (3,3) source into a (2,2) outer selection raises ValueError.
**masspcf:** No exception (verified): walks the (3,3) values and indexes `rs.indices[val_idx]` out of bounds → undefined `dst` write / memory corruption.
**Root cause:** `include/mpcf/tensor.tpp:843-855` `outer_assign` — `resolve_selectors` is called but `values.shape()` is never validated against the selected shape; the walk indexes `rs.indices[...]` unchecked.
**Proposed fix:** Compute `expected_shape` from selected sizes and throw `std::invalid_argument` on mismatch before the walk (mirror `axis_assign`/`index_assign`).

#### `outer-select-no-bounds-check` — `outer_select` (`resolve_selectors`) performs no bounds check / negative resolution on int indices (OOB read) 🟥 crash

```python
import numpy as np; from masspcf.tensor import FloatTensor, IntTensor
d=FloatTensor(np.arange(24.).reshape(4,6))._data
i0=IntTensor(np.array([0,1000],dtype=np.int64)); i1=IntTensor(np.array([0,1],dtype=np.int64))
d.outer_select([(0,i0._data),(1,i1._data)])  # idx 1000 OOB; numpy would IndexError; masspcf returns (2,2) reading OOB
```

**NumPy:** An OOB index along an axis raises IndexError.
**masspcf:** No exception (verified): returns a (2,2) tensor whose OOB row is read out of bounds; a raw negative index would `static_cast` to `SIZE_MAX` and read far OOB. The Python layer pre-resolves negatives, but the C++ entry point is unguarded.
**Root cause:** `include/mpcf/tensor.tpp:757-792` `detail::resolve_selectors` — the int branch (784-787) does `static_cast<size_t>(idx_tensor({i}))` with no bounds check and no negative resolution, unlike `validate_axis_indices` used by `index_select`.
**Proposed fix:** In `resolve_selectors`'s int branch, resolve negatives (`v += shape[axis]` if `v<0`) and throw `std::out_of_range` if `v<0 || v>=shape[axis]` before pushing. Reuse/parameterize `validate_axis_indices`.

### Category: Silently Wrong Results (🟧)

#### `slicerange-neg-bounds-posstep` — Negative slice start/stop with positive step clamped to 0 instead of resolved 🟧 wrong-result

```python
import numpy as np; from masspcf.tensor import FloatTensor
a=np.arange(24.).reshape(4,6); t=FloatTensor(a)
print(np.asarray(t[-3:-1]).shape)  # numpy (2,6); masspcf (0,6)
print(np.asarray(t[-2:]).shape)    # numpy (2,6); masspcf (4,6) full from front
print(np.asarray(t[:,-2:]).shape)  # numpy (4,2); masspcf (4,6)
```

**NumPy:** `a[-3:-1]`→(2,6) rows 1,2; `a[:-1]`→(3,6); `a[-2:]`→(2,6) last two; `a[1:-1]`→(2,6); `a[:,-2:]`→(4,2); `a[:,-3:-1]`→(4,2).
**masspcf:** Silently wrong: negative stop → empty (`t[-3:-1]`→(0,6), `t[:-1]`→(0,6), `t[1:-1]`→(0,6), `t[:,-3:-1]`→(4,0)); negative start clamped to 0 → full from front (`t[-2:]`→(4,6) rows 0..3, `t[:,-2:]`→(4,6)). Verified all.
**Root cause:** `include/mpcf/tensor.tpp:1475-1476` (step>0 branch): `if (start < 0) start = 0;` clamps negative start instead of resolving (`start += dim_size`); negative stop is never resolved (only `if (stop > dim_size) stop = dim_size;`), so a negative stop stays negative and triggers the `stop<=start` empty branch at 1478-1481.
**Proposed fix:** Before clamping in the step>0 branch: `if (start < 0) start += dim_size; if (stop < 0) stop += dim_size;` then clamp both into `[0,dim_size]` per Python `slice.indices(n)`. Single fix repairs both read and write of negative-bound positive-step slices.

#### `slicerange-neg-bounds-negstep` — Negative slice bounds with negative step resolved incorrectly 🟧 wrong-result

```python
import numpy as np; from masspcf.tensor import FloatTensor
a=np.arange(24.).reshape(4,6); t=FloatTensor(a)
print(np.asarray(t[-1:-4:-1]).shape)  # numpy (3,6) rows 3,2,1; masspcf (1,6) row 0
print(np.asarray(t[-1:0:-1]).shape)   # numpy (3,6); masspcf (0,6)
```

**NumPy:** `a[-1:-4:-1]`→(3,6) rows 3,2,1; `a[-1:0:-1]`→(3,6) rows 3,2,1.
**masspcf:** `t[-1:-4:-1]` → shape (1,6) returning row 0 (verified values `[0..5]`); `t[-1:0:-1]` → shape (0,6) empty. Both silently wrong.
**Root cause:** `include/mpcf/tensor.tpp:1491-1496` (step<0 branch): negative start hits `if (start < 0) start = 0;` (collapses −1 to 0) and negative stop hits `if (stop < -1) stop = -1;` without first adding `dim_size`, so length and offset are computed from wrong endpoints.
**Proposed fix:** In the step<0 branch resolve first: `if (start < 0) start += dim_size;` then clamp to `[-1, dim_size-1]`; `if (stop < 0) stop += dim_size;` then clamp to `[-1, dim_size-1]`. Mirror Python `slice.indices(n)` for negative step.

#### `slicerange-step-zero` — Slice `step == 0` silently returns empty instead of raising ValueError 🟧 wrong-result

```python
import numpy as np; from masspcf.tensor import FloatTensor
print(np.asarray(FloatTensor(np.arange(24.).reshape(4,6))[::0]).shape)  # numpy ValueError; masspcf (0,6)
```

**NumPy:** `a[::0]` raises `ValueError: slice step cannot be zero`.
**masspcf:** Returns an empty tensor of shape (0,6) with no exception. Verified.
**Root cause:** `include/mpcf/tensor.tpp:1509-1512` — the `step==0` else-branch sets `ret.m_shape[i]=0;` instead of throwing.
**Proposed fix:** Replace the `step==0` else-branch with `throw pybind11::value_error("slice step cannot be zero");`; the `slice_range` binding can also reject `step==0` earlier.

#### `setitem-negative-slice-bounds` — `x[-3:-1]=block` / `x[:,-2:]=block` fail (negative slice bounds clamped → wrong target region) 🟧 wrong-result

```python
import numpy as np; from masspcf.tensor import FloatTensor
t=FloatTensor(np.arange(24.).reshape(4,6))
t[-3:-1]=FloatTensor(np.zeros((2,6)))  # numpy sets rows 1,2; masspcf ValueError shapes (0,6) vs (2,6)
```

**NumPy:** `x[-3:-1]` resolves to rows 1..2; assignment fills those rows. `x[:,-2:]` → last two columns.
**masspcf:** The target view computes shape (0,6) [or full], so `assign_from` raises `ValueError: Shapes are not broadcast-compatible: (0, 6) and (2, 6)`. Verified. The assignment target region is wrong (silent on the full-clamp side).
**Root cause:** `include/mpcf/tensor.tpp:1475-1476` (and 1491-1496 for negative step) — the same negative-bound clamping defect on the read side propagates to the assign target view.
**Proposed fix:** Resolve negatives in `extract()`'s `SliceRange` handling (the `slicerange-neg-bounds-posstep`/`negstep` fix). One fix repairs both read and write of negative-bounded slices.

#### `fancy-multi-adv-outer-vs-vectorized` — Multiple advanced indices use outer-product (`np.ix_`) instead of NumPy vectorized/zip semantics (getitem + setitem) 🟧 wrong-result

```python
import numpy as np; from masspcf.tensor import FloatTensor
a=np.arange(24.).reshape(4,6)
FloatTensor(a)[np.array([0,2]),np.array([1,3])]  # numpy [1,15] shape (2,); masspcf (2,2) [[1,3],[13,15]]
t=FloatTensor(a.copy()); t[np.array([0,2]),np.array([1,3])]=9.0  # numpy sets (0,1)&(2,3); masspcf sets 2x2 grid
```

**NumPy:** Multiple advanced indices broadcast together and zip: `result[i]=a[idx0[i],idx1[i]]`; `x[[0,2],[1,3]]`→`[1,15]` shape (2,). Assignment writes exactly the 2 zipped cells.
**masspcf:** Returns/writes the outer/cross product `a[np.ix_([0,2],[1,3])]` → shape (2,2). Verified getitem (2,2) and setitem (sets indices 1 and 3 of row 0 to 9 → 4 cells instead of 2). Confirmed equal to NumPy's `np.ix_` form. This is the single largest divergence and affects every multi-advanced expression.
**Root cause:** `masspcf/_tensor_base.py:179-187` (get) and 244-270 (set) apply each advanced index sequentially via `index_select`/`outer_assign` (outer/Cartesian by construction); `tensor.tpp:796` `outer_select` and 843 `outer_assign` implement cartesian semantics. No broadcast/zip of index arrays.
**Proposed fix:** When 2+ integer advanced indices appear, broadcast them to a common shape and gather/scatter element-wise (vectorized) into a single combined advanced axis-block. Add a C++ `vectorized_select`/`vectorized_assign`. Keep outer/`ix_` behavior behind an explicit opt-in API (e.g. an `outer_index` helper), not the default.

#### `fancy-multi-adv-mismatched-len` — Mismatched-length advanced index arrays silently cross-product instead of raising IndexError 🟧 wrong-result

```python
import numpy as np; from masspcf.tensor import FloatTensor
FloatTensor(np.arange(24.).reshape(4,6))[np.array([0,2]),np.array([1,3,5])]  # numpy IndexError; masspcf (2,3)
```

**NumPy:** Non-broadcastable index-array shapes raise `IndexError: shape mismatch: indexing arrays could not be broadcast together`.
**masspcf:** Silently returns the (2,3) outer product, no error. Verified.
**Root cause:** Direct consequence of the outer-product semantics (`masspcf/_tensor_base.py:179-187`, `tensor.tpp:796`); index-array shapes are never compared.
**Proposed fix:** Covered by `fancy-multi-adv-outer-vs-vectorized`: vectorized broadcasting will naturally raise on incompatible shapes. Until then add an explicit shape-broadcast check across advanced indices in `__getitem__`/`__setitem__`.

#### `fancy-separated-adv-no-front-move` — Advanced indices separated by a slice/int not moved to front (transposed/wrong shape) 🟧 wrong-result

```python
import numpy as np; from masspcf.tensor import FloatTensor
b=np.arange(24.).reshape(2,3,4)
FloatTensor(b)[np.array([0,1]),:,np.array([0,1])]  # numpy (2,3); masspcf (2,3,2)
```

**NumPy:** When advanced indices are separated by a slice, NumPy broadcasts them (vectorized) and places the advanced axis-block at the FRONT of the result, then the sliced axes: shape (2,3).
**masspcf:** Sequential outer indexing keeps each advanced axis in place: shape (2,3,2). Verified.
**Root cause:** `masspcf/_tensor_base.py:179-189` loops advanced indices in original axis order and never relocates the resulting axes; `dims_dropped` only accounts for int-dropped axes, not NumPy's advanced-axis placement rule.
**Proposed fix:** Implement NumPy's two rules together: (a) vectorized broadcast of advanced indices into one block, (b) if the advanced indices are non-contiguous in the index tuple, move that block to position 0. Requires the vectorized-gather rewrite plus a transpose/placement step.

#### `fancy-two-bool-masks-outer` — Two boolean masks on two axes do outer product instead of NumPy vectorized (which usually errors) 🟧 wrong-result

```python
import numpy as np; from masspcf.tensor import FloatTensor, BoolTensor
a=np.arange(24.).reshape(4,6); rm=np.array([True,False,True,False]); cm=np.array([True,False,True,False,True,False])
FloatTensor(a)[BoolTensor(rm),BoolTensor(cm)]  # numpy IndexError (2 vs 3 true); masspcf (2,3) outer
```

**NumPy:** NumPy converts each bool mask to its true-index array and applies VECTORIZED indexing; with 2 vs 3 true values it raises IndexError; with equal counts it zips them.
**masspcf:** Returns the (2,3) outer/cross product, no error. Verified.
**Root cause:** `masspcf/_tensor_base.py:179-189` applies each `BoolTensor` via `axis_select` sequentially (outer). Same outer-vs-vectorized root cause as integer multi-advanced.
**Proposed fix:** Convert bool masks to int index arrays and route through the unified vectorized multi-advanced path so they broadcast/zip like NumPy and raise on incompatible true counts.

#### `bool-scalar-index` — `x[True]`/`x[False]` treated as integer index instead of NumPy scalar-bool newaxis-mask 🟧 wrong-result

```python
import numpy as np; from masspcf.tensor import FloatTensor
t=FloatTensor(np.arange(24.).reshape(4,6))
print(np.asarray(t[True]).shape)   # numpy (1,4,6); masspcf (6,) (== x[1])
print(np.asarray(t[False]).shape)  # numpy (0,4,6); masspcf (6,) (== x[0])
```

**NumPy:** `a[True]` adds a leading length-1 axis → (1,4,6); `a[False]` → (0,4,6).
**masspcf:** `t[True]`→(6,) (row 1), `t[False]`→(6,) (row 0). Python `bool` is an `int` subclass so it is silently used as an integer index. Verified. (`np.bool_` scalar instead raises "Unhandled slice type".)
**Root cause:** `masspcf/_tensor_base.py:193 & 202` use `isinstance(s, int)`, which matches Python `bool`; `np.bool_` is not caught by `_coerce_index_arrays` so it hits `_pyslice_to_slice`.
**Proposed fix:** Detect Python `bool` and `np.bool_` scalars BEFORE the int check: `True` → `expand_dims(0)`; `False` → length-0 leading axis. Use `isinstance(s,(bool,np.bool_))` explicitly.

### Category: Errors Where NumPy Succeeds (🟨)

#### `neg-int-multidim-squeeze` — Negative single int on N-D tensor raises squeeze ValueError instead of selecting from the end 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor
t=FloatTensor(np.arange(24.).reshape(4,6))
t[-1]  # numpy last row (6,); masspcf ValueError
```

**NumPy:** `a[-1]` returns the last row (6,); `a[-2]` second-to-last.
**masspcf:** `ValueError 'squeeze: cannot squeeze axis 0 with size 4'` for `t[-1]`; `'size 0'` for `t[-2]`. Verified across FloatTensor/IntTensor/BoolTensor/DistanceMatrixTensor/SymmetricMatrixTensor (uniform).
**Root cause:** `masspcf/_tensor_base.py:197-199` builds `slice(idx, idx+1 if idx!=-1 else None)`; for `idx=-1` → `slice(-1,None)`, for `idx=-2` → `slice(-2,-1)`. These negatives reach `extract()` which clamps/empties them, so `squeeze(0)` rejects the resulting axis. The `slice(idx,idx+1)` trick cannot represent a negative scalar index.
**Proposed fix:** In `_getitem_slices` resolve the negative axis-0 int first (`if idx<0: idx += self.shape[0]`; bounds-check → IndexError) before building `slice(idx, idx+1)`+squeeze. Becomes moot once `extract()` resolves `SliceIndex` negatives.

#### `neg-int-in-int-tuple` — Negative int inside an all-int index tuple rejected by `_get_element` (TypeError) 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor
t=FloatTensor(np.arange(24.).reshape(4,6))
t[1,-1]  # numpy 11.0; masspcf TypeError
```

**NumPy:** `a[1,-1]`→11.0, `a[-1,-1]`→23.0, `a[-1,0]`→18.0.
**masspcf:** `TypeError: _get_element() incompatible function arguments` (negative ints can't convert to `size_t`). Verified for `t[1,-1]`, `t[-1,-1]`, `t[-1,0]` across all types.
**Root cause:** `masspcf/_tensor_base.py:202-203` — the all-int fast path passes the raw tuple straight to `self._data._get_element(slices)`; the C++ `size_t`-typed `_get_element` rejects negatives. No negative resolution is performed.
**Proposed fix:** Before the all-int path, resolve each negative int against `self.shape[i]` (raise IndexError if still out of range), then call `_get_element` with the non-negative tuple. Add a `_resolve_int_tuple` helper.

#### `neg-int-1d-getelement` — Negative single int on 1-D tensor rejected by `_get_element(size_t)`; OOB negative raises TypeError not IndexError 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor
tb=FloatTensor(np.arange(6.))
tb[-1]  # numpy 5.0; masspcf TypeError
tb[-7]  # numpy IndexError; masspcf TypeError
```

**NumPy:** `b[-1]`→5.0, `b[-2]`→4.0; `b[-7]`→IndexError.
**masspcf:** `TypeError: _get_element() incompatible function arguments` for both in-range and OOB negative 1-D ints (the `size_t` overload rejects negatives before any bounds check). Verified.
**Root cause:** `masspcf/_tensor_base.py:193-195` — the 1-D single-int branch passes `slices[0]` (possibly negative) straight to `_data._get_element`; the `size_t`-typed C++ overload rejects negatives.
**Proposed fix:** Resolve negative 1-D int to `idx+shape[0]` with an IndexError bounds check before calling `_get_element`; raise NumPy-style IndexError for OOB.

#### `numpy-scalar-int-unhandled` — NumPy integer scalars (`np.int64`/`np.int32`) as index raise "Unhandled slice type" 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor
t=FloatTensor(np.arange(24.).reshape(4,6))
t[np.int64(1)]; t[np.int32(1),np.int32(2)]  # numpy ok; masspcf TypeError
```

**NumPy:** `a[np.int64(1)]==a[1]` (row 1); `a[np.int32(1),np.int32(2)]==8.0`. NumPy treats integer scalars like Python ints.
**masspcf:** `TypeError 'Unhandled slice type'`. Verified.
**Root cause:** `masspcf/_tensor_base.py:108 & 202` use `isinstance(s, int)`, which is False for `np.integer`; `_coerce_index_arrays` (149) only catches `np.ndarray`, not 0-d numpy scalars.
**Proposed fix:** Use the `__index__` protocol (`operator.index`) for integer detection so any index-like object works; normalize `np.integer` to `int` in `_coerce_index_arrays`/`_pyslice_to_slice`.

#### `partial-multiint-tuple` — Partial all-int tuple shorter than ndim raises IndexError instead of returning a subtensor 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor
t=FloatTensor(np.arange(24.).reshape(2,3,4))
t[1,2]  # numpy shape (4,); masspcf IndexError
```

**NumPy:** On a 3-D array `a[1,2]` returns shape (4,), keeping the remaining axis.
**masspcf:** `IndexError 'Index out of range'` (the all-int path calls `_get_element((1,2))`; `assert_valid_index` requires `index.size()==ndim`). Verified.
**Root cause:** `masspcf/_tensor_base.py:202-203` — the all-int branch always calls `_get_element` and never checks `len(slices)==ndim` or pads trailing axes with `slice(None)`.
**Proposed fix:** Take the scalar `_get_element` path only when `len(slices)==ndim`; otherwise pad missing trailing axes with `slice(None)` and route through the slice/extract path with int axes dropped.

#### `empty-tuple-index` — `x[()]` (empty tuple) raises IndexError instead of returning the full array 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor
FloatTensor(np.arange(24.).reshape(4,6))[()]  # numpy full (4,6); masspcf IndexError
```

**NumPy:** `a[()]` returns the array unchanged (4,6); for 0-d arrays returns the scalar.
**masspcf:** `IndexError 'Index out of range'` — an empty tuple satisfies `all(isinstance(s,int))` vacuously, so `_get_element(())` is called with a zero-length index. Verified.
**Root cause:** `masspcf/_tensor_base.py:202` — `all(isinstance(s,int) for s in slices)` is True for the empty tuple, routing to `_get_element` with a zero-length index.
**Proposed fix:** Special-case the empty tuple (and a lone Ellipsis) to return `self` (full view) before the all-int branch, or guard the all-int path with `len(slices)>0 and len(slices)==ndim`.

#### `fancy-python-list-index` — Python list integer/bool fancy index raises "Unhandled slice type" (getitem + setitem) 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor
t=FloatTensor(np.arange(24.).reshape(4,6))
t[[0,2]]  # numpy (2,6); masspcf TypeError
t[[True,False,True,False]]  # numpy (2,6); masspcf TypeError
t[[0,2]]=9.0  # setitem also TypeError (but t[np.array([0,2])]=9.0 works -> inconsistent)
```

**NumPy:** A top-level list of ints is a single advanced index (`x[[0,2]]`→(2,6)); a length-N list of bools is an axis-0 mask (→(2,6)). Same for assignment.
**masspcf:** `TypeError 'Unhandled slice type'` for both getitem and setitem. The equivalent `np.array` form works, so list vs ndarray is inconsistent. Verified across all types.
**Root cause:** `masspcf/_tensor_base.py:148-157` `_coerce_index_arrays` only converts `np.ndarray`; a Python list is left unchanged, so `__getitem__`/`__setitem__` see no advanced entry and the list reaches `_pyslice_to_slice` (113) which raises.
**Proposed fix:** In `_coerce_index_arrays`, coerce a top-level list/sequence (not tuple) of ints to `IntTensor` and all-bool list to `BoolTensor` via `np.asarray` + dtype inspection. Preserve the tuple==multi-axis-basic distinction (tuples must NOT be treated as a single fancy list).

#### `fancy-1d-rowmask-misrouted` — 1-D boolean mask matching only axis 0 errors instead of selecting rows; sub-shape masks also error 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor
a=np.arange(24.).reshape(4,6); m=np.array([True,False,True,False])
FloatTensor(a)[m]  # numpy (2,6); masspcf ValueError
b=np.arange(24.).reshape(2,3,4); mm=(b.sum(2)%2==0)  # (2,3) mask
FloatTensor(b)[mm]  # numpy (6,4); masspcf ValueError
```

**NumPy:** A k-D bool mask matching `tensor.shape[:k]` collapses the first k axes into one of length `n_true`, keeping trailing axes: shape `(n_true, *shape[k:])`. 1-D row mask → (n_true,6); 2-D submask on 3-D → (6,4).
**masspcf:** `ValueError 'masked_select: mask shape does not match tensor shape'` for both. Verified.
**Root cause:** `masspcf/_tensor_base.py:171-173` routes ANY single `BoolTensor` index to `masked_select`, which (`tensor.tpp:511`) requires `mask.shape()==src.shape()`. There is no path for a leading-axes (sub-shape) mask.
**Proposed fix:** In the single-`BoolTensor` branch compare `mask.ndim` to `tensor.ndim`: equal → `masked_select` (flat); `mask.shape==tensor.shape[:k]` → collapse first k axes (gather true multi-index positions, result `(n_true,*trailing)`) via a new `leading_mask_select` or `index_select` over a leading-collapsed view; else raise IndexError matching NumPy's "boolean index did not match" message.

#### `fancy-multidim-int-index` — Multi-dimensional (N-D) or 0-d integer index array rejected ("indices must be 1D") 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor
FloatTensor(np.arange(24.).reshape(4,6))[np.array([[0,1],[2,3]])]  # numpy (2,2,6); masspcf ValueError
FloatTensor(np.arange(24.).reshape(4,6))[np.array(2)]  # numpy (6,); masspcf ValueError
```

**NumPy:** `x[idx]` where `idx.shape=(2,2)` → result `idx.shape + x.shape[1:]` = (2,2,6); a 0-d int array indexes like the scalar (drops the axis) → (6,).
**masspcf:** `ValueError 'indices must be 1D'` for both N-D and 0-d index arrays. Verified.
**Root cause:** `index_select` (`py_tensor.hpp:368`) → `validate_axis_indices` (`tensor.tpp:1063`) requires `indices.shape().size()==1`. masspcf supports only 1-D index arrays. `_coerce_index_arrays` wraps a 0-d ndarray into a 0-d `IntTensor` which then fails the same check.
**Proposed fix:** Generalize `index_select` (and the Python path) to accept an N-D index tensor: flatten `idx`, run the existing 1-D gather, reshape the selected axis back to `idx.shape`. Special-case 0-d integer arrays as a scalar index (route through int semantics, dropping the axis).

#### `fancy-broadcast-2d-index` — Broadcasting advanced indices `x[[[0],[1]],[0,2]]` rejected (2-D index unsupported) 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor
FloatTensor(np.arange(24.).reshape(4,6))[np.array([[0],[1]]),np.array([0,2])]  # numpy (2,2) [[0,2],[6,8]]; masspcf ValueError
```

**NumPy:** The two index arrays broadcast to (2,2) and gather elementwise → `[[0,2],[6,8]]`.
**masspcf:** `ValueError 'indices must be 1D'`. Verified.
**Root cause:** Compounds `fancy-multidim-int-index` (1-D-only index arrays, `tensor.tpp:1063`) and `fancy-multi-adv-outer-vs-vectorized` (no inter-index broadcasting).
**Proposed fix:** Requires both N-D index support and vectorized multi-index broadcasting: broadcast all advanced index arrays to a common shape, then gather elementwise into one advanced axis-block.

#### `setitem-single-int-multidim` — `x[i]=v` / `x[i]+=v` on a multi-dim tensor broken (single int treated as full multi-index) 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor
t=FloatTensor(np.arange(24.).reshape(4,6))
t[1]=7.0   # numpy sets row 1; masspcf IndexError
t[1]+=5    # numpy adds 5 to row 1; masspcf TypeError (desugars to broken set)
```

**NumPy:** `x[1]=scalar` broadcasts the scalar into the whole row; `x[1]=row` assigns the row; `x[1]+=5` reads row, adds, writes back.
**masspcf:** `IndexError 'Index out of range'` for scalar RHS; `TypeError _set_element` for Tensor RHS / augmented assignment. Verified (uniform across all types). Note getitem `x[i]` correctly returns a row, so setitem is asymmetric with getitem.
**Root cause:** `masspcf/_tensor_base.py:274-275` — `_setitem_slices` single-int branch always calls `_set_element([slices[0]], ...)`, treating the single int as a complete multi-index, with no `ndim>1` sub-tensor branch (unlike `_getitem_slices:196-199`).
**Proposed fix:** Mirror `_getitem_slices`: for a single int on `ndim>1`, resolve negatives, build a `SliceIndex` view (`slice(idx,idx+1)` over axis 0, squeeze) and broadcast-assign `val` into it (handling scalar and Tensor RHS); use `_set_element` only when `ndim==1`. This also fixes augmented assignment.

#### `setitem-negative-int` — Negative single int and negative ints in an all-int tuple rejected on assignment (TypeError) 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor
v=FloatTensor(np.arange(5.)); v[-1]=50.0  # numpy sets last; masspcf TypeError
t=FloatTensor(np.arange(24.).reshape(4,6)); t[1,-1]=5.0; t[-1,-1]=5.0  # numpy ok; masspcf TypeError
```

**NumPy:** `v[-1]=50` sets the last element; `a[1,-1]` sets (1,5); `a[-1,-1]` sets (3,5).
**masspcf:** `TypeError _set_element incompatible arguments` — negatives are passed straight to the `size_t`-indexed setter and rejected. Verified for 1-D negative and 2-D negative tuples across all types.
**Root cause:** `masspcf/_tensor_base.py:274-275` (single-int) and 279-280 (all-int tuple) forward raw negative ints to `_set_element` without resolution (mirror of the getitem gaps).
**Proposed fix:** Resolve every negative int against its axis size (bounds-check → IndexError) in both setitem branches before calling `_set_element`.

#### `setitem-scalar-broadcast-into-slice` — `x[1:3]=scalar` / `x[:]=scalar` / `x[1:3,2]=scalar` raise AttributeError (no scalar RHS for slice/region assign) 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor
t=FloatTensor(np.arange(24.).reshape(4,6))
t[1:3]=5.0  # numpy broadcasts; masspcf AttributeError 'float' has no attribute '_data'
```

**NumPy:** Scalar RHS is broadcast to fill the selected slice/region (`x[1:3]=5`, `x[:]=5`, `x[1:3,2]=0` all succeed).
**masspcf:** `AttributeError: 'float' object has no attribute '_data'` — the slice and multi-component else branches call `val._data` unconditionally. Verified for `x[1:3]=5`, `x[:]=5`, `x[1:3,2]=0`.
**Root cause:** `masspcf/_tensor_base.py:276-278 and 281-283` do `self._data[real_slices] = val._data` with no scalar path; the C++ `__setitem__` (`py_tensor.hpp:214`) only accepts a TTensor RHS.
**Proposed fix:** Detect non-Tensor RHS: decay it and broadcast a 1-element tensor into the target via `assign_from`, or add a C++ scalar-fill setitem overload (`self[slices]=scalar`).

#### `setitem-cross-dtype-tensor` — `x[1:3]=IntTensor` (cross-dtype Tensor RHS) raises confusing C++ TypeError instead of coercing 🟨 error

```python
import numpy as np; from masspcf.tensor import FloatTensor, IntTensor
t=FloatTensor(np.arange(24.).reshape(4,6))
t[1:3]=IntTensor(np.ones((2,6),dtype=np.int64))  # numpy casts int->float; masspcf TypeError
```

**NumPy:** Integer RHS is cast to the target float dtype and assigned.
**masspcf:** `TypeError '__setitem__() incompatible function arguments'` — `_validate_setitem_dtype` accepts the `IntTensor` but the C++ `__setitem__` only accepts a same-precision TTensor. Verified.
**Root cause:** `masspcf/_tensor_base.py:278/283` passes `val._data` straight to the same-type C++ `__setitem__` (`py_tensor.hpp:214`); `assign_from` supports cross-type (`tensor.tpp:143`) but the Python layer never reaches it.
**Proposed fix:** When RHS is a `NumericTensor` of a different dtype, coerce via `val.astype(self.dtype)` before the C++ assign, or expose cross-type `assign_from` bindings.

### Category: Missing Features (🟦)

#### `ellipsis-unhandled` — Ellipsis (`...`) not supported in getitem or setitem 🟦 missing

```python
import numpy as np; from masspcf.tensor import FloatTensor
t=FloatTensor(np.arange(24.).reshape(4,6))
t[...]; t[...,0]; t[0,...]; t[...,1:3]  # numpy ok; masspcf TypeError
t[...,0]=9.0  # setitem also TypeError
```

**NumPy:** `x[...]` full array; `x[...,0]`→(4,); `x[0,...]`→(6,); `x[...,1:3]`→(4,2); assignment `x[...,0]=9` sets column 0. Ellipsis expands to `(ndim - explicit)` full slices.
**masspcf:** `TypeError 'Unhandled slice type'` for every Ellipsis form, both getitem and setitem. Verified.
**Root cause:** `masspcf/_tensor_base.py:107-113` `_pyslice_to_slice` handles only int and slice; Ellipsis falls through to `raise TypeError('Unhandled slice type')`. No Ellipsis-expansion pass exists in `__getitem__`/`__setitem__`.
**Proposed fix:** Add an Ellipsis-normalization pass at the top of `__getitem__`/`__setitem__`: count non-Ellipsis non-None axis-consuming entries, replace a single Ellipsis with `(ndim - count)` `slice(None)`, and reject more than one Ellipsis with IndexError.

#### `newaxis-none-unhandled` — `None` / `np.newaxis` not supported in getitem or setitem 🟦 missing

```python
import numpy as np; from masspcf.tensor import FloatTensor
t=FloatTensor(np.arange(24.).reshape(4,6))
t[None]; t[:,None]; t[:,None,:]; t[0,None]  # numpy inserts length-1 axis; masspcf TypeError
```

**NumPy:** `None`/`np.newaxis` inserts a length-1 axis: `x[None]`→(1,4,6); `x[:,None]`→(4,1,6); `x[0,None]`→(1,6).
**masspcf:** `TypeError 'Unhandled slice type'` for all `None`/newaxis forms (`np.newaxis` is `None`). Verified.
**Root cause:** `masspcf/_tensor_base.py:107-113` `_pyslice_to_slice` has no `None` case; the C++ `Slice` variant (`tensor.hpp:54`) has no newaxis member.
**Proposed fix:** Handle `None` in the Python layer: strip `None` entries before building the C++ slice list, record their positions (counting only axis-consuming entries, accounting for int-dropped axes), run `extract`, then call `expand_dims` (already exists at `_tensor_base.py:427`) at each recorded output position.

#### `docs-test-gaps` — No tests/docs for negative basic indices, negative slice bounds, ellipsis, newaxis, list-fancy, vectorized multi-index, or view-aliasing; a C++ test encodes the clamp bug 🟦 missing

```text
grep test/python for ellipsis/newaxis/None-index/list-fancy -> 0 hits
test/test_tensor.cpp:288 SliceRangeStartNegativeClampsToZero asserts the buggy clamp
docs/indexing.rst documents only negative-step and advanced-array negatives
```

**NumPy:** These are core NumPy indexing features users expect.
**masspcf:** Untested and largely undocumented; the negative-slice clamp behavior is asserted as correct in a GoogleTest, which will resist a correctness fix. The documented view-aliasing guarantee (`docs/indexing.rst:35`) has no test. Multi-advanced is locked to `np.ix_` parity, not NumPy zip.
**Root cause:** `test/python/test_tensor_extract.py` (only non-negative single ints / no negative bounds), `test/python/test_advanced_indexing.py` (only `np.array`/`IntTensor`, `_assert_outer_index` locks `ix_`), `test/test_tensor.cpp:288` (clamp asserted correct); `docs/indexing.rst` lacks negative-basic/ellipsis/newaxis/list sections.
**Proposed fix:** Add parametrized parity tests for negative basic ints, negative slice bounds (replacing the clamp test with a resolution test once fixed), ellipsis, newaxis, list-fancy, vectorized multi-index, and view write-through; add docs sections / limitations notes. Add C++ tests for `index_select`/`outer_select` bounds and negative resolution.

### Category: Minor / Inconsistent (⬜)

#### `oob-single-int-wrong-exception` — Out-of-range single int raises ValueError from squeeze / TypeError instead of IndexError ⬜ inconsistent

```python
import numpy as np; from masspcf.tensor import FloatTensor
FloatTensor(np.arange(24.).reshape(4,6))[10]  # numpy IndexError; masspcf ValueError 'squeeze: cannot squeeze axis 0 with size 0'
FloatTensor(np.arange(6.))[-7]  # numpy IndexError; masspcf TypeError
```

**NumPy:** `a[10]` / `b[-7]` raise `IndexError: index out of bounds for axis 0 with size N`.
**masspcf:** Positive OOB single int → ValueError from squeeze (empty slice path); negative OOB 1-D → TypeError from `_get_element`. Multi-int OOB (`a[5,5]`) correctly raises IndexError, so single vs multi paths are inconsistent. Verified.
**Root cause:** `masspcf/_tensor_base.py:197-199` — no bounds check before slicing; OOB positive int yields an empty slice that `squeeze(0)` rejects; negative 1-D reaches the `size_t` `_get_element` overload.
**Proposed fix:** Bounds-check the resolved scalar index against `self.shape[0]` in `_getitem_slices` (and the all-int tuple path) and raise IndexError with a NumPy-style message before any slice construction.

#### `distmat-symmat-scalar-getitem` — DistanceMatrix/SymmetricMatrix scalar classes have a restrictive `__getitem__` (only positive (i,j) tuple) ⬜ inconsistent

```python
from masspcf.distance_matrix import DistanceMatrix
dm=DistanceMatrix(4); dm[0,1]=5.0
dm[0]       # TypeError cannot unpack int
dm[-1,-1]   # TypeError (C++ tuple[size_t,size_t] rejects -1)
dm[0:2,0:2] # TypeError (slices rejected)
```

**NumPy:** A dense 2-D matrix supports `m[i]`, `m[-1,-1]`, `m[0:2,0:2]` with full semantics.
**masspcf:** These are NOT Tensor subclasses; their `__getitem__`/`__setitem__` unconditionally unpack a 2-tuple and forward to a C++ `(size_t,size_t)` binding. Single int → "cannot unpack non-iterable int object"; negative → TypeError; slices → TypeError; no IndexError on OOB. Their `*Tensor` counterparts go through the shared path. Verified. Divergent from the tensor-of-matrices types.
**Root cause:** `masspcf/distance_matrix.py:94-100` and `masspcf/symmetric_matrix.py:94-100` — both unpack a 2-tuple (`i,j=ij`) and pass to a `size_t`-only C++ binding; no negative resolution, no slice/single-int handling, no bounds-check translation.
**Proposed fix:** Decide the intended contract. At minimum resolve negative `i/j` against size and raise IndexError (not TypeError) on OOB; give a clear error for single-int/slice forms; keep consistent with the `*Tensor` versions.

#### `pointcloud-element-degenerate` — PointCloudTensor scalar element is a sub-FloatTensor (degenerate 0-d when empty), not a dedicated wrapper ⬜ inconsistent

```python
import masspcf as mp; from masspcf.typing import pcloud64
pc=mp.zeros((3,),dtype=pcloud64)
print(type(pc[1]).__name__, list(pc[1].shape))  # FloatTensor [] (0-d, degenerate)
```

**NumPy:** N/A (no NumPy oracle for container types).
**masspcf:** `PointCloudTensor._represent_element` returns a `FloatTensor` wrapping the per-element points tensor; for an empty (zeros) element this is a 0-d `FloatTensor` (shape `[]`), which is meaningless for a point cloud (n_points × n_dims). Contrast `PcfTensor[i]`→`Pcf`, `DistanceMatrixTensor[i]`→`DistanceMatrix`, `SymmetricMatrixTensor[i]`→`SymmetricMatrix`. Verified `pc[1]` type=FloatTensor shape=`[]`.
**Root cause:** `masspcf/tensor.py:287` `PointCloudTensor._represent_element` returns `FloatTensor(element)`; empty point clouds have 0 points so the wrapped element collapses to 0-d.
**Proposed fix:** Either introduce a PointCloud scalar wrapper (mirroring DistanceMatrix/SymmetricMatrix), or guarantee point-cloud elements are always 2-D FloatTensors (0 × n_dims even when empty). The degenerate 0-d shape is the concrete bug.

#### `scalar-result-python-float` — Scalar index result returns a Python float, not a 0-d tensor/numpy scalar ⬜ minor

```python
import numpy as np; from masspcf.tensor import FloatTensor
print(type(FloatTensor(np.arange(24.).reshape(4,6))[1,2]))  # <class 'float'>; numpy a[1,2] -> np.float64
```

**NumPy:** Indexing to a single element yields a 0-d numpy scalar (`np.float64`) supporting `.shape`/`.dtype`/`.ndim`.
**masspcf:** Returns a bare Python float (right value, no `.shape`/`.dtype`/`.ndim`). Verified type is `float`. API/semantics divergence, not a wrong value.
**Root cause:** `masspcf/_tensor_base.py:195 & 203` call `_represent_element`, which for `NumericTensor` (`tensor.py`) returns the raw element; no 0-d tensor representation.
**Proposed fix:** Decide policy: return a 0-d tensor (NumPy-consistent) or at least a typed numpy scalar; or document the divergence.

#### `len-0d-wrong-exception` — `len()` of a 0-d tensor raises IndexError instead of TypeError ⬜ minor

```python
import numpy as np; from masspcf.tensor import FloatTensor
len(FloatTensor(np.array(5.0)))  # numpy TypeError 'len() of unsized object'; masspcf IndexError
```

**NumPy:** `len()` of a 0-d array raises `TypeError: len() of unsized object`.
**masspcf:** `IndexError 'Attempted to get index >= len'` — `__len__` unconditionally indexes `shape[0]`. Verified.
**Root cause:** `masspcf/_tensor_base.py:285-287` `__len__`/`__iter__` index `shape[0]` without an `ndim==0` guard.
**Proposed fix:** In `__len__`/`__iter__`, raise `TypeError('len() of unsized object')` / appropriate error when `ndim==0`.

#### `too-many-indices-wrong-message` — Too many indices raises generic "Index out of range" instead of NumPy's message ⬜ minor

```python
import numpy as np; from masspcf.tensor import FloatTensor
FloatTensor(np.arange(24.).reshape(4,6))[0,1,2]  # numpy 'too many indices...'; masspcf IndexError 'Index out of range'
```

**NumPy:** `IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed`.
**masspcf:** `IndexError 'Index out of range'` (correct type, unhelpful non-NumPy message). Verified.
**Root cause:** `masspcf/_tensor_base.py:202-203` passes the 3-tuple to `_get_element` with no arity check; the C++ side raises the generic message.
**Proposed fix:** Check the number of axis-consuming entries against `self.ndim` and raise the NumPy-style IndexError early.

#### `fancy-float-index-wrong-error` — Float index array raises TypeError "Unhandled slice type" instead of NumPy's IndexError ⬜ minor

```python
import numpy as np; from masspcf.tensor import FloatTensor
FloatTensor(np.arange(24.).reshape(4,6))[np.array([0.0,1.0])]  # numpy IndexError; masspcf TypeError
```

**NumPy:** `IndexError: arrays used as indices must be of integer (or boolean) type`.
**masspcf:** `TypeError 'Unhandled slice type'` — both reject float indices but masspcf's exception type/message is misleading. Verified.
**Root cause:** `masspcf/_tensor_base.py:154-155` leaves a non-int/non-bool ndarray unchanged, so it reaches `_pyslice_to_slice` (113).
**Proposed fix:** In `_coerce_index_arrays`, when an ndarray is neither bool nor integer, raise `IndexError('arrays used as indices must be of integer (or boolean) type')`.

#### `float-scalar-index-wrong-error` — Float scalar index `x[1.0]` raises TypeError "Unhandled slice type" instead of NumPy IndexError ⬜ minor

```python
import numpy as np; from masspcf.tensor import FloatTensor
FloatTensor(np.arange(24.).reshape(4,6))[1.0]  # numpy IndexError 'only integers...'; masspcf TypeError
```

**NumPy:** `IndexError` explaining only integers/slices/ellipsis/newaxis/integer-or-bool arrays are valid.
**masspcf:** `TypeError 'Unhandled slice type'` — correct that it errors, wrong type/message.
**Root cause:** `masspcf/_tensor_base.py:113` catch-all `raise TypeError('Unhandled slice type')` is the single error path for every unrecognized index.
**Proposed fix:** Replace the catch-all with an IndexError mirroring NumPy's message including `repr(type(s))`.

#### `slice-all-dead-code` — `SliceAll` C++ type / `slice_all()` binding is dead code (excluded from the `Slice` variant) ⬜ minor

```python
import masspcf._mpcf_cpp as cpp
cpp.slice_all()  # exists but slice(None) maps to slice_range(None,None,None) (SliceRange), not SliceAll
```

**NumPy:** N/A (internal).
**masspcf:** `_pyslice_to_slice` maps `slice(None)` to `SliceRange`; `mpcf::all()` returns `SliceRange{}` not `SliceAll`. `slice_all` binding result cannot be placed in a slice vector since `Slice = variant<SliceIndex,SliceRange>`. Verified.
**Root cause:** `include/mpcf/tensor.hpp:54` `using Slice = std::variant<SliceIndex, SliceRange>` excludes `SliceAll` (defined at line 40); `_tensor_base.py:110-111` always emits `SliceRange` for full slices; `py_tensor.cpp:84` registers `slice_all` unused.
**Proposed fix:** Remove the `SliceAll` struct / `all()` / `slice_all` binding, or add `SliceAll` to the variant and emit it for `slice(None)`. Cosmetic; no behavioral bug.

### Category: Confirmed Parity / No Fix Needed (🟩)

#### `crosstype-uniform-shared-path` — All eight tensor types share one `__getitem__`/`__setitem__`; gaps and working patterns are identical 🟩 ok

All wrappers delegate to `Tensor.__getitem__`/`__setitem__` in `_tensor_base.py`. Uniform: positive slices/steps, single positive int, single `np.array` fancy index, full-shape bool mask all work on every type; every gap (neg int, list fancy, ellipsis, newaxis, multi-advanced, axis-0 bool mask, empty tuple) reproduces identically on every type. **Implication:** fixing the gaps in `_tensor_base.py` fixes them for all types at once.

#### `ok-positive-int-slice` — Positive ints, positive/empty slices, steps, and negative-step with omitted/positive bounds match NumPy 🟩 ok

`t[0]`, `t[1:3]`, `t[::2]`, `t[::-1]`, `t[3:0:-1]`, `t[:,2:4]`, `t[1:3,2:4]` all match (shapes and values). Positive OOB (`a[4,0]`, `a[0,6]`) correctly raises IndexError. (`include/mpcf/tensor.tpp:1470-1508` step branches are correct for non-negative/omitted bounds.)

#### `ok-single-fancy-and-negatives` — Single 1-D integer-array fancy index (ndarray/IntTensor), negative entries, OOB, and per-axis bool mask via `x[:,mask]` match NumPy 🟩 ok

`a[[0,2]]`, `a[[-1,-2]]`, `a[:,[1,3]]`, duplicate/empty arrays, full-shape bool mask, `x[:,colmask]`, `a[1:3,[0,2]]` (combined basic+single-advanced) all match; `a[[0,10]]` raises IndexError. `_resolve_negative_indices` (`_tensor_base.py:116-125`) and `validate_axis_indices` (`tensor.tpp:1063`) handle negatives/bounds correctly. With exactly one advanced index, outer == vectorized so it coincides with NumPy. **Ensure fixes don't regress these.**

#### `ok-view-writethrough` — View semantics: slice and single-int views write through; mask/fancy reads copy and assigns write through (matches NumPy) 🟩 ok

`sub=t[1:3]; sub[0,0]=999` updates the parent. `extract()` shares the `shared_ptr` storage (`tensor.tpp:1432`); `masked_select`/`index_select` copy on read (`tensor.tpp:509/1081`); `masked_assign`/`index_assign` write through; wrong-length mask RHS raises ValueError.

#### `ok-masked-select-corder` — `masked_select` returns correct C-order flat output even on non-contiguous (transposed) views 🟩 ok

Matches NumPy exactly on a transposed/non-contiguous view. `walk()` iterates logical C-order regardless of physical strides; `mask.shape()==src.shape()` validated up front. (`include/mpcf/tensor.tpp:509-533`.)

#### `ok-setitem-basic` — Scalar/int element assignment, Tensor-RHS broadcast into a slice, and ndarray fancy assign match NumPy 🟩 ok

`t[2,3]=5` (stored 5.0), Tensor-RHS row/col broadcast into a (2,6) slice, `t[np.array([0,2])]=9.0` sets rows 0,2. `assign_from` (`tensor.tpp:143`) broadcasts and rejects only true mismatches. (Scalar-RHS-into-slice is the gap; see `setitem-scalar-broadcast-into-slice`.)

#### `ok-index-select-bounds-safe` — `index_select` / `axis_select` bounds-check and reject raw negatives safely 🟩 ok

`index_select` raises IndexError for `1000` and for raw `-1`; the Python layer resolves negatives before calling. Internally consistent and memory-safe (unlike `outer_select`). (`include/mpcf/tensor.tpp:1054-1077` `validate_axis_indices` + `_tensor_base.py:116-125`.) **`outer_select` should adopt the same validation (see `outer-select-no-bounds-check`).**

## 5. Root-Cause Clusters

The crucial observation is that a **centralized NumPy-style index/slice normalization** removes a whole class of bugs at once. The defects are concentrated in five undersized helpers plus the C++ `extract()` and the outer-assignment kernels; each cluster below names where the fix lands, the findings it resolves, and the unified fix.

### Cluster A — Centralize NumPy-style negative-index + `slice.indices()` resolution and bounds checking in `extract()`
- **Where:** `include/mpcf/tensor.tpp:1454-1512` (`extract()`: `SliceIndex` + `SliceRange` step>0/step<0/step==0 branches).
- **Resolves:** `neg-sliceindex-oob-write-segv`, `sliceindex-positive-oob-no-check`, `slicerange-neg-bounds-posstep`, `slicerange-neg-bounds-negstep`, `slicerange-step-zero`, `neg-int-multidim-squeeze`, `setitem-negative-slice-bounds`, `oob-single-int-wrong-exception`, `docs-test-gaps`.
- **Unified fix:** Rewrite `extract()` to mirror Python's `slice.indices(n)` and single-int semantics: (a) **SliceIndex** — capture dim size `n` before overwriting `m_shape[i]=1`, resolve `if (ix<0) ix+=n`, throw `pybind11::index_error` if `ix∉[0,n)`; (b) **SliceRange step>0** — resolve negatives (`start+=n`, `stop+=n`) before clamping into `[0,n]`; (c) **step<0** — resolve negatives before clamping into `[-1,n-1]`; (d) **step==0** — throw `pybind11::value_error("slice step cannot be zero")`. This single function rewrite fixes the SIGSEGV write crash, the OOB reads, all negative-bound wrong-results (read and write), `step==0`, and — combined with the Python single-int branch using `slice(idx,idx+1)` — the negative-single-int multidim squeeze error. Also replace the C++ test `SliceRangeStartNegativeClampsToZero` (`test_tensor.cpp:288`) with a resolution expectation.

### Cluster B — Broaden index-object recognition in the Python normalization layer with an Ellipsis/newaxis pre-pass
- **Where:** `masspcf/_tensor_base.py:107-113` (`_pyslice_to_slice`), 140-158 (`_coerce_index_arrays`), 160-189 (`__getitem__`), 218-283 (`__setitem__`).
- **Resolves:** `ellipsis-unhandled`, `newaxis-none-unhandled`, `numpy-scalar-int-unhandled`, `fancy-python-list-index`, `bool-scalar-index`, `fancy-float-index-wrong-error`, `float-scalar-index-wrong-error`, `empty-tuple-index`, `partial-multiint-tuple`, `too-many-indices-wrong-message`.
- **Unified fix:** Add a normalization pass at the top of `__getitem__`/`__setitem__`: expand a single Ellipsis into `(ndim - axis-consuming-count)` `slice(None)` and reject multiple; strip `None`/`np.newaxis`, record positions, and `expand_dims` afterward; treat Python `bool` / `np.bool_` scalars as 0-d boolean indices (`True`→newaxis, `False`→len-0 axis) BEFORE the int check; use `operator.index`/`__index__` for integer detection (covers `np.integer`); coerce top-level lists of ints→`IntTensor` and of bools→`BoolTensor` (preserving the tuple==basic-multi-axis distinction); raise NumPy-style IndexError for float/unrecognized indices; special-case empty tuple to return `self`; pad partial all-int tuples with trailing `slice(None)`; arity-check against `ndim`.

### Cluster C — Resolve negative ints and route single-int multidim through a sliced sub-tensor on the assignment side (symmetry with getitem)
- **Where:** `masspcf/_tensor_base.py:272-283` (`_setitem_slices`) and 193-205 (`_getitem_slices` int paths).
- **Resolves:** `neg-int-in-int-tuple`, `neg-int-1d-getelement`, `setitem-single-int-multidim`, `setitem-negative-int`, `setitem-scalar-broadcast-into-slice`, `setitem-cross-dtype-tensor`.
- **Unified fix:** In both `_getitem_slices` and `_setitem_slices`: resolve negative ints against the matching shape dim (IndexError on OOB) before any `_get_element`/`_set_element` call; for a single int on `ndim>1`, build a `SliceIndex` sub-tensor view (`slice(idx,idx+1)`+squeeze) and broadcast-assign rather than calling `_set_element` with a 1-element index; accept scalar RHS by decaying+broadcasting (or a C++ scalar-fill overload); coerce a cross-dtype `NumericTensor` RHS via `astype(self.dtype)`. Use `_get_element`/`_set_element` only when `len==ndim`.

### Cluster D — Replace sequential outer/Cartesian advanced indexing with NumPy vectorized (broadcast+zip) gather/scatter, plus N-D index support and front-placement
- **Where:** `masspcf/_tensor_base.py:179-189` (`__getitem__` advanced loop), 244-270 (`__setitem__` advanced loop); `include/mpcf/tensor.tpp:757-855` (`resolve_selectors`/`outer_select`/`outer_assign`), 1054-1077 (`validate_axis_indices` for N-D).
- **Resolves:** `fancy-multi-adv-outer-vs-vectorized`, `fancy-multi-adv-mismatched-len`, `fancy-separated-adv-no-front-move`, `fancy-broadcast-2d-index`, `fancy-two-bool-masks-outer`, `fancy-multidim-int-index`, `fancy-1d-rowmask-misrouted`, `outer-select-no-bounds-check`.
- **Unified fix:** Add a C++ `vectorized_select`/`vectorized_assign` that takes a list of `(axis, index_tensor)`: broadcast all advanced index arrays to a common shape (raising IndexError on incompatible shapes), generalize indices to N-D (flatten+gather+reshape), gather/scatter element-wise into a single combined advanced axis-block, and place that block at the front when the advanced indices are non-contiguous (else in-position). Convert bool masks to int index arrays before this path; route a sole sub-shape bool mask through a leading-axes collapse. Bounds-check+resolve negatives in `resolve_selectors`. Keep the existing `outer_select`/`outer_assign` behind an explicit `np.ix_`-style API only.

### Cluster E — Add values-shape validation to the multi-axis/outer assignment C++ kernels (defense against OOB writes)
- **Where:** `include/mpcf/tensor.tpp:731-743` (`multi_axis_assign`), 843-855 (`outer_assign`).
- **Resolves:** `multi-axis-assign-no-shape-validation`, `outer-assign-no-shape-validation`.
- **Unified fix:** Before the assignment walk in `multi_axis_assign` and `outer_assign`, compute `expected_shape` from `dst.shape()` with each selected axis set to its selector length (`true_indices.size()` / resolved indices count) and throw `std::invalid_argument` if `values.shape() != expected_shape`, mirroring `axis_assign` (`tensor.tpp:643-647`). Prevents OOB writes / memory corruption.

### Cluster F — Reconcile the scalar matrix wrappers and the PointCloud element representation with the shared tensor indexing contract
- **Where:** `masspcf/distance_matrix.py:94-100`, `masspcf/symmetric_matrix.py:94-100`, `masspcf/tensor.py:287` (`PointCloudTensor._represent_element`); `masspcf/_tensor_base.py:285-287` (`__len__`).
- **Resolves:** `distmat-symmat-scalar-getitem`, `pointcloud-element-degenerate`, `len-0d-wrong-exception`, `scalar-result-python-float`.
- **Unified fix:** For `DistanceMatrix`/`SymmetricMatrix`, resolve negative `(i,j)`, raise IndexError on OOB, and give clear errors for single-int/slice forms (or support them) consistent with the `*Tensor` versions. For `PointCloudTensor`, either add a PointCloud scalar wrapper or guarantee elements are always 2-D (0 × n_dims when empty). Guard `__len__`/`__iter__` for `ndim==0` (TypeError). Optionally return a 0-d tensor / numpy scalar from scalar indexing for closer NumPy parity.

## 6. Implementation Roadmap

| Priority | Step | Addresses (ids) | Effort | Where |
|---|---|---|:---:|---|
| **P0** | Fix the memory-unsafe negative/OOB `SliceIndex` in `extract()`: resolve negative single-int indices against dim size and bounds-check, throwing `pybind11::index_error`. Eliminates the SIGSEGV write crash and the silent OOB-garbage reads. | `neg-sliceindex-oob-write-segv`, `sliceindex-positive-oob-no-check` | S | `include/mpcf/tensor.tpp:1454-1459` (SliceIndex branch of `extract()`) |
| **P0** | Rewrite the `SliceRange` branches in `extract()` to mirror Python `slice.indices(n)`: resolve negative start/stop (add `dim_size`) before clamping for both positive and negative step, and throw ValueError on `step==0`. Fixes all silently-wrong negative-slice results (read and write) and the missing `step==0` error. | `slicerange-neg-bounds-posstep`, `slicerange-neg-bounds-negstep`, `slicerange-step-zero`, `setitem-negative-slice-bounds` | M | `include/mpcf/tensor.tpp:1460-1512`; update C++ test `test/test_tensor.cpp:288` (`SliceRangeStartNegativeClampsToZero`) to expect resolution |
| **P0** | Add values-shape validation to `multi_axis_assign` and `outer_assign` before the assignment walk, throwing `std::invalid_argument` on mismatch. Prevents OOB writes / memory corruption. | `multi-axis-assign-no-shape-validation`, `outer-assign-no-shape-validation` | S | `include/mpcf/tensor.tpp:731-743, 843-855` |
| **P0** | Resolve negative ints in the Python int / `_get_element` and `_set_element` paths (single int and all-int tuples), and route single-int multidim setitem through a sliced sub-tensor with broadcast assign. Raise IndexError with NumPy-style messages for OOB. Fixes the most common user-facing negative-index and single-int-assignment failures across all 8 types. | `neg-int-multidim-squeeze`, `neg-int-in-int-tuple`, `neg-int-1d-getelement`, `setitem-single-int-multidim`, `setitem-negative-int`, `oob-single-int-wrong-exception` | M | `masspcf/_tensor_base.py:191-205` (`_getitem_slices`), 272-283 (`_setitem_slices`) |
| **P0** | Add bounds checking + negative resolution to `detail::resolve_selectors` (`outer_select` int branch) so the C++ entry point is memory-safe regardless of caller. | `outer-select-no-bounds-check` | S | `include/mpcf/tensor.tpp:757-792` (`resolve_selectors`) |
| **P1** | Replace sequential outer/Cartesian multi-advanced indexing with NumPy vectorized semantics: broadcast index arrays to a common shape, gather/scatter element-wise into one combined advanced axis-block, raise on incompatible shapes, and move the block to the front when advanced indices are non-contiguous. Add a C++ `vectorized_select`/`vectorized_assign` and an explicit `np.ix_`-style opt-in for the old outer behavior. Convert bool masks to int arrays for this path. | `fancy-multi-adv-outer-vs-vectorized`, `fancy-multi-adv-mismatched-len`, `fancy-separated-adv-no-front-move`, `fancy-two-bool-masks-outer` | L | `masspcf/_tensor_base.py:179-189, 244-270`; `include/mpcf/tensor.tpp:757-855` (new vectorized kernels) |
| **P1** | Add an Ellipsis/newaxis normalization pass and broaden index recognition in the Python layer: expand a single Ellipsis (reject multiple); strip `None` and `expand_dims` afterward; treat Python `bool` / `np.bool_` scalars as 0-d boolean indices; use `__index__` for integer detection (`np.integer`); coerce top-level int/bool lists to `IntTensor`/`BoolTensor` (preserving tuple==basic distinction); special-case empty tuple; pad partial all-int tuples; arity-check; raise NumPy-style IndexError for float/unrecognized indices. | `ellipsis-unhandled`, `newaxis-none-unhandled`, `numpy-scalar-int-unhandled`, `fancy-python-list-index`, `bool-scalar-index`, `empty-tuple-index`, `partial-multiint-tuple`, `fancy-float-index-wrong-error`, `float-scalar-index-wrong-error`, `too-many-indices-wrong-message` | L | `masspcf/_tensor_base.py:107-113, 140-158, 160-189, 218-283` |
| **P1** | Support sub-shape / leading-axes boolean masks: when a sole `BoolTensor` index has `ndim < tensor.ndim` and matches the leading axes, collapse those axes into one of length `n_true` keeping trailing axes (a `leading_mask_select` C++ helper or `index_select` over a leading-collapsed view); error with NumPy-style message otherwise. | `fancy-1d-rowmask-misrouted` | M | `masspcf/_tensor_base.py:171-173`; `include/mpcf/tensor.tpp` (new leading-mask select) |
| **P1** | Support N-D and 0-d integer index arrays: generalize `index_select`/`validate_axis_indices` to flatten+gather+reshape to `idx.shape`; treat 0-d int arrays as scalar indices (drop the axis). Unblocks broadcast advanced indexing once vectorized gather lands. | `fancy-multidim-int-index`, `fancy-broadcast-2d-index` | M | `include/mpcf/tensor.tpp:1054-1077` (`validate_axis_indices`), 1081 (`index_select`); `masspcf/_tensor_base.py:152, 180-187` |
| **P2** | Coerce cross-dtype Tensor RHS on assignment (`astype(self.dtype)` or cross-type `assign_from` bindings) and accept scalar RHS into slice/region assigns via decay+broadcast or a C++ scalar-fill setitem overload. | `setitem-cross-dtype-tensor`, `setitem-scalar-broadcast-into-slice` | M | `masspcf/_tensor_base.py:276-283`; `include/mpcf/tensor.tpp:143` (`assign_from`), `src/python/py_tensor.hpp:214` (`__setitem__` overloads) |
| **P2** | Reconcile scalar matrix wrappers and PointCloud element representation: resolve negatives + raise IndexError in `DistanceMatrix`/`SymmetricMatrix` `__getitem__`/`__setitem__` (or support single-int/slice forms); give `PointCloudTensor` elements a stable 2-D shape or a dedicated PointCloud scalar wrapper; guard `__len__`/`__iter__` for `ndim==0` (TypeError); optionally return a 0-d tensor / numpy scalar from scalar indexing. | `distmat-symmat-scalar-getitem`, `pointcloud-element-degenerate`, `len-0d-wrong-exception`, `scalar-result-python-float` | M | `masspcf/distance_matrix.py:94-100`, `masspcf/symmetric_matrix.py:94-100`, `masspcf/tensor.py:287`, `masspcf/_tensor_base.py:285-287` |
| **P2** | Add parity tests (negative basic ints, negative slice bounds, ellipsis, newaxis, list-fancy, vectorized multi-index, scalar bool, view write-through, C++ `index_select`/`outer_select` bounds) and update `docs/indexing.rst` with the new behaviors / limitations. Replace the clamp-asserting C++ test with a resolution test. | `docs-test-gaps` | M | `test/python/test_tensor_extract.py`, `test/python/test_advanced_indexing.py`, `test/python/test_tensor_mask.py`, `test/test_tensor.cpp:288`, `test/test_tensor_masked.cpp`, `docs/indexing.rst` |
| **P3** | Remove dead `SliceAll` code or include it in the `Slice` variant and emit it for `slice(None)`. | `slice-all-dead-code` | S | `include/mpcf/tensor.hpp:40,54,56`; `src/python/py_tensor.cpp:84`; `masspcf/_tensor_base.py:110-111` |

**Suggested sequencing.** Land the five **P0** items first and in roughly this order: the two `extract()` rewrites (Cluster A) are the keystone — they remove the SIGSEGV and the entire negative-slice wrong-result family for both reads and writes, and they make the negative-single-int Python fix trivial; do the two C++ shape-validation guards and the `resolve_selectors` bounds check alongside them since they are small, independent, and close memory-safety holes that the test suite cannot otherwise catch. Then add the Python negative-int / single-int-setitem resolution (Cluster C's P0 slice), which depends on the `extract()` SliceIndex fix to be fully clean. After P0 the package is memory-safe and correct for all basic indexing. **P1** is the larger correctness/feature wave: the vectorized multi-advanced rewrite (Cluster D) is the single largest behavior change and should be developed behind tests against NumPy's `np.ix_` for the opt-in path and NumPy's default for the new path; the Ellipsis/newaxis/index-recognition pass (Cluster B) is independent and can proceed in parallel, but should land before or with the N-D index and leading-mask work so the normalized index stream feeds the new gather. Defer **P2** (cross-dtype/scalar-RHS assignment, scalar-wrapper reconciliation, and the full parity test/doc sweep) until the semantics are settled, and treat **P3** (`SliceAll` dead code) as cleanup. Throughout, guard the confirmed-parity cases (`ok-*` findings) with regression tests so the rewrites do not silently break the working positive-path and single-advanced behaviors.

## 7. Appendix: Reproduction Script

Run from `test/` after building/installing (`cd test && micromamba run -n py313 python repro_indexing.py`). The script avoids the SIGSEGV repros (those terminate the process — run them separately per Section 4) and prints NumPy-vs-masspcf for the non-crashing gaps so the maintainer can diff before/after fixing.

```python
import numpy as np
from masspcf.tensor import FloatTensor, IntTensor, BoolTensor

def show(label, fn):
    try:
        r = fn()
        try:
            r = np.asarray(r)
            print(f"{label}: OK shape={r.shape} vals={r.ravel()[:6]}")
        except Exception:
            print(f"{label}: OK (non-array) -> {r!r} type={type(r).__name__}")
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {e}")

a = np.arange(24.).reshape(4, 6)
b = np.arange(24.).reshape(2, 3, 4)

cases = [
    # (label, numpy lambda, masspcf lambda)
    ("neg single int (N-D) x[-1]",        lambda: a[-1],                 lambda: FloatTensor(a)[-1]),
    ("neg single int (1-D) v[-1]",        lambda: np.arange(6.)[-1],     lambda: FloatTensor(np.arange(6.))[-1]),
    ("neg in int tuple x[1,-1]",          lambda: a[1, -1],              lambda: FloatTensor(a)[1, -1]),
    ("neg slice posstep x[-3:-1]",        lambda: a[-3:-1],              lambda: FloatTensor(a)[-3:-1]),
    ("neg slice posstep x[-2:]",          lambda: a[-2:],                lambda: FloatTensor(a)[-2:]),
    ("neg slice col x[:,-2:]",            lambda: a[:, -2:],             lambda: FloatTensor(a)[:, -2:]),
    ("neg slice negstep x[-1:-4:-1]",     lambda: a[-1:-4:-1],           lambda: FloatTensor(a)[-1:-4:-1]),
    ("step==0 x[::0]",                    lambda: a[::0],                lambda: FloatTensor(a)[::0]),
    ("ellipsis x[...,0]",                 lambda: a[..., 0],             lambda: FloatTensor(a)[..., 0]),
    ("newaxis x[:,None]",                 lambda: a[:, None],            lambda: FloatTensor(a)[:, None]),
    ("np scalar int x[np.int64(1)]",      lambda: a[np.int64(1)],        lambda: FloatTensor(a)[np.int64(1)]),
    ("list fancy x[[0,2]]",               lambda: a[[0, 2]],             lambda: FloatTensor(a)[[0, 2]]),
    ("scalar bool x[True]",               lambda: a[True],               lambda: FloatTensor(a)[True]),
    ("empty tuple x[()]",                 lambda: a[()],                 lambda: FloatTensor(a)[()]),
    ("partial int tuple b[1,2]",          lambda: b[1, 2],               lambda: FloatTensor(b)[1, 2]),
    ("N-D index x[[[0,1],[2,3]]]",        lambda: a[np.array([[0,1],[2,3]])],
                                          lambda: FloatTensor(a)[np.array([[0,1],[2,3]])]),
    ("0-d index x[np.array(2)]",          lambda: a[np.array(2)],        lambda: FloatTensor(a)[np.array(2)]),
    ("multi-adv zip x[[0,2],[1,3]]",      lambda: a[np.array([0,2]), np.array([1,3])],
                                          lambda: FloatTensor(a)[np.array([0,2]), np.array([1,3])]),
    ("multi-adv mismatch x[[0,2],[1,3,5]]", lambda: a[np.array([0,2]), np.array([1,3,5])],
                                          lambda: FloatTensor(a)[np.array([0,2]), np.array([1,3,5])]),
    ("adv separated b[i,:,j]",            lambda: b[np.array([0,1]), :, np.array([0,1])],
                                          lambda: FloatTensor(b)[np.array([0,1]), :, np.array([0,1])]),
    ("broadcast adv x[[[0],[1]],[0,2]]",  lambda: a[np.array([[0],[1]]), np.array([0,2])],
                                          lambda: FloatTensor(a)[np.array([[0],[1]]), np.array([0,2])]),
    ("1-D rowmask x[mask]",               lambda: a[np.array([True,False,True,False])],
                                          lambda: FloatTensor(a)[BoolTensor(np.array([True,False,True,False]))]),
    ("float index x[1.0]",                lambda: a[1.0],                lambda: FloatTensor(a)[1.0]),
    ("too many idx x[0,1,2]",             lambda: a[0, 1, 2],            lambda: FloatTensor(a)[0, 1, 2]),
    ("OOB single int x[10]",              lambda: a[10],                 lambda: FloatTensor(a)[10]),
    ("scalar result type x[1,2]",         lambda: a[1, 2],               lambda: FloatTensor(a)[1, 2]),
]

for label, np_fn, mp_fn in cases:
    print(f"\n### {label}")
    show("  numpy  ", np_fn)
    show("  masspcf", mp_fn)

# setitem gaps (each on a fresh copy)
print("\n### setitem gaps")
def set_show(label, np_fn, mp_fn):
    na = a.copy(); 
    try:
        np_fn(na); print(f"{label}: numpy OK")
    except Exception as e:
        print(f"{label}: numpy {type(e).__name__}: {e}")
    mt = FloatTensor(a.copy())
    try:
        mp_fn(mt); print(f"{label}: masspcf OK")
    except Exception as e:
        print(f"{label}: masspcf {type(e).__name__}: {e}")

set_show("  x[1]=7.0 (single int row)",   lambda na: na.__setitem__(1, 7.0),
                                          lambda mt: mt.__setitem__(1, 7.0))
set_show("  x[-1]=7.0 (neg int)",         lambda na: na.__setitem__(-1, 7.0),
                                          lambda mt: mt.__setitem__(-1, 7.0))
set_show("  x[1:3]=5.0 (scalar->slice)",  lambda na: na.__setitem__(slice(1,3), 5.0),
                                          lambda mt: mt.__setitem__(slice(1,3), 5.0))
set_show("  x[-3:-1]=block (neg slice)",  lambda na: na.__setitem__(slice(-3,-1), np.zeros((2,6))),
                                          lambda mt: mt.__setitem__(slice(-3,-1), FloatTensor(np.zeros((2,6)))))
set_show("  x[[0,2],[1,3]]=9 (multi-adv)", lambda na: na.__setitem__((np.array([0,2]), np.array([1,3])), 9.0),
                                          lambda mt: mt.__setitem__((np.array([0,2]), np.array([1,3])), 9.0))
```