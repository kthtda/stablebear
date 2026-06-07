# masspcf API friction triage

Usability friction found during a broad API scan (2026-06-07): places where a
reasonable user task needs more steps than it should, has no API at all, or
behaves inconsistently with the rest of the library / NumPy. These are *not*
correctness bugs (those live in `bug-scan-findings.md` and in
`test/python/test_bugscan_*.py`); a handful that straddle the line are
cross-referenced below.

## How to read this

Each item has a **priority** and an **effort** estimate:

- **P0** - breaks or forces an awkward detour around a *common, natural*
  workflow, or silently does the wrong thing; biggest user-experience wins.
- **P1** - real ergonomic gap or NumPy-parity inconsistency on a path users
  will hit, but with a tolerable workaround.
- **P2** - polish, edge-case parity, message quality, cleanup.

- **Effort** S = Python-only wrapper change; M = new method, possibly a small
  C++/pybind binding; L = backend work or API design.

Many items collapse into a few **root fixes** - tackling a cluster clears
several rows at once. Those are described first.

---

## Highest-leverage fixes (clusters)

### C1. numpy <-> tensor bridges for the non-numeric tensor families  (P0)

The single biggest theme. `FloatTensor`/`IntTensor`/`BoolTensor` accept ndarrays
and (for numeric) expose `np.asarray`, but the PCF/point-cloud/barcode/matrix
families do not, so users are forced into per-element Python loops - directly
against the project's own "prefer native operations, never loop over elements"
guideline.

Resolves: **item 00, item 39, item 31, item 30, item 40, item 41, item 32, item 22, item 25, item 36, item 19, item 13, item 38**.

Suggested surface:

- **Construct from numpy/list** (the seed example):
  - `PointCloudTensor(arr, cloud_ndim=2, dtype=pcloud64)` - trailing
    `cloud_ndim` axes are each cloud's `(npts, dim)`, leading axes are the
    tensor shape: `(m,n,k,l) -> (m,n)` tensor of `(k,l)` clouds.
  - `BarcodeTensor([...])` / `DistanceMatrixTensor.from_numpy((N,n,n))` /
    `SymmetricMatrixTensor.from_numpy(...)` accepting raw ndarrays.
  - `Pcf(times, values, dtype=...)` / `Pcf.from_arrays(...)` (item 13);
    `PcfTensor([...])` accepting integer (`pcf*i`) `Pcf` elements (item 38).
  - A 0-d numeric tensor from a Python/NumPy scalar: `FloatTensor(3.0)` (item 19).
  - Optional unifying top-level `mpcf.array(data, dtype=...)`.
- **Export to numpy**:
  - `PointCloudTensor.to_dense()` (and `np.asarray` for the equal-shape case)
    (item 40); `DistanceMatrixTensor.to_dense()/from_dense()`,
    `SymmetricMatrixTensor.to_dense()/from_dense()` over the leading axes (item 41);
    `BarcodeTensor.to_numpy()`/`tolist()` (ragged -> object array) (item 32).
  - `to_serial_content(t)` as the documented inverse of `from_serial_content`
    (item 36).
  - **`DistanceMatrix`/`SymmetricMatrix` must implement `__array__`** - today
    `np.asarray(D)` silently returns a 0-d *object* array wrapping the C++
    handle instead of the dense matrix (item 22, item 25). This is the dangerous,
    silent member of the cluster and is really a correctness defect (see
    cross-refs).

### C2. comparison + masking + index discovery  (P0/P1)

The canonical NumPy masking idiom is unavailable and partly crashes.

Resolves: **item 04 (also a bug), item 05, item 07, item 08**.

- `t > 3` (scalar RHS) should return a `BoolTensor` so `t[t > 3]` works; today
  the ordering ops crash with `AttributeError: 'int' has no attribute '_data'`
  and `==`/`!=` fall back to a plain Python `bool` (breaks `t[t == x]`). See
  bug-scan-findings.md (interop comparison bug) - fixing the operators fixes
  this friction too.
- `Tensor.nonzero()`, top-level `mpcf.where`/`argwhere`, `Tensor.take`/
  `compress` for index discovery (item 05) - especially important for PCF/point-
  cloud tensors, which cannot even round-trip through numpy.
- Tensor-RHS masked/index/outer assignment should broadcast a size-1 tensor
  (`a[mask] = FloatTensor([99.])`), matching the `__setitem__` docstring and
  NumPy (item 07).
- `Tensor.shares_memory(other)` (or automatic overlap-copy in `assign_from`)
  so users can guard view-aliasing (item 08; tied to the aliasing bug).

### C3. signed-axis / dim normalization everywhere  (P1)

`stack` and integer indexing already accept negatives; `concatenate`, `split`,
`array_split`, the reduction `dim`, `squeeze`, and `expand_dims` do not, raising
raw pybind `TypeError`s. One pass making the bindings take a signed axis and
normalizing (`axis % ndim`) brings them in line.

Resolves: **item 20, item 28** (and removes the friction face of the reduction-`dim`
crash bug; see bug-scan-findings.md). Related: `squeeze`/`expand_dims` should
also accept a *tuple* of axes (item 17) and `squeeze(0)` on a 0-d tensor should
be a no-op (item 18).

### C4. dtype promotion + a complete astype matrix  (P1)

The library positions itself as numpy-like but binary ops and `astype` don't
promote/convert across dtypes, leaking pybind overload dumps.

Resolves: **item 09, item 10, item 11, item 12, item 21, item 42, item 45, item 46, item 47**.

- Numeric binary ops should accept an ndarray RHS (`f + arr`, today only
  `arr + f` works) (item 09) and promote mixed precision (`float32 + float64`)
  (item 10), or at least raise a clean message.
- `PcfTensor + PcfTensor` dtype mismatch should give the clean
  "Mismatched PCF types" error the scalar path already gives (item 11).
- `IntTensor ** 0.5` should promote to float (item 12).
- `concatenate`/`stack` of mismatched precision: clean error or numpy-style
  upcast (item 21).
- `astype` should cover the full numeric matrix incl. `float<->uint`,
  `bool<->numeric`, and `distmat/symmat` precision casts (item 42, item 45, item 46) - its
  own docstring promises same-family precision casts.
- `allclose` should accept `IntTensor`/`BoolTensor` and an ndarray on either
  side (item 47).

### C5. random Generator semantics  (P0)

`masspcf.random.Generator` **does not advance its state between calls** - two
consecutive `noisy_sin(..., generator=g)` calls return *identical* tensors
(item 33). This silently breaks every repeated-sampling workflow (bootstrap,
Monte Carlo). Either make the generator advance per call (numpy/torch
convention) or document the per-element-seed design loudly and add
`Generator.spawn(n)`. Also expose the seed in use (`g.seed_value` /
`initial_seed()`), `copy()`, and `spawn()` so default-seeded experiments are
reproducible (item 34).

### C6. result-object consumability (matrices) + introspection  (P1)

Resolves: **item 23, item 26, item 35**.

- `pdist` returns a `DistanceMatrix` with no `shape`/`len`/iteration/row access
  and no tensor ops, while `cdist`/`lp_norm` return `FloatTensor` -
  inconsistent. Add `shape`/`__len__`/row access / `to_tensor()` or a
  `pdist(..., dense=True)` (item 23). Same for `SymmetricMatrix` plus `.diagonal()`
  (item 26).
- `PointCloudTensor.counts()/npoints()` -> `IntTensor` of per-cloud point
  counts, computed on the backend (item 35).

### C7. error-message quality (cross-cutting)  (P1/P2)

Many paths leak raw pybind11 overload dumps, bare `KeyError`s, taskflow
assertions with absolute source paths, or `AttributeError: '... _data'`.
Validate in the Python wrappers and raise clean, actionable errors.

Resolves/affects: **item 01, item 02, item 11, item 21, item 27, item 48** (and improves the friction
edge of several bugs).

---

## Full triage table

| # | Item | Area | Pri | Effort |
|---|------|------|-----|--------|
| 00 | No constructor for PointCloud/Barcode/DistanceMatrix/SymmetricMatrix tensors from numpy/list (seed example) | construction | P0 | L |
| 39 | No constructor to build a PointCloudTensor from a numpy array (`(m,n,k,l)->(m,n)` of `(k,l)`) | pcloud | P0 | M |
| 31 | No vectorized PointCloudTensor/DistanceMatrixTensor construction from numpy for batch PH | persistence | P0 | M |
| 22 | `np.asarray(pdist result)` silently returns a 0-d object scalar, not the dense matrix (also a bug) | distance | P0 | S |
| 25 | `np.asarray(l2_kernel result)` silently returns a 0-d object scalar (also a bug) | kernel | P0 | S |
| 04 | `t[t > scalar]` impossible; comparison ops crash on scalar RHS (also a bug) | getitem | P0 | S |
| 33 | `random.Generator` never advances state -> repeated sampling returns identical data | pointprocess | P0 | M |
| 40 | No `PointCloudTensor.to_dense()` / `np.asarray` for point-cloud tensors | pcloud | P0 | M |
| 09 | `tensor + ndarray` fails though `ndarray + tensor` works (operator asymmetry) | arithmetic | P0 | S |
| 05 | No `nonzero`/`where`/`argwhere`/`take`/`compress` for index discovery | getitem | P1 | M |
| 07 | Masked/index/outer tensor-RHS assignment won't broadcast a size-1 tensor | setitem | P1 | M |
| 10 | Mixed-precision tensor arithmetic raises a raw pybind error instead of promoting | arithmetic | P1 | M |
| 11 | `PcfTensor + PcfTensor` dtype mismatch gives an ugly pybind error (scalar path is clean) | arithmetic | P1 | S |
| 20 | No negative-axis support on `concatenate`/`split`/`array_split` | joinsplit | P1 | S |
| 28 | Reductions (`mean`/`max_time`) don't accept negative `dim` | reductions | P1 | S |
| 23 | `DistanceMatrix` has no shape/len/iteration/row access/tensor ops (inconsistent with cdist) | distance | P1 | M |
| 26 | `SymmetricMatrix` has no shape/len/row/diagonal/slice access | kernel | P1 | M |
| 30 | Cannot build a `BarcodeTensor` from a list/array of numpy barcodes | persistence | P1 | M |
| 32 | No `BarcodeTensor.to_numpy()`/`tolist()` (need a manual nested loop) | persistence | P1 | S |
| 34 | `Generator` exposes no seed readback / copy / spawn | pointprocess | P1 | M |
| 35 | No native per-cloud point-count accessor on `PointCloudTensor` | pointprocess | P1 | M |
| 36 | No `to_serial_content` inverse of `from_serial_content` | io | P1 | M |
| 41 | No bulk `from_dense`/`to_dense` on `DistanceMatrixTensor`/`SymmetricMatrixTensor` | pcloud | P1 | M |
| 45 | No numeric cross-cast to/from the unsigned family (`float<->uint`) | interop | P1 | M |
| 01 | `FloatTensor` silently ignores any `dtype` that isn't exactly `float32` | construction | P1 | S |
| 27 | Mixed-dtype list to `l2_kernel` raises an opaque internal `_set_element` error | kernel | P1 | S |
| 21 | Joining tensors of different dtype/precision: no path, cryptic error | joinsplit | P1 | S |
| 48 | Negative/zero args to system setters leak pybind/taskflow internals | interop | P1 | S |
| 12 | Fractional exponent on an `IntTensor` raises a raw pybind error instead of promoting | arithmetic | P1 | S |
| 17 | `squeeze`/`expand_dims` don't accept a tuple of axes | shapeops | P2 | S |
| 13 | No `Pcf` constructor from separate times and values arrays | pcf | P2 | S |
| 14 | No accessors for `Pcf` time/value arrays or domain bounds | pcf | P2 | S |
| 15 | `Pcf` has no `.dtype` (only `.ttype`/`.vtype`), inconsistent with tensors | pcf | P2 | S |
| 16 | `Pcf` has no `__repr__` (shows object address in REPL/notebooks) | pcf | P2 | S |
| 43 | `BoolTensor` has no `__repr__`/`__str__` (sibling numeric tensors have it) | interop | P2 | S |
| 02 | `IntTensor`/`PcfTensor` leak `KeyError`/pybind/`AttributeError` for bad input | construction | P2 | S |
| 03 | `Pcf` silently sorts non-monotonic / duplicate breakpoints (docstring says strict) | pcf | P2 | S |
| 06 | Writing into a `broadcast_to` view silently scatters (no read-only guard; tied to a bug) | getitem | P2 | M |
| 08 | No public `shares_memory`/`may_share_memory` to guard view-aliasing | setitem | P2 | M |
| 18 | `squeeze(0)` on a 0-d tensor raises instead of being a no-op | shapeops | P2 | S |
| 19 | No 0-d numeric tensor constructor from a python/numpy scalar (`FloatTensor(3.0)`) | shapeops | P2 | S |
| 24 | No Lp distance/norm for integer PCFs (must pre-cast to float) | distance | P2 | M |
| 29 | No documented error contract / validator for a bad reduction `dim` | reductions | P2 | S |
| 37 | `from_serial_content` cannot represent an empty PCF (`start == stop` rejected) | io | P2 | S |
| 38 | `PcfTensor([...])` accepts float `Pcf` but rejects integer (`pcf*i`) `Pcf` | io | P2 | S |
| 42 | `astype` between `distmat32`/`distmat64` (and `symmat`) is unsupported | pcloud | P2 | M |
| 46 | `bool` dtype cannot be `astype`-cast to/from numeric | interop | P2 | M |
| 47 | `allclose` rejects `IntTensor`/`BoolTensor` and ndarray operands | interop | P2 | S |
| 44 | `masspcf.timeseries` is an empty/orphaned namespace package (stale `.pyc` in tree) | interop | P2 | S |

---

## Items that are also correctness bugs

These were filed as friction (a workflow is awkward) but the underlying
behavior is a defect; they also have failing regression tests under
`test/python/test_bugscan_*.py`:

- **item 22 / item 25** - `np.asarray()` on `DistanceMatrix`/`SymmetricMatrix`
  *silently* yields a broken 0-d object array. Silent wrong shape, not just
  friction.
- **item 04** - scalar comparison crashes (`AttributeError`) and `==`/`!=` return a
  plain `bool`, corrupting `t[t == x]` masks. See the interop comparison bug.
- **item 06** - assigning through a `broadcast_to` view silently scatters to all
  aliased cells. See the broadcast-view write bug.
- **item 01** - `FloatTensor` silently downgrading a requested dtype to `float64`
  is a silent-precision footgun, not only friction.

## Notes on scope

- The `(m,n,k,l) -> (m,n)` of `(k,l)` point-cloud constructor (the seed
  example) is real: `PointCloudTensor` is the only first-class tensor family
  whose constructor rejects ndarrays, and the vectorized `t[...] = arr`
  shortcut is itself buggy (stores the whole array in every cell), so the
  Python double-loop is currently the *only correct* path.
- Several P0/P1 items share root fixes (clusters C1-C7); sequencing by cluster
  rather than by row will clear the table faster.

## Filed GitHub issues (49)

All filed on `bwehlin/masspcf` on 2026-06-07 (label `bug-scan`).

| Issue | Priority | Title |
|---|---|---|
| [#53](https://github.com/bwehlin/masspcf/issues/53) | high | [friction/construction] No way to build PointCloud/Barcode/DistanceMatrix/SymmetricMatrix tensors from a numpy array or |
| [#54](https://github.com/bwehlin/masspcf/issues/54) | medium | [friction/construction] FloatTensor silently ignores any dtype argument that is not exactly mpcf.float32 |
| [#55](https://github.com/bwehlin/masspcf/issues/55) | low | [friction/construction] IntTensor and PcfTensor raise leaky low-level errors (bare KeyError / pybind _set_element |
| [#56](https://github.com/bwehlin/masspcf/issues/56) | low | [friction/construction] Pcf silently sorts non-monotonic / duplicate breakpoints instead of raising as documented |
| [#57](https://github.com/bwehlin/masspcf/issues/57) | high | [friction/indexing] Cannot build a boolean mask with the natural t[t > scalar] idiom; comparison ops crash on |
| [#58](https://github.com/bwehlin/masspcf/issues/58) | medium | [friction/indexing] No nonzero / where / argwhere / take / compress for index discovery |
| [#59](https://github.com/bwehlin/masspcf/issues/59) | low | [friction/indexing] Writing into a broadcast view silently scatters to multiple elements (no read-only guard like |
| [#60](https://github.com/bwehlin/masspcf/issues/60) | medium | [friction/setitem] Masked/index/outer tensor-RHS assignment does not broadcast a scalar-shaped (size-1) tensor, |
| [#61](https://github.com/bwehlin/masspcf/issues/61) | low | [friction/setitem] No public way to detect whether two tensors share storage (aliasing), so users cannot |
| [#62](https://github.com/bwehlin/masspcf/issues/62) | high | [friction/arithmetic] Tensor + numpy.ndarray fails even though numpy.ndarray + tensor works (operator asymmetry) |
| [#63](https://github.com/bwehlin/masspcf/issues/63) | medium | [friction/arithmetic] Mixed-precision tensor-tensor arithmetic raises a raw pybind error instead of promoting |
| [#64](https://github.com/bwehlin/masspcf/issues/64) | medium | [friction/arithmetic] PcfTensor + PcfTensor dtype mismatch gives an ugly pybind error, unlike the clean Pcf + Pcf |
| [#65](https://github.com/bwehlin/masspcf/issues/65) | low | [friction/arithmetic] Fractional exponent on an IntTensor raises a raw pybind error instead of promoting to float |
| [#66](https://github.com/bwehlin/masspcf/issues/66) | low | [friction/pcf] No constructor from separate times and values arrays |
| [#67](https://github.com/bwehlin/masspcf/issues/67) | low | [friction/pcf] No accessors for the time/value arrays or domain bounds |
| [#68](https://github.com/bwehlin/masspcf/issues/68) | low | [friction/pcf] Pcf has no .dtype property (only .ttype/.vtype), inconsistent with tensors |
| [#69](https://github.com/bwehlin/masspcf/issues/69) | low | [friction/pcf] Pcf has no __repr__, so it shows the default object address in REPL/notebooks |
| [#70](https://github.com/bwehlin/masspcf/issues/70) | low | [friction/shapeops] squeeze and expand_dims do not accept a tuple of axes (NumPy does) |
| [#71](https://github.com/bwehlin/masspcf/issues/71) | low | [friction/shapeops] squeeze(0) on a 0-d tensor raises instead of returning the 0-d tensor (NumPy parity gap) |
| [#72](https://github.com/bwehlin/masspcf/issues/72) | low | [friction/shapeops] No constructor for a 0-d numeric tensor from a numpy/python scalar |
| [#73](https://github.com/bwehlin/masspcf/issues/73) | medium | [friction/joinsplit] No negative-axis support on concatenate/split/array_split forces manual axis arithmetic |
| [#74](https://github.com/bwehlin/masspcf/issues/74) | low | [friction/joinsplit] Joining tensors of different dtype/precision (e.g. float32 + float64, or int + float) has no |
| [#75](https://github.com/bwehlin/masspcf/issues/75) | high | [friction/distance] np.asarray() on a pdist result (DistanceMatrix) silently returns a useless object scalar |
| [#76](https://github.com/bwehlin/masspcf/issues/76) | medium | [friction/distance] pdist returns DistanceMatrix (no shape/len/iteration/row access/tensor ops); cdist/lp_norm |
| [#77](https://github.com/bwehlin/masspcf/issues/77) | low | [friction/distance] No Lp distance/norm support for integer PCFs; user must pre-cast to float |
| [#78](https://github.com/bwehlin/masspcf/issues/78) | medium | [friction/kernel] np.asarray(SymmetricMatrix) silently produces a 0-d object array instead of the dense matrix |
| [#79](https://github.com/bwehlin/masspcf/issues/79) | low | [friction/kernel] SymmetricMatrix has no shape/len/row access/slicing - extracting a row or diagonal needs a |
| [#80](https://github.com/bwehlin/masspcf/issues/80) | low | [friction/kernel] Mixed-dtype list passed to l2_kernel raises an opaque internal pybind error instead of a clear |
| [#81](https://github.com/bwehlin/masspcf/issues/81) | medium | [friction/reductions] Reductions do not accept negative dim (no NumPy-style axis-from-end), inconsistent with the |
| [#82](https://github.com/bwehlin/masspcf/issues/82) | low | [friction/reductions] No way to validate/clamp dim before calling, and no documented error contract for bad dim |
| [#83](https://github.com/bwehlin/masspcf/issues/83) | medium | [friction/persistence] Cannot build a BarcodeTensor from a list/array of numpy barcode arrays |
| [#84](https://github.com/bwehlin/masspcf/issues/84) | high | [friction/persistence] No vectorized way to construct a PointCloudTensor / DistanceMatrixTensor from existing numpy |
| [#85](https://github.com/bwehlin/masspcf/issues/85) | low | [friction/persistence] No way to get all bars out of a BarcodeTensor without a manual nested loop |
| [#86](https://github.com/bwehlin/masspcf/issues/86) | high | [friction/pointprocess] masspcf.random.Generator never advances its state between calls (silent footgun) |
| [#87](https://github.com/bwehlin/masspcf/issues/87) | medium | [friction/pointprocess] Generator exposes no way to read back its seed, copy, or spawn child generators |
| [#88](https://github.com/bwehlin/masspcf/issues/88) | medium | [friction/pointprocess] No native API to get per-cloud point counts (or total point count) from a PointCloudTensor |
| [#89](https://github.com/bwehlin/masspcf/issues/89) | medium | [friction/io] No to_serial_content inverse for from_serial_content; round-tripping a PcfTensor through numpy |
| [#90](https://github.com/bwehlin/masspcf/issues/90) | low | [friction/io] from_serial_content cannot represent an empty PCF (start == stop is rejected) |
| [#91](https://github.com/bwehlin/masspcf/issues/91) | low | [friction/io] PcfTensor([...]) accepts a list of float-valued Pcf objects but rejects int-valued (intpcf) Pcf |
| [#92](https://github.com/bwehlin/masspcf/issues/92) | high | [friction/matrices] No constructor to build a PointCloudTensor from a numpy array |
| [#93](https://github.com/bwehlin/masspcf/issues/93) | medium | [friction/matrices] No way to convert a PointCloudTensor (of equal-sized clouds) back to a single numpy array |
| [#94](https://github.com/bwehlin/masspcf/issues/94) | medium | [friction/matrices] No way to read or set the storage as a numpy array on DistanceMatrixTensor / |
| [#95](https://github.com/bwehlin/masspcf/issues/95) | low | [friction/matrices] astype between distmat32 and distmat64 (and symmat32/64) is unsupported |
| [#96](https://github.com/bwehlin/masspcf/issues/96) | medium | [friction/interop] BoolTensor has no __repr__/__str__, so it prints as <masspcf.tensor.BoolTensor object at 0x...> |
| [#97](https://github.com/bwehlin/masspcf/issues/97) | low | [friction/interop] masspcf.timeseries is an empty/orphaned namespace package (importable but contains nothing); |
| [#98](https://github.com/bwehlin/masspcf/issues/98) | medium | [friction/interop] No numeric cross-cast to/from the unsigned family (float<->uint, uint<->float), unlike NumPy |
| [#99](https://github.com/bwehlin/masspcf/issues/99) | low | [friction/interop] bool dtype cannot be astype-cast to/from numeric (bool->int, int->bool, bool->float all raise |
| [#100](https://github.com/bwehlin/masspcf/issues/100) | low | [friction/interop] allclose only accepts FloatTensor/DistanceMatrix/SymmetricMatrix; IntTensor and BoolTensor are |
| [#101](https://github.com/bwehlin/masspcf/issues/101) | low | [friction/interop] Negative / zero arguments to system setters raise leaky pybind/taskflow errors instead of clean |
