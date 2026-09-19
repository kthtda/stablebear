# Test plan: subsampling, nested tensors, and unified persistence

Planning only: this document does not implement tests or report test results.
All checklist entries are initially unchecked.

## Scope and execution rules

Reviewed on 2026-09-19 against:

- [#229: uniform point-cloud subsampling with tensor-level indexed storage](https://github.com/kthtda/stablebear/issues/229), including every acceptance criterion.
- [#230: all pickling through Stablebear IO with legacy compatibility](https://github.com/kthtda/stablebear/issues/230), including every acceptance criterion.
- [#231: runtime-nested tensors and ragged point-cloud indexing](https://github.com/kthtda/stablebear/issues/231), including every acceptance criterion.
- Branch `issue-229-uniform-subsampling`, HEAD `44acc9f97667025beb0c63fe5499ba759b435c14`.
- Freshly fetched `origin/main`, `4a9c284f4f48d6711083d99ee529819d4c546384`, which is also the merge base. The net diff contains 53 changed files, including one deleted test file. Issue comments were empty at review time.

Execute numbered entries in order. Each entry is a self-contained work unit:
create its own inputs, generators, temporary files, and any required historical
fixtures; restore global settings afterward. No entry consumes state, helpers,
fixtures, or tests produced by another entry. References to other entries below
are coverage mappings, not prerequisites. Existing repository helpers may be
reused. A failed or unavailable case is recorded and does not prevent independent
later entries from being performed.

For each entry, record test locations, cases covered, backend/build identity,
commands and results, and any remaining failures or blocked cases. Mark an entry
complete only when its acceptance checks pass; an expected failure or skipped
CUDA run is not acceptance evidence. These are instructions for future test
work, not authorization to implement tests as part of creating this plan.

Common environment prerequisite, independently applicable to every executable
entry: follow [CLAUDE.md](../CLAUDE.md), build and install before testing, and run
pytest and the C++ executable from `test/`. A full install is required before
the minimal CMake build. Use the repository's existing build/dependency setup.

```bash
# From the repository root, for a full install:
python -m pip install .

# For subsequent C++ edits in an already configured minimal build:
cmake --build cmake-build-debug -j$(nproc)
cmake --install cmake-build-debug

# Run Python tests from test/ (narrow the path for individual entries):
cd test
python -m pytest python
```

Common case matrix: point-cloud cases use both `pcloud32` and `pcloud64`;
nested cases use all six supported numeric leaves (`float32`, `float64`,
`int32`, `int64`, `uint32`, `uint64`). Run applicable Python checks with default
execution and forced CPU. Exercise genuinely loaded CPU and CUDA extension
modules in separate environments/processes where available: `force_cpu(True)`
inside a CUDA build alone does not test `_sb_cpu`. Do not imply that sampling
or recursive IO has a GPU kernel merely because the CUDA module is loaded.
Use exact comparisons for selections, shapes, integer data, and lossless IO;
use explicit precision-appropriate tolerances for computed floating results.

## Contract discrepancies to keep visible

These are source-review observations, not executed-test results. Test the issue
requirements even where the current implementation appears incomplete. Record
an explicit approved contract change if the intended behavior changes; do not
turn current behavior into the expected result solely to make a test pass.

| Observation at reviewed HEAD | Required treatment | Entries |
| --- | --- | --- |
| #229 describes full-element access returning a writable `FloatTensor`; this branch intentionally returns a write-aware `PointCloud` façade. | Cover the new façade and retain the underlying requirement that no mutable escape bypasses shared-state materialization. Record the public return-type change explicitly. | A2–A4, E3 |
| `PointCloud` in C++ still contains `m_indices` and local materialization; Python `_ensure_writeable()` replaces the calling wrapper's `_data`. | Verify tensor-level storage and whole shared-state transition, including pre-existing sibling views. Do not accept per-wrapper detachment as equivalent. | D4, E1–E3 |
| Generic `make_indexed_tensor` allows singleton source-axis broadcasting. #231 requires an exact public prefix match. | Test both contracts at their respective boundaries; public `(2, 1)` versus `(2, 3)` selection shapes must fail. | C1–C2 |
| Indexed Python save currently materializes; IO describes subtype `1001` as per-element source IDs and indices. | Require a distinct tensor-level indexed encoding, indexed round trips, and readers for old `1000`/`1001` files. | G1–G3 |
| General nested format dispatch has no visible reader for legacy `(5, 64)`. | Use fixed legacy bytes to test the required reader. | G2 |
| Standalone `PointCloud` is absent from object IO dispatch; dtype pickling still uses a name lookup. | Audit all public pickleable types, including default pickling. Do not silently exempt either from #230. | H1–H4 |
| C++ IO format version changes from 2 to 3, attributed to shared-source point clouds. #231 forbids a version bump merely for nested tensors. | Distinguish the reasons, verify supported old headers, and require new nested depths/types to work within the chosen current version. | G1–G3 |
| `CHANGELOG.md` still advertises `IndexTensor`; saving/indexing docs describe older or inconsistent ownership. | Verify documentation against the accepted API/storage contracts. | I3 |

## Ordered checklist

Items are labeled by section (A1, A2, B1, B2, and so on). Execute sections
A–K in order; each item remains independently executable.

### A. Build and public point-cloud API

- [ ] **A1. Build, packaging, and public import smoke checks.**

   Scope: branch additions in `CMakeLists.txt`, `src/python/pymodule.cpp`,
   `py_tensor.cpp`, the new point-cloud/subsampling bindings, and package exports.
   In a clean installed environment, import `stablebear`, `stablebear.io`,
   `stablebear.point_process`, and persistence modules in multiple orders.
   Exercise a minimal `PointCloud`, `NestedTensor`, and `subsample` call. Verify
   both precision bindings and all six nested bindings exist in each supported
   extension build, generated stubs reflect them, and imports do not cycle.
   Check `sb.PointCloud`, `sb.PointCloudTensor`, `sb.NestedTensor`,
   `sb.point_process.subsample`, dtype-to-wrapper construction, and `__all__`.
   No public `IndexTensor` should remain. Audit the moved
   `stablebear.base_tensor.PointCloudTensor` import path for documentation and
   historical pickle compatibility. Pass: installed builds expose the intended
   API without relying on source-tree shadowing or stale binaries.

- [ ] **A2. Point-cloud construction and strict rank-two cells.**

   Scope: branch-only façade and shape fix in `stablebear/point_cloud.py`.
   Construct standalone clouds, scalar cloud tensors from `(N, D)` arrays,
   outer tensors from `(2, 3, N, D)` arrays, and ragged lists/tuples. Include
   `(0, D)`, `(N, 0)`, `(0, 0)`, singleton clouds, empty outer tensors, different
   ambient dimensions in different cells, noncontiguous arrays, and dtype
   inference/explicit casts. Reject rank-0/1/3 cell data, including `(30, 2, 20)`,
   invalid dtypes, mixed inferred precisions, and `cloud_ndim != 2`; higher-rank
   dense input is valid when its final two axes are the cloud. Check assignment
   through ordinary, sliced, broadcast, and masked destinations. Pass: every
   stored cloud is rank two, shape errors are clear, and rejected writes do not
   corrupt existing values.

- [ ] **A3. Standalone and owner-backed `PointCloud` behavior.**

   Scope: branch-only public façade, replacing full-cell `FloatTensor` results.
   Check `shape`, `size`, floating dtype, representation, equality, integer/row/
   column indexing, negative indices, slices, empty slices, and bounds errors.
   For a scalar `PointCloudTensor`, distinguish `points[()]`/`points[...]` from
   coordinate indexing; for outer tensors, complete indices select clouds.
   Coordinate assignment through a retained façade updates the correct parent
   cell. `copy()` and `materialize()` produce independent values; materialize
   returns a `FloatTensor`. Check NumPy dtype/copy behavior and ownership of
   arrays and row views. Rank-changing methods such as reshape/squeeze must not
   be exposed as point-cloud operations. Delete the original Python owner and
   force garbage collection while retaining a façade; it must stay valid.
   Pass: all public reads/writes and lifetimes are consistent and documented.

- [ ] **A4. Mutable escapes from indexed cloud façades.**

   Scope: #229 plus the branch's changed element-access API. Independently create
   an indexed tensor and retain a full-cell façade, a row/column result, and
   NumPy conversions, including conversions requested with `copy=False` where
   supported. Reading a façade must not materialize the whole shared state.
   For each writable returned value, verify either documented detached-copy
   semantics or a safe shared-state transition before mutation is possible.
   Writing must never alter source storage, another sampled cloud, or repeated
   selected rows unintentionally. Unsupported no-copy requests must fail clearly.
   Pass: no mutable coordinate route bypasses the ownership contract; do not
   assume that NumPy conversion is zero-copy.

### B. Nested tensors and C++ tensor properties

- [ ] **B1. Runtime-nested construction, depth, and representation.**

   Scope: #231; `nested_tensor.py`, `tensor_create.py`, `nested_tensor.hpp`.
   Construct the issue's ragged vector and `2 x 2` matrix examples with lists
   and tuples, nested numeric tensors with differing child ranks/shapes, and
   depth-3 and substantially deeper values (for example depth 8). Numeric
   lists must still construct ordinary scalar tensors; Stablebear tensor
   objects mark nested boundaries. Exercise direct `NestedTensor` construction
   from numeric tensors/NumPy arrays as supported by this branch. For empty
   outer shapes `(0,)`, `(2, 0)` and scalar outer shape `()`, verify explicit
   dtype/depth construction and unambiguous element type. The branch counts the
   leaf tensor as depth 1; a tensor of numeric tensors has depth 2. Pass: full
   `Tensor<Tensor<...>>` representations use public dtype names even when there
   is no first child, with no pre-registered maximum template depth.

- [ ] **B2. Nested validation and checked C++ access.**

   Scope: #231. Reject mixed leaf dtypes, mixed child depths, irregular outer
   list shapes, numeric/tensor mixtures, unsupported leaf types, missing empty
   descriptors, incompatible explicit descriptors, and invalid depth values
   (including zero, negative, noninteger, and inconsistent scalar depth).
   Include a bad child late in a multidimensional input, plus empty children
   with incompatible stored depth. Exercise checked leaf/nested accessors and
   visitation in C++; a wrong alternative must report an error, not cast
   unchecked. Pass: concise Python type/value errors and safe C++ failures,
   without partly initialized values escaping.

- [ ] **B3. Nested ownership, assignment, copy, and deepcopy.**

   Scope: #231. Independently construct one-level/deep nested values, including
   the same caller-owned child supplied twice. Mutate original children after
   construction and assignment; stored values must stay independent. Mutate
   children in `copy()` and `copy.deepcopy()` and confirm recursive independence,
   including copies of scalar, empty, and noncontiguous values. Mutations through
   ordinary outer slices must be visible through their parent and sibling views.
   Replace a child with a different shape of the same type/depth; reject wrong
   dtype/depth without changes. Include slice and masked assignment. Pass:
   outer views share storage, while value construction/assignment and explicit
   deep copies do not alias caller-owned children.

- [ ] **B4. Nested outer operations and generic tensor compatibility.**

   Scope: #231 and branch integration with the base tensor interface. Check
   complete/partial indexing, negative indices, ellipsis/newaxis, masks and
   advanced indices where ordinary tensors support them, iteration, flatten,
   reshape, transpose, swapaxes, squeeze, expand-dims, broadcast, stack,
   concatenate, split, and array-split. Use scalar/empty/deep values and
   transposed, reversed, and stepped outer views. Validate illegal axes,
   incompatible shapes, reshape inference, and mixed type/depth joins. Compare
   children with independently built expected values, not just `repr`. Pass:
   depth, dtype, shape, contents, and ordinary view/copy semantics survive every
   valid operation; depth is retained even after operations yielding emptiness.

- [ ] **B5. C++ recursive tensor interface and property composition.**

   Scope: #229/#231 and the broad `tensor.hpp`/`tensor.tpp` refactor. Cover
   `IsTensor`, shape/rank/size/strides, const coordinate and flat access,
   iteration, and conversion from statically nested tensor types at leaf and
   nested depths, including scalar/empty values. Test an ordinary tensor with
   an unrelated property bit and an indexable test element with
   `Indexed | OtherProperty`. Verify removing `Indexed` preserves the other
   bit in the source type and operations preserve appropriate properties.
   Non-indexable element types must not enable indexed construction. Review
   that properties use the primary template/concepts, not a family of complete
   specializations. Pass: compile-time assertions and runtime values both match
   the property and recursive-type contracts.

### C. Ragged point-cloud indexing

- [ ] **C1. Point-cloud selection shape contract and contents.**

    Scope: #231. For independently constructed rank-one `uint64` children, cover
    every issue shape pair: `() -> ()`, `() -> (3,)`, `(2,) -> (2,)`,
    `(2,) -> (2, 4)`, `(2, 3) -> (2, 3, 5)`. Also cover multiple extra trailing
    axes, zero-sized outer axes, ragged selection lengths, and source/selection
    outer views. Use distinct coordinates per source cell; compare each result
    to NumPy selection from the exact prefix cell. Preserve `[3, 3, 1]` order
    and repetitions, empty children, ambient dimension, and source dtype.
    Pass: output outer shape is exactly the selection shape and child lengths
    never become outer axes.

- [ ] **C2. Point-cloud selection rejection and ownership.**

    Scope: #231. Reject fewer selection axes, mismatched prefixes, and singleton
    prefix broadcasting such as source `(2, 1)` versus selections `(2, 3)` at
    the public point-cloud boundary. Check wrong leaf dtype, deeper nesting,
    rank-0/2 children, index equal to row count, `uint64` maximum, and selection
    from an empty cloud. Include invalid late children and strided inputs.
    Expect `IndexError` for bounds, `ValueError` for outer shapes, and clear
    `TypeError`/`ValueError` for rank/dtype. Mutate/delete caller-owned selections
    after successful indexing: results must retain independently owned
    selections. Direct indexing retains an aligned source view; verify its
    documented source-view behavior separately from subsampling's snapshot
    guarantee. Pass: bad inputs never expose unsafe results or change inputs.

### D. Uniform subsampling and random generators

- [ ] **D1. Uniform subsampling API, shapes, and size boundaries.**

    Scope: #229; extend coverage beyond existing `test_subsample.py` size cases.
    Test defaults and explicit arguments on scalar, multidimensional, ragged,
    empty-outer, sliced, transposed, and reversed inputs. Always append the
    sample axis, including default `n_samples=1`. For `N=0,1,4`, test requested
    counts below/equal/above `N`, replacement on/off, and partial on/off;
    include `(0, D)`, `(N, 0)`, `(0, 0)`, and larger dimensions. Without
    replacement partial count is `min(n_points, N)`; replacement gives the
    requested count unless an allowed empty input gives zero. Pass: exact
    shapes/dtypes/counts and documented errors, with the input unchanged.

- [ ] **D2. Subsampling argument validation and failure atomicity.**

    Scope: #229 and branch-specific strict boolean validation. Test `n_points`
    and `n_samples` with Python/NumPy integers and valid `__index__` objects;
    reject booleans, floats, strings, `None`, zero and negative values with the
    specified type/value distinction. Check invalid `points`, nonboolean flags
    (including integer truthy values and NumPy booleans under the current strict
    contract), invalid generators, keyword-only misuse, and integers beyond
    binding limits. Put an insufficient/invalid cloud last in a ragged input.
    For every validation failure, compare the next draw to a fresh control
    generator with the same prior state, including the global generator.
    Pass: validation finishes before stream reservation and no input or random
    state changes on validation failure.

- [ ] **D3. Uniform draws, draw order, and duplicate filtering.**

    Scope: #229. Use unique row IDs to check membership, no repeated logical
    row without replacement, possible repetitions with replacement, random
    order when drawing the entire population, and fresh full populations for
    every output sample. Separately use coordinate-identical source rows to
    confirm they remain distinct candidates. Compare runs from identical seeds
    with duplicate filtering disabled/enabled: filtered output must equal a
    stable exact-coordinate filter of the unfiltered draw, with no redraw.
    Cover all-equal rows, signed zero, precision-rounding collisions, and
    zero-coordinate rows; explicitly document IEEE equality expectations for
    NaNs/infinities if supported. Use a fixed collection of seeds to assess
    per-position frequencies and ordered pairs on a small population, with a
    predeclared generous statistical bound (for example a union-bounded
    Hoeffding threshold at family error probability `1e-6`). Pass: semantic
    checks are exact and uniformity detects bias without a flaky tight cutoff.

- [ ] **D4. Generator stream allocation and scheduling independence.**

    Scope: #229. With explicit and global generators, check reseeding and replay
    of several calls, advancement after success, and isolation of explicit
    generators from the global one. Independently verify the reserved stream
    count/order against the C++ generator's seed-block contract: one slot per
    output cloud in row-major logical output order. Compare batched draws to
    row-major individual draws from the same generator. Empty outer outputs
    consume no slots; zero-row clouds consume their slots. Toggle duplicate
    filtering and compare the next call as well as current draws. Repeat with
    CPU limits 1, 2, and multiple workers and varied execution scheduling,
    including noncontiguous inputs. Pass: identical logical draws/advancement
    regardless of scheduling, without asserting unrelated toolchain-specific
    `std::distribution` byte sequences.

### E. Indexed storage, views, and mutation

- [ ] **E1. Tensor-level indexed structure and sampling snapshots.**

    Scope: #229/#231. Inspect structure using C++ ownership checks or narrowly
    scoped instrumentation, not Python coordinate reads that may copy. Require
    one materialized source tensor, an aligned broadcast/view of it, and one
    nested `uint64` selection tensor; source cells are materialized and no
    per-output indexed point-cloud objects are eagerly stored. Instrument
    coordinate copies to verify one logical copy per input cloud per sampling
    call, independent of sample count. Distinct input cells remain distinct;
    separate calls own distinct sources. Mutate input coordinates and previously
    obtained writable input views after return, then delete inputs: samples
    must remain unchanged and valid. Empty outer inputs need no coordinate
    copy. Pass: structural and copy-count requirements hold, not merely equal
    numerical results. Flag the issue's prohibition on local `PointCloud`
    selection state if the implementation still relies on it.

- [ ] **E2. Indexed outer views preserve alignment and sharing.**

    Scope: #229/#231. Independently create distinguishable indexed clouds and
    apply slicing, negative steps, empty slices, reshape, flatten, transpose,
    swapaxes, squeeze/expand-dims, and valid outer indexing/broadcasts. Chain
    views across both source and sample axes, including noncontiguous reshape.
    Verify each logical cell against the corresponding independently selected
    dense value and retain simultaneous parent/sibling views. Inspect storage
    identities/copy counts: metadata-only operations must keep selections and
    coordinates aligned without duplicating them. Pass: shape/type/content
    and shared backing state survive, including scalar and empty views.

- [ ] **E3. Whole-state copy-on-write and mutation visibility.**

    Scope: #229. Independently retain a sampled parent, several outer views,
    and cell façades before any write. In separate cases write a coordinate,
    row, whole cloud, outer slice, masked destination, and any low-level mutable
    accessor. Require exactly one materialization of the entire shared backing
    state, including clouds outside the visible slice, with adapter metadata
    removed. Existing views/façades must now resolve the same materialized
    state. Subsequent writes must not rematerialize. Every output cloud and
    repeated selected row must own independent coordinate storage. Pass:
    aliases of the same logical cell see the write; other sampled cells, the
    original input, and a separate sampling result do not. No mixture of local
    indexed/materialized cells is accepted.

- [ ] **E4. Repeated indexing and resampling indexed inputs.**

    Scope: #229/#231. Independently create an indexed input with reordered,
    repeated, and empty selections, then index it again and subsample it.
    Include outer views and already materialized results. Compare nested
    selections against explicit NumPy composition; validate second-stage
    indices against logical selected lengths, not original source sizes.
    Re-indexing may compose or materialize once before creating a fresh adapter.
    Resampling must copy current logical coordinates once into a fresh source,
    even when the input is indexed, and must not create adapter chains or
    perform both a preliminary full copy and a second source copy. Mutate both
    generations afterward. Pass: correct values, permitted structure, and
    complete sampling snapshot independence.

- [ ] **E5. Point-cloud copying, casting, joins, masks, and equality.**

    Scope: #229 compatibility and branch changes to `pcloud_cast`/copy bindings.
    Use independently built ordinary/indexed/strided tensors and compare
    copy/deepcopy, both precision casts, stack, concatenate, split/array-split,
    masked select/assignment, equality, and NumPy conversion to dense references.
    Check mutation independence and shape errors, not just resulting values.
    Operations must preserve a valid adapter or materialize the whole affected
    shared state according to their semantics. For retained low-level
    `copy(keep_source=True/False)` bindings, verify their distinct documented
    ownership and index-copy behavior without making them substitutes for
    public deep copies. Exercise source sharing in low-level casts and sources
    with the same owner but different offsets/strides. Pass: no accidental
    aliasing, dropped selections, unsupported-overload failures for promised
    operations, or incorrect cast deduplication.

### F. Downstream algorithms and task lifetimes

- [ ] **F1. Distances, persistent homology, and homological kernels.**

    Scope: #229/#231 and all changed distance/persistence wrappers and bindings.
    Independently create direct selections and sampled tensors, then compare
    whole-tensor and single-`PointCloud` algorithm paths with explicitly dense
    equivalents. Include scalar/ragged/multidimensional outer shapes, reordered
    and repeated rows, empty/singleton clouds, zero-dimensional coordinates,
    and outer/coordinate strides. Check `SquaredEuclideanDistance` against
    NumPy and equivalent distance-matrix persistence/kernel paths. Cover
    persistent homology dimensions 0 and 1, reduced/unreduced modes, and
    kernel pairs with mismatched logical row counts/dimensions/dtypes.
    Record any unsupported mixed input forms explicitly. Inspect storage/copy
    counts before and after const algorithms: selected values must be used
    without materializing the indexed source tensor. Pass: correct barcode
    values, precision and output outer shapes, clear invalid-pair errors, and
    no const-read storage transition. Existing single-cloud tests alone do not
    establish whole indexed-tensor support.

- [ ] **F2. Persistence task lifetime and error propagation.**

    Scope: branch changes from referenced to owned C++ task inputs and new
    standalone-cloud overloads. Using the existing async/task interfaces where
    available, submit work from temporary single clouds and temporary tensor
    views, release Python references/force collection, then await completion.
    Repeat with both persistence and homological-kernel tasks, materialized and
    indexed inputs. Check task exceptions, cancellation, output lifetime, and
    repeated runs; use ASan/UBSan for C++ lifetime-sensitive cases if supported.
    Pass: no dangling inputs, crashes, silent exceptions, or missing results.
    Do not assume a documented snapshot against concurrent user mutation unless
    that contract exists.

### G. Binary IO and format compatibility

- [ ] **G1. Current binary IO round trips and indexed representation.**

    Scope: #229/#231, `io.hpp`, `tensor_io.hpp`, and Python dispatch. Independently
    exercise C++ typed/untyped readers and Python `sb.save/load` and
    `sb.io.save/load`, using file paths and binary streams. Cover materialized
    clouds, indexed clouds, scalar and empty outers, non-owning/reversed/
    transposed views, both precisions, and nested tensors at several depths
    with scalar/ragged/empty children and all leaf dtypes. Preserve shape,
    values, depth, ambient dimensions, and normalized logical row-major order.
    Inspect indexed payload and loaded state: a new subtype must retain one
    materialized source, source-view mapping, and ordered/repeated/empty
    selections, not silently flatten to ordinary coordinates. Delete original
    owners before loading and exercise loaded mutation semantics. Pass:
    self-contained files, correct type dispatch and structure, and no
    reinterpretation of existing subtype IDs or new format version per depth.

- [ ] **G2. Fixed legacy binary compatibility.**

    Scope: #229/#231. Within this entry, obtain and commit immutable fixtures
    from historical writers for materialized point clouds (`1000`, both
    precisions), shared-source indexed point clouds (`1001`, both precisions),
    and one-level nested `uint64` format `(5, 64)`. Include supported historical
    header versions 1 and 2 and branch version 3 where applicable, plus ordinary
    tensor/object controls. Record producer commit/version, backend, generation
    command, checksum, and expected contents alongside each fixture. Use
    isolated historical checkouts/environments; do not fabricate compatibility
    bytes using today's writer. Load with current Python and appropriate C++
    readers, verifying types/shapes/dtypes/values. Legacy indexed data may
    normalize or materialize, but must retain logical values and safe ownership.
    Pass: all promised old formats load and new writes use current formats.

- [ ] **G3. Malformed binary input and boolean encoding.**

    Scope: #229/#231 and the new canonical bool IO. Independently create valid
    serialized seeds, then corrupt source counts/IDs, source and outer shapes,
    source-view mappings, coordinate ranks, selection rank/dtype/row bounds,
    strides, nested leaf formats/depth descriptors/child homogeneity, magic,
    version, and truncation points. Include empty payloads and checked size
    overflow cases, using subprocess resource limits for dangerous lengths.
    Invalid data must be rejected before exposing a usable value, without
    hangs, out-of-bounds access, or uncontrolled allocation. Boolean writers
    emit canonical 0/1 bytes; test the branch reader's nonzero-is-true behavior
    on noncanonical flag bytes separately from structural validation. Exercise
    Python's tensor/object load fallback to ensure corruption is not accepted
    as a different type. Pass: safe, informative failures and unchanged valid
    legacy bool reading.

### H. Pickling and backend interoperability

- [ ] **H1. Complete pickleable-type inventory and binary-path guard.**

    Scope: #230, even where existing `main` already uses IO reducers. Audit
    public exports and modules, inherited `__reduce__`/`__reduce_ex__`,
    `__getstate__`, registered reducers, pybind pickling, and default Python
    pickling. Cover every public tensor, standalone data object, the new
    `PointCloud`, and metadata/state objects such as dtype singletons and
    `Generator` when pickleable. Record supported/non-pickleable status; do not
    silently exclude types outside an existing hardcoded reducer list. Design
    an automated guard that catches a newly introduced bespoke/default-state
    serialization path, with an explicit inventory of supported types and
    inspection of actual reductions/payloads. Each newly written supported
    object's payload must use Stablebear binary IO and be loadable through its
    public IO path. Pass: complete inventory, no uncovered pickleable public
    type, and a guard that fails for a deliberately introduced bypass.

- [ ] **H2. New pickle and public IO parity for all types.**

    Scope: #230. Independently enumerate `BoolTensor`; float/signed/unsigned
    numeric tensors at every width; `PcfTensor`/`IntPcfTensor`; point-cloud,
    barcode, distance-matrix and symmetric-matrix tensors at both precisions;
    all supported nested leaf types; standalone `Pcf` variants, `Barcode`,
    `DistanceMatrix`, `SymmetricMatrix`, and `PointCloud`; and any other public
    pickleable types found in the source audit performed within this entry.
    Exercise supported pickle protocols including default/highest, scalar and
    empty shapes, multidimensional/noncontiguous views, ragged/deep nesting,
    and indexed clouds. Test dumps/loads, file dump/load, and public binary IO
    for each independently. Inspect the embedded binary representation rather
    than merely testing round-trip equality. Pass: type, dtype, shape, contents,
    type/depth metadata, special values, and appropriate singleton identity or
    object independence survive; restored objects work in normal operations.

- [ ] **H3. Fixed historical pickle fixtures and reconstruction paths.**

    Scope: #230. Independently audit Git history for every replaced pickle
    representation, including changes already present on `main`; current
    round-trip tests are not a replacement for historical fixtures. Produce
    fixtures with each actual previous implementation and commit the fixed
    bytes with producer/protocol/backend/checksum/expected-value metadata.
    Cover changed reducers, historical reconstruction function/module paths,
    moved classes where relevant, both precisions, and representative views/
    empties. Run only loading through the current implementation in compatibility
    tests. Pass: every replaced representation has fixed-byte coverage with
    exact type/shape/dtype/content checks and retained reconstruction symbols;
    missing historical bytes remain an explicit coverage gap.

- [ ] **H4. CPU/CUDA serialization interoperability.**

    Scope: #230 plus #229/#231 backend acceptance. Within this entry, independently
    construct representative examples of every supported public type and fixed
    historical fixtures needed for backend checks. In separate processes, write
    on `_sb_cpu` and read on each available CUDA-backed module, then reverse
    direction; exercise CUDA 12/13 cross-reading when both are supported and
    available. Run both binary and pickle paths, nested and indexed structures,
    and forced-CPU execution within CUDA builds. Pass: public types and values
    restore without serialized backend-specific class dependencies. Record
    unavailable modules/hardware as unverified, never as passing CUDA coverage.

### I. Regression coverage and documentation

- [ ] **I1. Ordinary tensor and point-process regression coverage.**

    Scope: branch-wide generic tensor/binding changes beyond the issue examples.
    Exercise existing materialized numeric, bool, PCF, barcode, compressed
    matrix and point-cloud construction, indexing, broadcasting, equality,
    arithmetic, reductions, joins/splits, assignment, casts, NumPy conversion,
    and IO. Include scalar/empty/strided views, since the template and shared
    binding refactors affect all types. Run Poisson point-process regressions
    for bounds, precision, shape, empty draws, and deterministic generators;
    it now constructs coordinate tensors and wraps `PointCloud`. Compare
    compressed-matrix view `storage_count` and `allclose` against existing
    expectations as adjacent regressions. Earlier branch commit subjects about
    `subsample_relative` and compressed-matrix rewrites are not net changes at
    this reviewed HEAD; do not invent APIs from those subjects. Pass: no
    regressions from the 53-file net diff.

- [ ] **I2. Recover the intent of deleted or rewritten tests.**

    Scope: deletion of `test/python/test_bugscan_persistence.py` and changes to
    point-cloud and C++ tests. Independently compare the deleted tests with
    remaining coverage. Retain validation for rank-1 and invalid rank-3 single
    clouds, mixed valid/invalid assignments, ordinary valid homology results,
    and exception propagation at the appropriate current boundary. Distinguish
    a valid rank-3 dense tensor-of-clouds constructor from an invalid rank-3
    single-cloud argument. Check that old arbitrary-rank point-cloud tests were
    replaced by rejection coverage and that newly added tests inspect logical
    selections rather than enshrining obsolete per-element indexed storage.
    Pass: every still-valid regression intent has a current test location;
    record intentional removals and changed exception timing explicitly.

- [ ] **I3. Documentation, examples, and release-note accuracy.**

    Scope: all seven changed documentation pages and `CHANGELOG.md`, plus public
    docstrings/stubs. Independently run the issue examples and documented
    construction/indexing/subsampling/saving/persistence examples against the
    installed package. Check recursive dtype/depth and empty/scalar creation,
    exact prefix indexing, ownership and shared-state writes, `PointCloud`
    return types, supported standalone IO/pickle types, fixed legacy-reader
    promises, and actual subtype/version descriptions. Remove stale public
    `IndexTensor` terminology from the proposed documentation changes when
    implementation is undertaken. Build Sphinx with `make html` from `docs/`
    using the TeX/pdf2svg/Pandoc prerequisites in `CLAUDE.md`. Pass: examples
    execute, links resolve, and prose matches the accepted contracts rather
    than intermediate implementation details.

### J. Golden tensor files and expected values

- [ ] **J1. Small golden tensor files from the current version and 0.4.7.**

    Scope: commit two permanent golden corpora under
    `test/fixtures/tensors/`: one produced by the current branch version, pinned
    to its exact generation commit, and one produced by the actual released
    Stablebear `0.4.7`. Give their directories immutable producer identities
    such as `current-<commit>/` and `0.4.7/`; do not use a rolling directory
    whose bytes are overwritten on every release. This entry includes its own
    inventory, data descriptions, generation procedure, binary/pickle files,
    and load tests, so it has no dependency on other checklist entries.

    **Type and format coverage.** Independently inventory every supported tensor
    type and dtype in each producer. For the current version, include
    `BoolTensor`; `FloatTensor` (`float32`, `float64`); `IntTensor` (`int32`,
    `int64`, `uint32`, `uint64`); `PcfTensor` (`pcf32`, `pcf64`);
    `IntPcfTensor` (`pcf32i`, `pcf64i`); `PointCloudTensor` (`pcloud32`,
    `pcloud64`); `BarcodeTensor` (`barcode32`, `barcode64`);
    `DistanceMatrixTensor` (`distmat32`, `distmat64`);
    `SymmetricMatrixTensor` (`symmat32`, `symmat64`); and `NestedTensor`
    with each of its six supported numeric leaf dtypes. Include any additional
    supported tensor types discovered during generation. Each supported
    type/dtype must have a nonempty `.sb` file and a `.pkl` file containing the
    same logical tensor, generated by that version's actual writers.

    Maintain an explicit producer/type/dtype/format coverage table. Types or
    representations genuinely absent from 0.4.7, such as new recursive/indexed
    representations, belong only in the current corpus, with the historical
    absence recorded. A failed save or pickle for a type that exists must be
    investigated and recorded as a gap, not relabeled as an absent type. Do not
    backport a writer or use the current writer to manufacture 0.4.7 fixtures.
    Use matching logical examples for types shared by the two versions.

    **Small, discriminating examples.** Prefer outer shapes `(2,)` or `(2, 2)`,
    two or three points/breakpoints/bars per cell, and matrices of order two or
    three. Use distinct, hand-chosen values and shapes to reveal ordering,
    truncation, signedness, precision, and cell-mapping errors. Include exact
    powers-of-two fractions, negative signed values, an unsigned value above
    the signed range, and representative empty/scalar/noncontiguous cases.
    Avoid multiplying every edge case across the entire corpus: retain one
    ordinary case per type/dtype, then a few targeted extras for zero axes,
    ragged and deeper nested tensors, and indexed clouds at both precisions
    with ordered/repeated/empty selections. Do not use large random samples.
    Set a default limit of 4 KiB per binary/pickle file and 256 KiB for the
    complete new corpus including manifests and descriptions; check byte sizes
    automatically and document any justified exception. Commit raw small files
    directly to Git, without large-file storage or opaque archives.

    **Independent JSON oracle.** Commit a versioned JSON schema and human-readable
    expected-data descriptions. Each logical case describes public tensor type,
    dtype, outer shape, and row-major elements; both its `.sb` and `.pkl` files
    reference that same description. Describe values independently of the
    serialized bytes and the current loader; never derive expected values by
    reading the files under test. Specify child shapes explicitly so `()` is
    distinct from `(0,)`, and an empty `(0, D)` cloud retains its dimension.
    Numeric/bool leaves use explicit values; PCFs use ordered breakpoint/value
    pairs; barcodes use ordered birth/death pairs and any stored metadata;
    point clouds use coordinate shapes and rows; compressed matrices use their
    full small logical matrix, including the diagonal. Nested elements
    recursively carry shape, dtype/depth, and contents. Indexed-cloud cases
    describe both expected logical coordinates and required source/selection
    structure, so correct values alone cannot hide lost indexed storage.

    For example, a shared logical description can contain:

    ```json
    {
      "schema_version": 1,
      "case_id": "float32-two-by-two",
      "tensor_type": "FloatTensor",
      "dtype": "float32",
      "shape": [2, 2],
      "values": [0.5, -2.0, 3.25, 0.0]
    }
    ```

    Define tagged encodings for infinities, NaN, negative zero, and integers
    beyond JSON consumers' exact numeric range; prohibit nonstandard JSON
    `NaN`/`Infinity` literals. Lossless values must compare exactly after
    interpreting the declared dtype; check NaNs and zero signs explicitly.
    Do not use broad floating tolerances that could hide serialization errors.

    **Provenance and generation.** Provide a small reproducible generation
    script and README, run separately in isolated installed environments for
    the pinned current commit and release 0.4.7. Verify the imported version,
    package path, and extension build before writing. Record producer version,
    commit or release artifact identity/hash, Python/NumPy versions, backend,
    generation command, binary format version, and explicit pickle protocol
    (use protocol 4 for this compact compatibility corpus unless the supported
    Python matrix requires otherwise). Record each file's path, logical case
    ID, byte count, and SHA-256 in a manifest. Other pickle protocols remain
    runtime-test cases; do not duplicate every golden unnecessarily. Validate
    each corpus against its JSON descriptions in its producer environment as
    well as with the current reader. Generation is a deliberate maintenance
    action, never part of normal test startup or an automatic repair on failure.

    **Fixed-file loading tests.** Discover every committed manifest entry and
    check completeness against the producer coverage table, file existence,
    checksums, sizes, schema validity, and absence of unlisted golden files.
    Load every `.sb` through public Stablebear IO and every `.pkl` through
    `pickle.load`/`pickle.loads` with the current implementation. Compare each
    loaded object independently against JSON: public type, dtype, outer/child
    shapes, recursive depth, all values, and required representation metadata.
    Checking only `.sb` versus `.pkl` equality is insufficient because both
    readers could share a defect. Exercise normal indexing on loaded tensors
    and mutation independence where applicable. Run on CPU and available
    CUDA-backed modules, recording unavailable coverage. Tests use committed
    bytes without requiring 0.4.7 to be installed and never regenerate fixtures.
    The compatibility direction is both producer versions into the current
    reader; 0.4.7 need not read newly introduced current formats.

    Pass: every supported producer/type/dtype has both small committed formats,
    all files load with correct JSON-described values and metadata, all
    exclusions have explicit historical evidence, and the corpus fits its size
    budget. Preserve existing goldens when serialization changes; add a new
    pinned corpus or targeted case rather than refreshing old bytes. This
    corpus supplements historical-format and standalone-object coverage, not
    just current writer/reader round trips.

### K. Final validation and acceptance

- [ ] **K1. Independent full regression and acceptance report.**

    Scope: final validation, independently runnable using the tests and build
    available at execution time; it does not consume another entry's artifacts.
    Build/install, run all Python tests from `test/`, build target `sb_test`,
    and run `../cmake-build-debug/sb_test` from `test/`. Repeat applicable runs
    across actual CPU/CUDA builds and runtime modes, using `SB_REQUIRE_CUDA=1`
    for a job that claims CUDA coverage. Run `./run_coverage.sh` from the root
    when coverage tooling is available and inspect unexercised branches in
    selection validation, nested ownership, shared materialization, and IO
    dispatch rather than using a percentage as proof. Report failures/skips,
    remaining issue-contract deviations, legacy fixture gaps, and exact tested
    commits. Pass: regression suites pass and every acceptance requirement has
    concrete evidence; an unresolved requirement stays open.

## Coverage map

| Source or net-diff area | Checklist entries |
| --- | --- |
| #229 API, uniformity, validation, duplicate semantics, random streams | D1–D4 |
| #229 copies, tensor-level adapter, views, whole-state mutation, resampling | A4, B5, E1–E5 |
| #229 downstream algorithms and compatibility | A2–A4, E5, F1–F2, G1–G3, H4, I1–I2 |
| #230 universal IO reducers, guard, all-type round trips | H1–H2 |
| #230 fixed legacy pickle readers, CPU/CUDA, documentation | H3–H4, I3 |
| #231 runtime recursion, homogeneous descriptor, empty/scalar construction | B1–B2, B5 |
| #231 ownership and outer operations | B3–B4 |
| #231 prefix indexing, lazy alignment, repeated indexing | C1–C2, E1–E4 |
| #231 recursive IO, `(5, 64)` compatibility, backend/docs | G1–G3, H4, I3 |
| New Python `PointCloud`, strict rank-two validation, moved exports | A1–A4, H1–H3, I2–I3 |
| C++ tensor properties, generic views/bindings, numeric wrapper dispatch | A1, B4–B5, E2, E5, I1 |
| Distance oracle, persistence dispatch, task input ownership | F1–F2 |
| Poisson coordinate wrapping and general tensor regressions | I1 |
| IO bool encoding, version change, recursive dispatch, legacy source tables | G1–G3 |
| Existing added/changed tests and deleted persistence tests | D1, F1, G1, I1–I2 |
| CMake bindings, generated API availability, all changed docs/changelog | A1, I3 |
| Small current-version and 0.4.7 golden files, all tensor types/dtypes, binary and pickle, independent JSON expectations | J1 |
| Full Python/C++ regression, coverage, acceptance evidence | K1 |
