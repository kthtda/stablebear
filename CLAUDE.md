# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**masspcf** is a Python package with a C++20/CUDA backend for massively parallel
computation on collections of piecewise constant functions (PCFs), aimed at TDA
(Topological Data Analysis) practitioners and statisticians. It is
GPU-accelerated when an NVIDIA GPU is present and falls back to a multi-threaded
CPU backend; the choice is automatic and tunable.

The core objects are numpy-like tensors holding PCFs, floats, ints, point
clouds, persistence barcodes, or distance/symmetric matrices. On top of them it
provides Lp distance matrices and norms, L2 kernels, pointwise reductions, full
numpy-style tensor manipulation, persistent homology (Ripser) with barcode
summaries (stable rank, Betti curves, accumulated persistence), point-process
sampling, plotting, and binary IO. `masspcf/__init__.py` is the source of truth
for the full public surface.

## Build and test

Three things trip up almost every change here -- internalize them first:

1. **Re-run `cmake --install` before testing, even for pure-Python edits.** Tests
   import the *installed* package from site-packages, and `cmake --install`
   copies the `.py` files there. Editing `masspcf/*.py` without re-installing has
   **no effect** on tests -- the most common reason a change seems to "do nothing".
2. **`cd test` before pytest.** From the repo root the local `masspcf/` directory
   shadows the installed package.
3. **Use `cmake --build <dir> -j` (no number), not `-j$(nproc)`** -- `nproc`
   doesn't exist on macOS; `-j` / `--parallel` lets the generator (Ninja)
   auto-detect cores on every platform.

### Fresh checkout (once)
`pip install .` installs dependencies and registers the package. Not needed again
afterward -- use the cycle below for all iteration, C++ and Python.

### Iterate
```bash
cmake -B cmake-build-debug          # or: cmake --preset debug
cmake --build cmake-build-debug -j
cmake --install cmake-build-debug
```
Builds `_mpcf_cpu` (always) and `_mpcf_cuda{12,13}` (when CUDA is present),
installs the extensions plus all `masspcf/*.py` into site-packages, and
back-copies the `.so`s into the source tree for IDE completion (unless
`SKIP_BACK_COPY=1`). Presets: `debug`, `release` (benchmarks/perf),
`debug-nocuda`, `tests`.

### Run tests (Python-first)
Prefer Python tests -- users interact almost entirely through the Python API. Add
or extend C++ GoogleTest tests only for things Python can't reach (internal
C++-only logic, low-level invariants, CUDA kernels).
```bash
cd test && python -m pytest python                            # all Python tests
cd test && python -m pytest python/test_pdist.py::test_name   # one test
```
Fixtures (`test/python/conftest.py`): `device` (runs CPU + CUDA, auto-skips CUDA
when absent), `pcf_dtype` (pcf32/pcf64). `MPCF_REQUIRE_CUDA=1` makes the suite
hard-fail instead of skip when no GPU is available (GPU CI).

C++ tests: `cmake --build cmake-build-debug -j --target mpcf_test`, then
`cd test && ../cmake-build-debug/mpcf_test`. Coverage: `./run_coverage.sh [--open]`.

### CUDA and env vars
- `BUILD_WITH_CUDA=0` forces a CPU-only build (auto-off on macOS). Pip CUDA
  toolkits `nvidia.cu12` / `nvidia.cu13` produce `_mpcf_cuda12` / `_mpcf_cuda13`.
- `SKIP_STUBGEN=1` skips pybind11_stubgen; `SKIP_BACK_COPY=1` skips the
  source-tree symlink (CI); `ENABLE_COVERAGE=1` builds with GCC/Clang coverage.

## Changes need tests and docs

For new or changed **public** behavior, a change isn't done until it ships with:
- **Thorough tests** (Python-first; cover edge cases, not just the happy path).
- **Docstrings** on new/changed public functions, classes, and parameters.
- **Updated Sphinx docs** -- the matching page under `docs/` (see "Documentation").

For small internal fixes with no user-visible surface, add a regression test; new
`.rst` usually isn't needed. When in doubt, treat the change as public.

## Do it right, not just done

Favor the correct, maintainable solution over the quickest path to a green
result. This is a research-grade numerical library; subtle wrongness is worse
than slowness.
- Fix root causes, not symptoms -- don't paper over a failure by loosening a
  tolerance, skipping or `xfail`ing a test, broadening an `except`, or
  special-casing the input that happened to break.
- Use the library's own abstractions (see "prefer native operations") instead of
  a quick hack that bypasses them, and match the surrounding code's idioms.
- Don't leave debris: no TODOs, commented-out code, disabled tests, or
  "temporary" workarounds. If something is genuinely out of scope, say so rather
  than stubbing it silently.
- When the right approach is unclear or costly, surface the tradeoff and ask --
  don't quietly ship the easy-but-wrong version.

## Working with tensors and PCFs: prefer native operations

**The single most important coding guideline.** A recurring failure mode is
reaching for raw pointers, manual Python loops over elements, or hand-rolled
C++/pybind code to do something the library already does as a vectorized native
operation. Before writing a loop, check the public API (`masspcf/__init__.py`,
and `include/mpcf/algorithms/` on the C++ side) -- the operation almost certainly
exists and runs on the parallel backend. The tensors are numpy-like: arithmetic
with broadcasting, comparisons, boolean/fancy indexing, slicing, reshaping, and
stacking/splitting are all available as operators and methods.

Common traps:
- Building a distance/kernel matrix by looping pairwise -- use `pdist`, `cdist`,
  or `l2_kernel` (they dispatch to the parallel CPU/CUDA backend).
- Reimplementing a reduction over PCFs -- use `reductions.py`.
- Dropping to raw buffers to add/scale tensors -- use the tensor operators.
- Looping to gather/scatter or reshape -- use indexing (`t[mask]`, `t[idx]`)
  and the shape methods.

The same applies in C++: use the tensor API and `include/mpcf/algorithms/`
rather than hand-written pointer loops; PCF storage and device transfer are
managed by the tensor.

### Parallel traversal and reproducible randomness (`walk` / `parallel_walk`)
Element-wise traversal in the C++ backend goes through `walk` / `parallel_walk`
(`include/mpcf/walk.hpp`) -- don't hand-roll index loops over tensor storage.
`walk` is sequential (row-major, with optional early exit); `parallel_walk`
distributes flat indices across threads via the taskflow executor.

For randomness, use the `RandomGenerator` overloads (`walk(t, gen, f)` /
`parallel_walk(t, gen, f, exec)`). Each element gets a fresh engine
`gen.sub_generator(flat)` seeded from `base_seed + flat_index` (splitmix64; see
`random_generator.hpp`). Because each element's stream depends only on the seed
and its flat index -- not on the thread or the thread count -- `walk` and
`parallel_walk` produce identical output, so randomized generation is
**reproducible across thread counts** for a fixed seed.

**Do NOT** share one engine across a `parallel_walk` body -- that reintroduces
thread-order dependence (and a data race) and breaks reproducibility. Always use
the per-element `engine`, and build any new randomized generation on these
overloads. Tests: `test/test_walk_random.cpp`.

## Architecture

Non-obvious wiring only (to find a file, list the directory rather than trusting
a path here):

- **Backend dispatch.** All Python imports go through `masspcf/_mpcf_cpp.py`,
  which detects the GPU and imports the CUDA backend or falls back to the CPU
  one. C++ lives in `include/mpcf/` (headers `.hpp`); pybind11 bindings in `src/`.
- **Public API.** `masspcf/__init__.py` is the source of truth for what's public;
  `persistence` and `point_process` are used as submodules, not from the
  top-level namespace.
- **Precision.** Each tensor class takes a `dtype`; the Python class dispatches to
  a separate per-precision C++ type internally.
- **Runtime control.** `masspcf/system.py` configures CPU/GPU at runtime. The CUDA
  threshold (default 500) is the number of PCFs a matrix computation must involve
  before it moves from CPU to GPU.

## Third-party code (`3rd/`)
Git submodules (see `.gitmodules`): **pybind11**, **taskflow** (header-only CPU
task parallelism), **googletest**. Also present but vendored / SDK-included (not
submodules): `ripser/`, `xoroshiro/` and `splitmix64/` (RNGs), `cuda/`,
`gcc-runtime/`, `msvc-runtime/`.

## Documentation (`docs/`)
- Sphinx docs in `docs/`, built with `make html` from there. Keep them in sync:
  when you change public API, defaults, or behavior, update the matching `.rst`
  page (find it under `docs/`) and the docstrings in the same change.
- Plots: include a `.. dropdown:: Show code` with a `.. literalinclude::`
  referencing snippet markers in the generation script (`docs/_static/`).

## Key files
- `pyproject.toml` -- single source of truth for version (`[project].version`),
  dependencies, and wheel build config.
- `BUILDING.md` -- authoritative build notes (version bumping, minimal builds, docs).
- `version.cpp.in` -- CMake template embedding version + build date into the binary.
- `.clang-format` -- C++ formatting (Microsoft style).
