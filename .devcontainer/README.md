# Devcontainer: masspcf

Dev environment with Claude Code preinstalled. Two configs share one Dockerfile:

| Config | Folder | Base image | GPU |
|---|---|---|---|
| `masspcf-cuda` (default) | `.devcontainer/` | `nvidia/cuda:12.6.3-devel-ubuntu22.04` | yes (`--gpus all`) |
| `masspcf-cpu` | `.devcontainer/cpu/` | `ubuntu:22.04` (multi-arch) | no -- runs on **macOS** too |

The CPU config builds only `_mpcf_cpu` (CUDA auto-off, `BUILD_WITH_CUDA=0`) and is
arch-agnostic, so it works on Apple Silicon and Intel Macs (and any Linux without
an NVIDIA GPU). Everything below is shared unless noted.

## What's inside

- **Base**: see table above. Both are Ubuntu 22.04 => system Python is **3.10**.
- **Toolchain**: gcc/g++-13 (default cc/c++; full C++20, within CUDA 12.6's host-compiler range), cmake, ninja, git, ripgrep.
- **Python 3.10** in a venv at `/opt/venv` (auto on `PATH`). Deps are
  installed at image-build time straight from the repo's canonical lists, so
  there's no second copy to maintain:
  - runtime + test deps from `pyproject.toml` (`[project].dependencies` +
    `[tool.cibuildwheel].test-requires`), read via `tomli`;
  - Sphinx docs deps from `docs/requirements.txt`.
- **Node 20** + `@anthropic-ai/claude-code` (run `claude` in the terminal).
- **GPU** (cuda config only): passed through via `--gpus all`. Builds `_mpcf_cuda12`, arch auto-detected (`native` -> sm_89 for the RTX 4070 Ti).

## Host prerequisites

**GPU config (Linux):**
- Docker.
- `nvidia-container-toolkit` (so `--gpus all` works).
- Verify: `docker run --rm --gpus all nvidia/cuda:12.6.3-devel-ubuntu22.04 nvidia-smi`

**CPU config (incl. macOS):**
- Docker Desktop (macOS) or Docker (Linux). No GPU / nvidia toolkit needed.

## Start it

- **VS Code**: Command Palette -> "Dev Containers: Reopen in Container". With two
  configs present, VS Code prompts which to use (`masspcf-cuda` or `masspcf-cpu`).
  On macOS, pick `masspcf-cpu`.
- **CLI (GPU)**: `devcontainer up --workspace-folder .`
- **CLI (CPU)**: `devcontainer up --workspace-folder . --config .devcontainer/cpu/devcontainer.json`
- Then `devcontainer exec --workspace-folder . bash`.

First start runs `postCreateCommand`: submodule init, then the cmake build + install
(`debug` preset for GPU, `debug-nocuda` for CPU). The extensions are built and
symlinked into `masspcf/` -- not pip-installed.

## Verify

```bash
python -c "import masspcf, masspcf._mpcf_cpp as m; print(m.__file__)"
nvidia-smi   # GPU config only
```

## Build loop (iterative C++)

Presets live in repo-root `CMakePresets.json`:

```bash
cmake --preset debug          # configure
cmake --build --preset debug  # build _mpcf_cpu (+_mpcf_cuda12)
cmake --install cmake-build-debug   # symlink extensions into masspcf/
```

C++ tests:

```bash
cmake --build --preset tests
cd test && ../cmake-build-debug/mpcf_test
```

Python tests (must `cd test` first, else local `masspcf/` shadows install):

```bash
cd test && python -m pytest python
```

Benchmarks/perf -> always use the release preset:

```bash
cmake --preset release && cmake --build --preset release && cmake --install cmake-build-release
```

CUDA-off build: `cmake --preset debug-nocuda`.

## Docs (Sphinx)

Deps are already in the image. Build:

```bash
cd docs && make html   # output in docs/_build/html
```

## Claude Code auth

`~/.claude` and `~/.claude.json` are bind-mounted from the host, so login + config
persist across rebuilds. To share **credentials only** (not full host config/hooks),
edit `devcontainer.json` and narrow the first mount to
`~/.claude/.credentials.json`.

## Notes

- GPU must be visible at configure time for `native` arch detection. It is (`--gpus all`),
  so no manual `CMAKE_CUDA_ARCHITECTURES` needed.
- The base image already sets `NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES`;
  they're pinned in `devcontainer.json` for clarity.
- `cmake-build-debug/` is a **Docker volume**, not the host dir. The workspace is
  bind-mounted, but its `CMakeCache.txt` is baked with host paths and CMake rejects
  a cache from a different source/binary path -- so the container keeps its own build
  dir. Consequence: the extension symlinks `cmake --install` writes into `masspcf/`
  use absolute build-dir paths, so they're valid in whichever environment last ran
  the install. Re-run `cmake --install` after switching between host and container.
