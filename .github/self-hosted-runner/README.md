# Self-hosted GPU runner

Ephemeral, containerized GitHub Actions runner with NVIDIA GPU support.
Each job gets a fresh container that is destroyed after completion.

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- `gh` CLI, authenticated (`gh auth login`)
- `nvidia-smi` working on the host

## Quick start

```bash
# 1. Build images for CUDA 12 and 13
./build.sh

# 2. Start runners (runs in foreground, Ctrl-C to stop)
REPO=owner/stablebear ./run.sh

# Or run only one CUDA version
REPO=owner/stablebear ./run.sh cuda12
```

## Workflow usage

Target these runners with the `self-hosted` and CUDA version labels:

```yaml
jobs:
  gpu-test-cuda12:
    runs-on: [self-hosted, cuda12]
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - run: pip install .
      - run: cd test && python -m pytest python

  gpu-test-cuda13:
    runs-on: [self-hosted, cuda13]
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - run: pip install .
      - run: cd test && python -m pytest python
```
