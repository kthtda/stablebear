#!/bin/bash
set -euo pipefail

# Build runner image for CUDA 12.
# Usage: ./build.sh  (set RUNNER_VERSION to pin; defaults to the latest release)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNNER_VERSION="${RUNNER_VERSION:-$(curl -fsSLI -o /dev/null -w '%{url_effective}' \
    https://github.com/actions/runner/releases/latest | sed 's#.*/v##')}"

echo "=== Building CUDA 12 runner image (actions runner ${RUNNER_VERSION}) ==="
docker build \
    --build-arg CUDA_VERSION=12.8.0 \
    --build-arg RUNNER_VERSION="${RUNNER_VERSION}" \
    -t stablebear-runner:cuda12 \
    "$SCRIPT_DIR"

echo ""
echo "Done. Image available:"
echo "  stablebear-runner:cuda12"
