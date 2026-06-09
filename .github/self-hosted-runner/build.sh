#!/bin/bash
set -euo pipefail

# Build runner image for CUDA 12.
# Usage: ./build.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Building CUDA 12 runner image ==="
docker build \
    --build-arg CUDA_VERSION=12.8.0 \
    -t stablebear-runner:cuda12 \
    "$SCRIPT_DIR"

echo ""
echo "Done. Image available:"
echo "  stablebear-runner:cuda12"
