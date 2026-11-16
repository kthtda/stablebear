#!/usr/bin/env bash
# Configure a build directory anywhere, for the active Python environment and
# the CUDA toolkit installed in it with pip (see pip-cuda-env.sh):
#
#   ./configure-build.sh ~/build/stablebear-release [Release|RelWithDebInfo|Debug]
#   cmake --build ~/build/stablebear-release -j8
#   cmake --install ~/build/stablebear-release
#
# Extra arguments go to cmake, such as -DCMAKE_CUDA_ARCHITECTURES="80-real;90".
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 BUILD_DIR [BUILD_TYPE] [CMAKE_ARGS...]" >&2
    exit 1
fi
build_dir=$1
build_type=${2:-Release}
shift $(( $# < 2 ? $# : 2 ))

repo=$(cd "$(dirname "$0")" && pwd)
source "$repo/pip-cuda-env.sh"

generator=()
if command -v ninja >/dev/null; then
    generator=(-G Ninja)
fi

cmake -S "$repo" -B "$build_dir" "${generator[@]}" \
    -DCMAKE_BUILD_TYPE="$build_type" \
    -DPython_EXECUTABLE="$(command -v python)" \
    "$@"
