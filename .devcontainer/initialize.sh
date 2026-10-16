#!/usr/bin/env bash
# Run on the host before the devcontainer is built or started. Stages host
# credentials and state in a local directory that Docker can read.
set -euo pipefail

copy_root=${1:?Expected a local staging directory}
script_dir=$(dirname -- "${BASH_SOURCE[0]}")

bash "$script_dir/prepare-gh.sh" "$copy_root"
bash "$script_dir/prepare-home.sh" "$copy_root"
