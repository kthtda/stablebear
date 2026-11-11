#!/usr/bin/env bash
# Run a command on the simulated GPUs that the PantheonSim action sets up,
# in a `vgpu shell` session.
#
# `vgpu run` only puts the simulator's CUDA libraries first, so the GPUs are
# visible to CUDA but not to stablebear's GPU detection, which reads
# /sys/class/drm. A session also mounts the GPUs' sysfs and /proc entries,
# in user namespaces that the job allows first. The session takes the GPU
# model and count from VGPU_GPU and VGPU_DEVICE_COUNT, which the action
# sets, and exits with the command's status.
#
# Usage:
#   bash .github/scripts/vgpu-shell.sh <command> [args...]
set -euo pipefail
exec vgpu shell -y -c "$(printf '%q ' "$@")"
