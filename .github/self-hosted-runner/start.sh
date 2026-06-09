#!/bin/bash
set -euo pipefail

# Spawn an ephemeral GPU runner for CUDA 12.
# The runner picks up one job then exits; the loop respawns it.
#
# Prerequisites:
#   - Run build.sh first
#   - gh CLI authenticated (used to fetch registration tokens)
#
# Usage: ./run.sh
#    or: REPO=owner/repo ./run.sh

if [[ -z "${REPO:-}" ]]; then
    REPO="$(git remote get-url origin 2>/dev/null | sed -E 's#(https://github\.com/|git@github\.com:)##; s#\.git$##')"
fi
: "${REPO:?Could not detect repo. Set REPO=owner/repo}"

get_token() {
    gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq .token
}

echo "[cuda12] Waiting for jobs..."
while true; do
    TOKEN="$(get_token)"
    docker run --rm --gpus all \
        -e REPO_URL="https://github.com/${REPO}" \
        -e TOKEN="$TOKEN" \
        -e RUNNER_NAME="$(hostname)-cuda12" \
        -e LABELS="self-hosted,gpu,cuda12" \
        "stablebear-runner:cuda12"
    echo "[cuda12] Job finished. Respawning..."
    sleep 1
done
