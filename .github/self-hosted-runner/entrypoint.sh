#!/bin/bash
set -euo pipefail

if [[ -z "${REPO_URL:-}" ]]; then
    echo "ERROR: REPO_URL is required (e.g. https://github.com/you/stablebear)"
    exit 1
fi

if [[ -z "${TOKEN:-}" ]]; then
    echo "ERROR: TOKEN is required (a runner registration token)"
    exit 1
fi

LABELS="${LABELS:-self-hosted,gpu,cuda}"
RUNNER_NAME="${RUNNER_NAME:-gpu-runner}"

./config.sh \
    --url "$REPO_URL" \
    --token "$TOKEN" \
    --name "$RUNNER_NAME" \
    --labels "$LABELS" \
    --ephemeral \
    --disableupdate \
    --unattended \
    --replace

./run.sh

./config.sh remove --token "$TOKEN" 2>/dev/null || true
