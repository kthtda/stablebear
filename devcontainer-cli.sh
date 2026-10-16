#!/usr/bin/env bash
# Start the CUDA devcontainer with the devcontainers CLI and open a shell in it.
# Extra arguments go to `devcontainer up`, e.g. --remove-existing-container.
set -euo pipefail

workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Scripts do not run the interactive shell setup that usually loads nvm, so
# load it here if present. Otherwise the node on PATH is used.
export NVM_DIR=${NVM_DIR:-$HOME/.nvm}
if [[ -s "$NVM_DIR/nvm.sh" ]]; then
    . "$NVM_DIR/nvm.sh"
    nvm use default --silent >/dev/null 2>&1 || true
fi

if ! node_major=$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null) \
    || ((node_major < 18)) || ! command -v npx >/dev/null; then
    echo "The devcontainers CLI needs Node.js 18 or newer with npx on PATH." >&2
    exit 1
fi

devcontainer() {
    npx --yes --package @devcontainers/cli devcontainer "$1" \
        --workspace-folder "$workspace" \
        --config "$workspace/.devcontainer/devcontainer.json" "${@:2}"
}

devcontainer up "$@"
devcontainer exec bash -l
