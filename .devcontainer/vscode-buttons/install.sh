#!/usr/bin/env bash
# Package and install the Stablebear panel extension unless this version is
# already installed. The devcontainers run it as postAttachCommand, since
# rebuilding a container discards installed extensions.
set -euo pipefail

dir=$(cd "$(dirname "$0")" && pwd)
version=$(node -p "require('$dir/package.json').version")

# The code command needs the window's IPC socket, which lifecycle commands may
# not have; the server's own CLI does not.
code=$(command -v code-server || ls -t ~/.vscode-server/bin/*/bin/code-server 2>/dev/null | head -1 || true)
if [ -z "$code" ]; then
  code=code
fi

if "$code" --list-extensions --show-versions | grep -qx "stablebear.stablebear-buttons@$version"; then
  exit 0
fi

vsix=$(mktemp -d)/stablebear-buttons.vsix
(cd "$dir" && npx --yes @vscode/vsce package --allow-missing-repository --skip-license -o "$vsix")
"$code" --install-extension "$vsix" --force
rm -r "$(dirname "$vsix")"
