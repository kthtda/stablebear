#!/usr/bin/env bash
# Stage host config in the build context before the image build. Never print credentials.
set -euo pipefail
umask 077

copy_root=${1:?Expected a local staging directory}
source_dir=${GH_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/gh}

# Reject staging paths owned by someone else or symlinks.
for directory in "$copy_root" "$copy_root/gh"; do
    if [[ -L "$directory" ]]; then
        echo "Refusing symlink staging directory: $directory" >&2
        exit 1
    fi
    if [[ ! -e "$directory" ]]; then
        mkdir -m 700 -- "$directory"
    fi
    if [[ ! -d "$directory" || ! -O "$directory" ]]; then
        echo "Staging directory must be owned by the current user: $directory" >&2
        exit 1
    fi
    chmod 700 -- "$directory"
done

temporary_file=
trap 'if [[ -n "$temporary_file" ]]; then rm -f -- "$temporary_file"; fi' EXIT
for name in hosts.yml config.yml; do
    destination="$copy_root/gh/$name"
    if [[ -f "$source_dir/$name" ]]; then
        temporary_file=$(mktemp "$copy_root/gh/.copy.XXXXXX")
        cat -- "$source_dir/$name" > "$temporary_file"
        chmod 600 -- "$temporary_file"
        # Replace files rather than following any destination symlinks.
        if [[ -d "$destination" ]]; then
            echo "Expected a file at $destination" >&2
            exit 1
        fi
        mv -f -- "$temporary_file" "$destination"
        temporary_file=
    else
        rm -f -- "$destination"
    fi
done
