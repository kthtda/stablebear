#!/usr/bin/env bash
# Stage host Claude Code and Codex state for the devcontainer's bind mounts.
#
# Docker reads bind-mount sources as root, and root holds no AFS tokens. A home
# directory on AFS therefore mounts without error, but every file in it reads as
# "Permission denied" inside the container. For sources on AFS, this script
# stages a copy that Docker can read. For other sources, it stages a symlink,
# so the mount still shows the live host files. For a source missing on the
# host, it stages an empty one, so the mount works and the tool starts logged
# out; what the container saves there stays in the staging directory.
#
# Set SB_DEVCONTAINER_COPY_HOME=1 to copy regardless of filesystem.
set -euo pipefail
umask 077

copy_root=${1:?Expected a local staging directory}
stage_root=$copy_root/home

# Reject staging paths owned by someone else or symlinks.
for directory in "$copy_root" "$stage_root"; do
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

needs_copy() {
    if [[ ${SB_DEVCONTAINER_COPY_HOME:-} == 1 ]]; then
        return 0
    fi
    [[ $(uname -s) == Linux && $(stat -f -c %T -- "$1") == afs ]]
}

stage() {
    local source=$HOME/$1
    local destination=$stage_root/$2
    local kind=$3

    if [[ ! -e "$source" ]]; then
        # A symlink left from a run when the host had the source.
        if [[ -L "$destination" ]]; then
            rm -f -- "$destination"
        fi
        if [[ ! -e "$destination" ]]; then
            if [[ $kind == directory ]]; then
                mkdir -- "$destination"
            else
                printf '{}\n' > "$destination"
            fi
            echo "No $source on the host; the devcontainer keeps its own in $destination"
        fi
        return
    fi

    if needs_copy "$source"; then
        # Never copy through a symlink staged by an earlier non-copy run.
        if [[ -L "$destination" ]]; then
            rm -f -- "$destination"
        fi
        # -u keeps files the container changed after the host did.
        if [[ -d "$source" ]]; then
            mkdir -p -- "$destination"
            cp -R -p -u -- "$source/." "$destination/"
        else
            cp -p -u -- "$source" "$destination"
        fi
        echo "Copied $source for the devcontainer (container changes stay in $destination)"
    else
        # Keep a copy from an earlier copy run; the container may have changed it.
        if [[ -e "$destination" && ! -L "$destination" ]]; then
            local kept
            kept=$destination.kept-$(date +%Y%m%d-%H%M%S)
            mv -- "$destination" "$kept"
            echo "Kept the earlier copy of $source at $kept; merge any changes you need by hand"
        fi
        ln -sfn -- "$source" "$destination"
    fi
}

stage .claude claude directory
stage .claude.json claude.json json
stage .codex/auth.json codex-auth.json json
