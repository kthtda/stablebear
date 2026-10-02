#!/usr/bin/env python3
"""Archive changed and untracked files at their repository-relative paths."""

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile


def git(root, *args):
    return subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(root), *args]
    )


def backup(root, destination):
    destination = destination.expanduser().resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Choose a destination outside the repository.")

    base = git(root, "rev-parse", "HEAD").decode().strip()
    status = git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    metadata = {
        "base-commit.txt": (base + "\n").encode(),
        "branch.txt": git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "status.txt": git(root, "status", "--short", "--untracked-files=all"),
        "status.z": status,
        "staged.patch": git(
            root, "diff", "--cached", "--binary", "--full-index",
            "--no-ext-diff", "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/",
        ),
        "unstaged.patch": git(
            root, "diff", "--binary", "--full-index", "--no-ext-diff",
            "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/",
        ),
        "submodules.txt": git(root, "submodule", "status", "--recursive"),
    }
    names = set()
    entries = iter(status.split(b"\0"))
    for entry in entries:
        if not entry:
            continue
        names.add(entry[3:])
        if b"R" in entry[:2] or b"C" in entry[:2]:
            names.add(next(entries))  # Original path follows a rename/copy.
    # Gitlinks name directories; do not recurse into submodule worktrees.
    paths = [os.fsdecode(name) for name in sorted(names)
             if (root / os.fsdecode(name)).is_symlink()
             or (root / os.fsdecode(name)).is_file()]
    deleted = [os.fsdecode(name) for name in sorted(names)
               if not os.path.lexists(root / os.fsdecode(name))]
    if any(p == ".worktree-backup" or p.startswith(".worktree-backup/")
           for p in paths):
        raise ValueError(".worktree-backup is reserved for backup metadata.")
    metadata["deleted-files.json"] = (json.dumps(deleted, indent=2) + "\n").encode()
    metadata["files.z"] = b"\0".join(os.fsencode(p) for p in paths) + b"\0"
    metadata["RESTORE.txt"] = f"""Worktree backup
===============
Base commit: {base}

The archive contains only changed and non-ignored untracked files from
`git status`, at their normal repository-relative paths. Binary files,
symlinks, and executable permissions are preserved. Unchanged tracked files,
ignored files, Git history, and submodule worktrees are not included.
Submodule revisions are in .worktree-backup/submodules.txt; back up a modified
submodule separately by running this script inside it.

To restore the files over your existing branch, run from its repository root:
    tar -xzf /path/to/backup.tar.gz
This overwrites archived paths with their saved versions. Use the base commit
above for an exact reconstruction; other commits or later edits may conflict.
Unpacking cannot delete files: remove the paths listed in
.worktree-backup/deleted-files.json, including old names of renamed files.
Files added after the backup are also not removed automatically.

Unpacking leaves your Git index unchanged. To also recover what was staged,
start with an index matching the base commit, then apply the saved patch:
    git apply --cached .worktree-backup/staged.patch
Skip this command if the patch is empty. Compare status with
.worktree-backup/status.txt. Merge conflicts and special index flags may
require manual recovery; this is not a copy of the index. The unstaged patch
is included for inspection or patch-based recovery, not for applying after
unpacking the saved files.

To recover individual files instead, extract into a separate directory and
copy only the files you need. .worktree-backup/ contains recovery metadata.

Pause edits while backing up: a filesystem archive is not an atomic snapshot.
""".encode()

    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive = destination / f"{root.name}-{stamp}.tar.gz"
    fd, temporary = tempfile.mkstemp(prefix=".backup-", dir=destination)
    os.close(fd)
    try:
        with tarfile.open(temporary, "w:gz", dereference=False) as tar:
            for name, content in metadata.items():
                info = tarfile.TarInfo(".worktree-backup/" + name)
                info.size = len(content)
                info.mode = 0o600
                tar.addfile(info, io.BytesIO(content))
            for path in paths:
                tar.add(root / path, arcname=path, recursive=False)
        if git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all") != status:
            raise RuntimeError("Git status changed during backup; pause edits and retry.")
        os.replace(temporary, archive)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return archive, len(paths)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Run from anywhere in the repository. Ignored files and submodule "
               "worktrees are excluded; back up modified submodules separately.",
    )
    parser.add_argument(
        "destination", nargs="?", type=Path,
        help="directory outside the repository (default: ~/<repo>-backups)",
    )
    args = parser.parse_args()
    try:
        root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
        destination = args.destination or Path.home() / (root.name + "-backups")
        archive, count = backup(root, destination)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Backup failed: {error}\n")
    print(f"Saved {count} files and Git metadata to {archive}")


if __name__ == "__main__":
    main()
