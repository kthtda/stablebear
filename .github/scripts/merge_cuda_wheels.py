import argparse
import base64
import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def _extract_wheel(wheel_path, dest_dir):
    with zipfile.ZipFile(wheel_path, "r") as zf:
        zf.extractall(dest_dir)


def _find_dist_info(root):
    for entry in os.listdir(root):
        if entry.endswith(".dist-info") and os.path.isdir(os.path.join(root, entry)):
            return entry
    raise RuntimeError("Could not find .dist-info directory in wheel")


def _iter_files(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            yield os.path.join(dirpath, name)


def _record_hash_and_size(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = base64.urlsafe_b64encode(h.digest()).decode("ascii").rstrip("=")
    size = os.path.getsize(path)
    return f"sha256={digest}", str(size)


def _write_record(root, dist_info_dir):
    record_path = os.path.join(root, dist_info_dir, "RECORD")
    lines = []
    for path in sorted(_iter_files(root)):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        if rel == f"{dist_info_dir}/RECORD":
            lines.append(f"{rel},,\n")
            continue
        h, size = _record_hash_and_size(path)
        lines.append(f"{rel},{h},{size}\n")
    with open(record_path, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)


def _merge_cuda_modules(base_root, extra_root, cuda_major):
    prefix = f"stablebear/_sb_cuda{cuda_major}"
    for path in _iter_files(extra_root):
        rel = os.path.relpath(path, extra_root).replace(os.sep, "/")
        if (
            rel == prefix
            or rel.startswith(prefix + ".")
            or rel.startswith(prefix + "/")
        ):
            dest = os.path.join(base_root, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(path, dest)


def merge_wheels(base_wheel, extra_wheel, output_dir, cuda_major):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        tempfile.TemporaryDirectory() as base_dir,
        tempfile.TemporaryDirectory() as extra_dir,
    ):
        _extract_wheel(base_wheel, base_dir)
        _extract_wheel(extra_wheel, extra_dir)

        _merge_cuda_modules(base_dir, extra_dir, cuda_major)

        dist_info = _find_dist_info(base_dir)
        _write_record(base_dir, dist_info)

        out_path = output_dir / Path(base_wheel).name
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in _iter_files(base_dir):
                rel = os.path.relpath(path, base_dir).replace(os.sep, "/")
                zf.write(path, rel)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge CUDA wheel variants into one wheel."
    )
    parser.add_argument(
        "--base-wheel", required=True, help="Wheel containing _sb_cuda12"
    )
    parser.add_argument(
        "--extra-wheel", required=True, help="Wheel containing _sb_cuda13"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write merged wheel"
    )
    parser.add_argument(
        "--cuda-major",
        required=True,
        type=int,
        help="CUDA major version of extra wheel (e.g. 13)",
    )
    args = parser.parse_args()

    merge_wheels(args.base_wheel, args.extra_wheel, args.output_dir, args.cuda_major)


if __name__ == "__main__":
    main()
