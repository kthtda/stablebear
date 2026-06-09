import io
import sys
import tarfile
import zipfile
from fnmatch import fnmatch
from pathlib import Path

IGNORE_GLOBS = [
    "stablebear/_sb_cpu*.so*",  # Linux/Mac Python extension module (CPU)
    "stablebear/_sb_cpu*.pyd",  # Windows Python extension module (CPU)
    "stablebear/_sb_cuda*.so*",  # Linux/Mac Python extension module (CUDA)
    "stablebear/_sb_cuda*.pyd",  # Windows Python extension module (CUDA)
    "stablebear/_gpu_detect*.so*",  # Linux/Mac Python extension module (GPU detect)
    "stablebear/_gpu_detect*.pyd",  # Windows Python extension module (GPU detect)
    "*/sboms/auditwheel.cdx.json",  # auditwheel metadata, varies by platform
    "stablebear.libs/libgcc*.so*",  # bundled gcc library
    "stablebear.libs/libstdc++*.so*",  # bundled stdc++ library
    "stablebear.libs/*.dll",  # delvewheel-bundled DLLs (hash-mangled, vary per wheel)
    "*.dist-info/DELVEWHEEL",  # delvewheel marker file, Windows wheels only
]

BUNDLED_LIB_GLOBS = [
    "stablebear.libs/*.so*",
    "stablebear.libs/*.dll",
    "stablebear.libs/*.dylib",
]


def should_ignore(filename):
    return any(fnmatch(filename, pattern) for pattern in IGNORE_GLOBS)


def read_wheel(whl):
    all_files = whl.namelist()
    raw = sorted(f for f in all_files if not f.endswith("/"))
    filtered = {f for f in all_files if not should_ignore(f) and not f.endswith("/")}
    return filtered, raw


def get_wheels_from_zip(zip_path):
    wheels = {}
    raw = {}
    with zipfile.ZipFile(zip_path) as zf:
        wheel_names = [n for n in zf.namelist() if n.endswith(".whl")]
        for wheel_name in wheel_names:
            with zf.open(wheel_name) as wf:
                wheel_data = io.BytesIO(wf.read())
                with zipfile.ZipFile(wheel_data) as whl:
                    wheels[Path(wheel_name).name], raw[Path(wheel_name).name] = (
                        read_wheel(whl)
                    )
    return wheels, raw


def get_wheels_from_dir(dir_path):
    wheels = {}
    raw = {}
    for wheel_path in sorted(Path(dir_path).glob("*.whl")):
        with zipfile.ZipFile(wheel_path) as whl:
            wheels[wheel_path.name], raw[wheel_path.name] = read_wheel(whl)
    return wheels, raw


def dump_contents(name, files):
    print(f"  Contents of {name}:")
    for f in files:
        print(f"    {f}")


def dump_sdist(path):
    """List contents of any .tar.gz source distributions found."""
    if path.is_dir():
        sdists = sorted(path.glob("*.tar.gz"))
    else:
        sdists = []

    for sdist_path in sdists:
        print(f"📦 Source distribution: {sdist_path.name}")
        with tarfile.open(sdist_path, "r:gz") as tf:
            members = sorted(m.name for m in tf.getmembers() if not m.isdir())
            for m in members:
                print(f"    {m}")
        print()


def main(input_path, verbose=False, cpu_only=False):
    path = Path(input_path)
    if path.is_dir():
        wheels, raw = get_wheels_from_dir(path)
        dump_sdist(path)
    elif path.suffix == ".zip":
        wheels, raw = get_wheels_from_zip(path)
    else:
        print(f"Error: {input_path} is not a directory or a .zip file")
        return 1

    names = list(wheels.keys())

    if len(names) < 2:
        print("Need at least 2 wheels to compare")
        return 1

    reference_name = names[0]
    reference = wheels[reference_name]

    print(f"Reference: {reference_name}\n")

    if verbose:
        dump_contents(reference_name, raw[reference_name])
        print()

    failed = False
    for name in names[1:]:
        files = wheels[name]
        only_in_ref = reference - files
        only_in_this = files - reference

        if not only_in_ref and not only_in_this:
            print(f"✅ {name}: identical file list")
        else:
            failed = True
            print(f"❌ {name}:")
            for f in sorted(only_in_ref):
                print(f"   only in reference: {f}")
            for f in sorted(only_in_this):
                print(f"   only in this:      {f}")
            if verbose:
                dump_contents(name, raw[name])
        print()

    # Bundled vendored libraries per wheel (license audit surface).
    print("Bundled libraries (audit licensing before publishing):")
    for name in names:
        bundled = sorted(
            f for f in raw[name]
            if any(fnmatch(f, g) for g in BUNDLED_LIB_GLOBS)
        )
        if not bundled:
            print(f"  {name}: (none)")
            continue
        print(f"  {name}:")
        for f in bundled:
            print(f"    {f}")
    print()

    # Verify CUDA wheels contain both CUDA 12 and CUDA 13 modules.
    # Skipped for CPU-only builds (ci.yaml), which ship no CUDA modules.
    CUDA_PLATFORMS = ("manylinux_*_x86_64", "win_amd64")
    if cpu_only:
        print("CUDA module check: skipped (--cpu-only)\n")
        return 1 if failed else 0
    print("CUDA module check:")
    for name in names:
        if not any(fnmatch(name, f"*{p}*") for p in CUDA_PLATFORMS):
            print(f"  ⏭️  {name}: non-CUDA platform, skipped")
            continue
        all_files = raw[name]
        has_cuda12 = any(fnmatch(f, "stablebear/_sb_cuda12*") for f in all_files)
        has_cuda13 = any(fnmatch(f, "stablebear/_sb_cuda13*") for f in all_files)
        if has_cuda12 and has_cuda13:
            print(f"  ✅ {name}: contains CUDA 12 and CUDA 13 modules")
        else:
            failed = True
            missing = []
            if not has_cuda12:
                missing.append("CUDA 12")
            if not has_cuda13:
                missing.append("CUDA 13")
            print(f"  ❌ {name}: missing {', '.join(missing)} module!")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    verbose = "--verbose" in args or "-v" in args
    cpu_only = "--cpu-only" in args
    paths = [a for a in args if not a.startswith("-")]

    if len(paths) != 1:
        print(
            "Usage: python compare_wheels.py [-v|--verbose] [--cpu-only] "
            "<wheels.zip|wheels_dir/>"
        )
        sys.exit(1)

    sys.exit(main(paths[0], verbose=verbose, cpu_only=cpu_only))
