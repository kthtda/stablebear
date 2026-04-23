#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Backend selector: loads a versioned _mpcf_cudaXX module if an NVIDIA GPU is detected,
otherwise falls back to _mpcf_cpu."""

import ctypes
import ctypes.util
import importlib
import importlib.util
import os
import pathlib
import pkgutil
import platform
import re
import sys

from .gpu import has_nvidia_gpu as _has_nvidia_gpu

# The compiled backend is built for x86-64-v3 by default (AVX2, FMA, BMI1/2,
# F16C). Validate the running CPU before attempting to import it, otherwise
# dlopen() would fail with SIGILL and no useful error message.
_REQUIRED_X86_64_FEATURES = ("avx2", "fma")


def _detect_x86_cpu_flags():
    """Return a lowercased set of feature flag names for the current x86-64 CPU.

    Returns an empty set when detection is not possible (unknown OS, restricted
    sandbox, etc.); callers treat that as "skip the check" rather than fail.
    """
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith(("flags", "Features")):
                        return {w.lower() for w in line.split(":", 1)[1].split()}
        except OSError:
            pass
        return set()

    if sys.platform == "darwin":
        import subprocess

        flags = set()
        for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
                )
                flags.update(w.lower() for w in out.split())
            except (OSError, subprocess.SubprocessError):
                pass
        return flags

    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            # Windows processor feature constants.
            PF_SSE4_2_INSTRUCTIONS_AVAILABLE = 38
            PF_AVX_INSTRUCTIONS_AVAILABLE = 39
            PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
            flags = set()
            if kernel32.IsProcessorFeaturePresent(PF_SSE4_2_INSTRUCTIONS_AVAILABLE):
                flags.add("sse4_2")
            if kernel32.IsProcessorFeaturePresent(PF_AVX_INSTRUCTIONS_AVAILABLE):
                flags.add("avx")
            if kernel32.IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE):
                # Any x86 CPU with AVX2 in the consumer market also has FMA3.
                flags.update(("avx2", "fma"))
            return flags
        except OSError:
            return set()

    return set()


def _check_x86_64_baseline():
    """Raise ImportError with a clear message if the CPU lacks the required
    x86-64-v3 extensions the extension modules were compiled against.

    On non-x86-64 hosts this is a no-op. Set MPCF_SKIP_CPU_CHECK=1 to bypass
    (useful for debugging detection problems; the extension will still SIGILL
    on a genuinely unsupported CPU)."""
    if os.environ.get("MPCF_SKIP_CPU_CHECK", "0") != "0":
        return

    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64", "x64"):
        return

    flags = _detect_x86_cpu_flags()
    if not flags:
        # Detection failed; don't block loading on a false negative.
        return

    missing = [f for f in _REQUIRED_X86_64_FEATURES if f not in flags]
    if not missing:
        return

    raise ImportError(
        "masspcf was built for x86-64-v3 CPUs (Haswell / Excavator / Zen 1+, "
        "2013+), but this CPU is missing required instruction set "
        f"extensions: {', '.join(missing)}. "
        "To run on this CPU, rebuild from source with a lower baseline, e.g.:\n"
        "    pip install --no-binary=masspcf masspcf "
        '--config-settings=cmake.args="-DMPCF_X86_64_LEVEL=v2"\n'
        "Set MPCF_SKIP_CPU_CHECK=1 to bypass this check (not recommended)."
    )


def _preload_cudart():
    """Preload libcudart from a pip-installed cuda-toolkit (nvidia-* packages),
    or fall back to a system-installed libcudart."""
    import sys as _sys

    _rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)
    _is_windows = _sys.platform == "win32"
    _glob = "cudart64_*.dll" if _is_windows else "libcudart.so*"
    _skip_suffixes = {".lib"} if _is_windows else {".a"}  # skip static libs

    loaded = False
    nvidia_spec = importlib.util.find_spec("nvidia")
    if nvidia_spec is not None and nvidia_spec.submodule_search_locations:
        for nvidia_root in nvidia_spec.submodule_search_locations:
            for lib in sorted(pathlib.Path(nvidia_root).rglob(_glob)):
                if lib.suffix not in _skip_suffixes:
                    ctypes.CDLL(str(lib), mode=_rtld_global)
                    loaded = True
    if not loaded:
        sys_lib = ctypes.util.find_library("cudart")
        if sys_lib:
            ctypes.CDLL(sys_lib, mode=_rtld_global)


def _find_cuda_backend_name():
    package_dir = os.path.dirname(__file__)
    candidates = []
    for mod in pkgutil.iter_modules([package_dir]):
        name = mod.name
        if not name.startswith("_mpcf_cuda"):
            continue
        match = re.fullmatch(r"_mpcf_cuda(\d+)?", name)
        if not match:
            continue
        version = int(match.group(1) or 0)
        candidates.append((version, name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


_check_x86_64_baseline()

_backend = None

if os.environ.get("MPCF_FORCE_CPU", "0") == "0" and _has_nvidia_gpu():
    try:
        _preload_cudart()
        _cuda_backend = _find_cuda_backend_name()
        if _cuda_backend is None:
            raise ImportError("No _mpcf_cuda* backend found")
        _backend = importlib.import_module(f".{_cuda_backend}", package="masspcf")
    except Exception as _e:
        import warnings

        warnings.warn(
            f"Failed to load CUDA backend ({_e}); falling back to CPU. "
            f"Make sure CUDA is installed (system) or via pip install cuda-toolkit[cudart].",
            RuntimeWarning,
            stacklevel=2,
        )

if _backend is None:
    _backend = importlib.import_module("._mpcf_cpu", package="masspcf")

# Populate this module's namespace with everything from the backend
_this = sys.modules[__name__]
for _attr in dir(_backend):
    setattr(_this, _attr, getattr(_backend, _attr))

# Make cpp.persistence work by aliasing the submodule
if hasattr(_backend, "persistence"):
    persistence = _backend.persistence
