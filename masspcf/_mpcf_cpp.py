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
import re
import sys

from . import _debug_load as _dbg
from .gpu import has_nvidia_gpu as _has_nvidia_gpu


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


_backend = None
_cuda_backend = None

if _dbg.is_debug_enabled():
    _dbg.diagnose("_mpcf_cpu")

if os.environ.get("MPCF_FORCE_CPU", "0") == "0" and _has_nvidia_gpu():
    try:
        _preload_cudart()
        _cuda_backend = _find_cuda_backend_name()
        if _cuda_backend is None:
            raise ImportError("No _mpcf_cuda* backend found")
        _backend = importlib.import_module(f".{_cuda_backend}", package="masspcf")
    except Exception as _e:
        _dbg.diagnose(_cuda_backend or "_mpcf_cudaXX", _e)
        import warnings

        warnings.warn(
            f"Failed to load CUDA backend ({_e}); falling back to CPU. "
            f"Make sure CUDA is installed (system) or via pip install cuda-toolkit[cudart].",
            RuntimeWarning,
            stacklevel=2,
        )

if _backend is None:
    try:
        _backend = importlib.import_module("._mpcf_cpu", package="masspcf")
    except ImportError as _e:
        _dbg.diagnose("_mpcf_cpu", _e)
        raise

# Populate this module's namespace with everything from the backend
_this = sys.modules[__name__]
for _attr in dir(_backend):
    setattr(_this, _attr, getattr(_backend, _attr))

# Make cpp.persistence work by aliasing the submodule
if hasattr(_backend, "persistence"):
    persistence = _backend.persistence
