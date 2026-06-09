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

"""Preload libcudart so that the CUDA extension module can be imported."""

import ctypes
import ctypes.util
import importlib.util
import os
import pathlib
import sys


def _add_delvewheel_libs_dir():
    """Add the sibling stablebear.libs directory to the DLL search path on Windows.

    delvewheel mangles bundled DLL filenames (e.g. msvcp140.dll ->
    msvcp140-<hash>.dll) and patches stablebear/__init__.py to call
    os.add_dll_directory at package import time. Callers that load
    _sb_cuda*.pyd directly via importlib (bypassing the package init)
    must replicate that step or the loader cannot resolve the mangled CRT.
    """
    if sys.platform != "win32":
        return
    libs_dir = pathlib.Path(__file__).resolve().parent.parent / "stablebear.libs"
    if libs_dir.is_dir():
        os.add_dll_directory(str(libs_dir))


def _preload_cudart():
    """Preload libcudart from a pip-installed cuda-toolkit (nvidia-* packages),
    or fall back to a system-installed libcudart."""
    _add_delvewheel_libs_dir()

    _rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)
    _is_windows = sys.platform == "win32"
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
