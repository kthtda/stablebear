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

import subprocess
import sys

import pytest

_SUBPROC = r"""
import ctypes.util
import importlib.util
import pathlib
import sys

def _find_cuda_extension(stablebear_dir: pathlib.Path):
    candidates = list(stablebear_dir.glob("_sb_cuda*.pyd")) + list(stablebear_dir.glob("_sb_cuda*.so"))
    return candidates[0] if candidates else None

def _module_name_from_path(path: pathlib.Path) -> str:
    return path.name.split(".")[0]

spec = importlib.util.find_spec("stablebear")
if spec is None or not spec.origin:
    print("stablebear package not importable", file=sys.stderr)
    sys.exit(1)

stablebear_dir = pathlib.Path(spec.origin).parent
cuda_ext = _find_cuda_extension(stablebear_dir)
if cuda_ext is None:
    print("No CUDA extension module in this wheel", file=sys.stderr)
    sys.exit(10)

_cudart_path = stablebear_dir / "_cudart_preload.py"
if not _cudart_path.exists():
    print("stablebear._cudart_preload.py not found", file=sys.stderr)
    sys.exit(1)
_cudart_spec = importlib.util.spec_from_file_location("_cudart_preload", _cudart_path)
if _cudart_spec is None or _cudart_spec.loader is None:
    print("Failed to load _cudart_preload spec", file=sys.stderr)
    sys.exit(1)
_cudart_mod = importlib.util.module_from_spec(_cudart_spec)
_cudart_spec.loader.exec_module(_cudart_mod)
_preload_cudart = _cudart_mod._preload_cudart

try:
    _preload_cudart()
    mod_name = _module_name_from_path(cuda_ext)
    cuda_spec = importlib.util.spec_from_file_location(mod_name, cuda_ext)
    if cuda_spec is None or cuda_spec.loader is None:
        print("Failed to create spec for CUDA module", file=sys.stderr)
        sys.exit(1)
    cuda_mod = importlib.util.module_from_spec(cuda_spec)
    cuda_spec.loader.exec_module(cuda_mod)
except Exception:
    has_pip_cuda = importlib.util.find_spec("nvidia") is not None
    has_system_cuda = ctypes.util.find_library("cudart") is not None
    if not has_pip_cuda and not has_system_cuda:
        print("No cudart found (pip or system); skipping CUDA load test", file=sys.stderr)
        sys.exit(11)
    raise
"""


def test_cuda_module_loads_with_pip_cudart():
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROC],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode in (10, 11):
        pytest.skip(result.stderr.strip() or "CUDA load test skipped")

    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise AssertionError(details or "CUDA load test failed")
