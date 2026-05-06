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

"""Diagnostic helpers for native extension load failures.

When a .pyd/.so import fails Python typically reports only "DLL load failed:
the specified module could not be found", which is unhelpful. This module
prints environment, package layout, the .pyd's PE import table, and the
result of attempting to resolve each imported DLL by name.

Triggered automatically on backend ImportError and forced on whenever
MPCF_DEBUG_LOAD=1.
"""

import os
import pathlib
import struct
import sys
import traceback

_BANNER = "=" * 72


def _w(msg=""):
    sys.stderr.write(msg + "\n")


def _safe(label, fn):
    try:
        fn()
    except Exception as e:
        _w(f"  [diagnostic '{label}' failed: {type(e).__name__}: {e}]")


def _read_pe_imports(path):
    """Parse a PE file and return the list of imported DLL names.

    Minimal parser - just enough for the import table. Handles PE32 and PE32+.
    """
    with open(path, "rb") as f:
        data = f.read()

    if data[:2] != b"MZ":
        raise ValueError("not a PE file (missing MZ)")
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_off:pe_off + 4] != b"PE\x00\x00":
        raise ValueError("not a PE file (missing PE signature)")

    coff = pe_off + 4
    _, num_sections, _, _, _, opt_size, _ = struct.unpack_from(
        "<HHIIIHH", data, coff
    )
    opt_off = coff + 20

    magic = struct.unpack_from("<H", data, opt_off)[0]
    if magic == 0x10B:
        dd_off = opt_off + 96
    elif magic == 0x20B:
        dd_off = opt_off + 112
    else:
        raise ValueError(f"unknown optional header magic 0x{magic:x}")

    import_rva, _ = struct.unpack_from("<II", data, dd_off + 8)
    if import_rva == 0:
        return []

    sections_off = opt_off + opt_size
    sections = []
    for i in range(num_sections):
        h = sections_off + i * 40
        v_size, v_addr, r_size, r_off = struct.unpack_from("<IIII", data, h + 8)
        sections.append((v_addr, v_size, r_off, r_size))

    def rva_to_off(rva):
        for v_addr, v_size, r_off, r_size in sections:
            if v_addr <= rva < v_addr + max(v_size, r_size):
                return r_off + (rva - v_addr)
        raise ValueError(f"RVA 0x{rva:x} not in any section")

    base = rva_to_off(import_rva)
    names = []
    for i in range(4096):
        d = base + i * 20
        ilt, _, _, name_rva, iat = struct.unpack_from("<IIIII", data, d)
        if ilt == 0 and name_rva == 0 and iat == 0:
            break
        try:
            n_off = rva_to_off(name_rva)
            end = data.index(b"\x00", n_off)
            names.append(data[n_off:end].decode("ascii", errors="replace"))
        except Exception:
            names.append("<unreadable>")
    return names


def _try_load_dll(name):
    import ctypes
    try:
        h = ctypes.WinDLL(name)
        return f"OK ({getattr(h, '_name', name)!r})"
    except OSError as e:
        return f"FAIL winerror={getattr(e, 'winerror', None)} ({e})"


def _print_environment():
    _w(f"  python: {sys.version}")
    _w(f"  executable: {sys.executable}")
    _w(f"  platform: {sys.platform}")
    if sys.platform == "win32":
        try:
            wv = sys.getwindowsversion()
            _w(f"  windows version: {wv.major}.{wv.minor}.{wv.build} "
               f"(platform={wv.platform}, service_pack={wv.service_pack!r})")
        except Exception as e:
            _w(f"  windows version: <unavailable: {e}>")
    _w(f"  arch: {getattr(sys, 'maxsize', 0).bit_length() + 1}-bit "
       f"(maxsize={sys.maxsize})")
    _w(f"  cwd: {os.getcwd()}")


def _print_package_layout(pkg_dir):
    _w(f"  package dir: {pkg_dir}")
    try:
        entries = sorted(p.name for p in pkg_dir.iterdir())
        _w(f"  package entries ({len(entries)}):")
        for e in entries:
            _w(f"    {e}")
    except Exception as e:
        _w(f"  [could not list package dir: {e}]")


def _print_libs_dir(pkg_dir):
    libs = pkg_dir.parent / "masspcf.libs"
    _w(f"  masspcf.libs: {libs} (exists={libs.is_dir()})")
    if libs.is_dir():
        try:
            for f in sorted(libs.iterdir()):
                size = f.stat().st_size if f.is_file() else "-"
                _w(f"    {f.name}  ({size} bytes)")
        except Exception as e:
            _w(f"    [list failed: {e}]")
    else:
        _w("    [delvewheel-bundled DLLs missing - wheel may be old or "
           "corrupt; reinstall with: pip install --force-reinstall masspcf]")


def _print_dll_directories():
    if sys.platform != "win32":
        return
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    _w(f"  PATH ({len(parts)} entries):")
    for p in parts:
        marker = "" if (p and os.path.isdir(p)) else "  [missing]"
        _w(f"    {p}{marker}")


def _print_pe_imports(backend_path):
    if sys.platform != "win32":
        return
    if not backend_path.is_file():
        _w(f"  [backend file not found: {backend_path}]")
        return
    try:
        imports = _read_pe_imports(backend_path)
    except Exception as e:
        _w(f"  [PE parse failed: {type(e).__name__}: {e}]")
        return
    _w(f"  PE imports of {backend_path.name} ({len(imports)}):")
    for name in imports:
        _w(f"    {name}: {_try_load_dll(name)}")


def _try_winsxs_load(backend_path):
    """Attempt to load the .pyd via WinDLL to surface a Windows error code."""
    if sys.platform != "win32":
        return
    if not backend_path.is_file():
        return
    import ctypes
    try:
        ctypes.WinDLL(str(backend_path))
        _w(f"  WinDLL({backend_path.name}): loaded successfully "
           "(import path failure may be unrelated to DLL resolution)")
    except OSError as e:
        _w(f"  WinDLL({backend_path.name}): "
           f"winerror={getattr(e, 'winerror', None)} "
           f"strerror={getattr(e, 'strerror', None)!r}")
        _w(f"    raw: {e}")


def _print_delvewheel_patch_status(pkg_dir):
    """Check whether delvewheel injected its DLL-search-path patch."""
    if sys.platform != "win32":
        return
    init_py = pkg_dir / "__init__.py"
    patches = sorted(pkg_dir.glob("_delvewheel_*.py"))
    _w(f"  delvewheel patch files: {[p.name for p in patches] or 'NONE'}")
    if init_py.is_file():
        try:
            head = init_py.read_text(encoding="utf-8", errors="replace")[:2000]
            patched = ("delvewheel" in head) or ("add_dll_directory" in head)
            _w(f"  __init__.py mentions delvewheel/add_dll_directory: {patched}")
        except Exception as e:
            _w(f"  [could not read __init__.py: {e}]")


def diagnose(backend_name, exc=None):
    """Print extensive diagnostics about a backend load attempt.

    backend_name: e.g. "_mpcf_cpu" or "_mpcf_cuda12"
    exc: the ImportError that triggered diagnostics, if any
    """
    pkg_dir = pathlib.Path(__file__).resolve().parent
    backend_path = pkg_dir / f"{backend_name}.pyd"
    if not backend_path.exists():
        # fall back to .so for non-Windows
        for suffix in (".so", ".dylib"):
            cand = list(pkg_dir.glob(f"{backend_name}*{suffix}"))
            if cand:
                backend_path = cand[0]
                break

    _w(_BANNER)
    _w(f"masspcf load diagnostics for {backend_name}")
    if exc is not None:
        _w(f"  exception: {type(exc).__name__}: {exc}")
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        for line in tb.rstrip().splitlines():
            _w(f"    {line}")
    _w("")
    _safe("environment", _print_environment)
    _w("")
    _safe("package layout", lambda: _print_package_layout(pkg_dir))
    _w("")
    _safe("masspcf.libs", lambda: _print_libs_dir(pkg_dir))
    _w("")
    _safe("delvewheel patch", lambda: _print_delvewheel_patch_status(pkg_dir))
    _w("")
    _safe("backend file",
          lambda: _w(f"  backend path: {backend_path} "
                     f"(exists={backend_path.exists()}, "
                     f"size={backend_path.stat().st_size if backend_path.exists() else 0})"))
    _w("")
    _safe("WinDLL probe", lambda: _try_winsxs_load(backend_path))
    _w("")
    _safe("PE imports", lambda: _print_pe_imports(backend_path))
    _w("")
    _safe("PATH", _print_dll_directories)
    _w(_BANNER)
    sys.stderr.flush()


def is_debug_enabled():
    return os.environ.get("MPCF_DEBUG_LOAD", "0") not in ("", "0", "false", "False")
