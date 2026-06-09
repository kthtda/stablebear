#    Copyright 2024-2026 Bjorn Wehlin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Detect CUDA-capable NVIDIA GPUs without requiring CUDA libraries.

Uses the C++ _gpu_detect module (direct OS API calls) when available,
falling back to a pure-Python implementation using subprocess.
"""

try:
    from _gpu_detect import detect_nvidia_gpus, has_nvidia_gpu, nvidia_gpu_count
except ImportError:
    import os
    import platform
    import re
    import subprocess

    def _run(cmd):
        """Run a command and return stdout, or None on failure."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout
            return None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    def _detect_linux():
        """Detect NVIDIA GPUs on Linux via sysfs and lspci."""
        gpus = []

        # Try lspci first (works on most Linux systems)
        output = _run(["lspci"])
        if output:
            for line in output.splitlines():
                if "NVIDIA" in line and ("VGA" in line or "3D" in line):
                    match = re.search(r"NVIDIA\s+(.*)", line)
                    name = match.group(1).strip() if match else "NVIDIA GPU"
                    gpus.append({"name": name})

        if gpus:
            return gpus

        # Fallback: check sysfs for NVIDIA vendor ID (0x10de)
        drm_path = "/sys/class/drm"
        if os.path.isdir(drm_path):
            for entry in os.listdir(drm_path):
                vendor_file = os.path.join(drm_path, entry, "device", "vendor")
                if os.path.isfile(vendor_file):
                    try:
                        with open(vendor_file) as f:
                            vendor = f.read().strip()
                        if vendor == "0x10de":
                            gpus.append({"name": "NVIDIA GPU"})
                    except OSError:
                        pass

        return gpus

    def _detect_windows():
        """Detect NVIDIA GPUs on Windows via WMIC/PowerShell."""
        gpus = []

        output = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Where-Object { $_.Name -match 'NVIDIA' } | "
                "Select-Object -ExpandProperty Name",
            ]
        )
        if output:
            for line in output.strip().splitlines():
                line = line.strip()
                if line:
                    gpus.append({"name": line})
            return gpus

        output = _run(["wmic", "path", "win32_videocontroller", "get", "name"])
        if output:
            for line in output.strip().splitlines()[1:]:
                line = line.strip()
                if "NVIDIA" in line.upper():
                    gpus.append({"name": line})

        return gpus

    def _detect_macos():
        gpus = []

        return gpus

    def detect_nvidia_gpus():
        """Detect NVIDIA GPUs present on the system.

        Uses OS-level tools (lspci, sysfs, PowerShell, system_profiler).
        Does not require CUDA or any NVIDIA drivers/libraries.

        Returns
        -------
        list[dict]
            A list of dicts, each with a ``"name"`` key describing the GPU.
            An empty list means no NVIDIA GPUs were found.
        """
        system = platform.system()
        if system == "Linux":
            return _detect_linux()
        elif system == "Windows":
            return _detect_windows()
        elif system == "Darwin":
            return _detect_macos()
        return []

    def has_nvidia_gpu():
        """Check whether the system has at least one NVIDIA GPU.

        Returns
        -------
        bool
            ``True`` if at least one NVIDIA GPU is detected.
        """
        return len(detect_nvidia_gpus()) > 0

    def nvidia_gpu_count():
        """Return the number of NVIDIA GPUs detected.

        Returns
        -------
        int
            Number of NVIDIA GPUs found.
        """
        return len(detect_nvidia_gpus())
