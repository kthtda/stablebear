"""Detect CUDA-capable NVIDIA GPUs without requiring CUDA libraries.

Uses the C++ _gpu_detect module (direct OS API calls) when available,
falling back to a pure-Python implementation using subprocess.
"""

try:
    # The extension is installed inside the package (site-packages/stablebear/),
    # so it must be imported relatively; a top-level import never resolves.
    from ._gpu_detect import detect_nvidia_gpus, has_nvidia_gpu, nvidia_gpu_count
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

        # Fallback: check sysfs for NVIDIA vendor ID (0x10de). Each physical
        # GPU exposes several drm nodes (cardN, renderDN, connectors); count
        # only the cardN nodes to avoid double-counting.
        drm_path = "/sys/class/drm"
        if os.path.isdir(drm_path):
            for entry in os.listdir(drm_path):
                if not re.fullmatch(r"card\d+", entry):
                    continue
                vendor_file = os.path.join(drm_path, entry, "device", "vendor")
                if os.path.isfile(vendor_file):
                    try:
                        with open(vendor_file) as f:
                            vendor = f.read().strip()
                        if vendor == "0x10de":
                            gpus.append({"name": "NVIDIA GPU"})
                    except OSError:
                        pass

        if gpus:
            return gpus

        # Fallback: the driver's procfs (one directory per GPU). Covers slim
        # containers without lspci or drm entries.
        nvidia_proc = "/proc/driver/nvidia/gpus"
        if os.path.isdir(nvidia_proc):
            for entry in sorted(os.listdir(nvidia_proc)):
                name = "NVIDIA GPU [" + entry + "]"
                info_file = os.path.join(nvidia_proc, entry, "information")
                try:
                    with open(info_file) as f:
                        for line in f:
                            if line.startswith("Model:"):
                                name = line[len("Model:"):].strip()
                                break
                except OSError:
                    pass
                gpus.append({"name": name})

        if gpus:
            return gpus

        # Fallback: WSL2 exposes the GPU through /dev/dxg with no NVIDIA PCI
        # drm nodes and no useful lspci. /dev/dxg alone could be any vendor,
        # so require the WSL NVIDIA driver libraries as well.
        if os.path.exists("/dev/dxg"):
            wsl_lib = "/usr/lib/wsl/lib"
            if any(
                os.path.exists(os.path.join(wsl_lib, lib))
                for lib in ("libcuda.so.1", "libcuda.so", "nvidia-smi")
            ):
                gpus.append({"name": "NVIDIA GPU (WSL2)"})

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
