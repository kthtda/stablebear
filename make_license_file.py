#!/usr/bin/env python3
#
# Copyright 2024-2026 Bjorn Wehlin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate LICENSE.rst from the project license and bundled 3rd-party licenses."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

APACHE2_SPDX = "Apache-2.0"

LICENSE_INPUT = REPO_ROOT / "mpcf.LICENSE.input"
_SEP = "=" * 20  # minimum prefix for separator lines in mpcf.LICENSE.input


def _read_apache2_text() -> str:
    """Read the Apache 2.0 license text from mpcf.LICENSE.input."""
    lines = LICENSE_INPUT.read_text().splitlines()
    sep_indices = [i for i, l in enumerate(lines) if l.startswith(_SEP)]
    if len(sep_indices) < 2:
        raise RuntimeError("Could not find license text delimiters in mpcf.LICENSE.input")
    return "\n".join(lines[sep_indices[0] + 1 : sep_indices[1]]).strip()

# 3rd-party packages bundled in the repository.
# Each entry: (display name, license file path relative to repo root, URL or None)
THIRD_PARTY = [
    ("Taskflow", "3rd/taskflow/LICENSE", "https://github.com/taskflow/taskflow"),
    ("pybind11", "3rd/pybind11/LICENSE", "https://github.com/pybind/pybind11"),
    ("GoogleTest", "3rd/googletest/LICENSE", "https://github.com/google/googletest"),
    ("Ripser", "3rd/ripser/LICENSE", "https://github.com/Ripser/ripser"),
    ("xoroshiro128++", "3rd/xoroshiro/LICENSE", "https://prng.di.unimi.it/xoroshiro128plusplus.c"),
    ("splitmix64", "3rd/splitmix64/LICENSE", "https://prng.di.unimi.it/splitmix64.c"),
]

# Platform-specific notices (not always bundled, but relevant depending on build).
# Each entry: (display name, license file path relative to repo root, URL, description or None)
PLATFORM_NOTICES = [
    (
        "GCC runtime libraries",
        "3rd/gcc-runtime/LICENSE",
        "https://www.gnu.org/licenses/gcc-exception-3.1.html",
        "On platforms where masspcf is compiled with GCC, the binary may ship "
        "with GCC runtime libraries covered by the GCC Runtime Library Exception 3.1.",
    ),
    (
        "Microsoft Visual C++ runtime libraries",
        "3rd/msvc-runtime/LICENSE",
        "https://visualstudio.microsoft.com/license-terms/",
        "On Windows, the binary wheels ship with Microsoft Visual C++ "
        "Redistributable runtime libraries, redistributed under the Microsoft "
        "Visual Studio license terms.",
    ),
    (
        "NVIDIA CUDA",
        "3rd/cuda/LICENSE",
        "https://docs.nvidia.com/cuda/eula/index.html",
        None,
    ),
]


def rst_heading(text: str, char: str) -> str:
    line = char * len(text)
    return f"{text}\n{line}"


def rst_section(text: str) -> str:
    return rst_heading(text, "=")


def rst_subsection(text: str) -> str:
    return rst_heading(text, "-")


def rst_subsubsection(text: str) -> str:
    return rst_heading(text, "~")


def rst_paragraph(text: str) -> str:
    return rst_heading(text, "^")


def indent(text: str, prefix: str = "   ") -> str:
    return "\n".join(prefix + line if line.strip() else "" for line in text.splitlines())


def build_rst() -> str:
    parts: list[str] = []

    parts.append(".. highlight:: none")
    parts.append("")
    parts.append(rst_section("License"))
    parts.append("")
    parts.append(
        "masspcf is Copyright 2024-2026 Bjorn H. Wehlin and is distributed "
        "under the Apache License, Version 2.0."
    )
    parts.append("")
    parts.append(
        "In addition, the masspcf repository and source distributions bundle "
        "several third-party libraries under compatible licenses, listed below."
    )

    # -- masspcf license --
    parts.append("")
    parts.append("")
    parts.append(rst_subsection("masspcf license (Apache-2.0)"))
    parts.append("")
    parts.append("::")
    parts.append("")
    parts.append(indent(_read_apache2_text()))

    # -- 3rd-party licenses --
    parts.append("")
    parts.append("")
    parts.append(rst_subsection("Third-party licenses"))

    for name, license_path, url in THIRD_PARTY:
        license_text = (REPO_ROOT / license_path).read_text().strip()
        parts.append("")
        if url:
            parts.append(rst_subsubsection(f"{name} ({license_path})"))
            parts.append("")
            parts.append(f"Homepage: {url}")
        else:
            parts.append(rst_subsubsection(f"{name} ({license_path})"))
        parts.append("")
        parts.append("::")
        parts.append("")
        parts.append(indent(license_text))

    # -- Platform-specific notices (under 3rd-party) --
    parts.append("")
    parts.append("")
    parts.append(rst_subsubsection("Platform-specific notices"))

    for name, license_path, url, description in PLATFORM_NOTICES:
        license_text = (REPO_ROOT / license_path).read_text().strip()
        parts.append("")
        parts.append(rst_paragraph(f"{name} ({license_path})"))
        parts.append("")
        if description:
            parts.append(f"{description} (see {url}).")
            parts.append("")
        parts.append("::")
        parts.append("")
        parts.append(indent(license_text))

    parts.append("")
    return "\n".join(parts)


def main() -> None:
    content = build_rst()
    out = REPO_ROOT / "LICENSE.rst"
    out.write_text(content)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
