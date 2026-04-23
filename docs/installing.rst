==========
Installing
==========

Installing using pip
=====================

The easiest way to get started with `masspcf` is using `pip`::

    pip install masspcf

CPU requirements
=================

Pre-built x86-64 wheels target the `x86-64-v3 <https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels>`_ microarchitecture level: AVX, AVX2, FMA, BMI1/2, F16C, LZCNT, and MOVBE. This covers essentially every x86-64 laptop and workstation from 2013 onwards (Haswell / Excavator / Zen 1 and newer). The main exception is pre-2022 Atom-derived chips such as Celeron N, Pentium Silver, and Jasper Lake, which lack AVX2.

On import, `masspcf` verifies that the CPU supports the required instruction set and raises a clear ``ImportError`` if it does not — rebuild from source as described below to run on such hardware.

arm64 / aarch64 wheels (Apple Silicon, Linux aarch64) have no equivalent baseline issue.

Building from source
=====================

Building from source is useful in two situations:

1. **Your CPU is older than x86-64-v3** and the pre-built wheel refuses to load.
2. **You want maximum performance.** A source build defaults to ``-march=native`` (or a CPUID-probed ``/arch:`` flag on MSVC), so the resulting extension can use every instruction set your CPU supports — including AVX-512 where available — rather than stopping at the v3 baseline shipped in the wheel. Speedups are workload-dependent but often measurable on AVX-512-capable and newer AMD/Intel parts.

To install the latest released version from source::

    pip install --no-binary=masspcf masspcf

To build a specific branch or tag from a Git checkout::

    git clone https://github.com/bwehlin/masspcf.git
    cd masspcf

    # optional: select a specific tagged version to build
    git checkout tags/v0.4.1

    pip install .

The CPU target can be pinned explicitly at configure time with ``-DMPCF_X86_64_LEVEL=v1|v2|v3|v4|native`` if you need to build for a specific microarchitecture level rather than the host CPU.

On Windows and Linux, `masspcf` is built with CUDA enabled by default. To override this behavior, please set the environment variable ``BUILD_WITH_CUDA=0``. To force a CUDA build, set ``BUILD_WITH_CUDA=1``.

.. note:: On OSX, ``masspcf`` is CPU-only.
