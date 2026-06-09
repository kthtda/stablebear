==================
GPU Acceleration
==================

stablebear can use NVIDIA GPUs to accelerate computationally intensive operations, particularly pairwise distance matrix computations. This guide explains how GPU support works and how to control it.


Supported platforms
===================================

CUDA support is provided on Linux and Windows only. On macOS, execution is CPU-only. If you have an NVIDIA GPU but no system-wide CUDA installation, you can install the CUDA runtime via pip::

   pip install cuda-toolkit[cudart]

stablebear will fall back to CPU execution with a warning message if a CUDA-capable GPU is detected but the CUDA runtime library cannot be loaded.

Automatic CPU/GPU dispatch
===========================

By default, stablebear decides at runtime whether to run each operation on the CPU or GPU. The decision is based on problem size: small computations run on the CPU (where launch overhead would dominate), while large computations are dispatched to the GPU.

For pairwise distance matrix computations (:py:func:`~stablebear.pdist`), the default threshold is **500 PCFs** -- below this, the CPU is used; above it, the GPU is used (if available).

You do not need to change any code to benefit from GPU acceleration. The same code runs correctly on machines with or without GPUs.


Controlling execution
======================

The :py:mod:`stablebear.system` module provides options for controlling how stablebear uses hardware resources. These settings are per-session and must be set each time you start a new Python process.

.. note::

   Most users should not need to change any of these settings. They are provided for advanced use cases such as benchmarking, debugging, or multi-user environments.

Forcing CPU execution
---------------------

To disable GPU acceleration entirely::

   import stablebear as sb

   sb.system.force_cpu(True)

To re-enable GPU dispatch::

   sb.system.force_cpu(False)

This is useful for benchmarking CPU vs. GPU performance, or when the GPU is being used by another process.

Adjusting the CUDA threshold
------------------------------

To change the problem-size threshold at which computations move from CPU to GPU::

   # Use GPU for any matrix computation with >= 100 PCFs
   sb.system.set_cuda_threshold(100)

   # Use GPU for all matrix computations (threshold of 1)
   sb.system.set_cuda_threshold(1)

Limiting CPU threads
---------------------

By default, stablebear uses all available hardware threads. To restrict this::

   # Use at most 4 CPU threads
   sb.system.limit_cpus(4)

This can be useful in multi-user environments or when running multiple processes in parallel.

Limiting GPU count
-------------------

If you have multiple GPUs but want stablebear to use only some of them::

   # Use at most 1 GPU
   sb.system.limit_gpus(1)

Setting CUDA block size
------------------------

For expert users, the CUDA block dimensions for matrix computations can be tuned::

   sb.system.set_block_size(16, 16)

The optimal block size depends on your GPU architecture and problem characteristics. The default values work well for most cases.

Verbose device logging
-----------------------

To see which device (CPU or GPU) is being used for each operation::

   sb.system.set_device_verbose(True)

   # Now operations will print whether they run on CPU or GPU
   D = sb.pdist(X)
   # e.g. "Running on GPU"

Set to ``False`` to disable (the default).

CPU and GPU execution
=====================

stablebear automatically detects available NVIDIA GPUs and uses them for computations when beneficial. The library decides at runtime whether to execute a given operation on the CPU or GPU based on problem size.

You can query GPU availability::

   from stablebear import gpu

   gpu.has_nvidia_gpu()       # True/False
   gpu.nvidia_gpu_count()     # Number of GPUs
   gpu.detect_nvidia_gpus()   # Detailed GPU info

Performance considerations
===========================

When to expect GPU speedups
----------------------------

GPU acceleration provides the largest benefit for computations with many PCFs -- the number of pairwise integrals grows as :math:`n(n-1)/2`, making this inherently parallel.

For small problems (fewer than a few hundred PCFs), CPU execution is typically faster due to GPU launch overhead.

Supported GPU architectures
----------------------------

Pre-built CUDA 12 wheels include native SASS for Maxwell through Hopper, plus PTX for forward compatibility:

- **Maxwell** (sm_50) -- e.g. GTX 980, Tesla M40
- **Turing** (sm_75) -- e.g. RTX 2080, Tesla T4
- **Ampere** (sm_80, sm_86) -- e.g. A100, RTX 3090
- **Hopper** (sm_90) -- e.g. H100 (also emits PTX for forward compat)

GPUs with architectures in between or newer (e.g. Pascal GTX 1080, Ada Lovelace RTX 4090, Blackwell B200) are supported via binary compatibility or PTX JIT compilation.

Pre-built CUDA 13 wheels additionally include native SASS for:

- **Blackwell** (sm_100, sm_120) -- e.g. B200, RTX 5090

32-bit vs. 64-bit
-------------------

Single-precision (32-bit) computation is significantly faster on most GPUs, where single-precision throughput is often 4x or more compared to double-precision. Use ``pcf32`` (the default) unless you specifically need 64-bit numerical precision.

.. note::

   stablebear is an independent open-source project and is not affiliated with, endorsed by, or sponsored by NVIDIA Corporation. NVIDIA, CUDA, GeForce, Tesla, and other product names are trademarks of NVIDIA Corporation.
