==============================
CUDA block execution engine
==============================

stablebear's CUDA engine computes pairwise matrix operations (distance matrices, kernel matrices, cross-distance tensors) by partitioning the work into 2D blocks and distributing them across GPUs. The framework is designed to be **function-type agnostic** -- the orchestration logic is generic, while function-specific behavior (data layout, kernels) is provided by a policy class.


File structure
===============

All files live under ``include/sbear/cuda/`` (headers) and ``src/cuda/`` (compiled sources).

**Generic framework** (function-type agnostic):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Purpose
   * - ``cuda_block_executor.cuh``
     - ``CudaBlockPipeline``: double-buffered block execution loop.
   * - ``cuda_block_scheduler.hpp``
     - ``CudaBlockScheduler``: 2D block decomposition for square or rectangular matrices. Pure C++.
   * - ``cuda_result_writer.hpp``
     - Scatter GPU results into ``DistanceMatrix``, ``SymmetricMatrix``, or ``Tensor`` (dense). Pure C++.
   * - ``offset_data_manager.hpp``
     - ``OffsetDataManager``: host-side data manager for variable-length objects. Pure C++.
   * - ``cuda_offset_data_manager.cuh``
     - ``CudaOffsetDataManager``: extends ``OffsetDataManager`` with GPU upload.
   * - ``cuda_device_array.cuh``
     - RAII wrapper for ``cudaMalloc``/``cudaFree``.
   * - ``block_matrix_support.cuh``
     - Shared utilities: grid dimension calculation, GPU memory query.

**PCF-specific:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Purpose
   * - ``cuda_pcf_kernel.cuh``
     - PCF integration kernel (``cuda_pcf_block_integrate``), rectangle iteration device function, ``TriangleSkipMode``, ``PcfBlockKernelParams``.
   * - ``pcf_block_op.cuh``
     - ``PcfBlockOp`` policy class, ``CudaPairwiseIntegrationTask``, ``CudaCrossIntegrationTask``.
   * - ``cuda_pcf_data_manager.cuh``
     - ``CudaPcfDataManager`` type alias and ``init_pcf_data()`` flattener.

**Public API and wiring:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Purpose
   * - ``cuda_matrix_integrate_api.hpp``
     - Factory function declarations for pdist and cdist (pure C++ header, no NVCC).
   * - ``cuda_matrix_integrate.cu``
     - NVCC-compiled factory implementations.


Block decomposition
====================

The ``CudaBlockScheduler`` partitions an ``nRows x nCols`` matrix into 2D blocks. It supports two modes:

- **LowerTriangle** -- for symmetric operations (``pdist``, ``l2_kernel``) where ``nRows == nCols``. Blocks entirely above the diagonal are skipped.
- **Full** -- for cross-distance operations (``cdist``) where every ``(i, j)`` pair is computed. The matrix may be rectangular (``nRows != nCols``).


Algorithm
----------

The scheduler computes a block decomposition in three steps:

**Step 1: Compute block side length.** The block side determines how large each block can be. It is derived from ``maxOutputElements`` (the GPU memory budget, in number of output scalars) and ``nSplitsHint`` (how many blocks we want the budget to be divided into):

.. code-block:: text

   elementsPerBlock = maxOutputElements / nSplitsHint
   blockSide        = floor(sqrt(elementsPerBlock))
   blockSide        = clamp(blockSide, 1, max(nRows, nCols))

The square root targets roughly square blocks, which is a good default for 2D CUDA grids.

**Step 2: Subdivide rows and columns into bands.** The ``subdivide(blockSide, n)`` function chops a range ``[0, n)`` into consecutive bands of length ``blockSide``, with the last band absorbing the remainder:

.. code-block:: text

   subdivide(3, 10)  ->  [0,2], [3,5], [6,8], [9,9]
   subdivide(4, 8)   ->  [0,3], [4,7]
   subdivide(5, 3)   ->  [0,2]

Row bands and column bands are computed independently: ``rowBands = subdivide(blockSide, nRows)``, ``colBands = subdivide(blockSide, nCols)``. This means rectangular matrices are handled naturally -- the row and column band counts can differ.

**Step 3: Form the block grid and filter.** Every ``(rowBand, colBand)`` pair becomes a candidate block. In **Full** mode, all candidates are kept. In **LowerTriangle** mode, a block is discarded if its entire column range lies above the diagonal (i.e., ``colBand.first > rowBand.last``):

.. code-block:: text

   LowerTriangle, 9x9, blockSide=3:

            col 0-2   col 3-5   col 6-8
           +---------+---------+---------+
   row 0-2 |    A    |    -    |    -    |    colBand [3,5].first > rowBand [0,2].last -> skip
           +---------+---------+---------+
   row 3-5 |    B    |    C    |    -    |
           +---------+---------+---------+
   row 6-8 |    D    |    E    |    F    |
           +---------+---------+---------+

   A, C, F = diagonal blocks (straddle the diagonal, partially computed)
   B, D, E = below-diagonal blocks (fully computed)
       -   = above-diagonal blocks (skipped entirely)

Within each diagonal block (A, C, F), the kernel's ``TriangleSkipMode`` skips individual threads where ``i <= j`` (for distance matrices) or ``i < j`` (for kernel matrices).

**Full** example (``cdist``, 2 row functions x 3 column functions, block side length 2):

The matrix is rectangular, so all blocks are computed.

.. code-block:: text

            col 0-1   col 2
           +---------+------+
   row 0-1 |    A    |   B  |
           +---------+------+

   A = 2x2 block, B = 2x1 block

Finally, blocks are sorted by descending work (``rowHeight * colWidth``) for better load balancing when distributing across GPUs.


Memory budget and nSplitsHint
------------------------------

``maxOutputElements`` comes from the available GPU memory (via ``get_max_output_elements`` in ``block_matrix_support.cuh``):

.. code-block:: text

   maxOutputElements = (free GPU RAM * 0.8) / 2 / sizeof(T)

The factor of 2 reserves half of GPU memory for the output buffer and half for function data.

``nSplitsHint`` controls how many blocks the memory budget is split into. A larger hint produces smaller blocks, which matters for two reasons:

1. **Multi-GPU load balancing.** With only one or two large blocks, some GPUs sit idle while others finish. More blocks means work can be distributed more evenly.
2. **Pipeline overlap.** The double-buffered pipeline (see below) overlaps computation with data transfer. Smaller blocks allow the pipeline to start transferring results sooner.

In production, the hint is set to ``nGpus * 32`` -- enough blocks to keep all GPUs busy with a comfortable margin for uneven block sizes. For single-GPU or testing scenarios, ``nSplitsHint = 1`` uses the full memory budget as one block (unless the matrix is larger, in which case ``subdivide`` still produces multiple blocks).


Minimum block side (GPU occupancy floor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A high ``nSplitsHint`` can produce very small blocks, leading to poor GPU occupancy -- each kernel launch has too few threads to keep the SMs busy. To prevent this, ``Config::minBlockSide`` sets a floor on the block side length. The block side is clamped: ``side = max(side, minBlockSide)``, then capped at the matrix dimension as usual.

In production, ``minBlockSide`` is derived from the GPU hardware via ``get_min_block_side(nGpus)`` (in ``block_matrix_support.cuh``), which queries ``cudaDevAttrMultiProcessorCount`` across all devices and targets ~50% max occupancy:

.. code-block:: text

   smCount       = min SM count across GPUs
   targetThreads = smCount * 1024
   minBlockSide  = floor(sqrt(targetThreads))

For example, an A100 with 108 SMs gives ``minBlockSide = floor(sqrt(108 * 1024)) = 332``, ensuring each kernel launch has at least ~110K threads -- enough to keep the GPU busy even for compute-bound PCF integration kernels.

When the floor kicks in, the scheduler produces fewer blocks than ``nSplitsHint`` requested. This is the right tradeoff: slightly uneven multi-GPU distribution is much cheaper than underutilizing all GPUs on every kernel launch.

``minBlockSide`` is optional and defaults to 0 (no floor), which preserves the original behavior for tests and non-GPU use.


Higher-rank tensors
--------------------

For ``cdist(X, Y)`` where X and Y are tensors with rank > 1, the scheduler still operates on a 2D grid. The inputs are flattened: ``nRows = product(X.shape)``, ``nCols = product(Y.shape)``. The output tensor is created with the concatenated shape ``(*X.shape, *Y.shape)`` but its underlying storage is a flat contiguous buffer. Since the 2D row-major indexing ``i * nCols + j`` produces the same layout as the multi-dimensional shape, the scheduler does not need to be aware of batch dimensions.


Triangle skip modes
--------------------

The kernel uses a ``TriangleSkipMode`` enum (defined in ``cuda_pcf_kernel.cuh``) to control which ``(i, j)`` pairs are computed:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Mode
     - Computes
     - Used by
   * - ``None``
     - All ``(i, j)``
     - ``cdist``
   * - ``LowerTriangleSkipDiag``
     - ``i > j`` only
     - ``pdist`` (DistanceMatrix)
   * - ``LowerTriangle``
     - ``i >= j``
     - ``l2_kernel`` (SymmetricMatrix)


Double-buffered pipeline
=========================

``CudaBlockPipeline`` (in ``cuda_block_executor.cuh``) uses **double buffering** to overlap GPU computation with host-side result transfer:

Each GPU worker alternates between two buffers (``buf[0]`` and ``buf[1]``). While the GPU computes block N+1, the CPU scatters block N's results — this is the core overlap that double buffering enables.

.. mermaid::

   gantt
       title Double-buffered block pipeline (one GPU)
       dateFormat x
       axisFormat %s
       section Block 0 - buf 0
       Upload and launch kernel    :a1, 0, 1
       GPU kernel running          :a2, 1, 8
       Async D2H copy              :a3, 8, 9
       section Block 1 - buf 1
       Upload and launch kernel    :b1, 9, 10
       GPU kernel running          :b2, 10, 17
       D2H buf 0 finishing         :crit, b3, 10, 11
       Scatter buf 0 on CPU        :crit, b4, 11, 12
       Async D2H copy              :b5, 17, 18
       section Block 2 - buf 0
       Upload and launch kernel    :c1, 18, 19
       GPU kernel running          :c2, 19, 26
       D2H buf 1 finishing         :crit, c3, 19, 20
       Scatter buf 1 on CPU        :crit, c4, 20, 21
       Async D2H copy              :c5, 26, 27
       section Finalize
       D2H buf 0 finishing         :crit, d1, 27, 28
       Scatter buf 0 on CPU        :crit, d2, 28, 29

The red bars show work that overlaps with the GPU kernel. The async D2H copy from the previous block runs on a separate CUDA stream concurrently with the kernel, and the CPU scatter follows once the D2H completes — all while the GPU is still computing. The sequence for each block is:

1. Upload data and launch kernel (returns immediately — kernel runs async on GPU)
2. While the GPU computes: the previous block's async D2H transfer completes on the download stream, then the CPU scatters those results into the output matrix
3. Synchronize with the kernel (wait for GPU to finish)
4. Start async D2H copy of this block's results on the download stream
5. Move to the next block

Each GPU maintains:

- **Two output buffers** (device memory) -- alternated between blocks.
- **Two pinned host scratch buffers** (``cudaMallocHost``) -- required for truly asynchronous ``cudaMemcpyAsync``.
- **A download stream** -- separate from the compute stream, enabling D2H transfer to overlap with the next block's data upload and kernel launch.
- **A single set of function-specific storage** (e.g., PCF offset/point arrays) -- shared across blocks since the previous kernel is always complete before the next block begins.

The ``PinnedHostBuffer`` class wraps ``cudaMallocHost``/``cudaFreeHost`` with RAII semantics.


Streaming function data
=========================

For large datasets, all function data may not fit in GPU memory simultaneously. The data manager uploads only the subset needed for each block:

.. code-block:: text

   For a block covering rows [r0, r0+rH) x cols [c0, c0+cW):

   1. Extract host offset/element data for row functions [r0 .. r0+rH-1]
   2. Extract host offset/element data for col functions [c0 .. c0+cW-1]
   3. Re-index offsets to 0-based for each subset
   4. Upload to pre-allocated device arrays via cudaMemcpy

The data management is split into two layers:

- ``OffsetDataManager<ElementT>`` (pure C++, ``offset_data_manager.hpp``) -- host-side init, offset queries, element storage. No CUDA dependency.
- ``CudaOffsetDataManager<ElementT>`` (``cuda_offset_data_manager.cuh``) -- extends the base with ``upload_subset()`` for GPU device transfers.

Both are generic over the element type. Function-type-specific flattening (e.g., PCF breakpoints into ``SimplePoint`` structs) is done by a free function at init time:

.. code-block:: cpp

   // Generic: OffsetDataManager<ElementT> (base class)
   // sizeFn:    (const Obj&) -> size_t            -- number of elements
   // elementFn: (const Obj&, size_t i) -> ElementT -- extract i-th element
   manager.init(begin, end, sizeFn, elementFn);

   // PCF-specific convenience:
   init_pcf_data(manager, pcfBegin, pcfEnd);

For ``pdist``, both row and column uploads come from the same data manager. For ``cdist``, separate data managers are used for the row functions (X) and column functions (Y).

GPU device arrays are allocated once at their maximum required size (computed across all blocks) and reused.


Result scatter
===============

After each block's results are downloaded to the host scratch buffer, they are scattered into the final output. The scatter logic depends on the output type:

- **DistanceMatrix**: write only entries where :math:`i > j` (lower triangle, diagonal excluded). Uses compressed storage: ``max(i,j) * (max(i,j)-1) / 2 + min(i,j)``.
- **SymmetricMatrix**: write entries where :math:`i \geq j` (lower triangle, diagonal included).
- **Tensor** (dense): write all entries at their row-major position. Used by ``cdist`` -- the output tensor has the concatenated shape ``(*X.shape, *Y.shape)`` and is allocated before the computation begins.

For ``DistanceMatrix`` and ``SymmetricMatrix``, the scatter validates that skipped entries (upper triangle and, for distance matrices, the diagonal) are zero. A non-zero value at a skipped position indicates a bug in the computation and raises ``std::logic_error``.

Since blocks cover non-overlapping regions, scatter operations from different blocks never write to the same output element. No locking is required.


BlockOp policy
===============

``CudaBlockPipeline`` is generic over the function type. A **BlockOp** policy class encapsulates all function-specific behavior:

.. code-block:: cpp

   // BlockOp concept (duck-typed):
   //
   // typename BlockOp::GpuStorage
   //   Per-GPU device arrays for function-specific data. Move-constructible.
   //
   // GpuStorage init_gpu_storage(size_t gpuId, const CudaBlockScheduler& scheduler)
   //   Allocate device arrays for one GPU, sized for the largest block.
   //
   // void exec_block(GpuStorage& storage, const BlockInfo& block,
   //                 CudaDeviceArray<Tv>& gpuOutputMatrix, dim3 blockDim)
   //   Upload data for this block, launch kernel, synchronize.
   //   The output matrix is pre-cleared before this call.

For PCFs, the policy is ``PcfBlockOp`` (in ``pcf_block_op.cuh``). It takes separate row and column data managers, enabling both ``pdist`` (same source for both) and ``cdist`` (different sources).


PCF-specific implementation
=============================

PCF data layout
----------------

Each piecewise constant function is defined by a sequence of breakpoints :math:`(t_k, v_k)`, where the function takes value :math:`v_k` on the interval :math:`[t_k, t_{k+1})`. On the GPU, these are stored as flat arrays of ``SimplePoint`` structs:

.. code-block:: cpp

   template <typename Tt, typename Tv>
   struct SimplePoint { Tt t; Tv v; };

An offset array maps each function index to its starting position in the points array:

.. code-block:: text

   PCF 0: points[0..2]     offsets = [0, 3, 5, 8, ...]
   PCF 1: points[3..4]
   PCF 2: points[5..7]
   ...

``CudaPcfDataManager`` is a type alias for ``CudaOffsetDataManager<SimplePoint<Tt, Tv>>``, with a free function ``init_pcf_data()`` that handles the PCF-specific flattening.


Rectangle iteration kernel
----------------------------

The kernel (in ``cuda_pcf_kernel.cuh``) walks two PCFs simultaneously, advancing through their breakpoints in time order:

.. code-block:: text

   f: |---3---|---1---|---0---|
   g: |----2----|---0----|

   Rectangles: [0,1)x(3,2)  [1,2)x(1,2)  [2,3)x(1,0)  [3,4)x(0,0)

For each rectangle :math:`[l, r) \times (f_v, g_v)`, an operation-specific function computes the contribution. The result is accumulated:

.. code-block:: text

   L1 distance:       sum += (r - l) * |f_v - g_v|
   Lp distance:       sum += (r - l) * |f_v - g_v|^p,  finalize: sum^(1/p)
   L2 inner product:  sum += (r - l) * f_v * g_v

Each CUDA thread computes one :math:`(i, j)` pair. The ``TriangleSkipMode`` controls which pairs are skipped.


Operation data flows
======================

All operations share the same ``CudaBlockPipeline`` -- they differ in the task class, operation functor, output type, and triangle mode.

**pdist** — pairwise distance (symmetric, lower triangle, diagonal excluded):

.. code-block:: text

   Python: sb.pdist(X)
     -> C++: create DistanceMatrix(n), one CudaPcfDataManager
     -> CudaPairwiseIntegrationTask:
          One function set, one data manager (shared for row and col)
          Operation = OperationL1Dist or OperationLpDist
          TriangleSkipMode = LowerTriangleSkipDiag
          BlockTriangleMode = LowerTriangle
          ResultWriter = DistanceMatrixResultWriter
     -> Result: compressed n*(n-1)/2 DistanceMatrix

**l2_kernel** — inner product kernel matrix (symmetric, lower triangle, diagonal included):

.. code-block:: text

   Python: sb.l2_kernel(X)
     -> C++: create SymmetricMatrix(n), one CudaPcfDataManager
     -> CudaPairwiseIntegrationTask:
          One function set, one data manager (shared for row and col)
          Operation = OperationL2InnerProduct
          TriangleSkipMode = LowerTriangle  (includes diagonal)
          BlockTriangleMode = LowerTriangle
          ResultWriter = SymmetricMatrixResultWriter
     -> Result: compressed n*(n+1)/2 SymmetricMatrix

**cdist** — cross-distance or cross-kernel (rectangular, all pairs):

.. code-block:: text

   Python: sb.cdist(X, Y)
     -> C++: create Tensor<Tv> with shape (*X.shape, *Y.shape)
     -> C++: create separate row/col CudaPcfDataManagers
     -> CudaCrossIntegrationTask:
          Two function sets, two data managers
          Operation = any (L1, Lp, or L2 inner product)
          TriangleSkipMode = None
          BlockTriangleMode = Full
          Scheduler: nRows = product(X.shape), nCols = product(Y.shape)
          ResultWriter = DenseResultWriter
     -> Result: properly shaped Tensor, no reshape needed

.. list-table:: Summary of operation configurations
   :header-rows: 1
   :widths: 16 16 18 22 16 12

   * - Operation
     - Task class
     - Output type
     - TriangleSkipMode
     - ResultWriter
     - BlockMode
   * - ``pdist``
     - ``CudaPairwiseIntegrationTask``
     - ``DistanceMatrix``
     - ``LowerTriangleSkipDiag``
     - ``DistanceMatrixResultWriter``
     - ``LowerTriangle``
   * - ``l2_kernel``
     - ``CudaPairwiseIntegrationTask``
     - ``SymmetricMatrix``
     - ``LowerTriangle``
     - ``SymmetricMatrixResultWriter``
     - ``LowerTriangle``
   * - ``cdist``
     - ``CudaCrossIntegrationTask``
     - ``Tensor``
     - ``None``
     - ``DenseResultWriter``
     - ``Full``

The CPU fallback uses the same split: ``CpuPairwiseIntegrationTask`` for pdist/l2_kernel and ``CpuCrossIntegrationTask`` for cdist (in ``matrix_integrate.hpp``).


Adding a new function type
===========================

To add support for a new piecewise function type (e.g., piecewise linear functions):

1. **Define the GPU data layout** -- create a point struct and a ``CudaOffsetDataManager`` type alias with a flattener function (analogous to ``CudaPcfDataManager`` + ``init_pcf_data``).

2. **Write the kernel** -- implement the device-side iteration function and integration kernel (analogous to ``cuda_pcf_iterate_rectangles`` and ``cuda_pcf_block_integrate`` in ``cuda_pcf_kernel.cuh``).

3. **Create a BlockOp** -- implement the ``GpuStorage``/``init_gpu_storage``/``exec_block`` interface (analogous to ``PcfBlockOp``).

4. **Add task classes and factory functions** -- create pairwise and cross integration task classes (analogous to ``CudaPairwiseIntegrationTask`` / ``CudaCrossIntegrationTask``), and add factory functions in ``cuda_matrix_integrate_api.hpp`` / ``cuda_matrix_integrate.cu``.

The block scheduler, result writers, ``CudaBlockPipeline``, offset data manager, and Python integration patterns are all reused unchanged.
