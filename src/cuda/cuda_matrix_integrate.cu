// This file is compiled by NVCC. It provides concrete (non-template)
// factory functions for the block-based CUDA integration pipeline.

#include <sbear/cuda/cuda_matrix_integrate_api.hpp>
#include <sbear/cuda/pcf_block_op.cuh>
#include <sbear/cuda/cuda_result_writer.hpp>
#include <sbear/functional/operations.cuh>

namespace sb
{
  // Helper for pdist/l2_kernel factories
  template <typename Tv, typename OpT, typename WriterT>
  static std::unique_ptr<StoppableTask<void>> make_pdist_task(
      WriterT writer,
      const std::vector<Pcf<Tv, Tv>>& pcfs,
      OpT op, Tv a, Tv b,
      TriangleSkipMode skipMode = TriangleSkipMode::LowerTriangleSkipDiag)
  {
    using iter_t = typename std::vector<Pcf<Tv, Tv>>::const_iterator;
    return std::make_unique<CudaPairwiseIntegrationTask<iter_t, OpT, WriterT>>(
        *default_executor().cuda(), std::move(writer),
        pcfs.cbegin(), pcfs.cend(), op, a, b, skipMode);
  }

  // L1 → DistanceMatrix
  std::unique_ptr<StoppableTask<void>> create_cuda_block_integrate_l1_task(
      DistanceMatrix<float32_t>& out,
      const std::vector<Pcf<float32_t, float32_t>>& pcfs,
      float32_t a, float32_t b)
  { return make_pdist_task(DistanceMatrixResultWriter<float32_t>(out), pcfs, OperationL1Dist<float32_t, float32_t>{}, a, b); }

  std::unique_ptr<StoppableTask<void>> create_cuda_block_integrate_l1_task(
      DistanceMatrix<float64_t>& out,
      const std::vector<Pcf<float64_t, float64_t>>& pcfs,
      float64_t a, float64_t b)
  { return make_pdist_task(DistanceMatrixResultWriter<float64_t>(out), pcfs, OperationL1Dist<float64_t, float64_t>{}, a, b); }

  // Lp → DistanceMatrix
  std::unique_ptr<StoppableTask<void>> create_cuda_block_integrate_lp_task(
      DistanceMatrix<float32_t>& out,
      const std::vector<Pcf<float32_t, float32_t>>& pcfs,
      float32_t p, float32_t a, float32_t b)
  { return make_pdist_task(DistanceMatrixResultWriter<float32_t>(out), pcfs, OperationLpDist<float32_t, float32_t>(p), a, b); }

  std::unique_ptr<StoppableTask<void>> create_cuda_block_integrate_lp_task(
      DistanceMatrix<float64_t>& out,
      const std::vector<Pcf<float64_t, float64_t>>& pcfs,
      float64_t p, float64_t a, float64_t b)
  { return make_pdist_task(DistanceMatrixResultWriter<float64_t>(out), pcfs, OperationLpDist<float64_t, float64_t>(p), a, b); }

  // L2 → SymmetricMatrix
  std::unique_ptr<StoppableTask<void>> create_cuda_block_integrate_l2_kernel_task(
      SymmetricMatrix<float32_t>& out,
      const std::vector<Pcf<float32_t, float32_t>>& pcfs,
      float32_t a, float32_t b)
  { return make_pdist_task(SymmetricMatrixResultWriter<float32_t>(out), pcfs, OperationL2InnerProduct<float32_t, float32_t>{}, a, b, TriangleSkipMode::LowerTriangle); }

  std::unique_ptr<StoppableTask<void>> create_cuda_block_integrate_l2_kernel_task(
      SymmetricMatrix<float64_t>& out,
      const std::vector<Pcf<float64_t, float64_t>>& pcfs,
      float64_t a, float64_t b)
  { return make_pdist_task(SymmetricMatrixResultWriter<float64_t>(out), pcfs, OperationL2InnerProduct<float64_t, float64_t>{}, a, b, TriangleSkipMode::LowerTriangle); }

  // === cdist factories ===

  // Helper for cdist factories
  template <typename Tv, typename OpT>
  static std::unique_ptr<StoppableTask<void>> make_cdist_task(
      Tensor<Tv>& out,
      const std::vector<Pcf<Tv, Tv>>& rowPcfs,
      const std::vector<Pcf<Tv, Tv>>& colPcfs,
      OpT op, Tv a, Tv b)
  {
    using iter_t = typename std::vector<Pcf<Tv, Tv>>::const_iterator;
    using writer_t = DenseResultWriter<Tv>;

    return std::make_unique<CudaCrossIntegrationTask<iter_t, OpT, writer_t>>(
        *default_executor().cuda(), writer_t(DenseMatrixView<Tv>(out, colPcfs.size())),
        rowPcfs.cbegin(), rowPcfs.cend(),
        colPcfs.cbegin(), colPcfs.cend(),
        op, a, b);
  }

  // L1 cdist
  std::unique_ptr<StoppableTask<void>> create_cuda_block_cdist_l1_task(
      Tensor<float32_t>& out,
      const std::vector<Pcf<float32_t, float32_t>>& rowPcfs,
      const std::vector<Pcf<float32_t, float32_t>>& colPcfs,
      float32_t a, float32_t b)
  {
    return make_cdist_task(out, rowPcfs, colPcfs, OperationL1Dist<float32_t, float32_t>{}, a, b);
  }

  std::unique_ptr<StoppableTask<void>> create_cuda_block_cdist_l1_task(
      Tensor<float64_t>& out,
      const std::vector<Pcf<float64_t, float64_t>>& rowPcfs,
      const std::vector<Pcf<float64_t, float64_t>>& colPcfs,
      float64_t a, float64_t b)
  {
    return make_cdist_task(out, rowPcfs, colPcfs, OperationL1Dist<float64_t, float64_t>{}, a, b);
  }

  // Lp cdist
  std::unique_ptr<StoppableTask<void>> create_cuda_block_cdist_lp_task(
      Tensor<float32_t>& out,
      const std::vector<Pcf<float32_t, float32_t>>& rowPcfs,
      const std::vector<Pcf<float32_t, float32_t>>& colPcfs,
      float32_t p, float32_t a, float32_t b)
  {
    return make_cdist_task(out, rowPcfs, colPcfs, OperationLpDist<float32_t, float32_t>(p), a, b);
  }

  std::unique_ptr<StoppableTask<void>> create_cuda_block_cdist_lp_task(
      Tensor<float64_t>& out,
      const std::vector<Pcf<float64_t, float64_t>>& rowPcfs,
      const std::vector<Pcf<float64_t, float64_t>>& colPcfs,
      float64_t p, float64_t a, float64_t b)
  {
    return make_cdist_task(out, rowPcfs, colPcfs, OperationLpDist<float64_t, float64_t>(p), a, b);
  }

  // L2 cross-kernel
  std::unique_ptr<StoppableTask<void>> create_cuda_block_cdist_l2_kernel_task(
      Tensor<float32_t>& out,
      const std::vector<Pcf<float32_t, float32_t>>& rowPcfs,
      const std::vector<Pcf<float32_t, float32_t>>& colPcfs,
      float32_t a, float32_t b)
  {
    return make_cdist_task(out, rowPcfs, colPcfs, OperationL2InnerProduct<float32_t, float32_t>{}, a, b);
  }

  std::unique_ptr<StoppableTask<void>> create_cuda_block_cdist_l2_kernel_task(
      Tensor<float64_t>& out,
      const std::vector<Pcf<float64_t, float64_t>>& rowPcfs,
      const std::vector<Pcf<float64_t, float64_t>>& colPcfs,
      float64_t a, float64_t b)
  {
    return make_cdist_task(out, rowPcfs, colPcfs, OperationL2InnerProduct<float64_t, float64_t>{}, a, b);
  }
}
