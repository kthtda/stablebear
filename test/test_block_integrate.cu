#include <gtest/gtest.h>

#include <sbear/functional/pcf.hpp>
#include <sbear/distance_matrix.hpp>
#include <sbear/symmetric_matrix.hpp>
#include <sbear/tensor.hpp>
#include <sbear/task.hpp>
#include <sbear/functional/operations.cuh>
#include <sbear/algorithms/functional/lp_distance.hpp>
#include <sbear/algorithms/functional/matrix_integrate.hpp>

#ifdef BUILD_WITH_CUDA
#include <sbear/cuda/cuda_matrix_integrate_api.hpp>
#endif

#include <vector>
#include <memory>
#include <cmath>

namespace
{
  // --- Type traits for parameterization ---

  template <typename T>
  struct PrecisionTraits;

  template <>
  struct PrecisionTraits<float>
  {
    using Tv = float;
    using Pcf = sb::Pcf_f32;
    using Point = sb::TimePoint_f32;
    static constexpr double tolerance = 1e-5;
    static const char* name() { return "f32"; }
  };

  template <>
  struct PrecisionTraits<double>
  {
    using Tv = double;
    using Pcf = sb::Pcf_f64;
    using Point = sb::TimePoint_f64;
    static constexpr double tolerance = 1e-10;
    static const char* name() { return "f64"; }
  };

  // --- Typed test fixture ---

  template <typename T>
  class BlockIntegrateTyped : public ::testing::Test
  {
  protected:
    using Tv = typename PrecisionTraits<T>::Tv;
    using PcfT = typename PrecisionTraits<T>::Pcf;
    using PointT = typename PrecisionTraits<T>::Point;
    static constexpr double tol = PrecisionTraits<T>::tolerance;

    void SetUp() override
    {
#ifndef BUILD_WITH_CUDA
      GTEST_SKIP() << "CUDA not available";
#else
      if (sb::get_num_cuda_devices() == 0)
      {
        GTEST_SKIP() << "No CUDA devices available";
      }
#endif
    }

    // Helper to build PCFs from initializer lists
    PcfT make_pcf(std::initializer_list<std::pair<Tv, Tv>> pts)
    {
      std::vector<PointT> points;
      for (auto [t, v] : pts)
      {
        points.emplace_back(t, v);
      }
      return PcfT(std::move(points));
    }

    // Run a pairwise integration on CPU and return the result matrix
    template <typename Op, typename OutputT, bool includeDiagonal>
    OutputT cpu_pdist(const std::vector<PcfT>& pcfs, Op op)
    {
      OutputT out(pcfs.size());
      auto task = std::make_unique<sb::CpuPairwiseIntegrationTask<Op, typename std::vector<PcfT>::const_iterator, OutputT, includeDiagonal>>(
          out, pcfs.cbegin(), pcfs.cend(), op);
      task->start_async(sb::default_executor());
      task->future().get();
      return out;
    }

    // Run a cross-integration on CPU and return the result tensor
    template <typename Op>
    sb::Tensor<Tv> cpu_cdist(const std::vector<PcfT>& rowPcfs, const std::vector<PcfT>& colPcfs, Op op)
    {
      sb::Tensor<Tv> out({rowPcfs.size(), colPcfs.size()}, Tv(0));
      auto task = std::make_unique<sb::CpuCrossIntegrationTask<Op, typename std::vector<PcfT>::const_iterator>>(
          out, rowPcfs.cbegin(), rowPcfs.cend(), colPcfs.cbegin(), colPcfs.cend(), op);
      task->start_async(sb::default_executor());
      task->future().get();
      return out;
    }
  };

  using TestTypes = ::testing::Types<float, double>;
  TYPED_TEST_SUITE(BlockIntegrateTyped, TestTypes);

#ifdef BUILD_WITH_CUDA

  // ==================== pdist L1 ====================

  TYPED_TEST(BlockIntegrateTyped, PdistL1_TwoPcfs)
  {
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;

    std::vector<PcfT> pcfs;
    pcfs.push_back(this->make_pcf({{0, 3}, {1, 0}}));
    pcfs.push_back(this->make_pcf({{0, 1}, {2, 0}}));

    sb::DistanceMatrix<Tv> dm(2);
    auto task = sb::create_cuda_block_integrate_l1_task(dm, pcfs);
    task->start_async(sb::default_executor());
    task->future().get();

    auto expected = sb::lp_distance(pcfs[0], pcfs[1]);
    EXPECT_NEAR(dm(0, 0), 0, this->tol);
    EXPECT_NEAR(dm(0, 1), expected, this->tol);
    EXPECT_NEAR(dm(1, 0), expected, this->tol);
    EXPECT_NEAR(dm(1, 1), 0, this->tol);
  }

  TYPED_TEST(BlockIntegrateTyped, PdistL1_Empty)
  {
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;

    std::vector<PcfT> pcfs(2);
    sb::DistanceMatrix<Tv> dm(2);
    auto task = sb::create_cuda_block_integrate_l1_task(dm, pcfs);
    task->start_async(sb::default_executor());
    task->future().get();

    EXPECT_NEAR(dm(0, 1), 0, this->tol);
  }

  TYPED_TEST(BlockIntegrateTyped, PdistL1_FivePcfs_MatchesCpu)
  {
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;

    std::vector<PcfT> pcfs;
    pcfs.push_back(this->make_pcf({{0, 1}, {1, 0}}));
    pcfs.push_back(this->make_pcf({{0, 2}, {Tv(0.5), 1}, {Tv(1.5), 0}}));
    pcfs.push_back(this->make_pcf({{0, 0}, {1, 3}, {2, 0}}));
    pcfs.push_back(this->make_pcf({{0, 5}}));
    pcfs.push_back(this->make_pcf({{0, 1}, {Tv(0.25), 2}, {Tv(0.75), 1}, {1, 0}}));

    size_t n = pcfs.size();

    auto cpuDm = this->template cpu_pdist<sb::OperationL1Dist<Tv, Tv>, sb::DistanceMatrix<Tv>, false>(
        pcfs, sb::OperationL1Dist<Tv, Tv>{});

    sb::DistanceMatrix<Tv> gpuDm(n);
    auto task = sb::create_cuda_block_integrate_l1_task(gpuDm, pcfs);
    task->start_async(sb::default_executor());
    task->future().get();

    EXPECT_TRUE(sb::allclose(gpuDm, cpuDm, Tv(this->tol)));
  }

  // ==================== pdist Lp ====================

  TYPED_TEST(BlockIntegrateTyped, PdistLp_P2_MatchesCpu)
  {
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;

    std::vector<PcfT> pcfs;
    pcfs.push_back(this->make_pcf({{0, 3}, {1, 0}}));
    pcfs.push_back(this->make_pcf({{0, 1}, {2, 0}}));

    size_t n = 2;

    auto cpuDm = this->template cpu_pdist<sb::OperationLpDist<Tv, Tv>, sb::DistanceMatrix<Tv>, false>(
        pcfs, sb::OperationLpDist<Tv, Tv>(Tv(2)));

    sb::DistanceMatrix<Tv> gpuDm(n);
    auto task = sb::create_cuda_block_integrate_lp_task(gpuDm, pcfs, Tv(2));
    task->start_async(sb::default_executor());
    task->future().get();

    EXPECT_TRUE(sb::allclose(gpuDm, cpuDm, Tv(1e-4)));
  }

  // ==================== L2 kernel ====================

  TYPED_TEST(BlockIntegrateTyped, L2Kernel_MatchesCpu)
  {
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;

    std::vector<PcfT> pcfs;
    pcfs.push_back(this->make_pcf({{0, 3}, {1, 0}}));
    pcfs.push_back(this->make_pcf({{0, 1}, {2, 0}}));

    size_t n = 2;

    auto cpuSm = this->template cpu_pdist<sb::OperationL2InnerProduct<Tv, Tv>, sb::SymmetricMatrix<Tv>, true>(
        pcfs, sb::OperationL2InnerProduct<Tv, Tv>{});

    sb::SymmetricMatrix<Tv> gpuSm(n);
    auto task = sb::create_cuda_block_integrate_l2_kernel_task(gpuSm, pcfs);
    task->start_async(sb::default_executor());
    task->future().get();

    EXPECT_TRUE(sb::allclose(gpuSm, cpuSm, Tv(this->tol)));
  }

  // ==================== cdist L1 ====================

  TYPED_TEST(BlockIntegrateTyped, CdistL1_Rectangular_MatchesCpu)
  {
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;

    std::vector<PcfT> rowPcfs;
    rowPcfs.push_back(this->make_pcf({{0, 3}, {1, 0}}));
    rowPcfs.push_back(this->make_pcf({{0, 1}, {2, 0}}));

    std::vector<PcfT> colPcfs;
    colPcfs.push_back(this->make_pcf({{0, 0}}));
    colPcfs.push_back(this->make_pcf({{0, 3}, {1, 0}}));
    colPcfs.push_back(this->make_pcf({{0, 1}}));

    auto cpuDense = this->cpu_cdist(rowPcfs, colPcfs, sb::OperationL1Dist<Tv, Tv>{});

    sb::Tensor<Tv> out({rowPcfs.size(), colPcfs.size()}, Tv(0));
    auto task = sb::create_cuda_block_cdist_l1_task(out, rowPcfs, colPcfs);
    task->start_async(sb::default_executor());
    task->future().get();

    EXPECT_TRUE(sb::allclose(out, cpuDense, Tv(this->tol)));
  }

  TYPED_TEST(BlockIntegrateTyped, CdistL1_SameInput_MatchesPdist)
  {
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;

    std::vector<PcfT> pcfs;
    pcfs.push_back(this->make_pcf({{0, 3}, {1, 0}}));
    pcfs.push_back(this->make_pcf({{0, 1}, {2, 0}}));
    pcfs.push_back(this->make_pcf({{0, 0}}));

    size_t n = pcfs.size();

    // pdist
    sb::DistanceMatrix<Tv> dm(n);
    auto pdistTask = sb::create_cuda_block_integrate_l1_task(dm, pcfs);
    pdistTask->start_async(sb::default_executor());
    pdistTask->future().get();

    // cdist(X, X)
    sb::Tensor<Tv> cdistOut({n, n}, Tv(0));
    auto cdistTask = sb::create_cuda_block_cdist_l1_task(cdistOut, pcfs, pcfs);
    cdistTask->start_async(sb::default_executor());
    cdistTask->future().get();

    for (size_t i = 0; i < n; ++i)
    {
      for (size_t j = 0; j < n; ++j)
      {
        EXPECT_NEAR(cdistOut.data()[i * n + j], dm(i, j), this->tol)
          << "cdist vs pdist mismatch at (" << i << ", " << j << ")";
      }
    }
  }

  // ==================== cdist Lp ====================

  TYPED_TEST(BlockIntegrateTyped, CdistLp_P2_MatchesCpu)
  {
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;

    std::vector<PcfT> rowPcfs;
    rowPcfs.push_back(this->make_pcf({{0, 4}, {1, 0}}));

    std::vector<PcfT> colPcfs;
    colPcfs.push_back(this->make_pcf({{0, 1}, {1, 0}}));

    auto cpuDense = this->cpu_cdist(rowPcfs, colPcfs, sb::OperationLpDist<Tv, Tv>(Tv(2)));

    sb::Tensor<Tv> out({rowPcfs.size(), colPcfs.size()}, Tv(0));
    auto task = sb::create_cuda_block_cdist_lp_task(out, rowPcfs, colPcfs, Tv(2));
    task->start_async(sb::default_executor());
    task->future().get();

    EXPECT_TRUE(sb::allclose(out, cpuDense, Tv(1e-4)));
  }

  // ==================== cdist L2 kernel ====================

  TYPED_TEST(BlockIntegrateTyped, CdistL2Kernel_MatchesCpu)
  {
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;

    std::vector<PcfT> rowPcfs;
    rowPcfs.push_back(this->make_pcf({{0, 4}, {3, 0}}));

    std::vector<PcfT> colPcfs;
    colPcfs.push_back(this->make_pcf({{0, 2}, {3, 0}}));
    colPcfs.push_back(this->make_pcf({{0, 1}}));

    auto cpuDense = this->cpu_cdist(rowPcfs, colPcfs, sb::OperationL2InnerProduct<Tv, Tv>{});

    sb::Tensor<Tv> out({rowPcfs.size(), colPcfs.size()}, Tv(0));
    auto task = sb::create_cuda_block_cdist_l2_kernel_task(out, rowPcfs, colPcfs);
    task->start_async(sb::default_executor());
    task->future().get();

    EXPECT_TRUE(sb::allclose(out, cpuDense, Tv(this->tol)));
  }

#endif // BUILD_WITH_CUDA
}
