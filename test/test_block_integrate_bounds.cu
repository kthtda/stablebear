// Regression tests for issue #190: the device-side rectangle iteration must
// agree with the CPU reference for non-default integration bounds [a, b].
// Before the fix, the device walk started at the first breakpoint instead of
// the last breakpoint <= a, and never clamped the final rectangle to b.

#include <gtest/gtest.h>

#include <sbear/functional/pcf.hpp>
#include <sbear/distance_matrix.hpp>
#include <sbear/task.hpp>
#include <sbear/functional/operations.cuh>
#include <sbear/algorithms/functional/matrix_integrate.hpp>

#ifdef BUILD_WITH_CUDA
#include <sbear/cuda/cuda_matrix_integrate_api.hpp>
#endif

#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace
{

  template <typename T>
  class BlockIntegrateBoundsTyped : public ::testing::Test
  {
  protected:
    using Tv = T;
    using PcfT = sb::Pcf<T, T>;
    using PointT = sb::TimePoint<T, T>;

    static double tol() { return std::is_same_v<T, float> ? 1e-4 : 1e-9; }

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

    PcfT make_pcf(std::initializer_list<std::pair<Tv, Tv>> pts)
    {
      std::vector<PointT> points;
      for (auto [t, v] : pts)
      {
        points.emplace_back(t, v);
      }
      return PcfT(std::move(points));
    }

    // Reference: the CPU integrate() with the same bounds and operation.
    void expect_matches_cpu(const std::vector<PcfT>& pcfs, Tv a, Tv b)
    {
#ifdef BUILD_WITH_CUDA
      auto op = sb::OperationL1Dist<Tv, Tv>{};

      sb::DistanceMatrix<Tv> dm(pcfs.size());
      auto task = sb::create_cuda_block_integrate_l1_task(dm, pcfs, a, b);
      task->start_async(sb::default_executor());
      task->future().get();

      for (size_t i = 0; i < pcfs.size(); ++i)
      {
        for (size_t j = 0; j < i; ++j)
        {
          auto expected = op(sb::integrate<Tv, Tv>(pcfs[i], pcfs[j], op, a, b));
          EXPECT_NEAR(dm(i, j), expected, tol())
              << "pair (" << i << ", " << j << ") diverges for bounds [" << a << ", " << b << "]";
        }
      }
#endif
    }
  };

  using TestTypes = ::testing::Types<float, double>;
  TYPED_TEST_SUITE(BlockIntegrateBoundsTyped, TestTypes);

#ifdef BUILD_WITH_CUDA

  TYPED_TEST(BlockIntegrateBoundsTyped, BreakpointsBeforeLowerBound)
  {
    // Breakpoints in (0, a]: the pre-fix device walk started at index 0 and
    // emitted wrong values (and a negative-width leading rectangle).
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;

    std::vector<PcfT> pcfs;
    pcfs.push_back(this->make_pcf({{0, 3}, {1, 5}, {2, 1}, {4, 0}}));
    pcfs.push_back(this->make_pcf({{0, 1}, {1.5, 2}, {3, 4}, {5, 0}}));
    pcfs.push_back(this->make_pcf({{0, 2}, {2.5, 3}, {6, 0}}));

    this->expect_matches_cpu(pcfs, Tv(2), std::numeric_limits<Tv>::max());
  }

  TYPED_TEST(BlockIntegrateBoundsTyped, FiniteUpperBoundIsClamped)
  {
    // b inside the domain: the pre-fix device walk integrated the final
    // rectangle past b.
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;

    std::vector<PcfT> pcfs;
    pcfs.push_back(this->make_pcf({{0, 3}, {1, 5}, {4, 0}}));
    pcfs.push_back(this->make_pcf({{0, 1}, {3, 4}, {5, 0}}));

    this->expect_matches_cpu(pcfs, Tv(0), Tv(3.5));
  }

  TYPED_TEST(BlockIntegrateBoundsTyped, BothBoundsCustom)
  {
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;

    std::vector<PcfT> pcfs;
    pcfs.push_back(this->make_pcf({{0, 3}, {1, 5}, {2, 1}, {4, 0}}));
    pcfs.push_back(this->make_pcf({{0, 1}, {1.5, 2}, {3, 4}, {5, 0}}));

    this->expect_matches_cpu(pcfs, Tv(1.25), Tv(4.5));
  }

  TYPED_TEST(BlockIntegrateBoundsTyped, DefaultBoundsUnchanged)
  {
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;

    std::vector<PcfT> pcfs;
    pcfs.push_back(this->make_pcf({{0, 3}, {1, 0}}));
    pcfs.push_back(this->make_pcf({{0, 1}, {2, 0}}));

    this->expect_matches_cpu(pcfs, Tv(0), std::numeric_limits<Tv>::max());
  }

  TYPED_TEST(BlockIntegrateBoundsTyped, RandomizedAgainstCpu)
  {
    using Tv = typename TestFixture::Tv;
    using PcfT = typename TestFixture::PcfT;
    using PointT = typename TestFixture::PointT;

    std::mt19937 rng(20260702);
    std::uniform_int_distribution<int> szDist(1, 8);
    std::uniform_real_distribution<double> dtDist(0.1, 2.0);
    std::uniform_real_distribution<double> vDist(-4.0, 4.0);

    std::vector<PcfT> pcfs;
    for (int i = 0; i < 12; ++i)
    {
      std::vector<PointT> pts;
      double t = 0.0;
      int n = szDist(rng);
      for (int k = 0; k < n; ++k)
      {
        pts.emplace_back(static_cast<Tv>(t), static_cast<Tv>(vDist(rng)));
        t += dtDist(rng);
      }
      pcfs.emplace_back(std::move(pts));
    }

    for (auto [a, b] : {std::pair<Tv, Tv>{Tv(0.75), Tv(6)},
                        std::pair<Tv, Tv>{Tv(2), Tv(3)},
                        std::pair<Tv, Tv>{Tv(3.5), std::numeric_limits<Tv>::max()}})
    {
      this->expect_matches_cpu(pcfs, a, b);
    }
  }

#endif // BUILD_WITH_CUDA

} // namespace
