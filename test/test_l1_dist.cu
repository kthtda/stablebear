/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <gtest/gtest.h>

#include <sbear/functional/pcf.hpp>
#include <sbear/distance_matrix.hpp>
#include <sbear/task.hpp>
#include <sbear/functional/operations.cuh>
#include <sbear/algorithms/functional/matrix_integrate.hpp>

#ifdef BUILD_WITH_CUDA
#include <sbear/cuda/cuda_matrix_integrate_api.hpp>
#endif

#include <vector>
#include <memory>

#ifdef BUILD_WITH_CUDA
#pragma message("Building tester with CUDA")
#else
#pragma message("Building tester without CUDA")
#endif

namespace
{
  // Direct (non-task) integration test to verify the core algorithm
  template <typename T>
  class PcfL1DirectTest : public ::testing::Test {};

  using DirectTypes = ::testing::Types<float, double>;
  TYPED_TEST_SUITE(PcfL1DirectTest, DirectTypes);

  TYPED_TEST(PcfL1DirectTest, TwoPointPcfIntegrate)
  {
    using T = TypeParam;
    sb::Pcf<T, T> f(std::vector<sb::TimePoint<T, T>>({ {T(0), T(3)}, {T(1), T(0)} }));
    sb::Pcf<T, T> g(std::vector<sb::TimePoint<T, T>>({ {T(0), T(1)}, {T(2), T(0)} }));

    auto op = sb::OperationL1Dist<T, T>{};
    T result = op(sb::integrate(f, g, op));

    EXPECT_NEAR(result, T(3), T(1e-6));
  }

  // Parameterized on precision and hardware
  template <typename T, sb::Hardware Hw>
  struct TestConfig
  {
    using value_type = T;
    static constexpr sb::Hardware hw = Hw;
  };

  template <typename Cfg>
  class PcfL1IntegratorFixture : public ::testing::Test
  {
  public:
    using T = typename Cfg::value_type;
    using PcfT = sb::Pcf<T, T>;

    void compute_l1()
    {
      std::unique_ptr<sb::StoppableTask<void>> task;

      if constexpr (Cfg::hw == sb::Hardware::CUDA)
      {
#ifdef BUILD_WITH_CUDA
        if (sb::get_num_cuda_devices() == 0)
        {
          GTEST_SKIP() << "No CUDA devices available";
          return;
        }
        task = sb::create_cuda_block_integrate_l1_task(m_dm, m_pcfs);
#else
        GTEST_SKIP() << "CUDA not available";
#endif
      }
      else
      {
        auto op = sb::OperationL1Dist<T, T>{};
        task = std::make_unique<sb::CpuPairwiseIntegrationTask<decltype(op), decltype(m_pcfs.cbegin()), sb::DistanceMatrix<T>, false>>(
            m_dm, m_pcfs.cbegin(), m_pcfs.cend(), op);
      }

      task->start_async(sb::default_executor());
      task->future().get();
    }

    std::vector<PcfT> m_pcfs;
    sb::DistanceMatrix<T> m_dm{0};
  };

  using IntegratorConfigs = ::testing::Types<
      TestConfig<float, sb::Hardware::CPU>,
      TestConfig<double, sb::Hardware::CPU>,
      TestConfig<float, sb::Hardware::CUDA>,
      TestConfig<double, sb::Hardware::CUDA>
  >;
  TYPED_TEST_SUITE(PcfL1IntegratorFixture, IntegratorConfigs);

  TYPED_TEST(PcfL1IntegratorFixture, EmptyPcfPairL1dist)
  {
    using T = typename TypeParam::value_type;
    this->m_pcfs.resize(2);
    this->m_dm = sb::DistanceMatrix<T>(2);

    this->compute_l1();

    EXPECT_NEAR(this->m_dm(0, 1), T(0), T(1e-6));
  }

  TYPED_TEST(PcfL1IntegratorFixture, TwoPointPcfL1dist)
  {
    using T = typename TypeParam::value_type;
    using PointT = sb::TimePoint<T, T>;

    this->m_pcfs.emplace_back(std::vector<PointT>({{T(0), T(3)}, {T(1), T(0)}}));
    this->m_pcfs.emplace_back(std::vector<PointT>({{T(0), T(1)}, {T(2), T(0)}}));
    this->m_dm = sb::DistanceMatrix<T>(2);

    this->compute_l1();

    EXPECT_NEAR(this->m_dm(0, 0), T(0), T(1e-6));
    EXPECT_NEAR(this->m_dm(0, 1), T(3), T(1e-6));
    EXPECT_NEAR(this->m_dm(1, 0), T(3), T(1e-6));
    EXPECT_NEAR(this->m_dm(1, 1), T(0), T(1e-6));
  }
}
