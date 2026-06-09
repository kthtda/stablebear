// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <sbear/functional/pcf.hpp>
#include <sbear/algorithms/functional/apply_functional.hpp>

#include <vector>

namespace
{

  using PcfFloatTypes = ::testing::Types<sb::Pcf_f32, sb::Pcf_f64>;

  template<typename PcfT>
  class PcfAlgebraAndNormsTest : public ::testing::Test
  {
  };

  TYPED_TEST_SUITE(PcfAlgebraAndNormsTest, PcfFloatTypes);

  // ============================================================================
  // Basic algebra: +, -, *, / and average
  // ============================================================================

  TYPED_TEST(PcfAlgebraAndNormsTest, PlusMinusScaleOperators)
  {
    using PcfT = TypeParam;
    using T = typename PcfT::value_type;
    using Pt = typename PcfT::point_type;

    // f(t): 0 on [0,1), 1 on [1,2), 0 afterwards
    PcfT f({ { static_cast<T>(0), static_cast<T>(0) },
             { static_cast<T>(1), static_cast<T>(1) },
             { static_cast<T>(2), static_cast<T>(0) } });

    // g(t): 1 on [0,1), 0 on [1,2), 0 afterwards
    PcfT g({ { static_cast<T>(0), static_cast<T>(1) },
             { static_cast<T>(1), static_cast<T>(0) },
             { static_cast<T>(2), static_cast<T>(0) } });

    PcfT sum = f + g;
    PcfT diff = f - g;
    PcfT scaledUp = f * static_cast<T>(2);
    PcfT scaledDown = f / static_cast<T>(2);

    ASSERT_EQ(sum.size(), 2u);
    ASSERT_EQ(diff.size(), 3u);
    ASSERT_EQ(scaledUp.size(), f.size());
    ASSERT_EQ(scaledDown.size(), f.size());

    EXPECT_EQ(sum.points()[0], Pt(static_cast<T>(0), static_cast<T>(1)));  // 0+1 and 1+0
    EXPECT_EQ(sum.points()[1], Pt(static_cast<T>(2), static_cast<T>(0)));  // 0+0

    EXPECT_EQ(diff.points()[0], Pt(static_cast<T>(0), static_cast<T>(-1))); // 0-1
    EXPECT_EQ(diff.points()[1], Pt(static_cast<T>(1), static_cast<T>(1)));  // 1-0
    EXPECT_EQ(diff.points()[2], Pt(static_cast<T>(2), static_cast<T>(0)));  // 0-0

    EXPECT_EQ(scaledUp.points()[0], Pt(static_cast<T>(0), static_cast<T>(0)));
    EXPECT_EQ(scaledUp.points()[1], Pt(static_cast<T>(1), static_cast<T>(2)));
    EXPECT_EQ(scaledUp.points()[2], Pt(static_cast<T>(2), static_cast<T>(0)));

    EXPECT_EQ(scaledDown.points()[0], Pt(static_cast<T>(0), static_cast<T>(0)));
    EXPECT_EQ(scaledDown.points()[1], Pt(static_cast<T>(1), static_cast<T>(0.5)));
    EXPECT_EQ(scaledDown.points()[2], Pt(static_cast<T>(2), static_cast<T>(0)));
  }

  TYPED_TEST(PcfAlgebraAndNormsTest, AverageForSmallExample)
  {
    using PcfT = TypeParam;
    using T = typename PcfT::value_type;

    PcfT f1({ { static_cast<T>(0), static_cast<T>(1) },
              { static_cast<T>(1), static_cast<T>(0) } });
    PcfT f2({ { static_cast<T>(0), static_cast<T>(3) },
              { static_cast<T>(1), static_cast<T>(0) } });

    std::vector<PcfT> pcfs{ f1, f2 };

    auto avg = sb::average(pcfs);

    // Average should correspond to (f1+f2)/2
    PcfT manual = (f1 + f2) / static_cast<T>(2);

    EXPECT_EQ(avg, manual);
  }

  // ============================================================================
  // Norms: L1, L2, Lp, Linf
  // ============================================================================

  TYPED_TEST(PcfAlgebraAndNormsTest, NormsOnSimplePcf)
  {
    using PcfT = TypeParam;
    using T = typename PcfT::value_type;

    // f(t) = 1 on [0,1), 2 on [1,2), 0 afterwards
    PcfT f({ { static_cast<T>(0), static_cast<T>(1) },
             { static_cast<T>(1), static_cast<T>(2) },
             { static_cast<T>(2), static_cast<T>(0) } });

    auto l1 = sb::l1_norm(f);
    auto l2 = sb::l2_norm(f);
    auto lp = sb::lp_norm(f, static_cast<T>(2));
    auto linf = sb::linfinity_norm(f);

    T expectedL1 = static_cast<T>(1) * static_cast<T>(1) +
                   static_cast<T>(1) * static_cast<T>(2); // 3
    T expectedL2Sq = static_cast<T>(1) * static_cast<T>(1) * static_cast<T>(1) +
                     static_cast<T>(1) * static_cast<T>(4); // 5

    EXPECT_EQ(l1, expectedL1);
    EXPECT_NEAR(static_cast<double>(l2*l2),
                expectedL2Sq,
                1e-9);
    EXPECT_NEAR(static_cast<double>(lp),
                static_cast<double>(l2),
                1e-9);
    EXPECT_EQ(linf, static_cast<T>(2));
  }

  TYPED_TEST(PcfAlgebraAndNormsTest, ApplyFunctionalWithL1NormMultiplePcfs)
  {
    using PcfT = TypeParam;
    using T = typename PcfT::value_type;

    PcfT f0; // default PCF: identically zero
    PcfT f1({ { static_cast<T>(0), static_cast<T>(2) },
              { static_cast<T>(1), static_cast<T>(0) } });
    PcfT f2({ { static_cast<T>(0), static_cast<T>(2) },
              { static_cast<T>(1), static_cast<T>(1) },
              { static_cast<T>(3), static_cast<T>(0) } });

    std::vector<PcfT> fs{ f0, f1, f2 };
    std::vector<T> output(fs.size());

    sb::apply_functional(fs.begin(), fs.end(), output.begin(), [](const PcfT& f) {
      return sb::l1_norm(f);
    });

    EXPECT_EQ(output[0], static_cast<T>(0));
    EXPECT_EQ(output[1], static_cast<T>(2));
    EXPECT_EQ(output[2], static_cast<T>(4)); // 1*2 + 2*1 = 4
  }

} // namespace

