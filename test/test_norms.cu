#include <gtest/gtest.h>

#include <sbear/functional/pcf.hpp>
#include <sbear/algorithms/functional/apply_functional.hpp>

#include <vector>
#include <memory>

#ifndef __CUDACC__

namespace
{
  TEST(LpNorm, EmptyPcfHasZeroNorm)
  {
    sb::Pcf_f32 f;
    EXPECT_EQ(sb::l1_norm(f), 0.f);

  }

  TEST(LpNorm, TwoPointPcf)
  {
    sb::Pcf_f32 f({ {0.f, 2.f}, {1.5f, 0.f} });

    EXPECT_FLOAT_EQ(sb::l1_norm(f), 3.f);
  }

  TEST(LpNorm, FourPointPcf)
  {
    sb::Pcf_f32 f({ {0.f, 2.f}, {1.5f, 1.f}, {3.f, 7.f}, {4.f, 0.f} });

    EXPECT_FLOAT_EQ(sb::l1_norm(f), 11.5f);
  }

  TEST(LpNorm, ApplyL1Norm)
  {
    sb::Pcf_f32 f0;
    sb::Pcf_f32 f1({ {0.f, 2.f}, {1.5f, 0.f} });
    sb::Pcf_f32 f2({ {0.f, 2.f}, {1.5f, 1.f}, {3.f, 7.f}, {4.f, 0.f} });
    std::vector<sb::Pcf_f32> fs{ f0, f1, f2 };
    std::vector<float> output;
    output.resize(3);

    sb::apply_functional(fs.begin(), fs.end(), output.begin(), sb::l1_norm<sb::Pcf_f32>);

    EXPECT_FLOAT_EQ(output[0], 0.f);
    EXPECT_FLOAT_EQ(output[1], 3.f);
    EXPECT_FLOAT_EQ(output[2], 11.5f);
  }
}

#endif
