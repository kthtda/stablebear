#include <gtest/gtest.h>

#include <sbear/functional/pcf.hpp>
#include <sbear/algorithms/functional/reduce.hpp>

TEST(ParallelReduce, AddThreeFunctions)
{
  std::vector<sb::Pcf_f64> pcfs;

  pcfs.emplace_back(sb::Pcf_f64{ {0., 3.}, {1., 2.}, {4., 5.}, {6., 0.} });
  pcfs.emplace_back(sb::Pcf_f64{ {0., 2.}, {3., 4.}, {4., 2.}, {5., 1.}, {8., 3.} });
  pcfs.emplace_back(sb::Pcf_f64{ {0., 0.}, {3., 7.}, {5., 2.} });

  auto res = sb::parallel_reduce(pcfs.begin(), pcfs.end(), [](const typename sb::Pcf_f64::rectangle_type& rect) {
    return rect.f_value + rect.g_value;
    });

  EXPECT_EQ(res, (pcfs[0] + pcfs[1] + pcfs[2]));
}
