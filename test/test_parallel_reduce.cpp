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

// Regression for issue #186: an empty range used to reach subdivide's
// SIZE_MAX-wrapped block and read far out of bounds.
TEST(ParallelReduce, EmptyRangeThrows)
{
  std::vector<sb::Pcf_f64> pcfs;

  auto op = [](const typename sb::Pcf_f64::rectangle_type& rect) {
    return rect.f_value + rect.g_value;
  };

  EXPECT_THROW(sb::parallel_reduce(pcfs.begin(), pcfs.end(), op), std::invalid_argument);
}

TEST(ParallelReduce, AverageOfEmptyVectorThrows)
{
  std::vector<sb::Pcf_f64> pcfs;
  EXPECT_THROW(sb::average(pcfs), std::invalid_argument);
}
