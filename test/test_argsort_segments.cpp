#include <gtest/gtest.h>

#include <sbear/algorithms/argsort_segments.hpp>

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace
{
  std::vector<int64_t> argsort(const std::vector<double>& values, int64_t start)
  {
    std::vector<int64_t> order(values.size());
    sb::argsort_segment(std::span<const double>(values), start, std::span<int64_t>(order));
    return order;
  }
}

TEST(ArgsortSegment, OrdersPositionsByValue)
{
  EXPECT_EQ(argsort({3., 1., 2.}, 0), (std::vector<int64_t>{1, 2, 0}));
}

TEST(ArgsortSegment, PositionsCountFromStart)
{
  EXPECT_EQ(argsort({3., 1., 2.}, 10), (std::vector<int64_t>{11, 12, 10}));
}

TEST(ArgsortSegment, EqualValuesKeepTheirOrder)
{
  EXPECT_EQ(argsort({2., 1., 2., 1.}, 0), (std::vector<int64_t>{1, 3, 0, 2}));
}

TEST(ArgsortSegment, InfiniteValueComesLast)
{
  EXPECT_EQ(argsort({std::numeric_limits<double>::infinity(), 1.}, 0), (std::vector<int64_t>{1, 0}));
}

TEST(ArgsortSegment, EmptySegment)
{
  EXPECT_TRUE(argsort({}, 5).empty());
}
