#include <gtest/gtest.h>

#include <sbear/point_process/subsample.hpp>

namespace
{
  template <typename T>
  class SubsampleTest : public ::testing::Test { };

  using CoordinateTypes = ::testing::Types<float, double>;
  TYPED_TEST_SUITE(SubsampleTest, CoordinateTypes);

  TEST(SubsampleTest, IndexFilterPreservesFirstOccurrenceOrder)
  {
    sb::Tensor<uint64_t> unfiltered_indices({5});
    unfiltered_indices(0) = 2;
    unfiltered_indices(1) = 1;
    unfiltered_indices(2) = 2;
    unfiltered_indices(3) = 0;
    unfiltered_indices(4) = 1;

    std::vector<uint64_t> scratch;
    const auto filtered_indices = sb::pp::detail::discard_duplicate_indices(
      unfiltered_indices, scratch);

    // Keep [2, 1, 0], removing repeats without sorting the draw.
    ASSERT_EQ(filtered_indices.size(), 3);
    EXPECT_EQ(filtered_indices(0), 2);
    EXPECT_EQ(filtered_indices(1), 1);
    EXPECT_EQ(filtered_indices(2), 0);
  }

  TYPED_TEST(SubsampleTest, CoordinateFilterPreservesFirstOccurrenceOrder)
  {
    sb::PointCloud<TypeParam> source(std::vector<size_t>{3, 1});
    source(0, 0) = 10;
    source(1, 0) = 10;
    source(2, 0) = 20;

    sb::Tensor<uint64_t> unfiltered_indices({3});
    unfiltered_indices(0) = 2;
    unfiltered_indices(1) = 1;
    unfiltered_indices(2) = 0;

    std::vector<uint64_t> scratch;
    const auto filtered_indices = sb::pp::detail::discard_duplicate_coordinates(
      source.coords(), unfiltered_indices, scratch);

    // Keep [2, 1]: point 0 repeats point 1's coordinates; do not sort the draw.
    ASSERT_EQ(filtered_indices.size(), 2);
    EXPECT_EQ(filtered_indices(0), 2);
    EXPECT_EQ(filtered_indices(1), 1);
  }
}
