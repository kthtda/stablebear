#include <gtest/gtest.h>

#include <sbear/point_process/subsample.hpp>
#include <sbear/sampling/weighted_draw.hpp>

#include <vector>

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

  TEST(SubsampleTest, CdfBoundaryTargetsSelectPositiveWeightIndices)
  {
    // Probabilities [0, 1/4, 0, 3/4, 0]. Index i owns the targets in
    // [cdf[i - 1], cdf[i]), with cdf[-1] = 0:
    //   index 0: [0, 0)       empty
    //   index 1: [0, 0.25)
    //   index 2: [0.25, 0.25) empty
    //   index 3: [0.25, 1)
    //   index 4: [1, 1)       empty
    const std::vector<double> cdf{0, 0.25, 0.25, 1, 1};

    // A target of 1 selects the last positive weight, index 3.
    EXPECT_EQ(sb::sampling::detail::index_for_target<double>(cdf, 1.0), 3);
    // A target of 0 selects the first positive weight, index 1.
    EXPECT_EQ(sb::sampling::detail::index_for_target<double>(cdf, 0.0), 1);
  }
}
