#include <gtest/gtest.h>

#include <sbear/point_cloud.hpp>
#include <sbear/tensor.hpp>

namespace
{
  template <typename T>
  class PointCloudTest : public ::testing::Test { };

  using PointCloudTypes = ::testing::Types<float, double>;
  TYPED_TEST_SUITE(PointCloudTest, PointCloudTypes);

  TYPED_TEST(PointCloudTest, RejectsCoordinatesWithoutTwoDimensions)
  {
    using T = TypeParam;

    const sb::Tensor<T> oneDimensional({3});
    EXPECT_THROW((void)sb::PointCloud<T>{oneDimensional}, std::invalid_argument);
    EXPECT_THROW(
      (void)sb::PointCloud<T>{sb::Tensor<T>({3})}, std::invalid_argument);
    EXPECT_THROW(
      (void)sb::PointCloud<T>{std::vector<size_t>{0}}, std::invalid_argument);

    EXPECT_NO_THROW((void)sb::PointCloud<T>{});
    EXPECT_NO_THROW(((void)sb::PointCloud<T>{std::vector<size_t>{0, 2}}));
  }
}
