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

  TYPED_TEST(PointCloudTest, CopyMaterializesAnIndexedView)
  {
    using T = TypeParam;

    sb::Tensor<T> coordinates({3, 1});
    coordinates({0, 0}) = T{10};
    coordinates({1, 0}) = T{20};
    coordinates({2, 0}) = T{30};
    sb::Tensor<uint64_t> rows({2});
    rows(0) = 2;
    rows(1) = 0;

    const sb::PointCloud<T> view(coordinates, std::move(rows));
    const sb::PointCloud<T> copied = view.copy();

    EXPECT_TRUE(view.is_indexed());
    EXPECT_FALSE(copied.is_indexed());
    EXPECT_EQ(copied.n_points(), 2);
    EXPECT_EQ(copied(0, 0), T{30});
    EXPECT_EQ(copied(1, 0), T{10});
    EXPECT_NE(copied.coords().storage_owner(), coordinates.storage_owner());
  }
}
