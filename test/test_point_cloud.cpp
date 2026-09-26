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
    EXPECT_THROW(
      (void)sb::PointCloud<T>{sb::Tensor<T>({2, 3, 4})}, std::invalid_argument);

    EXPECT_NO_THROW((void)sb::PointCloud<T>{});
    EXPECT_NO_THROW(((void)sb::PointCloud<T>{std::vector<size_t>{0, 2}}));
    EXPECT_EQ(sb::PointCloud<T>{}.rank(), 2);
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

    sb::PointCloud<T> source(coordinates);
    const sb::PointCloud<T> view = source.index_into(rows);
    const sb::PointCloud<T> copied = view.copy();

    EXPECT_EQ(source.storage_data(), view.storage_data());
    EXPECT_EQ(source.n_points(), 3);
    EXPECT_EQ(view.n_points(), 2);
    source(2, 0) = T{40};
    EXPECT_EQ(view(0, 0), T{40});

    EXPECT_TRUE(view.is_indexed());
    EXPECT_FALSE(copied.is_indexed());
    EXPECT_EQ(copied.n_points(), 2);
    EXPECT_EQ(copied(0, 0), T{30});
    EXPECT_EQ(copied(1, 0), T{10});
    EXPECT_NE(copied.storage_data(), source.storage_data());
  }

  TYPED_TEST(PointCloudTest, CoordinateConstructionCopiesViews)
  {
    using T = TypeParam;

    sb::Tensor<T> coordinates({4, 2});
    for (size_t row = 0; row < 4; ++row)
    {
      for (size_t column = 0; column < 2; ++column)
      {
        coordinates({row, column}) = static_cast<T>(10 * row + column);
      }
    }

    const auto view = coordinates[std::vector<sb::Slice>{
      sb::range(0, 2, 1), sb::all()}];
    const sb::PointCloud<T> fromTensor(coordinates);
    const sb::PointCloud<T> fromLvalue(view);
    const sb::PointCloud<T> fromRvalue(
      coordinates[std::vector<sb::Slice>{
        sb::range(2, 4, 1), sb::all()}]);

    coordinates = T{-1};

    EXPECT_EQ(fromTensor(3, 1), T{31});
    EXPECT_EQ(fromLvalue(0, 0), T{0});
    EXPECT_EQ(fromLvalue(1, 1), T{11});
    EXPECT_EQ(fromRvalue(0, 0), T{20});
    EXPECT_EQ(fromRvalue(1, 1), T{31});
    EXPECT_NE(fromTensor.storage_data(), coordinates.data());
    EXPECT_NE(fromLvalue.storage_data(), coordinates.data());
    EXPECT_NE(fromRvalue.storage_data(), coordinates.data());
    EXPECT_NE(
      fromLvalue.coords().storage_data(), fromRvalue.coords().storage_data());
  }
}
