#include <gtest/gtest.h>

#include <sbear/algorithms/multiply_points.hpp>

#include <span>
#include <stdexcept>
#include <vector>

namespace
{
  sb::PointCloud<double> cloud(const std::vector<std::vector<double>>& points, size_t dim)
  {
    sb::PointCloud<double> result(std::vector<size_t>{points.size(), dim});
    for (size_t i = 0; i < points.size(); ++i)
    {
      for (size_t j = 0; j < dim; ++j)
      {
        result(i, j) = points[i][j];
      }
    }
    return result;
  }

  // Swaps the two coordinates and doubles what becomes the first:
  // (x, y) -> (2 y, x).
  const std::vector<double> swapAndDouble{0., 1.,
                                          2., 0.};

  std::span<const double> matrix()
  {
    return std::span<const double>(swapAndDouble);
  }
}

TEST(MultiplyPoints, EachPointIsMultipliedAsARow)
{
  auto result = sb::multiply_points(cloud({{1., 10.}, {2., 20.}}, 2), matrix(), 2, 2);

  ASSERT_EQ(result.shape(), (std::vector<size_t>{2, 2}));
  EXPECT_DOUBLE_EQ(result(0, 0), 20.);
  EXPECT_DOUBLE_EQ(result(0, 1), 1.);
  EXPECT_DOUBLE_EQ(result(1, 0), 40.);
  EXPECT_DOUBLE_EQ(result(1, 1), 2.);
}

TEST(MultiplyPoints, MatrixWithFewerColumnsLowersTheDimension)
{
  // (x, y) -> x + y.
  const std::vector<double> sum{1., 1.};

  auto result = sb::multiply_points(cloud({{1., 10.}, {2., 20.}}, 2), std::span<const double>(sum), 2, 1);

  ASSERT_EQ(result.shape(), (std::vector<size_t>{2, 1}));
  EXPECT_DOUBLE_EQ(result(0, 0), 11.);
  EXPECT_DOUBLE_EQ(result(1, 0), 22.);
}

TEST(MultiplyPoints, EmptyCloudStaysEmpty)
{
  auto result = sb::multiply_points(cloud({}, 2), matrix(), 2, 2);

  EXPECT_EQ(result.n_points(), 0u);
}

TEST(MultiplyPoints, IndexedCloudMultipliesItsSelectedPoints)
{
  sb::Tensor<uint64_t> indices(std::vector<size_t>{2});
  indices(std::vector<size_t>{0}) = 2;
  indices(std::vector<size_t>{1}) = 0;
  sb::PointCloud<double> selected(cloud({{1., 10.}, {2., 20.}, {3., 30.}}, 2), indices);

  auto result = sb::multiply_points(selected, matrix(), 2, 2);

  ASSERT_EQ(result.n_points(), 2u);
  EXPECT_DOUBLE_EQ(result(0, 0), 60.);
  EXPECT_DOUBLE_EQ(result(0, 1), 3.);
  EXPECT_DOUBLE_EQ(result(1, 0), 20.);
  EXPECT_DOUBLE_EQ(result(1, 1), 1.);
}

TEST(MultiplyPoints, CloudOfAnotherDimensionThrows)
{
  EXPECT_THROW(sb::multiply_points(cloud({{1., 10., 100.}}, 3), matrix(), 2, 2), std::invalid_argument);
}

TEST(MultiplyPoints, TensorOfCloudsOfDifferentSizes)
{
  sb::Tensor<sb::PointCloud<double>> clouds(std::vector<size_t>{3});
  clouds(std::vector<size_t>{0}) = cloud({{1., 10.}}, 2);
  clouds(std::vector<size_t>{1}) = cloud({{2., 20.}, {3., 30.}}, 2);
  clouds(std::vector<size_t>{2}) = cloud({}, 2);

  auto result = sb::multiply_points(clouds, matrix(), 2, 2, sb::default_executor());

  ASSERT_EQ(result.shape(), (std::vector<size_t>{3}));
  EXPECT_EQ(result(std::vector<size_t>{0}).n_points(), 1u);
  EXPECT_DOUBLE_EQ(result(std::vector<size_t>{0})(0, 0), 20.);
  EXPECT_EQ(result(std::vector<size_t>{1}).n_points(), 2u);
  EXPECT_DOUBLE_EQ(result(std::vector<size_t>{1})(1, 0), 60.);
  EXPECT_DOUBLE_EQ(result(std::vector<size_t>{1})(1, 1), 3.);
  EXPECT_EQ(result(std::vector<size_t>{2}).n_points(), 0u);
}
