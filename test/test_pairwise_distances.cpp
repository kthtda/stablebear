#include <gtest/gtest.h>

#include <sbear/algorithms/pairwise_distances.hpp>
#include <sbear/ufunc.hpp>

#include <cmath>
#include <limits>
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

  // The points (0, 0) and (3, 4), and (3, 0) between them.
  sb::PointCloud<double> triangle()
  {
    return cloud({{0., 0.}, {3., 4.}, {3., 0.}}, 2);
  }

  const double infinity = std::numeric_limits<double>::infinity();
}

TEST(PairwiseLpDistances, TaxicabDistancesAddTheCoordinates)
{
  auto distances = sb::pairwise_lp_distances(triangle(), 1.);

  EXPECT_EQ(distances.size(), 3u);
  EXPECT_DOUBLE_EQ(distances(1, 0), 7.);
  EXPECT_DOUBLE_EQ(distances(2, 0), 3.);
  EXPECT_DOUBLE_EQ(distances(2, 1), 4.);
}

TEST(PairwiseLpDistances, EuclideanDistances)
{
  auto distances = sb::pairwise_lp_distances(triangle(), 2.);

  EXPECT_DOUBLE_EQ(distances(1, 0), 5.);
  EXPECT_DOUBLE_EQ(distances(2, 0), 3.);
  EXPECT_DOUBLE_EQ(distances(2, 1), 4.);
}

TEST(PairwiseLpDistances, CubicDistances)
{
  auto distances = sb::pairwise_lp_distances(triangle(), 3.);

  // 3^3 + 4^3 = 91.
  EXPECT_DOUBLE_EQ(distances(1, 0), std::cbrt(91.));
  EXPECT_DOUBLE_EQ(distances(2, 0), 3.);
}

TEST(PairwiseLpDistances, InfinityTakesTheLargestCoordinate)
{
  auto distances = sb::pairwise_lp_distances(triangle(), infinity);

  EXPECT_DOUBLE_EQ(distances(1, 0), 4.);
  EXPECT_DOUBLE_EQ(distances(2, 0), 3.);
  EXPECT_DOUBLE_EQ(distances(2, 1), 4.);
}

TEST(PairwiseLpDistances, EmptyCloudGivesEmptyMatrix)
{
  EXPECT_EQ(sb::pairwise_lp_distances(cloud({}, 2), 1.).size(), 0u);
}

TEST(PairwiseLpDistances, ExponentBelowOneThrows)
{
  EXPECT_THROW(sb::pairwise_lp_distances(triangle(), 0.5), std::invalid_argument);
  EXPECT_THROW(sb::pairwise_lp_distances(triangle(), std::nan("")), std::invalid_argument);
}

TEST(PairwiseLpDistances, OperationWithExponentBelowOneThrows)
{
  EXPECT_THROW(sb::PairwiseLpDistancesOperation<double>(0.5), std::invalid_argument);
}

TEST(PairwiseLpDistances, TensorOfCloudsOfDifferentSizes)
{
  sb::Tensor<sb::PointCloud<double>> clouds(std::vector<size_t>{3});
  clouds(std::vector<size_t>{0}) = triangle();
  clouds(std::vector<size_t>{1}) = cloud({{0., 0.}, {1., 2.}}, 2);
  clouds(std::vector<size_t>{2}) = cloud({}, 2);

  auto distances = sb::apply_ufunc(clouds, sb::PairwiseLpDistancesOperation<double>(1.), sb::default_executor());

  ASSERT_EQ(distances.shape(), (std::vector<size_t>{3}));
  EXPECT_EQ(distances(std::vector<size_t>{0}).size(), 3u);
  EXPECT_DOUBLE_EQ(distances(std::vector<size_t>{0})(1, 0), 7.);
  EXPECT_EQ(distances(std::vector<size_t>{1}).size(), 2u);
  EXPECT_DOUBLE_EQ(distances(std::vector<size_t>{1})(1, 0), 3.);
  EXPECT_EQ(distances(std::vector<size_t>{2}).size(), 0u);
}
