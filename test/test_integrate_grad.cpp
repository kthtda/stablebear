#include <gtest/gtest.h>

#include <sbear/algorithms/functional/integrate_grad.hpp>

#include <vector>

namespace
{
  using Pcf = sb::Pcf<double, double>;

  // The weights of the breakpoints (T_i, F_i) of an integral in a number
  // L = sum of a_i T_i + b_i F_i.
  std::vector<double> weights(size_t n)
  {
    std::vector<double> g;
    for (size_t i = 0; i < n; ++i)
    {
      g.push_back(0.5 + 0.25 * static_cast<double>(i));
      g.push_back(1.5 - 0.75 * static_cast<double>(i));
    }
    return g;
  }

  double weighted(const Pcf& f, const std::vector<double>& g)
  {
    const auto integral = sb::integrate(f);
    double total = 0.;
    for (size_t i = 0; i < integral.points().size(); ++i)
    {
      total += g[2 * i] * integral.points()[i].t + g[2 * i + 1] * integral.points()[i].v;
    }
    return total;
  }

  Pcf pcf(const std::vector<double>& points)
  {
    std::vector<Pcf::point_type> pts;
    for (size_t i = 0; i < points.size(); i += 2)
    {
      pts.emplace_back(points[i], points[i + 1]);
    }
    return Pcf(std::move(pts));
  }

  // The gradient matches central differences in every time but the first,
  // which stays 0, and in every value.
  void expect_gradient_matches_differences(const std::vector<double>& points)
  {
    const auto f = pcf(points);
    const auto g = weights(sb::integral_size(f));
    std::vector<double> grad(points.size());
    sb::integrate_grad(f, g.data(), grad.data());

    const double eps = 1e-6;
    for (size_t k = 1; k < points.size(); ++k)
    {
      auto up = points;
      auto down = points;
      up[k] += eps;
      down[k] -= eps;
      EXPECT_NEAR(grad[k], (weighted(pcf(up), g) - weighted(pcf(down), g)) / (2 * eps), 1e-6) << "entry " << k;
    }
  }
}

TEST(IntegrateGrad, IntegralHasOneMoreBreakpointWhenLastValueDiffers)
{
  EXPECT_EQ(sb::integral_size(pcf({0., 2., 1., -1., 3., 0.5})), 4u);
  EXPECT_EQ(sb::integral_size(pcf({0., 2., 1., 0.5, 3., 0.5})), 3u);
  EXPECT_EQ(sb::integral_size(pcf({0., 2.})), 2u);
  EXPECT_EQ(sb::integral_size(pcf({0., 0.})), 1u);
}

TEST(IntegrateGrad, SizeIsThatOfTheIntegral)
{
  for (const auto& points : std::vector<std::vector<double>>{{0., 2., 1., -1., 3., 0.5}, {0., 2., 1., 0.5, 3., 0.5}, {0., 2.}, {0., 0.}})
  {
    const auto f = pcf(points);
    EXPECT_EQ(sb::integral_size(f), sb::integrate(f).points().size());
  }
}

TEST(IntegrateGrad, MatchesDifferencesWhenLastValueDiffers)
{
  expect_gradient_matches_differences({0., 2., 1., -1., 3., 0.5});
}

TEST(IntegrateGrad, MatchesDifferencesWhenLastValueContinues)
{
  expect_gradient_matches_differences({0., 2., 1., 0.5, 3., 0.5, 4.5, 0.5});
}

TEST(IntegrateGrad, MatchesDifferencesWithLastTimeBelowOne)
{
  expect_gradient_matches_differences({0., 2., 0.25, -1., 0.5, 3.});
}

TEST(IntegrateGrad, MatchesDifferencesForOneBreakpoint)
{
  expect_gradient_matches_differences({0., 2.});
}
