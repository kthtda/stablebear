#include <gtest/gtest.h>

#include <sbear/ufunc.hpp>

#include <span>
#include <stdexcept>
#include <vector>

namespace
{
  /// The sum and the product of a pair of numbers.
  struct SumAndProduct
  {
    using input_type = sb::ElementPair<double, double>;
    using output_type = double;

    [[nodiscard]] std::vector<size_t> extra_dims() const { return {2}; }

    void operator()(const input_type& in, std::span<double> out) const
    {
      out[0] = in.first + in.second;
      out[1] = in.first * in.second;
    }
  };

  /// The double of a number.
  struct Double
  {
    using input_type = double;
    using output_type = double;

    [[nodiscard]] std::vector<size_t> extra_dims() const { return {}; }

    void operator()(const double& in, std::span<double> out) const { out[0] = 2 * in; }
  };

  sb::Tensor<double> tensor(const std::vector<double>& values)
  {
    sb::Tensor<double> result(std::vector<size_t>{values.size()});
    for (size_t i = 0; i < values.size(); ++i)
    {
      result(std::vector<size_t>{i}) = values[i];
    }
    return result;
  }
}

TEST(UfuncZipped, OperationTakesOneElementOfEachTensor)
{
  sb::Zipped<sb::Tensor<double>, sb::Tensor<double>> pair{tensor({1., 2., 3.}), tensor({10., 20., 30.})};

  auto result = sb::apply_ufunc(pair, SumAndProduct{}, sb::default_executor());

  ASSERT_EQ(result.shape(), (std::vector<size_t>{3, 2}));
  EXPECT_DOUBLE_EQ(result(std::vector<size_t>{0, 0}), 11.);
  EXPECT_DOUBLE_EQ(result(std::vector<size_t>{0, 1}), 10.);
  EXPECT_DOUBLE_EQ(result(std::vector<size_t>{2, 0}), 33.);
  EXPECT_DOUBLE_EQ(result(std::vector<size_t>{2, 1}), 90.);
}

TEST(UfuncZipped, TensorsOfDifferentShapesThrow)
{
  sb::Zipped<sb::Tensor<double>, sb::Tensor<double>> pair{tensor({1., 2., 3.}), tensor({10., 20.})};

  EXPECT_THROW(sb::apply_ufunc(pair, SumAndProduct{}, sb::default_executor()), std::invalid_argument);
}

TEST(ApplyUfunc, GivesTheResultOfTheOperationOnEachElement)
{
  auto result = sb::apply_ufunc(tensor({1., 2., 3.}), Double{}, sb::default_executor());

  ASSERT_EQ(result.shape(), (std::vector<size_t>{3}));
  EXPECT_DOUBLE_EQ(result(std::vector<size_t>{0}), 2.);
  EXPECT_DOUBLE_EQ(result(std::vector<size_t>{2}), 6.);
}
