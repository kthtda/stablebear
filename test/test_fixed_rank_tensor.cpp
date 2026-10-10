#include <gtest/gtest.h>

#include <sbear/fixed_rank_tensor.hpp>

TEST(FixedRankTensor, StoresShapeAndUsesRowMajorIndexing)
{
  sb::FixedRankTensor<int, 2> values({2, 3});
  values(1, 2) = 7;

  static_assert(decltype(values)::rank() == 2);
  EXPECT_EQ(values.shape(), (decltype(values)::shape_type{2, 3}));
  EXPECT_EQ(values.size(), 6);
  EXPECT_EQ(values.flat(5), 7);
  EXPECT_THROW((void)values(2, 0), std::out_of_range);
  size_t index = 0;
  for (const auto& value : values.flat_view())
    EXPECT_EQ(&value, values.storage_data() + index++);
  EXPECT_EQ(index, values.size());
}

TEST(FixedRankTensor, CopiesShareStorageAndExplicitCopyIsIndependent)
{
  sb::FixedRankTensor<double, 1> values({3});
  values(1) = 4.0;

  auto shared = values;
  auto independent = values.copy();
  shared(1) = 8.0;

  EXPECT_EQ(values(1), 8.0);
  EXPECT_EQ(independent(1), 4.0);
  EXPECT_EQ(values.storage_data(), shared.storage_data());
  EXPECT_NE(values.storage_data(), independent.storage_data());
}

TEST(FixedRankTensor, DefaultShapeHasZeroExtentOnEveryAxis)
{
  using Tensor = sb::FixedRankTensor<float, 2>;
  const Tensor values;
  EXPECT_EQ(values.shape(), (Tensor::shape_type{0, 0}));
  EXPECT_EQ(values.size(), 0);
  const auto flat = values.flat_view();
  EXPECT_EQ(flat.begin(), flat.end());
}

TEST(FixedRankTensor, ScalarFlatViewContainsOneValue)
{
  const sb::FixedRankTensor<int, 0> scalar;
  const auto flat = scalar.flat_view();
  auto it = flat.begin();
  EXPECT_EQ(&*it, scalar.storage_data());
  ++it;
  EXPECT_EQ(it, flat.end());
}
