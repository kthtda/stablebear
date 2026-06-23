#include <gtest/gtest.h>

#include <sbear/tensor.hpp>

// Minimal C++ smoke tests for masked operations.
// Full coverage lives in test/python/test_tensor_mask.py.

namespace
{

  TEST(MaskedSelect, Smoke)
  {
    sb::Tensor<float> t({4});
    t({0}) = 10.0f; t({1}) = 20.0f; t({2}) = 30.0f; t({3}) = 40.0f;

    sb::Tensor<bool> mask({4});
    mask({0}) = true; mask({1}) = false; mask({2}) = true; mask({3}) = false;

    auto result = sb::masked_select(t, mask);
    ASSERT_EQ(result.shape(), (std::vector<size_t>{2}));
    EXPECT_FLOAT_EQ(result({0}), 10.0f);
    EXPECT_FLOAT_EQ(result({1}), 30.0f);
  }

  TEST(MaskedAssign, Smoke)
  {
    sb::Tensor<float> t({3}, 0.0f);
    sb::Tensor<bool> mask({3});
    mask({0}) = false; mask({1}) = true; mask({2}) = false;

    sb::Tensor<float> values({1});
    values({0}) = 99.0f;

    sb::masked_assign(t, mask, values);
    EXPECT_FLOAT_EQ(t({0}), 0.0f);
    EXPECT_FLOAT_EQ(t({1}), 99.0f);
    EXPECT_FLOAT_EQ(t({2}), 0.0f);
  }

  TEST(MaskedFill, Smoke)
  {
    sb::Tensor<float> t({3});
    t({0}) = 1.0f; t({1}) = 2.0f; t({2}) = 3.0f;

    sb::Tensor<bool> mask({3}, true);
    sb::masked_fill(t, mask, -1.0f);

    EXPECT_FLOAT_EQ(t({0}), -1.0f);
    EXPECT_FLOAT_EQ(t({1}), -1.0f);
    EXPECT_FLOAT_EQ(t({2}), -1.0f);
  }

  TEST(AxisSelect, Smoke2D)
  {
    // 2x3 tensor, select along axis 1 with mask [T, F, T]
    sb::Tensor<float> t({2, 3});
    t({0, 0}) = 0; t({0, 1}) = 1; t({0, 2}) = 2;
    t({1, 0}) = 3; t({1, 1}) = 4; t({1, 2}) = 5;

    sb::Tensor<bool> mask({3});
    mask({0}) = true; mask({1}) = false; mask({2}) = true;

    auto result = sb::axis_select(t, 1, mask);
    ASSERT_EQ(result.shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(result({0, 0}), 0);
    EXPECT_FLOAT_EQ(result({0, 1}), 2);
    EXPECT_FLOAT_EQ(result({1, 0}), 3);
    EXPECT_FLOAT_EQ(result({1, 1}), 5);
  }

  TEST(AxisAssign, Smoke2D)
  {
    sb::Tensor<float> t({2, 3}, 0.0f);
    sb::Tensor<bool> mask({3});
    mask({0}) = true; mask({1}) = false; mask({2}) = true;

    sb::Tensor<float> values({2, 2});
    values({0, 0}) = 10; values({0, 1}) = 20;
    values({1, 0}) = 30; values({1, 1}) = 40;

    sb::axis_assign(t, 1, mask, values);
    EXPECT_FLOAT_EQ(t({0, 0}), 10);
    EXPECT_FLOAT_EQ(t({0, 1}), 0);
    EXPECT_FLOAT_EQ(t({0, 2}), 20);
    EXPECT_FLOAT_EQ(t({1, 0}), 30);
    EXPECT_FLOAT_EQ(t({1, 1}), 0);
    EXPECT_FLOAT_EQ(t({1, 2}), 40);
  }

  TEST(AxisFill, Smoke2D)
  {
    sb::Tensor<float> t({2, 3}, 0.0f);
    sb::Tensor<bool> mask({3});
    mask({0}) = false; mask({1}) = true; mask({2}) = false;

    sb::axis_fill(t, 1, mask, 99.0f);
    EXPECT_FLOAT_EQ(t({0, 0}), 0);
    EXPECT_FLOAT_EQ(t({0, 1}), 99);
    EXPECT_FLOAT_EQ(t({0, 2}), 0);
    EXPECT_FLOAT_EQ(t({1, 0}), 0);
    EXPECT_FLOAT_EQ(t({1, 1}), 99);
    EXPECT_FLOAT_EQ(t({1, 2}), 0);
  }

} // namespace
