#include <gtest/gtest.h>

#include <sbear/tensor.hpp>
#include <sbear/walk.hpp>

namespace
{

  template<typename T>
  sb::Tensor<T> make_sequential(const std::vector<size_t>& shape)
  {
    sb::Tensor<T> t(shape);
    size_t n = 0;
    sb::walk(t, [&t, &n](const std::vector<size_t>& idx)
    {
      t(idx) = static_cast<T>(n++);
    });
    return t;
  }

// ============================================================================
// broadcast_shapes()
// ============================================================================

  TEST(BroadcastShapes, SameShape)
  {
    auto result = sb::broadcast_shapes({3, 4}, {3, 4});
    EXPECT_EQ(result, (std::vector<size_t>{3, 4}));
  }

  TEST(BroadcastShapes, ScalarAndVector)
  {
    auto result = sb::broadcast_shapes({1}, {5});
    EXPECT_EQ(result, (std::vector<size_t>{5}));
  }

  TEST(BroadcastShapes, RankMismatch)
  {
    auto result = sb::broadcast_shapes({3, 4}, {4});
    EXPECT_EQ(result, (std::vector<size_t>{3, 4}));
  }

  TEST(BroadcastShapes, BothExpand)
  {
    auto result = sb::broadcast_shapes({2, 1}, {1, 3});
    EXPECT_EQ(result, (std::vector<size_t>{2, 3}));
  }

  TEST(BroadcastShapes, ThreeDims)
  {
    auto result = sb::broadcast_shapes({1, 3, 1}, {2, 1, 4});
    EXPECT_EQ(result, (std::vector<size_t>{2, 3, 4}));
  }

  TEST(BroadcastShapes, IncompatibleThrows)
  {
    EXPECT_THROW(sb::broadcast_shapes({3}, {4}), std::invalid_argument);
  }

  TEST(BroadcastShapes, IncompatibleMultiDimThrows)
  {
    EXPECT_THROW(sb::broadcast_shapes({2, 3}, {2, 4}), std::invalid_argument);
  }

// ============================================================================
// broadcast_to()
// ============================================================================

  TEST(BroadcastTo, SameShapePreservesStrides)
  {
    auto t = make_sequential<double>({3, 4});
    auto view = t.broadcast_to({3, 4});
    EXPECT_EQ(view.shape(), t.shape());
    EXPECT_EQ(view.strides(), t.strides());
  }

  TEST(BroadcastTo, ExpandSize1Dim)
  {
    sb::Tensor<double> t({1, 3});
    t({0, 0}) = 10; t({0, 1}) = 20; t({0, 2}) = 30;
    auto view = t.broadcast_to({4, 3});
    EXPECT_EQ(view.shape(), (std::vector<size_t>{4, 3}));
    EXPECT_EQ(view.stride(0), 0u);
    EXPECT_EQ(view.stride(1), 1u);
    for (size_t i = 0; i < 4; ++i)
    {
      EXPECT_EQ(view({i, 0}), 10);
      EXPECT_EQ(view({i, 1}), 20);
      EXPECT_EQ(view({i, 2}), 30);
    }
  }

  TEST(BroadcastTo, PrependDim)
  {
    sb::Tensor<double> t({3});
    t(0) = 1; t(1) = 2; t(2) = 3;
    auto view = t.broadcast_to({5, 3});
    EXPECT_EQ(view.shape(), (std::vector<size_t>{5, 3}));
    EXPECT_EQ(view.stride(0), 0u);
    for (size_t i = 0; i < 5; ++i)
      for (size_t j = 0; j < 3; ++j)
        EXPECT_EQ(view({i, j}), t(j));
  }

  TEST(BroadcastTo, IsView)
  {
    sb::Tensor<double> t({3});
    t(0) = 1; t(1) = 2; t(2) = 3;
    auto view = t.broadcast_to({2, 3});
    EXPECT_EQ(view.data(), t.data());
    EXPECT_FALSE(view.is_contiguous());
  }

  TEST(BroadcastTo, IncompatibleThrows)
  {
    sb::Tensor<double> t({3, 4});
    EXPECT_THROW(t.broadcast_to({3, 5}), std::invalid_argument);
  }

  TEST(BroadcastTo, FewerDimsThrows)
  {
    sb::Tensor<double> t({3, 4});
    EXPECT_THROW(t.broadcast_to({4}), std::invalid_argument);
  }

// ============================================================================
// Tensor-Tensor arithmetic with broadcasting
// ============================================================================

  using FloatTypes = ::testing::Types<float, double>;

  template <typename T>
  class TensorBroadcastTyped : public ::testing::Test {};
  TYPED_TEST_SUITE(TensorBroadcastTyped, FloatTypes);

  TYPED_TEST(TensorBroadcastTyped, AddSameShape)
  {
    using T = TypeParam;
    sb::Tensor<T> a({3});
    a(0) = T(1); a(1) = T(2); a(2) = T(3);
    sb::Tensor<T> b({3});
    b(0) = T(10); b(1) = T(20); b(2) = T(30);
    auto result = a + b;
    EXPECT_EQ(result(0), T(11));
    EXPECT_EQ(result(1), T(22));
    EXPECT_EQ(result(2), T(33));
  }

  TYPED_TEST(TensorBroadcastTyped, AddBroadcast)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({3, 4});
    sb::Tensor<T> b({4});
    b(0) = T(100); b(1) = T(200); b(2) = T(300); b(3) = T(400);
    auto result = a + b;
    EXPECT_EQ(result.shape(), (std::vector<size_t>{3, 4}));
    EXPECT_EQ(result({0, 0}), T(100));
    EXPECT_EQ(result({0, 3}), T(403));
    EXPECT_EQ(result({2, 0}), T(108));
  }

  TYPED_TEST(TensorBroadcastTyped, AddBroadcastBothExpand)
  {
    using T = TypeParam;
    sb::Tensor<T> a({2, 1});
    a({0, 0}) = T(1); a({1, 0}) = T(2);
    sb::Tensor<T> b({1, 3});
    b({0, 0}) = T(10); b({0, 1}) = T(20); b({0, 2}) = T(30);
    auto result = a + b;
    EXPECT_EQ(result.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(result({0, 0}), T(11));
    EXPECT_EQ(result({0, 2}), T(31));
    EXPECT_EQ(result({1, 0}), T(12));
    EXPECT_EQ(result({1, 2}), T(32));
  }

  TYPED_TEST(TensorBroadcastTyped, SubtractBroadcast)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({3, 4});
    sb::Tensor<T> b({4});
    b(0) = T(0); b(1) = T(1); b(2) = T(2); b(3) = T(3);
    auto result = a - b;
    EXPECT_EQ(result({0, 0}), T(0));
    EXPECT_EQ(result({0, 3}), T(0));
    EXPECT_EQ(result({1, 0}), T(4));
  }

  TYPED_TEST(TensorBroadcastTyped, MultiplyBroadcast)
  {
    using T = TypeParam;
    sb::Tensor<T> a({2, 3});
    a({0, 0}) = T(1); a({0, 1}) = T(2); a({0, 2}) = T(3);
    a({1, 0}) = T(4); a({1, 1}) = T(5); a({1, 2}) = T(6);
    sb::Tensor<T> b({1, 3});
    b({0, 0}) = T(10); b({0, 1}) = T(20); b({0, 2}) = T(30);
    auto result = a * b;
    EXPECT_EQ(result({0, 0}), T(10));
    EXPECT_EQ(result({0, 2}), T(90));
    EXPECT_EQ(result({1, 1}), T(100));
  }

  TYPED_TEST(TensorBroadcastTyped, DivideBroadcast)
  {
    using T = TypeParam;
    sb::Tensor<T> a({2, 2});
    a({0, 0}) = T(10); a({0, 1}) = T(20);
    a({1, 0}) = T(30); a({1, 1}) = T(40);
    sb::Tensor<T> b({2});
    b(0) = T(2); b(1) = T(5);
    auto result = a / b;
    EXPECT_EQ(result({0, 0}), T(5));
    EXPECT_EQ(result({0, 1}), T(4));
    EXPECT_EQ(result({1, 0}), T(15));
    EXPECT_EQ(result({1, 1}), T(8));
  }

  TYPED_TEST(TensorBroadcastTyped, CompoundAddBroadcast)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({3, 4});
    sb::Tensor<T> b({4});
    b(0) = T(100); b(1) = T(200); b(2) = T(300); b(3) = T(400);
    a += b;
    EXPECT_EQ(a({0, 0}), T(100));
    EXPECT_EQ(a({0, 3}), T(403));
    EXPECT_EQ(a({2, 0}), T(108));
  }

  TYPED_TEST(TensorBroadcastTyped, CompoundAddIncompatibleShapeThrows)
  {
    using T = TypeParam;
    sb::Tensor<T> a({4});
    sb::Tensor<T> b({3, 4});
    EXPECT_THROW(a += b, std::invalid_argument);
  }

  TYPED_TEST(TensorBroadcastTyped, ScalarTensorBroadcast)
  {
    using T = TypeParam;
    sb::Tensor<T> a({1});
    a(0) = T(5);
    sb::Tensor<T> b({3});
    b(0) = T(1); b(1) = T(2); b(2) = T(3);
    auto result = a + b;
    EXPECT_EQ(result.shape(), (std::vector<size_t>{3}));
    EXPECT_EQ(result(0), T(6));
    EXPECT_EQ(result(1), T(7));
    EXPECT_EQ(result(2), T(8));
  }

} // namespace
