#include <gtest/gtest.h>

#include <sbear/tensor.hpp>
#include <sbear/walk.hpp>

static_assert(std::random_access_iterator<sb::Tensor1dValueIterator<sb::Tensor<double>>>);

TEST(TensorIteration, Iterate1dValues)
{
  sb::Tensor<int> x({3});
  x(0) = 1;
  x(1) = 2;
  x(2) = 3;

  auto begin = sb::begin1dValues(x);
  auto end = sb::end1dValues(x);

  EXPECT_EQ(*begin, 1);
  EXPECT_EQ(*(begin + 1), 2);
  EXPECT_EQ(*(begin + 2), 3);

  EXPECT_EQ(begin + 3, end);
}

TEST(TensorIteration, AxisIteration)
{
  sb::Tensor<int> x{{3, 3}};
  for (auto i = 0_uz; i < x.shape(0); ++i)
  {
    for (auto j = 0_uz; j < x.shape(1); ++j)
    {

    }
  }
  x({0, 0}) = 10;

}

// --- walk tests ---

TEST(TensorWalk, Walk1dOdometerOrder)
{
  sb::Tensor<int> x({4});
  std::vector<std::vector<size_t>> visited;
  sb::walk(x, [&](const std::vector<size_t>& idx) {
    visited.push_back(idx);
  });

  std::vector<std::vector<size_t>> expected = {{0}, {1}, {2}, {3}};
  EXPECT_EQ(visited, expected);
}

TEST(TensorWalk, Walk2dOdometerOrder)
{
  sb::Tensor<int> x({2, 3});
  std::vector<std::vector<size_t>> visited;
  sb::walk(x, [&](const std::vector<size_t>& idx) {
    visited.push_back(idx);
  });

  std::vector<std::vector<size_t>> expected = {
    {0, 0}, {0, 1}, {0, 2},
    {1, 0}, {1, 1}, {1, 2}
  };
  EXPECT_EQ(visited, expected);
}

TEST(TensorWalk, Walk3dOdometerOrder)
{
  sb::Tensor<int> x({2, 2, 2});
  std::vector<std::vector<size_t>> visited;
  sb::walk(x, [&](const std::vector<size_t>& idx) {
    visited.push_back(idx);
  });

  std::vector<std::vector<size_t>> expected = {
    {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
    {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}
  };
  EXPECT_EQ(visited, expected);
}

TEST(TensorWalk, WalkEmptyTensor)
{
  sb::Tensor<int> x({0});
  size_t count = 0;
  sb::walk(x, [&](const std::vector<size_t>&) { ++count; });
  EXPECT_EQ(count, 0);
}

TEST(TensorWalk, WalkEmptyShape)
{
  sb::Tensor<int> x(std::vector<size_t>{});
  size_t count = 0;
  sb::walk(x, [&](const std::vector<size_t>&) { ++count; });
  EXPECT_EQ(count, 0);
}

TEST(TensorWalk, WalkBoolEarlyTermination)
{
  sb::Tensor<int> x({10});
  std::vector<std::vector<size_t>> visited;
  sb::walk(x, [&](const std::vector<size_t>& idx) -> bool {
    visited.push_back(idx);
    return idx[0] < 3;
  });

  std::vector<std::vector<size_t>> expected = {{0}, {1}, {2}, {3}};
  EXPECT_EQ(visited, expected);
}

TEST(TensorWalk, WalkReadsCorrectValues)
{
  sb::Tensor<int> x({2, 3});
  int val = 0;
  x({0, 0}) = val++;
  x({0, 1}) = val++;
  x({0, 2}) = val++;
  x({1, 0}) = val++;
  x({1, 1}) = val++;
  x({1, 2}) = val++;

  std::vector<int> values;
  sb::walk(x, [&](const std::vector<size_t>& idx) {
    values.push_back(x(idx));
  });

  std::vector<int> expected = {0, 1, 2, 3, 4, 5};
  EXPECT_EQ(values, expected);
}

TEST(TensorWalk, WalkSingleElement)
{
  sb::Tensor<int> x({1});
  std::vector<std::vector<size_t>> visited;
  sb::walk(x, [&](const std::vector<size_t>& idx) {
    visited.push_back(idx);
  });

  std::vector<std::vector<size_t>> expected = {{0}};
  EXPECT_EQ(visited, expected);
}

TEST(TensorWalk, WalkZeroDimInMiddle)
{
  sb::Tensor<int> x({3, 0, 2});
  size_t count = 0;
  sb::walk(x, [&](const std::vector<size_t>&) { ++count; });
  EXPECT_EQ(count, 0);
}

TEST(TensorWalk, MemberWalkMatchesFreeWalk)
{
  sb::Tensor<int> x({2, 3});

  std::vector<std::vector<size_t>> from_member;
  sb::walk(x, [&](const std::vector<size_t>& idx) {
    from_member.push_back(idx);
  });

  std::vector<std::vector<size_t>> from_free;
  sb::walk(x, [&](const std::vector<size_t>& idx) {
    from_free.push_back(idx);
  });

  EXPECT_EQ(from_member, from_free);
}
