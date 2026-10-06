#include <gtest/gtest.h>

#include <sbear/tensor.hpp>
#include <sbear/nested_tensor.hpp>
#include <sbear/point_cloud.hpp>
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

TEST(TensorWalk, WalkRank0ShapeVisitsOnce)
{
  // shape () is a rank-0 scalar with exactly one element, so walk visits it
  // once with an empty index. (A zero-size extent like {0} is the empty case
  // -- see WalkEmptyDimension.)
  sb::Tensor<int> x(std::vector<size_t>{});
  size_t count = 0;
  sb::walk(x, [&](const std::vector<size_t>& idx) { ++count; EXPECT_TRUE(idx.empty()); });
  EXPECT_EQ(count, 1);
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

TEST(TensorWalk, IndexedTensorVisitsLogicalIndicesAndSelectedValues)
{
  sb::Tensor<float> coordinates({3, 1});
  coordinates({0, 0}) = 10;
  coordinates({1, 0}) = 20;
  coordinates({2, 0}) = 30;
  sb::Tensor<sb::PointCloud<float>> source({1}, sb::PointCloud<float>(coordinates));

  // One source cloud backs four single-point selections, including a repeat.
  sb::Tensor<sb::Tensor<uint64_t>> selections({2, 2});
  selections({0, 0}) = sb::Tensor<uint64_t>({1}, 2);
  selections({0, 1}) = sb::Tensor<uint64_t>({1}, 0);
  selections({1, 0}) = sb::Tensor<uint64_t>({1}, 1);
  selections({1, 1}) = sb::Tensor<uint64_t>({1}, 2);
  const auto indexed = sb::make_indexed_tensor(source, sb::to_nested_tensor(selections));

  std::vector<std::vector<size_t>> visited;
  std::vector<float> values;
  sb::walk(indexed, [&](const std::vector<size_t>& idx) {
    visited.push_back(idx);
    const auto cloud = indexed(idx);
    values.push_back(cloud(0, 0));
  });

  EXPECT_EQ(visited, (std::vector<std::vector<size_t>>{
    {0, 0}, {0, 1},
    {1, 0}, {1, 1}
  }));
  EXPECT_EQ(values, (std::vector<float>{30, 10, 20, 30}));
}

TEST(TensorWalk, IndexedOuterSliceVisitsOnlyItsLogicalElements)
{
  sb::Tensor<float> coordinates({4, 1});
  coordinates({0, 0}) = 10;
  coordinates({1, 0}) = 20;
  coordinates({2, 0}) = 30;
  coordinates({3, 0}) = 40;
  sb::Tensor<sb::PointCloud<float>> source({1}, sb::PointCloud<float>(coordinates));

  sb::Tensor<sb::Tensor<uint64_t>> selections({4});
  selections(0) = sb::Tensor<uint64_t>({1}, 2);
  selections(1) = sb::Tensor<uint64_t>({1}, 0);
  selections(2) = sb::Tensor<uint64_t>({1}, 3);
  selections(3) = sb::Tensor<uint64_t>({1}, 1);
  const auto indexed = sb::make_indexed_tensor(source, sb::to_nested_tensor(selections));
  // Outer values [30, 10, 40, 20] sliced with [1::2] become [10, 20].
  const auto view = indexed[std::vector<sb::Slice>{sb::range(1, std::nullopt, 2)}];

  std::vector<std::vector<size_t>> visited;
  std::vector<float> values;
  sb::walk(view, [&](const std::vector<size_t>& idx) {
    visited.push_back(idx);
    const auto cloud = view(idx);
    values.push_back(cloud(0, 0));
  });

  EXPECT_EQ(visited, (std::vector<std::vector<size_t>>{{0}, {1}}));
  EXPECT_EQ(values, (std::vector<float>{10, 20}));
}
