#include <gtest/gtest.h>

#include <sbear/algorithms/subdivide.hpp>

namespace
{

  TEST(Subdivide, SingleBlock)
  {
    auto blocks = sb::subdivide(100, 5);
    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_EQ(blocks[0].first, 0u);
    EXPECT_EQ(blocks[0].second, 4u);
  }

  TEST(Subdivide, ExactMultiple)
  {
    // 10 items, block size 5 => 2 blocks
    auto blocks = sb::subdivide(5, 10);
    ASSERT_EQ(blocks.size(), 2u);
    EXPECT_EQ(blocks[0].first, 0u);
    EXPECT_EQ(blocks[0].second, 4u);
    EXPECT_EQ(blocks[1].first, 5u);
    EXPECT_EQ(blocks[1].second, 9u);
  }

  TEST(Subdivide, NonExactMultiple)
  {
    // 7 items, block size 3 => 3 blocks: [0,2], [3,5], [6,6]
    auto blocks = sb::subdivide(3, 7);
    ASSERT_EQ(blocks.size(), 3u);
    EXPECT_EQ(blocks[0], std::make_pair(size_t(0), size_t(2)));
    EXPECT_EQ(blocks[1], std::make_pair(size_t(3), size_t(5)));
    EXPECT_EQ(blocks[2], std::make_pair(size_t(6), size_t(6)));
  }

  TEST(Subdivide, BlockSizeOne)
  {
    auto blocks = sb::subdivide(1, 4);
    ASSERT_EQ(blocks.size(), 4u);
    for (size_t i = 0; i < 4; ++i)
    {
      EXPECT_EQ(blocks[i].first, i);
      EXPECT_EQ(blocks[i].second, i);
    }
  }

  TEST(Subdivide, NoItems)
  {
    // Regression: nItems == 0 used to clamp the first block's end to
    // nItems - 1, wrapping to SIZE_MAX (issue #186).
    for (size_t bs = 1; bs <= 5; ++bs)
    {
      auto blocks = sb::subdivide(bs, 0);
      EXPECT_TRUE(blocks.empty());
    }
  }

  TEST(Subdivide, SingleItem)
  {
    auto blocks = sb::subdivide(5, 1);
    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_EQ(blocks[0].first, 0u);
    EXPECT_EQ(blocks[0].second, 0u);
  }

  TEST(Subdivide, CoverageComplete)
  {
    // Verify all items are covered for various sizes
    for (size_t n = 1; n <= 20; ++n)
    {
      for (size_t bs = 1; bs <= n + 5; ++bs)
      {
        auto blocks = sb::subdivide(bs, n);
        EXPECT_EQ(blocks.front().first, 0u);
        EXPECT_EQ(blocks.back().second, n - 1);
        // No gaps between blocks
        for (size_t i = 1; i < blocks.size(); ++i)
        {
          EXPECT_EQ(blocks[i].first, blocks[i - 1].second + 1);
        }
      }
    }
  }

} // namespace
