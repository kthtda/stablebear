#include <gtest/gtest.h>

#include <sbear/algorithms/union_find.hpp>

#include <cstddef>

namespace
{
  TEST(UnionFindTest, StartsAsSingletons)
  {
    sb::UnionFind uf(4);
    for (size_t x = 0; x < 4; ++x)
    {
      EXPECT_EQ(uf.find(x), x);
    }
  }

  TEST(UnionFindTest, UniteMergesComponentsWithChosenSurvivor)
  {
    sb::UnionFind uf(5);
    uf.unite(uf.find(0), uf.find(1)); // 0 survives
    EXPECT_EQ(uf.find(0), 0U);
    EXPECT_EQ(uf.find(1), 0U);

    uf.unite(uf.find(2), uf.find(3)); // 2 survives
    uf.unite(uf.find(2), uf.find(0)); // 2 survives, absorbs {0, 1}
    for (size_t x = 0; x < 4; ++x)
    {
      EXPECT_EQ(uf.find(x), 2U);
    }
    EXPECT_EQ(uf.find(4), 4U);
  }

  TEST(UnionFindTest, SelfUnionIsHarmless)
  {
    sb::UnionFind uf(3);
    uf.unite(uf.find(0), uf.find(1));
    const auto root = uf.find(0);
    uf.unite(root, root);
    EXPECT_EQ(uf.find(0), root);
    EXPECT_EQ(uf.find(1), root);
    EXPECT_EQ(uf.find(2), 2U);
  }

  TEST(UnionFindTest, ChainOfUnionsResolvesToFinalRoot)
  {
    const size_t n = 64;
    sb::UnionFind uf(n);
    for (size_t x = 1; x < n; ++x)
    {
      uf.unite(uf.find(x), uf.find(x - 1)); // the newest element always survives
    }
    for (size_t x = 0; x < n; ++x)
    {
      EXPECT_EQ(uf.find(x), n - 1);
    }
  }

  TEST(UnionFindTest, CopyAssignmentResetsScratchToSeed)
  {
    sb::UnionFind seed(4);
    seed.unite(seed.find(0), seed.find(1));

    sb::UnionFind scratch(4);
    scratch = seed;
    EXPECT_EQ(scratch.find(1), scratch.find(0));
    EXPECT_NE(scratch.find(2), scratch.find(3));

    // Mutating the copy must not leak back into the source...
    scratch.unite(scratch.find(2), scratch.find(3));
    EXPECT_NE(seed.find(2), seed.find(3));

    // ...and re-seeding discards the copy-only union.
    scratch = seed;
    EXPECT_NE(scratch.find(2), scratch.find(3));
    EXPECT_EQ(scratch.find(1), scratch.find(0));
  }
} // namespace
