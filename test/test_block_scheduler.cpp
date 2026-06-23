#include <gtest/gtest.h>

#include <sbear/cuda/cuda_block_scheduler.hpp>

#include <set>
#include <utility>

namespace
{
  void verify_block_coverage(const sb::CudaBlockScheduler::Config& config,
                              bool checkBudget = true)
  {
    sb::CudaBlockScheduler scheduler(config);
    auto const& blocks = scheduler.blocks();

    std::set<std::pair<size_t, size_t>> covered;
    for (auto const& block : blocks)
    {
      if (checkBudget)
      {
        EXPECT_LE(block.rowHeight * block.colWidth, config.maxOutputElements)
          << "Block (" << block.rowStart << "," << block.colStart
          << ") size " << block.rowHeight << "x" << block.colWidth << " exceeds budget";
      }

      for (size_t i = block.rowStart; i < block.rowStart + block.rowHeight; ++i)
      {
        for (size_t j = block.colStart; j < block.colStart + block.colWidth; ++j)
        {
          if (config.triangleMode == sb::BlockTriangleMode::LowerTriangle && i < j)
          {
            continue;
          }

          auto [it, inserted] = covered.insert({i, j});
          EXPECT_TRUE(inserted) << "Duplicate coverage at (" << i << ", " << j << ")";
        }
      }
    }

    size_t expected = (config.triangleMode == sb::BlockTriangleMode::LowerTriangle)
      ? config.nRows * (config.nRows + 1) / 2
      : config.nRows * config.nCols;

    EXPECT_EQ(covered.size(), expected) << "Incomplete coverage";
  }
}

TEST(CudaBlockScheduler, EmptyMatrix)
{
  sb::CudaBlockScheduler scheduler({
    .nRows = 0, .nCols = 0,
    .maxOutputElements = 1000,
    .nSplitsHint = 1,
    .triangleMode = sb::BlockTriangleMode::LowerTriangle
  });

  EXPECT_TRUE(scheduler.blocks().empty());
}

TEST(CudaBlockScheduler, SingleElement)
{
  sb::CudaBlockScheduler scheduler({
    .nRows = 1, .nCols = 1,
    .maxOutputElements = 1000,
    .nSplitsHint = 1,
    .triangleMode = sb::BlockTriangleMode::LowerTriangle
  });

  auto const& blocks = scheduler.blocks();
  ASSERT_EQ(blocks.size(), 1);
  EXPECT_EQ(blocks[0].rowStart, 0);
  EXPECT_EQ(blocks[0].rowHeight, 1);
  EXPECT_EQ(blocks[0].colStart, 0);
  EXPECT_EQ(blocks[0].colWidth, 1);
}

TEST(CudaBlockScheduler, SmallMatrix_LowerTriangle)
{
  verify_block_coverage({
    .nRows = 4, .nCols = 4,
    .maxOutputElements = 4,
    .nSplitsHint = 1,
    .triangleMode = sb::BlockTriangleMode::LowerTriangle
  });
}

TEST(CudaBlockScheduler, SmallMatrix_Full)
{
  verify_block_coverage({
    .nRows = 4, .nCols = 4,
    .maxOutputElements = 4,
    .nSplitsHint = 1,
    .triangleMode = sb::BlockTriangleMode::Full
  });
}

TEST(CudaBlockScheduler, LowerTriangleHasFewerBlocks)
{
  sb::CudaBlockScheduler lower({
    .nRows = 10, .nCols = 10,
    .maxOutputElements = 25,
    .nSplitsHint = 1,
    .triangleMode = sb::BlockTriangleMode::LowerTriangle
  });

  sb::CudaBlockScheduler full({
    .nRows = 10, .nCols = 10,
    .maxOutputElements = 25,
    .nSplitsHint = 1,
    .triangleMode = sb::BlockTriangleMode::Full
  });

  EXPECT_LT(lower.blocks().size(), full.blocks().size());
}

TEST(CudaBlockScheduler, LargeMatrixSmallBlocks)
{
  sb::CudaBlockScheduler::Config config = {
    .nRows = 100, .nCols = 100,
    .maxOutputElements = 100,
    .nSplitsHint = 4,
    .triangleMode = sb::BlockTriangleMode::LowerTriangle
  };

  sb::CudaBlockScheduler scheduler(config);
  EXPECT_GT(scheduler.blocks().size(), 1);

  verify_block_coverage(config);
}

TEST(CudaBlockScheduler, MaxRowHeightAndColWidth)
{
  sb::CudaBlockScheduler scheduler({
    .nRows = 10, .nCols = 10,
    .maxOutputElements = 10000,
    .nSplitsHint = 1,
    .triangleMode = sb::BlockTriangleMode::Full
  });

  // With budget=10000 and nPcfs=10, one block should cover everything
  EXPECT_EQ(scheduler.max_row_height(), 10);
  EXPECT_EQ(scheduler.max_col_width(), 10);
  EXPECT_EQ(scheduler.blocks().size(), 1);
}

TEST(CudaBlockScheduler, SortedByDescendingWork)
{
  sb::CudaBlockScheduler scheduler({
    .nRows = 20, .nCols = 20,
    .maxOutputElements = 100,
    .nSplitsHint = 1,
    .triangleMode = sb::BlockTriangleMode::LowerTriangle
  });

  auto const& blocks = scheduler.blocks();
  for (size_t i = 1; i < blocks.size(); ++i)
  {
    EXPECT_GE(blocks[i - 1].rowHeight * blocks[i - 1].colWidth,
              blocks[i].rowHeight * blocks[i].colWidth);
  }
}

TEST(CudaBlockScheduler, RectangularMatrix_Full)
{
  verify_block_coverage({
    .nRows = 3, .nCols = 5,
    .maxOutputElements = 100,
    .nSplitsHint = 1,
    .triangleMode = sb::BlockTriangleMode::Full
  });
}

TEST(CudaBlockScheduler, RectangularMatrix_ManyBlocks)
{
  sb::CudaBlockScheduler::Config config = {
    .nRows = 7, .nCols = 13,
    .maxOutputElements = 10,
    .nSplitsHint = 1,
    .triangleMode = sb::BlockTriangleMode::Full
  };

  sb::CudaBlockScheduler scheduler(config);
  EXPECT_GT(scheduler.blocks().size(), 1);

  verify_block_coverage(config);
}

// --- nSplitsHint > 1 tests ---

TEST(CudaBlockScheduler, SplitsHint_ProducesMoreBlocks)
{
  sb::CudaBlockScheduler hint1({
    .nRows = 20, .nCols = 20,
    .maxOutputElements = 400,
    .nSplitsHint = 1,
    .triangleMode = sb::BlockTriangleMode::Full
  });

  sb::CudaBlockScheduler hint4({
    .nRows = 20, .nCols = 20,
    .maxOutputElements = 400,
    .nSplitsHint = 4,
    .triangleMode = sb::BlockTriangleMode::Full
  });

  // More splits -> smaller blocks -> more blocks
  EXPECT_GT(hint4.blocks().size(), hint1.blocks().size());
}

TEST(CudaBlockScheduler, SplitsHint_SmallerMaxDimensions)
{
  sb::CudaBlockScheduler hint1({
    .nRows = 20, .nCols = 20,
    .maxOutputElements = 400,
    .nSplitsHint = 1,
    .triangleMode = sb::BlockTriangleMode::Full
  });

  sb::CudaBlockScheduler hint4({
    .nRows = 20, .nCols = 20,
    .maxOutputElements = 400,
    .nSplitsHint = 4,
    .triangleMode = sb::BlockTriangleMode::Full
  });

  // Higher hint should produce smaller block dimensions
  EXPECT_LE(hint4.max_row_height(), hint1.max_row_height());
  EXPECT_LE(hint4.max_col_width(), hint1.max_col_width());
}

TEST(CudaBlockScheduler, SplitsHint_FullCoverage)
{
  for (size_t nSplits : {2, 4, 8, 16})
  {
    verify_block_coverage({
      .nRows = 15, .nCols = 15,
      .maxOutputElements = 225,
      .nSplitsHint = nSplits,
      .triangleMode = sb::BlockTriangleMode::LowerTriangle
    });
  }
}

TEST(CudaBlockScheduler, SplitsHint_FullMatrix_Coverage)
{
  for (size_t nSplits : {2, 4, 8})
  {
    verify_block_coverage({
      .nRows = 10, .nCols = 10,
      .maxOutputElements = 100,
      .nSplitsHint = nSplits,
      .triangleMode = sb::BlockTriangleMode::Full
    });
  }
}

TEST(CudaBlockScheduler, SplitsHint_Rectangular_Coverage)
{
  verify_block_coverage({
    .nRows = 5, .nCols = 20,
    .maxOutputElements = 100,
    .nSplitsHint = 4,
    .triangleMode = sb::BlockTriangleMode::Full
  });
}

TEST(CudaBlockScheduler, SplitsHint_StillSorted)
{
  sb::CudaBlockScheduler scheduler({
    .nRows = 20, .nCols = 20,
    .maxOutputElements = 100,
    .nSplitsHint = 4,
    .triangleMode = sb::BlockTriangleMode::LowerTriangle
  });

  auto const& blocks = scheduler.blocks();
  for (size_t i = 1; i < blocks.size(); ++i)
  {
    EXPECT_GE(blocks[i - 1].rowHeight * blocks[i - 1].colWidth,
              blocks[i].rowHeight * blocks[i].colWidth)
      << "Blocks not sorted by descending work at index " << i;
  }
}

// --- minBlockSide tests ---

TEST(CudaBlockScheduler, MinBlockSide_EnforcesFloor)
{
  // Without floor: maxOutputElements=10000, nSplitsHint=100 -> elementsPerBlock=100, side=10
  sb::CudaBlockScheduler noFloor({
    .nRows = 500, .nCols = 500,
    .maxOutputElements = 10000,
    .nSplitsHint = 100,
    .triangleMode = sb::BlockTriangleMode::Full,
    .minBlockSide = 0
  });

  // With floor of 50: side should be raised from 10 to 50
  sb::CudaBlockScheduler withFloor({
    .nRows = 500, .nCols = 500,
    .maxOutputElements = 10000,
    .nSplitsHint = 100,
    .triangleMode = sb::BlockTriangleMode::Full,
    .minBlockSide = 50
  });

  EXPECT_LE(noFloor.max_row_height(), 10);
  EXPECT_GE(withFloor.max_row_height(), 50);
  EXPECT_GE(withFloor.max_col_width(), 50);

  // Fewer blocks when the floor is active
  EXPECT_LT(withFloor.blocks().size(), noFloor.blocks().size());
}

TEST(CudaBlockScheduler, MinBlockSide_ClampedByMatrixSize)
{
  // minBlockSide=200 but the matrix is only 30x30 -- should clamp to 30
  sb::CudaBlockScheduler scheduler({
    .nRows = 30, .nCols = 30,
    .maxOutputElements = 100,
    .nSplitsHint = 10,
    .triangleMode = sb::BlockTriangleMode::Full,
    .minBlockSide = 200
  });

  // Single block covering the entire 30x30 matrix
  EXPECT_EQ(scheduler.blocks().size(), 1);
  EXPECT_EQ(scheduler.max_row_height(), 30);
  EXPECT_EQ(scheduler.max_col_width(), 30);
}

TEST(CudaBlockScheduler, MinBlockSide_FullCoverage)
{
  // minBlockSide can force blocks larger than maxOutputElements, so skip budget check
  verify_block_coverage({
    .nRows = 100, .nCols = 100,
    .maxOutputElements = 1000,
    .nSplitsHint = 100,
    .triangleMode = sb::BlockTriangleMode::LowerTriangle,
    .minBlockSide = 40
  }, /*checkBudget=*/false);
}

TEST(CudaBlockScheduler, MinBlockSide_Rectangular_FullCoverage)
{
  sb::CudaBlockScheduler::Config config = {
    .nRows = 50, .nCols = 200,
    .maxOutputElements = 1000,
    .nSplitsHint = 50,
    .triangleMode = sb::BlockTriangleMode::Full,
    .minBlockSide = 30
  };

  sb::CudaBlockScheduler scheduler(config);

  // Blocks should respect the floor (up to maxDim clamping)
  EXPECT_GE(scheduler.max_row_height(), 30);
  EXPECT_GE(scheduler.max_col_width(), 30);

  verify_block_coverage(config);
}

TEST(CudaBlockScheduler, MinBlockSide_Zero_HasNoEffect)
{
  // minBlockSide=0 should behave identically to omitting it
  sb::CudaBlockScheduler withZero({
    .nRows = 20, .nCols = 20,
    .maxOutputElements = 400,
    .nSplitsHint = 4,
    .triangleMode = sb::BlockTriangleMode::Full,
    .minBlockSide = 0
  });

  sb::CudaBlockScheduler withoutField({
    .nRows = 20, .nCols = 20,
    .maxOutputElements = 400,
    .nSplitsHint = 4,
    .triangleMode = sb::BlockTriangleMode::Full
  });

  EXPECT_EQ(withZero.blocks().size(), withoutField.blocks().size());
  EXPECT_EQ(withZero.max_row_height(), withoutField.max_row_height());
  EXPECT_EQ(withZero.max_col_width(), withoutField.max_col_width());
}
