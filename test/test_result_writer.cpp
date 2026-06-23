#include <gtest/gtest.h>

#include <sbear/cuda/cuda_result_writer.hpp>

template <typename T>
class DistanceMatrixResultWriterTyped : public ::testing::Test {};

using ResultWriterTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DistanceMatrixResultWriterTyped, ResultWriterTypes);

TYPED_TEST(DistanceMatrixResultWriterTyped, ScatterLowerTriangle)
{
  using Tv = TypeParam;
  sb::DistanceMatrix<Tv> dm(4);

  // Block covers rows [2,3], cols [0,1]
  Tv hostBlock[] = {
    Tv(10), Tv(20),   // row 2: (2,0)=10, (2,1)=20
    Tv(30), Tv(40)    // row 3: (3,0)=30, (3,1)=40
  };

  sb::BlockInfo block{};
  block.rowStart = 2;
  block.rowHeight = 2;
  block.colStart = 0;
  block.colWidth = 2;

  sb::DistanceMatrixResultWriter<Tv> writer(dm);
  writer.scatter(hostBlock, block);

  EXPECT_EQ(dm(2, 0), Tv(10));
  EXPECT_EQ(dm(2, 1), Tv(20));
  EXPECT_EQ(dm(3, 0), Tv(30));
  EXPECT_EQ(dm(3, 1), Tv(40));
  EXPECT_EQ(dm(0, 2), Tv(10));  // symmetric access

  // Untouched
  EXPECT_EQ(dm(1, 0), Tv(0));
  EXPECT_EQ(dm(2, 3), Tv(0));
}

TYPED_TEST(DistanceMatrixResultWriterTyped, DiagonalBlock)
{
  using Tv = TypeParam;
  sb::DistanceMatrix<Tv> dm(4);

  // Block on diagonal: rows [1,2], cols [1,2]
  // Lower triangle: only (2,1) has i > j
  Tv hostBlock[] = {
    Tv(0), Tv(0),     // (1,1)=skip(diag), (1,2)=skip(upper)
    Tv(60), Tv(0)     // (2,1)=60, (2,2)=skip(diag)
  };

  sb::BlockInfo block{};
  block.rowStart = 1;
  block.rowHeight = 2;
  block.colStart = 1;
  block.colWidth = 2;

  sb::DistanceMatrixResultWriter<Tv> writer(dm);
  writer.scatter(hostBlock, block);

  EXPECT_EQ(dm(2, 1), Tv(60));
  EXPECT_EQ(dm(1, 1), Tv(0));
}

TYPED_TEST(DistanceMatrixResultWriterTyped, ThrowsOnNonZeroSkippedEntry)
{
  using Tv = TypeParam;
  sb::DistanceMatrix<Tv> dm(4);

  // (1,2) is upper triangle and nonzero — should throw
  Tv hostBlock[] = {
    Tv(0), Tv(50),
    Tv(60), Tv(0)
  };

  sb::BlockInfo block{};
  block.rowStart = 1;
  block.rowHeight = 2;
  block.colStart = 1;
  block.colWidth = 2;

  sb::DistanceMatrixResultWriter<Tv> writer(dm);
  EXPECT_THROW(writer.scatter(hostBlock, block), std::logic_error);
}

TYPED_TEST(DistanceMatrixResultWriterTyped, NonOverlappingBlocks)
{
  using Tv = TypeParam;
  sb::DistanceMatrix<Tv> dm(4);
  sb::DistanceMatrixResultWriter<Tv> writer(dm);

  Tv block1[] = { Tv(1), Tv(2), Tv(3), Tv(4) };
  sb::BlockInfo bi1{.rowStart = 2, .rowHeight = 2, .colStart = 0, .colWidth = 2, .blockIndex = 0};
  writer.scatter(block1, bi1);

  Tv block2[] = { Tv(5) };
  sb::BlockInfo bi2{.rowStart = 1, .rowHeight = 1, .colStart = 0, .colWidth = 1, .blockIndex = 1};
  writer.scatter(block2, bi2);

  EXPECT_EQ(dm(1, 0), Tv(5));
  EXPECT_EQ(dm(2, 0), Tv(1));
  EXPECT_EQ(dm(2, 1), Tv(2));
  EXPECT_EQ(dm(3, 0), Tv(3));
  EXPECT_EQ(dm(3, 1), Tv(4));
}

template <typename T>
class SymmetricMatrixResultWriterTyped : public ::testing::Test {};
TYPED_TEST_SUITE(SymmetricMatrixResultWriterTyped, ResultWriterTypes);

TYPED_TEST(SymmetricMatrixResultWriterTyped, ScatterWithDiagonal)
{
  using Tv = TypeParam;
  sb::SymmetricMatrix<Tv> sm(4);

  // Block covers rows [0,1], cols [0,1]
  // Lower triangle + diagonal: (0,0), (1,0), (1,1)
  Tv hostBlock[] = {
    Tv(10), Tv(0),
    Tv(30), Tv(40)
  };

  sb::BlockInfo block{};
  block.rowStart = 0;
  block.rowHeight = 2;
  block.colStart = 0;
  block.colWidth = 2;

  sb::SymmetricMatrixResultWriter<Tv> writer(sm);
  writer.scatter(hostBlock, block);

  EXPECT_EQ(sm(0, 0), Tv(10));
  EXPECT_EQ(sm(1, 0), Tv(30));
  EXPECT_EQ(sm(0, 1), Tv(30));  // symmetric
  EXPECT_EQ(sm(1, 1), Tv(40));
}

TYPED_TEST(SymmetricMatrixResultWriterTyped, ThrowsOnNonZeroSkippedEntry)
{
  using Tv = TypeParam;
  sb::SymmetricMatrix<Tv> sm(4);

  // (0,1) is upper triangle and nonzero — should throw
  Tv hostBlock[] = {
    Tv(10), Tv(20),
    Tv(30), Tv(40)
  };

  sb::BlockInfo block{};
  block.rowStart = 0;
  block.rowHeight = 2;
  block.colStart = 0;
  block.colWidth = 2;

  sb::SymmetricMatrixResultWriter<Tv> writer(sm);
  EXPECT_THROW(writer.scatter(hostBlock, block), std::logic_error);
}

TYPED_TEST(SymmetricMatrixResultWriterTyped, OffDiagonalBlock)
{
  using Tv = TypeParam;
  sb::SymmetricMatrix<Tv> sm(4);

  // Block covers rows [2,3], cols [0,1] — entirely below diagonal
  Tv hostBlock[] = { Tv(1), Tv(2), Tv(3), Tv(4) };

  sb::BlockInfo block{};
  block.rowStart = 2;
  block.rowHeight = 2;
  block.colStart = 0;
  block.colWidth = 2;

  sb::SymmetricMatrixResultWriter<Tv> writer(sm);
  writer.scatter(hostBlock, block);

  EXPECT_EQ(sm(2, 0), Tv(1));
  EXPECT_EQ(sm(2, 1), Tv(2));
  EXPECT_EQ(sm(3, 0), Tv(3));
  EXPECT_EQ(sm(3, 1), Tv(4));
  EXPECT_EQ(sm(0, 2), Tv(1));  // symmetric
}

template <typename T>
class DenseResultWriterTyped : public ::testing::Test {};
TYPED_TEST_SUITE(DenseResultWriterTyped, ResultWriterTypes);

TYPED_TEST(DenseResultWriterTyped, ScatterAll)
{
  using Tv = TypeParam;
  sb::Tensor<Tv> dense({4, 4}, Tv(0));
  sb::DenseMatrixView<Tv> view(dense, 4);
  sb::DenseResultWriter<Tv> writer(view);

  Tv hostBlock[] = { Tv(1), Tv(2), Tv(3), Tv(4) };

  sb::BlockInfo block{};
  block.rowStart = 1;
  block.rowHeight = 2;
  block.colStart = 2;
  block.colWidth = 2;

  writer.scatter(hostBlock, block);

  EXPECT_EQ(view(1, 2), Tv(1));
  EXPECT_EQ(view(1, 3), Tv(2));
  EXPECT_EQ(view(2, 2), Tv(3));
  EXPECT_EQ(view(2, 3), Tv(4));
  EXPECT_EQ(view(0, 0), Tv(0));
}

TYPED_TEST(DenseResultWriterTyped, RectangularOutput)
{
  using Tv = TypeParam;
  sb::Tensor<Tv> dense({3, 5}, Tv(0));
  sb::DenseMatrixView<Tv> view(dense, 5);
  sb::DenseResultWriter<Tv> writer(view);

  Tv hostBlock[] = { Tv(7), Tv(8), Tv(9) };

  sb::BlockInfo block{};
  block.rowStart = 1;
  block.rowHeight = 1;
  block.colStart = 2;
  block.colWidth = 3;

  writer.scatter(hostBlock, block);

  EXPECT_EQ(view(1, 2), Tv(7));
  EXPECT_EQ(view(1, 3), Tv(8));
  EXPECT_EQ(view(1, 4), Tv(9));
  EXPECT_EQ(view(0, 2), Tv(0));  // untouched
}
