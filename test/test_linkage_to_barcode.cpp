#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

#include <sbear/persistence/linkage.hpp>
#include <sbear/tensor.hpp>

namespace
{
  using ScalarTypes = ::testing::Types<sb::float32_t, sb::float64_t>;

  template <typename T>
  class LinkageToBarcodeTest : public ::testing::Test
  {
  };

  TYPED_TEST_SUITE(LinkageToBarcodeTest, ScalarTypes);

  // Shape-only tensor: oversized inputs must be rejected before accessing data.
  struct OversizedLinkage
  {
    using value_type = double;
    std::vector<std::size_t> dimensions;

    const auto& shape() const { return dimensions; }
    std::size_t shape(std::size_t axis) const { return dimensions.at(axis); }
    std::vector<std::size_t> strides() const { return {4, 1}; }
    std::size_t rank() const { return 2; }
    std::size_t size() const { throw std::logic_error("Unexpected size query"); }
    value_type operator()(const std::vector<std::size_t>&) const
    {
      throw std::logic_error("Unexpected linkage data access");
    }
  };

  TEST(LinkageToBarcodeSizeTest, RejectsOversizedMatrix)
  {
    const auto maxSize = std::numeric_limits<std::size_t>::max();
    for (const auto rows : {(maxSize - 1) / 2 + 1, maxSize})
    {
      SCOPED_TRACE(rows);
      const OversizedLinkage linkage{{rows, 4}};
      try
      {
        sb::ph::linkage_to_barcode(linkage);
        FAIL() << "Expected oversized linkage matrix to be rejected";
      }
      catch (const std::invalid_argument& error)
      {
        EXPECT_STREQ(error.what(), "Linkage matrix is too large");
      }
    }
  }

  TYPED_TEST(LinkageToBarcodeTest, NonMonotoneFlag)
  {
    using T = TypeParam;
    sb::Tensor<T> linkage({2, 4});
    const T values[2][4] = {{0, 1, 3, 2}, {2, 3, 2, 3}};
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 4; ++j)
        linkage({i, j}) = values[i][j];

    bool hasNonMonotoneHeights = false;
    const auto barcode = sb::ph::linkage_to_barcode(linkage, false, &hasNonMonotoneHeights);
    EXPECT_TRUE(hasNonMonotoneHeights);
    EXPECT_EQ(barcode, sb::ph::linkage_to_barcode(linkage));
    EXPECT_EQ(barcode, sb::ph::linkage_to_barcode(linkage, false, nullptr));

    linkage({1, 2}) = T{4};
    hasNonMonotoneHeights = false;
    sb::ph::linkage_to_barcode(linkage, false, &hasNonMonotoneHeights);
    EXPECT_FALSE(hasNonMonotoneHeights);

    hasNonMonotoneHeights = true;
    sb::ph::linkage_to_barcode(linkage, false, &hasNonMonotoneHeights);
    EXPECT_FALSE(hasNonMonotoneHeights);
  }
}
