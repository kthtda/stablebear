#include <gtest/gtest.h>

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
