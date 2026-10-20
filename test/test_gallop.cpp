#include <gtest/gtest.h>

#include <sbear/algorithms/gallop.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace
{
  TEST(GallopTest, MatchesUpperBoundFromEveryHint)
  {
    // Repeated values, so the answer for a tie is the end of its run.
    const std::vector<int> values{1, 2, 2, 2, 5, 7, 7, 9};
    for (const int value : {0, 1, 2, 3, 5, 6, 7, 9, 10})
    {
      const auto expected = std::upper_bound(values.begin(), values.end(), value);
      for (size_t h = 0; h <= values.size(); ++h)
      {
        const auto hint = values.begin() + static_cast<std::ptrdiff_t>(h);
        EXPECT_EQ(sb::gallop_upper_bound(values.begin(), values.end(), hint, value), expected)
            << "value " << value << ", hint " << h;
      }
    }
  }

  TEST(GallopTest, EmptyRange)
  {
    const std::vector<int> values;
    EXPECT_EQ(sb::gallop_upper_bound(values.begin(), values.end(), values.end(), 1), values.end());
  }
}
