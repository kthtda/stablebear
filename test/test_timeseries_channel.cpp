#include <gtest/gtest.h>

#include <sbear/timeseries.hpp>

#include <cmath>
#include <vector>

namespace
{
  // Observations at 0 s, at one ulp before 2 s, and with @p atTwo at 2 s.
  sb::Channel<double> channel_up_to_two_seconds(bool atTwo)
  {
    std::vector<double> times{0.0, std::nextafter(2.0, 0.0)};
    if (atTwo)
    {
      times.push_back(2.0);
    }
    std::vector<double> values(times.size(), 1.0);
    return sb::Channel<double>(sb::Times(std::move(times)), {}, std::move(values));
  }

  TEST(ChannelTest, ConvertingKeepsAnOpenEndAfterTheLastTimestamp)
  {
    const auto channel = channel_up_to_two_seconds(false).until(2.0, false);

    const auto converted = channel.to(sb::TimeUnit::Seconds, sb::TimeUnit::Minutes);

    ASSERT_TRUE(converted.end());
    EXPECT_GT(*converted.end(), converted.time(1));
  }

  TEST(ChannelTest, ShiftingKeepsAnOpenEndAfterTheLastTimestamp)
  {
    const auto channel = channel_up_to_two_seconds(false).until(2.0, false);

    const auto shifted = channel.shifted(86400.0);

    ASSERT_TRUE(shifted.end());
    EXPECT_GT(*shifted.end(), shifted.time(1));
  }

  TEST(ChannelTest, ConvertingKeepsTheTimestampsIncreasing)
  {
    const auto converted = channel_up_to_two_seconds(true).to(sb::TimeUnit::Seconds, sb::TimeUnit::Minutes);

    EXPECT_LT(converted.time(1), converted.time(2));
  }

  TEST(ChannelTest, ShiftingKeepsTheTimestampsIncreasing)
  {
    const auto shifted = channel_up_to_two_seconds(true).shifted(86400.0);

    EXPECT_LT(shifted.time(1), shifted.time(2));
  }
}
