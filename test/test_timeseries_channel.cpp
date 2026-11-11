#include <gtest/gtest.h>

#include <sbear/timeseries.hpp>

#include <cmath>
#include <stdexcept>
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

  // Observations 0, 1, ..., 9 at 0 s, 1 s, ..., 9 s.
  sb::Channel<double> channel_of_ten()
  {
    std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<double> values(times);
    return sb::Channel<double>(sb::Times(std::move(times)), {}, std::move(values));
  }

  TEST(ChannelTest, ATakeOfATakeReadsTheObservationsBothPick)
  {
    const auto first = channel_of_ten().take(sb::Times(std::vector<double>{0.0, 1.0, 2.0}), {2, 5, 7});

    const auto second = first.take(sb::Times(std::vector<double>{0.0, 1.0}), {2, 0});

    // Observation 2 of first is 7, and observation 0 is 2.
    EXPECT_EQ(second.flat_value(0), 7.0);
    EXPECT_EQ(second.flat_value(1), 2.0);
  }

  TEST(ChannelTest, ATakeOfAViewAtOneLagReadsWhatTheViewReads)
  {
    const std::vector<double> lags{2.0};
    const auto lagged = channel_of_ten().stack_at(sb::Times(std::vector<double>{3.0, 5.0, 8.0}), lags);

    const auto taken = lagged.take(sb::Times(std::vector<double>{0.0, 1.0}), {2, 0});

    // The view reads 1, 3, and 6, two seconds before 3 s, 5 s, and 8 s.
    EXPECT_EQ(taken.flat_value(0), 6.0);
    EXPECT_EQ(taken.flat_value(1), 1.0);
  }

  TEST(TimeSinceTest, AUnitWithoutALengthThrows)
  {
    const sb::DateTime date(10, sb::TimeUnit::Seconds);
    const sb::DateTime epoch(0, sb::TimeUnit::Seconds);

    EXPECT_THROW((void)sb::time_since(date, sb::DateTime(0, sb::TimeUnit::Unitless), sb::TimeUnit::Seconds),
                 std::invalid_argument);
    EXPECT_THROW((void)sb::time_since(date, epoch, sb::TimeUnit::Unitless), std::invalid_argument);
  }
}
