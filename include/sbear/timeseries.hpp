#ifndef STABLEBEAR_TIMESERIES_H
#define STABLEBEAR_TIMESERIES_H

#include "config.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sb
{
  /// Physical time units, from attoseconds to days.
  enum class TimeUnit : std::uint8_t
  {
    Attoseconds,
    Femtoseconds,
    Picoseconds,
    Nanoseconds,
    Microseconds,
    Milliseconds,
    Seconds,
    Minutes,
    Hours,
    Days
  };

  namespace detail
  {
    /// A unit is mult * 10^exp seconds. The multipliers 1, 60, 3600, and
    /// 86400 divide each other, so a conversion factor reduces to one
    /// integer multiplying or dividing the value.
    struct TimeUnitInfo
    {
      TimeUnit unit;
      std::string_view name;
      int exp;
      std::int64_t mult;
    };

    inline constexpr auto TimeUnitTable = std::to_array<TimeUnitInfo>({
      {TimeUnit::Attoseconds, "as", -18, 1},
      {TimeUnit::Femtoseconds, "fs", -15, 1},
      {TimeUnit::Picoseconds, "ps", -12, 1},
      {TimeUnit::Nanoseconds, "ns", -9, 1},
      {TimeUnit::Microseconds, "us", -6, 1},
      {TimeUnit::Milliseconds, "ms", -3, 1},
      {TimeUnit::Seconds, "s", 0, 1},
      {TimeUnit::Minutes, "min", 0, 60},
      {TimeUnit::Hours, "h", 0, 3600},
      {TimeUnit::Days, "d", 0, 86400},
    });

    /// Accepted spellings besides the canonical names.
    inline constexpr auto TimeUnitAliases = std::to_array<std::pair<std::string_view, TimeUnit>>({
      {"attosecond", TimeUnit::Attoseconds}, {"attoseconds", TimeUnit::Attoseconds},
      {"femtosecond", TimeUnit::Femtoseconds}, {"femtoseconds", TimeUnit::Femtoseconds},
      {"picosecond", TimeUnit::Picoseconds}, {"picoseconds", TimeUnit::Picoseconds},
      {"nanosecond", TimeUnit::Nanoseconds}, {"nanoseconds", TimeUnit::Nanoseconds},
      {"microsecond", TimeUnit::Microseconds}, {"microseconds", TimeUnit::Microseconds},
      {"\xc2\xb5s", TimeUnit::Microseconds},  // micro sign
      {"\xce\xbcs", TimeUnit::Microseconds},  // Greek small letter mu
      {"millisecond", TimeUnit::Milliseconds}, {"milliseconds", TimeUnit::Milliseconds},
      {"sec", TimeUnit::Seconds}, {"second", TimeUnit::Seconds}, {"seconds", TimeUnit::Seconds},
      {"minute", TimeUnit::Minutes}, {"minutes", TimeUnit::Minutes},
      {"hr", TimeUnit::Hours}, {"hour", TimeUnit::Hours}, {"hours", TimeUnit::Hours},
      {"day", TimeUnit::Days}, {"days", TimeUnit::Days},
      {"msec", TimeUnit::Milliseconds}, {"usec", TimeUnit::Microseconds},
    });

    inline const TimeUnitInfo& time_unit_info(TimeUnit unit)
    {
      return TimeUnitTable[static_cast<size_t>(unit)];
    }

    /// 10^k, exact in double for k <= 22.
    inline float64_t exact_power_of_ten(int k)
    {
      float64_t result = 1;
      for (int i = 0; i < k; ++i)
      {
        result *= 10;
      }
      return result;
    }
  }

  [[nodiscard]] inline std::string_view time_unit_name(TimeUnit unit)
  {
    return detail::time_unit_info(unit).name;
  }

  [[nodiscard]] inline TimeUnit parse_time_unit(std::string_view name)
  {
    for (const auto& info : detail::TimeUnitTable)
    {
      if (info.name == name)
      {
        return info.unit;
      }
    }
    for (const auto& [alias, unit] : detail::TimeUnitAliases)
    {
      if (alias == name)
      {
        return unit;
      }
    }
    throw std::invalid_argument(
        "Unknown time unit '" + std::string(name)
        + "'; expected one of as, fs, ps, ns, us, ms, s, min, h, d, or a spelled-out"
          " name such as seconds");
  }

  /// Convert @p value from unit @p from to unit @p to. The factor is an
  /// integer that multiplies or divides the value, so the result is rounded
  /// once. The factor is exact unless it exceeds 2^53 (days and attoseconds).
  [[nodiscard]] inline float64_t convert_time(float64_t value, TimeUnit from, TimeUnit to)
  {
    const auto& src = detail::time_unit_info(from);
    const auto& dst = detail::time_unit_info(to);
    const auto g = std::gcd(src.mult, dst.mult);
    const int diff = src.exp - dst.exp;
    const float64_t num = static_cast<float64_t>(src.mult / g) * detail::exact_power_of_ten(std::max(diff, 0));
    const float64_t den = static_cast<float64_t>(dst.mult / g) * detail::exact_power_of_ten(std::max(-diff, 0));
    return den == 1 ? value * num : value / den;
  }

  /// One-dimensional, finite, strictly increasing timestamps in one unit.
  class Times
  {
  public:
    explicit Times(std::vector<float64_t> values, TimeUnit unit = TimeUnit::Seconds)
      : m_values(std::move(values)), m_unit(unit)
    {
      for (size_t i = 0; i < m_values.size(); ++i)
      {
        if (!std::isfinite(m_values[i]))
        {
          throw std::invalid_argument("Timestamps must be finite");
        }
        if (i > 0 && !(m_values[i - 1] < m_values[i]))
        {
          throw std::invalid_argument("Timestamps must be strictly increasing");
        }
      }
    }

    [[nodiscard]] const std::vector<float64_t>& values() const noexcept { return m_values; }
    [[nodiscard]] TimeUnit unit() const noexcept { return m_unit; }
    [[nodiscard]] size_t size() const noexcept { return m_values.size(); }

    /// The same timestamps in another unit.
    [[nodiscard]] Times to(TimeUnit unit) const
    {
      std::vector<float64_t> converted(m_values.size());
      for (size_t i = 0; i < m_values.size(); ++i)
      {
        converted[i] = convert_time(m_values[i], m_unit, unit);
      }
      return Times(std::move(converted), unit);
    }

    bool operator==(const Times&) const = default;

  private:
    std::vector<float64_t> m_values;
    TimeUnit m_unit;
  };
}

#endif // STABLEBEAR_TIMESERIES_H
