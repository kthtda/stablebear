#ifndef STABLEBEAR_TIMESERIES_H
#define STABLEBEAR_TIMESERIES_H

#include "config.hpp"
#include "tensor.hpp"
#include "walk.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
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

  /// The times at which a channel can be evaluated: [start, end] when the end
  /// is closed, otherwise [start, end).
  struct Domain
  {
    float64_t start;
    float64_t end;
    bool endClosed = true;

    [[nodiscard]] bool contains(float64_t t) const
    {
      return start <= t && (endClosed ? t <= end : t < end);
    }

    [[nodiscard]] std::string to_string() const
    {
      std::ostringstream ss;
      ss << "[" << start << ", " << end << (endClosed ? "]" : ")");
      return ss.str();
    }
  };

  /// Recorded observations of one channel: one observation per timestamp,
  /// each a tensor of shape `shape`, stored flat in row-major order.
  template <typename T>
  struct ChannelStorage
  {
    Times times;
    std::vector<size_t> shape;
    std::vector<T> values;
  };

  /// Default snap tolerance, as a fraction of the gap between observations.
  inline constexpr float64_t DefaultSnapTolerance = 1e-6;

  /// Default cap on the snap distance, in multiples of machine epsilon
  /// times the magnitude of the timestamps.
  inline constexpr float64_t DefaultSnapEps = 1024;

  /// One channel of a time series: a view of storage that other views may
  /// share. A write materializes the view first unless it owns the storage
  /// alone, so other views never see it. Timestamps never change.
  template <typename T>
  class Channel
  {
  public:
    /// Copy @p values, whose axis 0 is time, into new storage.
    Channel(Times times, const Tensor<T>& values)
    {
      if (values.rank() == 0)
      {
        throw std::invalid_argument("Values must have a time axis");
      }
      if (values.shape(0) != times.size())
      {
        throw std::invalid_argument(
            "Expected one timestamp per observation: got " + std::to_string(times.size())
            + " timestamps for " + std::to_string(values.shape(0)) + " observations");
      }

      auto storage = std::make_shared<ChannelStorage<T>>(ChannelStorage<T>{
          std::move(times),
          std::vector<size_t>(values.shape().begin() + 1, values.shape().end()),
          {}});
      // ponytail: copies on construction; adopt the caller's storage if this shows up in profiles
      auto owned = values.copy();
      storage->values.reserve(owned.size());
      walk(owned, [&](const std::vector<size_t>& index) {
        storage->values.push_back(owned(index));
      });
      m_storage = std::move(storage);
      if (size() > 0)
      {
        // A channel built from observations covers [t_0, t_{n-1}].
        const auto& recorded = m_storage->times.values();
        m_domain = Domain{recorded.front(), recorded.back(), true};
      }
    }

    /// Where the channel can be evaluated; empty for a channel without
    /// observations.
    [[nodiscard]] const std::optional<Domain>& domain() const noexcept { return m_domain; }

    /// The observation with the largest timestamp at or before @p t, which is
    /// in the channel's own unit. A query below an observation by at most
    /// @p snapTolerance times the gap to the previous observation, and by at
    /// most @p snapEps times machine epsilon times the larger magnitude
    /// of the two timestamps, counts as reaching it.
    [[nodiscard]] Tensor<T> evaluate(float64_t t, float64_t snapTolerance, float64_t snapEps) const
    {
      if (!m_domain || !m_domain->contains(t))
      {
        std::ostringstream ss;
        ss << "Time " << t << " is outside the domain "
           << (m_domain ? m_domain->to_string() : std::string("(empty)"));
        throw std::domain_error(ss.str());
      }
      const auto& times = m_storage->times.values();
      auto next = std::upper_bound(times.begin(), times.end(), t);
      // A query computed in floating point, such as 0.3 - 0.1, can land just
      // below the observation it means; snap it forward. Measuring against
      // the gap keeps the rule independent of the unit and of how large the
      // timestamps are, and a tolerance below 1 never skips an observation.
      // The epsilon cap keeps widely spaced observations from snapping
      // queries that are far off in absolute terms but small next to the gap.
      // t >= times.front(), so next has a predecessor.
      if (next != times.end())
      {
        const auto prev = *std::prev(next);
        const auto magnitude = std::max(std::abs(prev), std::abs(*next));
        const auto cap = snapEps * std::numeric_limits<float64_t>::epsilon() * magnitude;
        if (*next - t <= std::min(snapTolerance * (*next - prev), cap))
        {
          ++next;
        }
      }
      return observation(std::distance(times.begin(), next) - 1);
    }

    [[nodiscard]] const Times& times() const noexcept { return m_storage->times; }
    [[nodiscard]] const std::vector<size_t>& shape() const noexcept { return m_storage->shape; }
    [[nodiscard]] size_t size() const noexcept { return m_storage->times.size(); }

    /// A copy of the values as a tensor of shape (size(), *shape()).
    [[nodiscard]] Tensor<T> values() const
    {
      std::vector<size_t> fullShape{size()};
      fullShape.insert(fullShape.end(), shape().begin(), shape().end());
      Tensor<T> out(fullShape);
      size_t i = 0;
      walk(out, [&](const std::vector<size_t>& index) {
        out(index) = m_storage->values[i++];
      });
      return out;
    }

    /// A copy of observation @p i as a tensor of shape shape(). Negative
    /// indices count from the end.
    [[nodiscard]] Tensor<T> observation(std::ptrdiff_t i) const
    {
      const auto n = static_cast<std::ptrdiff_t>(size());
      if (i < -n || i >= n)
      {
        throw std::out_of_range(
            "Observation index " + std::to_string(i) + " is out of range for "
            + std::to_string(n) + " observations");
      }
      const size_t index = static_cast<size_t>(i < 0 ? i + n : i);
      const size_t blockSize = std::accumulate(
          shape().begin(), shape().end(), size_t{1}, std::multiplies<>());
      Tensor<T> out(shape());
      size_t k = index * blockSize;
      walk(out, [&](const std::vector<size_t>& outIndex) {
        out(outIndex) = m_storage->values[k++];
      });
      return out;
    }

  private:
    std::shared_ptr<ChannelStorage<T>> m_storage;
    std::optional<Domain> m_domain;
  };

  /// Named channels of one element type, queried in one time unit.
  template <typename T>
  class TimeSeries
  {
  public:
    using value_type = T;

    TimeSeries(std::vector<std::string> names, std::vector<Channel<T>> channels, TimeUnit unit)
      : m_names(std::move(names)), m_channels(std::move(channels)), m_unit(unit)
    {
      if (m_channels.empty())
      {
        throw std::invalid_argument("A time series needs at least one channel");
      }
      if (m_names.size() != m_channels.size())
      {
        throw std::invalid_argument("Expected one name per channel");
      }
    }

    /// A series with one channel named "0", queried in the unit of its times.
    TimeSeries(Times times, const Tensor<T>& values, float64_t snapTolerance, float64_t snapEps)
      : TimeSeries({"0"}, {Channel<T>(times, values)}, times.unit())
    {
      set_snap_tolerance(snapTolerance);
      set_snap_eps(snapEps);
    }

    /// The snap tolerance used when evaluating, as a fraction of the gap
    /// between observations; 0 disables snapping.
    [[nodiscard]] float64_t snap_tolerance() const noexcept { return m_snapTolerance; }

    void set_snap_tolerance(float64_t tolerance)
    {
      if (!(tolerance >= 0 && tolerance < 1))
      {
        throw std::invalid_argument("The snap tolerance must be in [0, 1)");
      }
      m_snapTolerance = tolerance;
    }

    /// The cap on the snap distance, in multiples of machine epsilon times
    /// the magnitude of the timestamps.
    [[nodiscard]] float64_t snap_eps() const noexcept { return m_snapEps; }

    void set_snap_eps(float64_t eps)
    {
      if (!(eps >= 0 && std::isfinite(eps)))
      {
        throw std::invalid_argument("snap_eps must be finite and nonnegative");
      }
      m_snapEps = eps;
    }

    [[nodiscard]] const std::vector<std::string>& channel_names() const noexcept { return m_names; }
    [[nodiscard]] const Channel<T>& channel(size_t i) const { return m_channels.at(i); }
    [[nodiscard]] size_t n_channels() const noexcept { return m_channels.size(); }
    [[nodiscard]] TimeUnit unit() const noexcept { return m_unit; }

    /// The same series queried in @p unit; the channels are shared, not
    /// converted.
    [[nodiscard]] TimeSeries to(TimeUnit unit) const
    {
      auto series = *this;
      series.m_unit = unit;
      return series;
    }

    /// The timestamps of channel @p i, converted to the query unit.
    [[nodiscard]] Times times(size_t i) const { return channel(i).times().to(m_unit); }

    /// The intersection of the channels' domains, in the series' unit; empty
    /// when the channels share no time.
    [[nodiscard]] std::optional<Domain> domain() const
    {
      std::optional<Domain> result;
      for (size_t i = 0; i < m_channels.size(); ++i)
      {
        const auto& channelDomain = m_channels[i].domain();
        if (!channelDomain)
        {
          return std::nullopt;
        }
        const auto channelUnit = m_channels[i].times().unit();
        const Domain d{convert_time(channelDomain->start, channelUnit, m_unit),
                       convert_time(channelDomain->end, channelUnit, m_unit),
                       channelDomain->endClosed};
        if (!result)
        {
          result = d;
          continue;
        }
        result->start = std::max(result->start, d.start);
        if (d.end < result->end)
        {
          result->end = d.end;
          result->endClosed = d.endClosed;
        }
        else if (d.end == result->end)
        {
          result->endClosed = result->endClosed && d.endClosed;
        }
      }
      if (result && (result->start > result->end
                     || (result->start == result->end && !result->endClosed)))
      {
        return std::nullopt;
      }
      return result;
    }

    /// Evaluate channel @p i at @p t, given in the series' unit.
    [[nodiscard]] Tensor<T> evaluate_channel(size_t i, float64_t t) const
    {
      const auto& c = channel(i);
      return c.evaluate(convert_time(t, m_unit, c.times().unit()), m_snapTolerance, m_snapEps);
    }

  private:
    std::vector<std::string> m_names;
    std::vector<Channel<T>> m_channels;
    TimeUnit m_unit;
    float64_t m_snapTolerance = DefaultSnapTolerance;
    float64_t m_snapEps = DefaultSnapEps;
  };
}

#endif // STABLEBEAR_TIMESERIES_H
