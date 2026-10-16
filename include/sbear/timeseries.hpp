#ifndef STABLEBEAR_TIMESERIES_H
#define STABLEBEAR_TIMESERIES_H

#include "config.hpp"
#include "nested_tensor.hpp"
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
  enum class TimeUnit : uint8_t
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
      int32_t exp;
      int64_t mult;
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

    constexpr const TimeUnitInfo& time_unit_info(TimeUnit unit)
    {
      return TimeUnitTable[static_cast<size_t>(unit)];
    }

    /// 10^k, exact in double for k <= 22.
    constexpr float64_t exact_power_of_ten(int32_t k)
    {
      float64_t result = 1;
      for (int32_t i = 0; i < k; ++i)
      {
        result *= 10;
      }
      return result;
    }
  }

  [[nodiscard]] constexpr std::string_view time_unit_name(TimeUnit unit)
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
  [[nodiscard]] constexpr float64_t convert_time(float64_t value, TimeUnit from, TimeUnit to)
  {
    const auto& src = detail::time_unit_info(from);
    const auto& dst = detail::time_unit_info(to);
    const auto g = std::gcd(src.mult, dst.mult);
    const int32_t diff = src.exp - dst.exp;
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
        if (i > 0 && m_values[i - 1] >= m_values[i])
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
    Channel(Times times, const Tensor<T>& values, float64_t snapTolerance = DefaultSnapTolerance,
            float64_t snapEps = DefaultSnapEps)
    {
      set_snap_tolerance(snapTolerance);
      set_snap_eps(snapEps);
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
    }

    /// Adopt @p values, the observations stored flat in row-major order with
    /// prod(@p shape) values each.
    Channel(Times times, std::vector<size_t> shape, std::vector<T> values,
            float64_t snapTolerance = DefaultSnapTolerance, float64_t snapEps = DefaultSnapEps)
    {
      set_snap_tolerance(snapTolerance);
      set_snap_eps(snapEps);
      const size_t blockSize = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
      if (values.size() != times.size() * blockSize)
      {
        throw std::invalid_argument(
            "Expected " + std::to_string(times.size() * blockSize) + " values for " + std::to_string(times.size())
            + " observations, got " + std::to_string(values.size()));
      }
      m_storage = std::make_shared<ChannelStorage<T>>(
          ChannelStorage<T>{std::move(times), std::move(shape), std::move(values)});
    }

    /// The snap tolerance used when evaluating, as a fraction of the gap
    /// between observations; 0 disables snapping.
    [[nodiscard]] float64_t snap_tolerance() const noexcept { return m_snapTolerance; }

    void set_snap_tolerance(float64_t tolerance)
    {
      if (std::isnan(tolerance) || tolerance < 0 || tolerance >= 1)
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
      if (std::isnan(eps) || std::isinf(eps) || eps < 0)
      {
        throw std::invalid_argument("snap_eps must be finite and nonnegative");
      }
      m_snapEps = eps;
    }

    /// The observation with the largest timestamp at or before @p t, which is
    /// in the channel's own unit. A query below an observation by at most
    /// snap_tolerance() times the gap to the previous observation, and by at
    /// most snap_eps() times machine epsilon times the larger magnitude
    /// of the two timestamps, counts as reaching it.
    ///
    /// The caller checks that @p t lies between the first and last
    /// observations, in its own unit: a time converted into the channel's
    /// unit may round just outside them. A time below the first observation
    /// gets the first. Requires size() > 0.
    [[nodiscard]] Tensor<T> lookup(float64_t t) const
    {
      const auto& times = m_storage->times.values();
      auto next = std::upper_bound(times.begin(), times.end(), t);
      if (next == times.begin())
      {
        return observation(0);
      }
      // A query computed in floating point, such as 0.3 - 0.1, can land just
      // below the observation it means; snap it forward. Measuring against
      // the gap keeps the rule independent of the unit and of how large the
      // timestamps are, and a tolerance below 1 never skips an observation.
      // The epsilon cap keeps widely spaced observations from snapping
      // queries that are far off in absolute terms but small next to the gap.
      if (next != times.end())
      {
        const auto prev = *std::prev(next);
        const auto magnitude = std::max(std::abs(prev), std::abs(*next));
        const auto cap = m_snapEps * std::numeric_limits<float64_t>::epsilon() * magnitude;
        if (*next - t <= std::min(m_snapTolerance * (*next - prev), cap))
        {
          ++next;
        }
      }
      return observation(std::distance(times.begin(), next) - 1);
    }

    [[nodiscard]] const Times& times() const noexcept { return m_storage->times; }
    [[nodiscard]] const std::vector<size_t>& shape() const noexcept { return m_storage->shape; }
    [[nodiscard]] size_t size() const noexcept { return m_storage->times.size(); }

    /// The observations stored flat in row-major order, prod(shape()) values
    /// each.
    [[nodiscard]] const std::vector<T>& flat_values() const noexcept { return m_storage->values; }

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
    float64_t m_snapTolerance = DefaultSnapTolerance;
    float64_t m_snapEps = DefaultSnapEps;
  };

  /// Named channels of one element type, queried in one time unit.
  template <typename T>
  class TimeSeries
  {
  public:
    using value_type = T;

    TimeSeries(std::vector<Channel<T>> channels, TimeUnit unit)
      : m_channels(std::move(channels)), m_unit(unit)
    {
      if (m_channels.empty())
      {
        throw std::invalid_argument("A time series needs at least one channel");
      }
    }

    /// A series with one channel, queried in the unit of its times.
    TimeSeries(Times times, const Tensor<T>& values, float64_t snapTolerance, float64_t snapEps)
      : TimeSeries({Channel<T>(times, values, snapTolerance, snapEps)}, times.unit())
    {
    }

    /// Combine one-channel series into one series; channel i is @p parts[i].
    /// Each channel keeps its own times, unit, and snap
    /// settings. @p unit defaults to the unit the parts share and is required
    /// when their units differ.
    [[nodiscard]] static TimeSeries from_channels(const std::vector<TimeSeries>& parts,
                                                  std::optional<TimeUnit> unit)
    {
      if (parts.empty())
      {
        throw std::invalid_argument("A time series needs at least one channel");
      }
      std::vector<Channel<T>> channels;
      channels.reserve(parts.size());
      for (const auto& part : parts)
      {
        if (part.n_channels() != 1)
        {
          throw std::invalid_argument("Each channel must be a one-channel series; select channels with series[i]");
        }
        channels.push_back(part.m_channels.front());
        if (!unit && part.unit() != parts.front().unit())
        {
          throw std::invalid_argument("The channels differ in unit; pass unit= to choose one");
        }
      }
      return TimeSeries(std::move(channels), unit.value_or(parts.front().unit()));
    }

    [[nodiscard]] const Channel<T>& channel(size_t i) const { return m_channels.at(i); }
    [[nodiscard]] size_t n_channels() const noexcept { return m_channels.size(); }

    /// The channels at @p indices, in that order, sharing their storage with
    /// this series and queried in the same unit. Negative indices count from
    /// the end.
    [[nodiscard]] TimeSeries select(const std::vector<std::ptrdiff_t>& indices) const
    {
      const auto n = static_cast<std::ptrdiff_t>(m_channels.size());
      std::vector<Channel<T>> channels;
      channels.reserve(indices.size());
      for (const auto i : indices)
      {
        if (i < -n || i >= n)
        {
          throw std::out_of_range(
              "Channel index " + std::to_string(i) + " is out of range for " + std::to_string(n) + " channels");
        }
        channels.push_back(m_channels[static_cast<size_t>(i < 0 ? i + n : i)]);
      }
      return TimeSeries(std::move(channels), m_unit);
    }
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

    /// Every channel's value at @p t, given in the series' unit: one leaf per
    /// channel, in channel order, each a tensor of that channel's shape.
    /// @p t must lie between the first and last observations of every
    /// channel, both included.
    [[nodiscard]] NestedTensor<T> evaluate(float64_t t) const
    {
      check_observed(t);
      Tensor<NestedTensor<T>> values({m_channels.size()});
      for (size_t i = 0; i < m_channels.size(); ++i)
      {
        const auto& c = m_channels[i];
        values(i) = NestedTensor<T>::from_leaf_view(c.lookup(convert_time(t, m_unit, c.times().unit())));
      }
      return NestedTensor<T>::from_outer_view(std::move(values), 1);
    }

  private:
    /// Throw unless @p t, in the series' unit, lies between the first and
    /// last observations of every channel. The check is exact: it compares
    /// against the timestamps converted to the series' unit, not the query
    /// converted to each channel's unit.
    void check_observed(float64_t t) const
    {
      auto first = -std::numeric_limits<float64_t>::infinity();
      auto last = std::numeric_limits<float64_t>::infinity();
      for (const auto& c : m_channels)
      {
        if (c.size() == 0)
        {
          throw std::domain_error("Cannot evaluate a channel without observations");
        }
        const auto& recorded = c.times().values();
        first = std::max(first, convert_time(recorded.front(), c.times().unit(), m_unit));
        last = std::min(last, convert_time(recorded.back(), c.times().unit(), m_unit));
      }
      if (t < first || t > last)
      {
        std::ostringstream ss;
        ss << "Time " << t << " is outside the domain [" << first << ", " << last << "]";
        throw std::domain_error(ss.str());
      }
    }

    std::vector<Channel<T>> m_channels;
    TimeUnit m_unit;
  };

  template <typename T>
  struct is_timeseries : std::false_type {};

  template <typename T>
  struct is_timeseries<TimeSeries<T>> : std::true_type { using value_type = T; };

  template <typename T>
  inline constexpr bool is_timeseries_v = is_timeseries<T>::value;
}

#endif // STABLEBEAR_TIMESERIES_H
