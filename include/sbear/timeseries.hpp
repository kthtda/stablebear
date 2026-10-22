#ifndef STABLEBEAR_TIMESERIES_H
#define STABLEBEAR_TIMESERIES_H

#include "algorithms/gallop.hpp"
#include "config.hpp"
#include "nested_tensor.hpp"
#include "tensor.hpp"
#include "walk.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
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

  /// One channel of a time series: an immutable view of storage that other
  /// views may share. Observations leave the channel as deep copies, so
  /// mutating them never changes the channel or its views.
  ///
  /// A channel either holds stored observations or is a view whose
  /// observations are those of a source channel: stacked at lagged times by
  /// stack_at, or picked by index by take. A view keeps its times, its lags,
  /// and for each observation a base: the source observation for the
  /// smallest lag. The other components are found by searching back from the
  /// base when read, so a view costs two numbers per observation and copies
  /// no values. A view evaluates as any channel does, by previous value on
  /// its own times with its own snapping, so it shows exactly the
  /// observations it holds and nothing else of its source.
  template <typename T>
  class Channel
  {
  public:
    /// Copy @p values, whose axis 0 is time, into new storage.
    Channel(Times times, const Tensor<T>& values, float64_t snapTolerance = DefaultSnapTolerance,
            float64_t snapEps = DefaultSnapEps);

    /// Adopt @p values, the observations stored flat in row-major order with
    /// prod(@p shape) values each.
    Channel(Times times, std::vector<size_t> shape, std::vector<T> values,
            float64_t snapTolerance = DefaultSnapTolerance, float64_t snapEps = DefaultSnapEps);

    /// The snap tolerance used when evaluating, as a fraction of the gap
    /// between observations; 0 disables snapping.
    [[nodiscard]] float64_t snap_tolerance() const noexcept { return m_snapTolerance; }

    void set_snap_tolerance(float64_t tolerance);

    /// The cap on the snap distance, in multiples of machine epsilon times
    /// the magnitude of the timestamps.
    [[nodiscard]] float64_t snap_eps() const noexcept { return m_snapEps; }

    void set_snap_eps(float64_t eps);

    /// A view of this channel whose times are @p times, in the channel's
    /// unit, and whose observation k stacks, along a new leading axis, this
    /// channel's observations in effect at times[k] - lags[j] for each j, in
    /// order. A lagged time before the first observation gets the first. It
    /// shares this channel and its snap settings.
    [[nodiscard]] Channel stack_at(Times times, std::span<const float64_t> lags) const;

    /// A view of this channel whose times are @p times and whose observation
    /// k is this channel's observation @p indices[k], with no new axis. It
    /// shares this channel and its snap settings, but not its start or end.
    [[nodiscard]] Channel take(Times times, std::vector<size_t> indices) const;

    /// This channel with the end @p end, in its unit: it can be evaluated
    /// after its last timestamp up to @p end, and at @p end if @p closed.
    /// @p end must come after the last timestamp. Views made by stack_at
    /// keep the end.
    [[nodiscard]] Channel until(float64_t end, bool closed = false) const;

    /// The end set by until(), if any; without one, the channel can be
    /// evaluated up to and at its last timestamp.
    [[nodiscard]] std::optional<float64_t> end() const noexcept { return m_end; }

    /// Whether the end, end() if set and else the last timestamp, is in the
    /// domain.
    [[nodiscard]] bool end_closed() const noexcept { return !m_end || m_endClosed; }

    /// This channel with its first timestamp in the domain if @p closed. If
    /// not, the first observation still stays in effect after it. Views made
    /// by stack_at keep the start.
    [[nodiscard]] Channel with_start_closed(bool closed) const;

    /// Whether the first timestamp is in the domain.
    [[nodiscard]] bool start_closed() const noexcept { return m_startClosed; }

    /// The observation with the largest timestamp at or before @p t, in the
    /// channel's own unit, a tensor of shape shape(). A query below an
    /// observation by at most snap_tolerance() times the gap to the previous
    /// observation, and by at most snap_eps() times machine epsilon times the
    /// larger magnitude of the two timestamps, counts as reaching it.
    ///
    /// The caller checks that @p t lies between the first and last
    /// observations, in its own unit: a time converted into the channel's
    /// unit may round just outside them. A time below the first observation
    /// gets the first. Requires size() > 0.
    [[nodiscard]] Tensor<T> lookup(float64_t t) const;

    /// The index of the observation in effect at @p t, in the channel's unit,
    /// with the snapping of lookup(); a time below the first observation gets
    /// the first. This is the only way from a time to an observation, so
    /// every lookup snaps: the timestamps are not handed out to search.
    /// The search gallops from @p hint, any index, so a hint d timestamps
    /// from the answer costs O(log d) comparisons; the answer does not depend
    /// on it. Passing the previous answer makes a run of nearby queries
    /// cheap. Requires size() > 0.
    [[nodiscard]] size_t index_at(float64_t t, size_t hint = 0) const;

    /// Write the observations at @p sortedTimes, ascending and in the
    /// channel's own unit, to @p out: the one at sortedTimes[k] goes to block
    /// order[k], each block prod(shape()) values long. @p out is a
    /// random-access iterator, such as a tensor's data() or a vector's
    /// begin(). One merged pass over the timestamps, as in lookup(). Requires
    /// size() > 0.
    template <std::random_access_iterator Out>
    void lookup_sorted(std::span<const float64_t> sortedTimes, std::span<const size_t> order, Out out) const;

    /// A copy of the timestamps, to save or report them; find observations
    /// with index_at().
    [[nodiscard]] Times times() const { return held_times(); }

    /// Timestamp @p k, in the channel's unit, to build new times from.
    [[nodiscard]] float64_t time(size_t k) const { return held_times().values()[k]; }

    [[nodiscard]] TimeUnit unit() const noexcept { return held_times().unit(); }
    [[nodiscard]] const std::vector<size_t>& shape() const noexcept { return m_shape; }
    [[nodiscard]] size_t size() const noexcept { return held_times().size(); }

    /// The same timestamps, start, end, shape, values, and snap settings.
    [[nodiscard]] bool operator==(const Channel& other) const;

    /// Element @p i of the observations at times(), flattened in row-major
    /// order: observation i / prod(shape()), element i % prod(shape()).
    [[nodiscard]] typename std::vector<T>::const_reference flat_value(size_t i) const;

    /// A copy of the values as a tensor of shape (size(), *shape()).
    [[nodiscard]] Tensor<T> values() const;

    /// A copy of observation @p i as a tensor of shape shape(). Negative
    /// indices count from the end.
    [[nodiscard]] Tensor<T> observation(std::ptrdiff_t i) const;

  private:
    /// The timestamps, kept inside the channel so that searches go through
    /// index_at().
    [[nodiscard]] const Times& held_times() const noexcept { return m_times ? *m_times : m_storage->times; }

    /// The number of values in one observation, prod(shape()).
    [[nodiscard]] size_t block_size() const noexcept
    {
      return std::accumulate(m_shape.begin(), m_shape.end(), size_t{1}, std::multiplies<>());
    }

    /// Write observation @p k, prod(shape()) values, to @p out.
    template <std::random_access_iterator Out>
    void copy_observation(size_t k, Out out) const;

    /// The observation indices for @p sortedTimes, ascending and in the
    /// channel's unit, found in one merged pass. Requires size() > 0.
    [[nodiscard]] std::vector<size_t> indices_sorted(std::span<const float64_t> sortedTimes) const;

    /// The index of the observation for @p t, given @p next, the index of the
    /// first timestamp after t.
    [[nodiscard]] size_t snap(size_t next, float64_t t) const;

    /// For a view, the source observation that component @p j of observation
    /// @p k reads.
    [[nodiscard]] size_t source_index(size_t k, size_t j) const;

    std::shared_ptr<ChannelStorage<T>> m_storage;  // null for a view
    std::shared_ptr<const Channel> m_source;       // null for stored observations
    std::shared_ptr<const Times> m_times;
    std::shared_ptr<const std::vector<size_t>> m_base;
    std::vector<float64_t> m_lags;
    std::vector<size_t> m_shape;
    std::optional<float64_t> m_end;
    bool m_endClosed = false;
    bool m_startClosed = true;
    float64_t m_snapTolerance = DefaultSnapTolerance;
    float64_t m_snapEps = DefaultSnapEps;
  };

  /// Named channels of one element type, queried in one time unit.
  template <typename T>
  class TimeSeries
  {
  public:
    using value_type = T;

    /// An empty one-channel series in seconds, so that series can be tensor
    /// elements.
    TimeSeries()
      : TimeSeries(Times(std::vector<float64_t>{}), Tensor<T>(std::vector<size_t>{0}), DefaultSnapTolerance,
                   DefaultSnapEps)
    {
    }

    TimeSeries(std::vector<Channel<T>> channels, TimeUnit unit);

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
                                                  std::optional<TimeUnit> unit);

    [[nodiscard]] const Channel<T>& channel(size_t i) const { return m_channels.at(i); }
    [[nodiscard]] size_t n_channels() const noexcept { return m_channels.size(); }

    /// The channels at @p indices, in that order, sharing their storage with
    /// this series and queried in the same unit. Negative indices count from
    /// the end.
    [[nodiscard]] TimeSeries select(const std::vector<std::ptrdiff_t>& indices) const;
    [[nodiscard]] TimeUnit unit() const noexcept { return m_unit; }

    /// The same series queried in @p unit; the channels are shared, not
    /// converted.
    [[nodiscard]] TimeSeries to(TimeUnit unit) const;

    /// The timestamps of channel @p i, converted to the query unit.
    [[nodiscard]] Times times(size_t i) const { return channel(i).times().to(m_unit); }

    /// The times, in the series' unit, at which every channel can be
    /// evaluated: from the latest first timestamp to the earliest end, a
    /// channel's last timestamp or its end, converted exactly as
    /// check_observed() compares.
    struct Domain
    {
      float64_t first;
      float64_t last;
      bool closedStart;  // first itself is in the domain
      bool closedEnd;    // last itself is in the domain

      /// Whether @p t is in the domain; NaN is not.
      [[nodiscard]] bool contains(float64_t t) const noexcept
      {
        return (t > first || (closedStart && t == first)) && (t < last || (closedEnd && t == last));
      }

      /// The domain as text, such as "[0, 2)".
      [[nodiscard]] std::string to_string() const;
    };

    /// The domain of the series. Requires every channel to have
    /// observations.
    [[nodiscard]] Domain domain() const;

    /// The same unit and channels.
    [[nodiscard]] bool operator==(const TimeSeries& other) const = default;

    /// Every channel's value at @p t, given in the series' unit: one leaf per
    /// channel, in channel order, each a tensor of that channel's shape.
    /// @p t must lie between the first and last observations of every
    /// channel, both included.
    [[nodiscard]] NestedTensor<T> evaluate(float64_t t) const;

    /// Every channel's values at the times @p t, given in the series' unit:
    /// one leaf per channel, in channel order, each a tensor of shape
    /// (*t.shape(), *shape). The queries are sorted once, in O(m log m) for m
    /// queries, and merged against each channel's timestamps, galloping
    /// between queries: at most O(m log n) and O(m + n) for n observations.
    /// Every time must lie in the domain.
    [[nodiscard]] NestedTensor<T> evaluate(const Tensor<float64_t>& t) const;

  private:
    /// Throw std::domain_error, naming the domain, unless @p t, in the
    /// series' unit, lies in the domain. The check is exact: it compares
    /// against the timestamps converted to the series' unit, not the query
    /// converted to each channel's unit.
    void check_observed(float64_t t) const;

    std::vector<Channel<T>> m_channels;
    TimeUnit m_unit;
  };

  template <typename T>
  struct is_timeseries : std::false_type {};

  template <typename T>
  struct is_timeseries<TimeSeries<T>> : std::true_type { using value_type = T; };

  template <typename T>
  inline constexpr bool is_timeseries_v = is_timeseries<T>::value;

  /// Every series of @p series evaluated at the times @p t, each in its own
  /// unit, as TimeSeries::evaluate does: one leaf per channel, in channel
  /// order, of shape (*series.shape(), *t.shape(), *shape) of that channel.
  /// Requires every series to have the same number of channels and channel
  /// shapes, and @p series to be non-empty. A time outside the domain of a
  /// series throws std::domain_error naming the first such series'
  /// position, as in "TimeSeries at position (1, 2)".
  /// The series are evaluated in parallel on @p exec.
  template <typename T>
  [[nodiscard]] NestedTensor<T> evaluate(const Tensor<TimeSeries<T>>& series, const Tensor<float64_t>& t,
                                         Executor& exec = default_executor());
}

#include "timeseries.tpp"

#endif // STABLEBEAR_TIMESERIES_H
