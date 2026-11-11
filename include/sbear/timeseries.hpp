#ifndef STABLEBEAR_TIMESERIES_H
#define STABLEBEAR_TIMESERIES_H

#include "algorithms/gallop.hpp"
#include "config.hpp"
#include "errors.hpp"
#include "nested_tensor.hpp"
#include "tensor.hpp"
#include "walk.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <optional>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <string_view>
#include <utility>
#include <vector>

namespace sb
{
  /// Physical time units, from attoseconds to days, and Unitless for times
  /// that are plain numbers, which convert only to themselves.
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
    Days,
    Unitless
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
      // No length; saved as 0 * 10^0 seconds.
      {TimeUnit::Unitless, "", 0, 0},
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
      {"day", TimeUnit::Days}, {"days", TimeUnit::Days}, {"D", TimeUnit::Days},  // NumPy's spelling
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
      if (info.unit != TimeUnit::Unitless && info.name == name)
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
  /// A plain number, from Unitless, becomes a count of @p to unchanged; a
  /// time in a unit cannot become a plain number.
  [[nodiscard]] constexpr float64_t convert_time(float64_t value, TimeUnit from, TimeUnit to)
  {
    if (from == to || from == TimeUnit::Unitless)
    {
      return value;
    }
    if (to == TimeUnit::Unitless)
    {
      throw std::invalid_argument("Times in a unit cannot become plain numbers");
    }
    const auto& src = detail::time_unit_info(from);
    const auto& dst = detail::time_unit_info(to);
    const auto g = std::gcd(src.mult, dst.mult);
    const int32_t diff = src.exp - dst.exp;
    const float64_t num = static_cast<float64_t>(src.mult / g) * detail::exact_power_of_ten(std::max(diff, 0));
    const float64_t den = static_cast<float64_t>(dst.mult / g) * detail::exact_power_of_ten(std::max(-diff, 0));
    return den == 1 ? value * num : value / den;
  }

  namespace detail
  {
    /// Raises each of @p values that rounding put at or before the one
    /// before it to just after that one, so that converted or shifted
    /// timestamps stay strictly increasing.
    inline void keep_increasing(std::vector<float64_t>& values)
    {
      for (size_t i = 1; i < values.size(); ++i)
      {
        values[i] = std::max(values[i], std::nextafter(values[i - 1], std::numeric_limits<float64_t>::infinity()));
      }
    }

    /// The time unit of a std::chrono period, at compile time.
    template <typename Period>
    constexpr TimeUnit time_unit_of()
    {
      if constexpr (std::ratio_equal_v<Period, std::atto>) return TimeUnit::Attoseconds;
      else if constexpr (std::ratio_equal_v<Period, std::femto>) return TimeUnit::Femtoseconds;
      else if constexpr (std::ratio_equal_v<Period, std::pico>) return TimeUnit::Picoseconds;
      else if constexpr (std::ratio_equal_v<Period, std::nano>) return TimeUnit::Nanoseconds;
      else if constexpr (std::ratio_equal_v<Period, std::micro>) return TimeUnit::Microseconds;
      else if constexpr (std::ratio_equal_v<Period, std::milli>) return TimeUnit::Milliseconds;
      else if constexpr (std::ratio_equal_v<Period, std::ratio<1>>) return TimeUnit::Seconds;
      else if constexpr (std::ratio_equal_v<Period, std::ratio<60>>) return TimeUnit::Minutes;
      else if constexpr (std::ratio_equal_v<Period, std::ratio<3600>>) return TimeUnit::Hours;
      else if constexpr (std::ratio_equal_v<Period, std::ratio<86400>>) return TimeUnit::Days;
      else static_assert(sizeof(Period) == 0, "No time unit has this period");
    }
  }

  /// Throw TypeError unless times in @p queryUnit suit a series in
  /// @p seriesUnit: durations, which come with a unit, for a series with
  /// one, and plain numbers for a series without one.
  inline void check_query_unit(TimeUnit queryUnit, TimeUnit seriesUnit)
  {
    if (queryUnit != TimeUnit::Unitless && seriesUnit == TimeUnit::Unitless)
    {
      throw TypeError("A series without a unit takes plain numbers, not durations");
    }
    if (queryUnit == TimeUnit::Unitless && seriesUnit != TimeUnit::Unitless)
    {
      throw TypeError("These series have a unit, so their times are durations, not plain numbers");
    }
  }

  /// Throw unless both @p a and @p b are units, or neither is.
  inline void check_same_kind_of_unit(TimeUnit a, TimeUnit b)
  {
    if ((a == TimeUnit::Unitless) != (b == TimeUnit::Unitless))
    {
      throw std::invalid_argument("A series without a unit and a series with one cannot be combined or converted");
    }
  }

  /// A calendar instant: @p count units after 1970-01-01T00:00, the value of
  /// a NumPy datetime64 at its own resolution. As in NumPy and
  /// std::chrono::sys_time, dates follow the Gregorian calendar and every day
  /// has 86400 seconds: leap seconds are not counted.
  struct DateTime
  {
    int64_t count;
    TimeUnit unit;

    constexpr DateTime(int64_t count, TimeUnit unit) noexcept : count(count), unit(unit) {}

    /// The instant @p time, at its resolution, such as
    /// DateTime(std::chrono::sys_days{2023y / January / 1}).
    template <typename Duration>
      requires std::integral<typename Duration::rep>
    constexpr explicit DateTime(std::chrono::sys_time<Duration> time) noexcept
      : DateTime(static_cast<int64_t>(time.time_since_epoch().count()), detail::time_unit_of<typename Duration::period>())
    {
    }

    /// The same count at the same resolution; the same instant at another
    /// resolution is a different DateTime.
    bool operator==(const DateTime&) const = default;
  };

  namespace detail
  {
    /// Whether @p a is a unit at most as long as @p b.
    constexpr bool not_coarser(const TimeUnitInfo& a, const TimeUnitInfo& b)
    {
      return a.exp < b.exp || (a.exp == b.exp && a.mult <= b.mult);
    }

    /// The number of @p fine in one @p coarse, a unit at least as long, or
    /// nullopt if it exceeds int64. It is a whole number, since the
    /// multipliers of coarser units are multiples of finer ones.
    constexpr std::optional<int64_t> units_per(const TimeUnitInfo& coarse, const TimeUnitInfo& fine)
    {
      int64_t ratio = coarse.mult / fine.mult;
      for (int32_t k = fine.exp; k < coarse.exp; ++k)
      {
        if (ratio > std::numeric_limits<int64_t>::max() / 10)
        {
          return std::nullopt;
        }
        ratio *= 10;
      }
      return ratio;
    }

    /// Whether @p count of a unit, @p factor of a finer one each, is below
    /// 2^62 in the finer unit, so that a difference of two such fits in int64.
    constexpr bool fits_in_fine(int64_t count, std::optional<int64_t> factor)
    {
      if (!factor)
      {
        return false;
      }
      const auto most = ((int64_t{1} << 62) - 1) / *factor;
      return count <= most && count >= -most;
    }
  }

  /// The time from @p epoch to @p date in @p unit. When both counts fit
  /// below 2^62 in the finest of the three units, the difference is taken
  /// there in integers and converted to @p unit as convert_time() converts
  /// it. Otherwise each count is split into whole days and a remainder in
  /// its own unit, and the difference of the days and that of the
  /// remainders, taken in integers when a day of the finer unit fits, are
  /// converted to @p unit separately and added, so that no resolution or
  /// span overflows. stablebear.timeseries computes
  /// dates the same way, so that a series and a tensor of it agree.
  /// Throws std::invalid_argument if a unit has no length, and
  /// std::overflow_error if the days differ by more than int64 holds.
  [[nodiscard]] inline float64_t time_since(const DateTime& date, const DateTime& epoch, TimeUnit unit)
  {
    if (date.unit == TimeUnit::Unitless || epoch.unit == TimeUnit::Unitless || unit == TimeUnit::Unitless)
    {
      throw std::invalid_argument("Dates, epochs, and the times between them need a unit with a length");
    }
    const auto& d = detail::time_unit_info(date.unit);
    const auto& e = detail::time_unit_info(epoch.unit);
    const auto& u = detail::time_unit_info(unit);
    const auto& fine = detail::not_coarser(d, e) ? (detail::not_coarser(d, u) ? d : u)
                                                 : (detail::not_coarser(e, u) ? e : u);
    const auto dateFactor = detail::units_per(d, fine);
    const auto epochFactor = detail::units_per(e, fine);
    if (detail::fits_in_fine(date.count, dateFactor) && detail::fits_in_fine(epoch.count, epochFactor))
    {
      return convert_time(static_cast<float64_t>(date.count * *dateFactor - epoch.count * *epochFactor), fine.unit,
                          unit);
    }
    // Each count as whole days and a remainder in its own unit.
    const auto& days = detail::time_unit_info(TimeUnit::Days);
    const auto split = [&](int64_t count, const detail::TimeUnitInfo& info) {
      // A ratio beyond int64 exceeds every count, which is then all remainder.
      const auto ratio = detail::units_per(days, info);
      return ratio ? std::pair{count / *ratio, count % *ratio} : std::pair{int64_t{0}, count};
    };
    const auto [dateDays, dateRest] = split(date.count, d);
    const auto [epochDays, epochRest] = split(epoch.count, e);
    constexpr auto max = std::numeric_limits<int64_t>::max();
    constexpr auto min = std::numeric_limits<int64_t>::min();
    if ((epochDays > 0 && dateDays < min + epochDays) || (epochDays < 0 && dateDays > max + epochDays))
    {
      throw std::overflow_error("A date is too many days from the epoch to count");
    }
    const auto dayDifference = convert_time(static_cast<float64_t>(dateDays - epochDays), TimeUnit::Days, unit);
    // The remainders, below a day each, subtract in int64 in the finer of
    // their units when a day of it is below 2^62, as down to picoseconds.
    const auto& finer = detail::not_coarser(d, e) ? d : e;
    if (detail::fits_in_fine(1, detail::units_per(days, finer)))
    {
      const auto restDifference = dateRest * *detail::units_per(d, finer) - epochRest * *detail::units_per(e, finer);
      return dayDifference + convert_time(static_cast<float64_t>(restDifference), finer.unit, unit);
    }
    return dayDifference + convert_time(static_cast<float64_t>(dateRest), date.unit, unit)
           - convert_time(static_cast<float64_t>(epochRest), epoch.unit, unit);
  }

  /// One-dimensional, finite, strictly increasing timestamps, placed by an
  /// origin in the series they were cut from. Their unit and epoch are those
  /// of the series that holds them.
  class Times
  {
  public:
    explicit Times(std::vector<float64_t> values) : m_values(std::move(values))
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
    [[nodiscard]] size_t size() const noexcept { return m_values.size(); }

    /// The time that time 0 stands for in the series these times were cut
    /// from; 0 for recorded observations. Evaluation does not use it.
    [[nodiscard]] float64_t origin() const noexcept { return m_origin; }

    /// The same timestamps with the origin @p origin, which must be finite.
    [[nodiscard]] Times with_origin(float64_t origin) const
    {
      if (!std::isfinite(origin))
      {
        throw std::invalid_argument("The origin of times must be finite");
      }
      auto times = *this;
      times.m_origin = origin;
      return times;
    }

    /// The timestamps @p values with this origin.
    [[nodiscard]] Times with_values(std::vector<float64_t> values) const
    {
      Times times(std::move(values));
      times.m_origin = m_origin;
      return times;
    }

    /// The same timestamps and origin converted from @p from to @p to, as
    /// convert_time() converts them.
    [[nodiscard]] Times to(TimeUnit from, TimeUnit to) const
    {
      std::vector<float64_t> converted(m_values.size());
      for (size_t i = 0; i < m_values.size(); ++i)
      {
        converted[i] = convert_time(m_values[i], from, to);
      }
      detail::keep_increasing(converted);
      Times times(std::move(converted));
      times.m_origin = convert_time(m_origin, from, to);
      return times;
    }

    bool operator==(const Times&) const = default;

  private:
    std::vector<float64_t> m_values;
    float64_t m_origin = 0.0;
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

  /// Default snap tolerance, relative to the magnitude of the timestamps:
  /// the default comparison tolerance of APL.
  inline constexpr float64_t DefaultSnapTolerance = 1e-14;

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
    Channel(Times times, const Tensor<T>& values, float64_t snapTolerance = DefaultSnapTolerance);

    /// Adopt @p values, the observations stored flat in row-major order with
    /// prod(@p shape) values each.
    Channel(Times times, std::vector<size_t> shape, std::vector<T> values,
            float64_t snapTolerance = DefaultSnapTolerance);

    /// The snap tolerance used when evaluating, relative to the magnitude
    /// of the timestamps; 0 disables snapping.
    [[nodiscard]] float64_t snap_tolerance() const noexcept { return m_snapTolerance; }

    void set_snap_tolerance(float64_t tolerance);

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
    /// @p end must come after the last timestamp, or be at it if @p closed,
    /// which is the same as setting no end. Views made by stack_at keep the
    /// end.
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

    /// This channel with its timestamps, end, and origin, and for a view its
    /// lags and source, converted from @p from to @p to as convert_time()
    /// converts them, so that timestamps without a unit become counts of
    /// @p to. It shares the observations.
    [[nodiscard]] Channel to(TimeUnit from, TimeUnit to) const;

    /// This channel with @p delta added to every timestamp and to its end. A
    /// view shifts its source alike, so it reads the same observations. It
    /// shares the observations; only the timestamps are copied.
    [[nodiscard]] Channel shifted(float64_t delta) const;

    /// Whether the first timestamp is in the domain.
    [[nodiscard]] bool start_closed() const noexcept { return m_startClosed; }

    /// The origin of the channel's times.
    [[nodiscard]] float64_t origin() const noexcept { return held_times().origin(); }

    /// The observation with the largest timestamp at or before @p t, in the
    /// channel's own unit, a tensor of shape shape(). As in the tolerant
    /// comparison of APL, a query below an observation by at most
    /// snap_tolerance() times the larger magnitude of it and the previous
    /// timestamp counts as reaching it, as long as it is also within a
    /// quarter of the gap to the previous timestamp, so that a query at the
    /// previous timestamp still gets the observation recorded there.
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

    /// The timestamps @p values with this channel's origin, without copying
    /// its own timestamps.
    [[nodiscard]] Times times_like(std::vector<float64_t> values) const
    {
      return held_times().with_values(std::move(values));
    }

    /// Timestamp @p k, in the channel's unit, to build new times from.
    [[nodiscard]] float64_t time(size_t k) const { return held_times().values()[k]; }

    [[nodiscard]] const std::vector<size_t>& shape() const noexcept { return m_shape; }
    [[nodiscard]] size_t size() const noexcept { return held_times().size(); }

    /// The same timestamps, start, end, shape, values, and snap settings;
    /// the timestamps include their epoch and origin.
    [[nodiscard]] bool operator==(const Channel& other) const;

    /// operator== without the origin of the timestamps, which places a
    /// channel but does not change how it evaluates.
    [[nodiscard]] bool same_observations(const Channel& other) const;

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

    /// This channel with an end that rounding put at or before its last
    /// timestamp placed as windows place it: a closed end is dropped, and an
    /// open one moves to just after the last timestamp.
    [[nodiscard]] Channel with_end_after_last() const;

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
    std::shared_ptr<const Times> m_times;  // set for a view, or to relabel stored times
    std::shared_ptr<const std::vector<size_t>> m_base;
    std::vector<float64_t> m_lags;
    std::vector<size_t> m_shape;
    std::optional<float64_t> m_end;
    bool m_endClosed = false;
    bool m_startClosed = true;
    float64_t m_snapTolerance = DefaultSnapTolerance;
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
      : TimeSeries(Times(std::vector<float64_t>{}), Tensor<T>(std::vector<size_t>{0}), DefaultSnapTolerance)
    {
    }

    /// A series of @p channels, whose times are in @p unit, counted from
    /// @p epoch, or from none. A series without a unit has no epoch.
    TimeSeries(std::vector<Channel<T>> channels, TimeUnit unit, std::optional<DateTime> epoch = std::nullopt);

    /// A series with one channel.
    TimeSeries(Times times, const Tensor<T>& values, float64_t snapTolerance, TimeUnit unit = TimeUnit::Seconds,
               std::optional<DateTime> epoch = std::nullopt)
      : TimeSeries({Channel<T>(std::move(times), values, snapTolerance)}, unit, epoch)
    {
    }

    /// Combine one-channel series into one series; channel i is @p parts[i]
    /// converted to @p unit, with its own snap settings. @p unit defaults to
    /// the unit the parts share and is required when their units differ.
    [[nodiscard]] static TimeSeries from_channels(const std::vector<TimeSeries>& parts,
                                                  std::optional<TimeUnit> unit);

    [[nodiscard]] const Channel<T>& channel(size_t i) const { return m_channels.at(i); }
    [[nodiscard]] size_t n_channels() const noexcept { return m_channels.size(); }

    /// The channels at @p indices, in that order, sharing their storage with
    /// this series and queried in the same unit. Negative indices count from
    /// the end.
    [[nodiscard]] TimeSeries select(const std::vector<std::ptrdiff_t>& indices) const;
    [[nodiscard]] TimeUnit unit() const noexcept { return m_unit; }

    /// The instant time 0 stands for, if any.
    [[nodiscard]] const std::optional<DateTime>& epoch() const noexcept { return m_epoch; }

    /// The same series in @p unit, with every channel converted by
    /// Channel::to(), sharing the observations. A series without a unit gets
    /// @p unit: its numbers become counts of it.
    [[nodiscard]] TimeSeries to(TimeUnit unit) const;

    /// The same series counted from @p epoch, or from none, sharing the
    /// observations; no time is shifted.
    [[nodiscard]] TimeSeries with_epoch(std::optional<DateTime> epoch) const;

    /// The same series with @p delta, in its unit, added to the times of
    /// every channel, counted from @p epoch, or from none.
    [[nodiscard]] TimeSeries shifted(float64_t delta, std::optional<DateTime> epoch) const;

    /// The times at which every channel can be evaluated: from the latest
    /// first timestamp to the earliest end, a channel's last timestamp or its
    /// end, as check_observed() compares them.
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

    /// The same unit, epoch, and channels.
    [[nodiscard]] bool operator==(const TimeSeries& other) const = default;

    /// The same unit, and channels with the same observations; see
    /// Channel::same_observations(). The epoch is not compared.
    [[nodiscard]] bool same_observations(const TimeSeries& other) const
    {
      return m_unit == other.m_unit && std::ranges::equal(m_channels, other.m_channels, [](const auto& a, const auto& b) {
               return a.same_observations(b);
             });
    }

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

    /// Throw unless a series in @p unit can have @p epoch.
    static void check_epoch(TimeUnit unit, const std::optional<DateTime>& epoch)
    {
      if (epoch && unit == TimeUnit::Unitless)
      {
        throw std::invalid_argument("A series without a unit cannot have an epoch; give its times as durations or dates");
      }
    }

    std::vector<Channel<T>> m_channels;
    TimeUnit m_unit;
    std::optional<DateTime> m_epoch;
  };

  template <typename T>
  struct is_timeseries : std::false_type {};

  template <typename T>
  struct is_timeseries<TimeSeries<T>> : std::true_type { using value_type = T; };

  template <typename T>
  inline constexpr bool is_timeseries_v = is_timeseries<T>::value;

  /// Every series of @p series evaluated at the times @p t, counts of
  /// @p unit, as TimeSeries::evaluate does: one leaf per channel, in channel
  /// order, of shape (*series.shape(), *t.shape(), *shape) of that channel.
  /// Without @p unit, @p t are plain numbers, which only series without a
  /// unit take; times of the wrong kind throw TypeError.
  /// Requires every series to have the same number of channels and channel
  /// shapes, and @p series to be non-empty. A time outside the domain of a
  /// series throws std::domain_error naming the first such series'
  /// position, as in "TimeSeries at position (1, 2)".
  /// The series are evaluated in parallel on @p exec.
  template <typename T>
  [[nodiscard]] NestedTensor<T> evaluate(const Tensor<TimeSeries<T>>& series, const Tensor<float64_t>& t,
                                         std::optional<TimeUnit> unit = std::nullopt,
                                         Executor& exec = default_executor());

  /// The time of @p date in channel @p c of @p series, which has an epoch:
  /// time_since(date, epoch, unit) - origin. The difference rounds on the
  /// scale of the time since the epoch and the origin, not of the result,
  /// so a date that misses a closed end of the channel's domain by at most
  /// snap_tolerance() times the larger magnitude of the two is placed at
  /// that end. A window's own start date, its origin, then lies in its
  /// domain although the two round differently.
  template <typename T>
  [[nodiscard]] float64_t time_of_date(const TimeSeries<T>& series, size_t c, const DateTime& date);

  /// Every series of @p series evaluated at the dates @p dates, counts of
  /// @p dateUnit after 1970-01-01, as evaluate() evaluates at times. Each
  /// series places a date in each channel by time_of_date(), so series with
  /// different epochs are evaluated at the same instants. Throws
  /// std::invalid_argument if a series has no epoch.
  template <typename T>
  [[nodiscard]] NestedTensor<T> evaluate_dates(const Tensor<TimeSeries<T>>& series, const Tensor<int64_t>& dates,
                                               TimeUnit dateUnit, Executor& exec = default_executor());
}

#include "timeseries.tpp"

#endif // STABLEBEAR_TIMESERIES_H
