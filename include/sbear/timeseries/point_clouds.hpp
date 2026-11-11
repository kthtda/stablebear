#ifndef STABLEBEAR_TIMESERIES_POINT_CLOUDS_H
#define STABLEBEAR_TIMESERIES_POINT_CLOUDS_H

#include "../errors.hpp"
#include "../executor.hpp"
#include "../point_cloud.hpp"
#include "../tensor.hpp"
#include "../timeseries.hpp"
#include "../walk.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace sb
{
  /// The coordinate type of point clouds from series of @p T: float32 for
  /// float32, and float64 for other numbers.
  template <typename T>
  using point_coordinate_t = std::conditional_t<std::is_same_v<T, float32_t>, float32_t, float64_t>;

  /// The times at which to_point_cloud() takes points. By default, every
  /// time a series stores an observation in its domain. With @p n > 0, @p n
  /// evenly spaced times over the domain, including each end it includes.
  /// With @p step > 0, the times start + k * step in the domain, for
  /// k = 0, 1, ... With @p times, those times, in their order. @p step and
  /// @p times are counts of @p unit, or plain numbers for a series without
  /// a unit. With @p dates, counts of @p dateUnit after 1970-01-01, the
  /// times of those dates in the series, as evaluate_dates() places them.
  struct PointTimes
  {
    size_t n = 0;
    float64_t step = 0;
    std::optional<std::vector<float64_t>> times;
    TimeUnit unit = TimeUnit::Unitless;
    std::optional<std::vector<int64_t>> dates;
    TimeUnit dateUnit = TimeUnit::Unitless;
  };

  namespace detail
  {
    /// The times of @p how in the domain of @p series, of one channel:
    /// ascending, or in their order for given times or dates, which throw
    /// std::domain_error if one is outside the domain.
    template <typename T>
    std::vector<float64_t> grid_times(const TimeSeries<T>& series, const PointTimes& how)
    {
      const auto domain = series.domain();
      const auto unit = series.unit();
      std::vector<float64_t> times;
      if (how.dates)
      {
        if (!series.epoch())
        {
          throw std::invalid_argument("Every series needs an epoch to place dates and times; set one with with_epoch");
        }
        for (const auto date : *how.dates)
        {
          const auto t = time_of_date(series, 0, DateTime(date, how.dateUnit));
          if (!domain.contains(t))
          {
            throw std::domain_error("Time " + detail::shortest_text(t) + " is outside the domain "
                                    + domain.to_string());
          }
          times.push_back(t);
        }
        return times;
      }
      if (how.n > 0)
      {
        if (how.n == 1 && domain.closedStart && domain.closedEnd)
        {
          throw std::invalid_argument("One point cannot include both ends of the closed domain "
                                      + domain.to_string() + "; take n of at least 2");
        }
        // The fractions of the domain's length at which the points lie: from
        // 0 if the start is closed, to 1 if the end is closed, evenly apart.
        const auto gaps = static_cast<float64_t>(how.n + 1) - (domain.closedStart ? 1 : 0) - (domain.closedEnd ? 1 : 0);
        const float64_t offset = domain.closedStart ? 0 : 1;
        for (size_t i = 0; i < how.n; ++i)
        {
          times.push_back(std::lerp(domain.first, domain.last, (static_cast<float64_t>(i) + offset) / gaps));
        }
        return times;
      }
      check_query_unit(how.unit, unit);
      if (how.times)
      {
        for (const auto query : *how.times)
        {
          const auto t = convert_time(query, how.unit, unit);
          if (!domain.contains(t))
          {
            throw std::domain_error("Time " + detail::shortest_text(t) + " is outside the domain "
                                    + domain.to_string());
          }
          times.push_back(t);
        }
        return times;
      }
      const auto step = convert_time(how.step, how.unit, unit);
      for (size_t k = domain.closedStart ? 0 : 1;; ++k)
      {
        const auto t = domain.first + static_cast<float64_t>(k) * step;
        if (!domain.contains(t))
        {
          return times;
        }
        times.push_back(t);
      }
    }
  }

  /// The point cloud of @p series at the times @p how picks, in the order of
  /// their times: each gives the observation in effect then as one point,
  /// an observation of shape () a point of dimension 1, and one of shape (m)
  /// a point of dimension m. By default the times are those at which the
  /// series stores an observation in its domain; a window with an open
  /// start leaves out the value it stores at its start. Requires one
  /// channel, and throws TypeError for observations of other shapes.
  template <typename T>
    requires(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
  PointCloud<point_coordinate_t<T>> to_point_cloud(const TimeSeries<T>& series, const PointTimes& how = {})
  {
    if (series.n_channels() != 1)
    {
      throw std::invalid_argument(
          "A point cloud comes from a series of one channel, not " + std::to_string(series.n_channels())
          + "; combine the channels with sb.stack or sb.concatenate, or select one with series[i]");
    }
    const auto& channel = series.channel(0);
    const auto& shape = channel.shape();
    if (shape.size() > 1)
    {
      throw TypeError("Point clouds come from observations of shape () or (m,), not of "
                      + std::to_string(shape.size()) + " dimensions");
    }
    const size_t dim = shape.empty() ? 1 : shape[0];
    std::vector<size_t> kept;
    if (how.n > 0 || how.step > 0 || how.times || how.dates)
    {
      size_t hint = 0;
      for (const auto t : detail::grid_times(series, how))
      {
        hint = channel.index_at(t, hint);
        kept.push_back(hint);
      }
    }
    else if (channel.size() > 0)
    {
      const auto domain = series.domain();
      for (size_t k = 0; k < channel.size(); ++k)
      {
        if (domain.contains(channel.time(k)))
        {
          kept.push_back(k);
        }
      }
    }
    PointCloud<point_coordinate_t<T>> cloud({kept.size(), dim});
    for (size_t i = 0; i < kept.size(); ++i)
    {
      for (size_t j = 0; j < dim; ++j)
      {
        cloud(i, j) = static_cast<point_coordinate_t<T>>(channel.flat_value(kept[i] * dim + j));
      }
    }
    return cloud;
  }

  /// The point cloud of every series of @p seriesTensor, as to_point_cloud()
  /// gives it at the times @p how picks, in a tensor of the same shape. The
  /// clouds can differ in size. The series are converted in parallel on
  /// @p exec.
  template <typename T>
    requires(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
  Tensor<PointCloud<point_coordinate_t<T>>> to_point_clouds(const Tensor<TimeSeries<T>>& seriesTensor,
                                                            const PointTimes& how = {},
                                                            Executor& exec = default_executor())
  {
    Tensor<PointCloud<point_coordinate_t<T>>> clouds(
        std::vector<size_t>(seriesTensor.shape().begin(), seriesTensor.shape().end()));
    parallel_walk(seriesTensor, [&](const std::vector<size_t>& index) {
      clouds(index) = to_point_cloud(seriesTensor(index), how);
    }, exec);
    return clouds;
  }
}

#endif // STABLEBEAR_TIMESERIES_POINT_CLOUDS_H
