#ifndef STABLEBEAR_TIMESERIES_STACK_CHANNELS_H
#define STABLEBEAR_TIMESERIES_STACK_CHANNELS_H

#include "../tensor.hpp"
#include "../timeseries.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace sb
{
  namespace detail
  {
    /// Throw unless @p axis, which counts from the end if negative, lies in
    /// [-size, size), and return it counted from the start.
    inline size_t channel_axis(std::ptrdiff_t axis, std::ptrdiff_t size, std::ptrdiff_t rank)
    {
      if (axis < -size || axis >= size)
      {
        throw std::invalid_argument("axis " + std::to_string(axis) + " is out of range for observations of "
                                    + std::to_string(rank) + " dimensions");
      }
      return static_cast<size_t>(axis < 0 ? axis + size : axis);
    }

    /// The channels of @p series joined into one channel by @p join, which
    /// takes each channel's observations at the same times, as tensors of
    /// shape (times, *shape), and returns the joined observations, with time
    /// still along axis 0. The times are the start of the series' domain
    /// and every time in the domain at which a channel stores an
    /// observation, and the domain is the series'. The snap tolerance is
    /// the smallest of the channels'. Requires every channel to have
    /// observations and the same origin.
    template <typename T, typename Join>
    TimeSeries<T> join_channels(const TimeSeries<T>& series, Join join)
    {
      const auto n = series.n_channels();
      for (size_t c = 0; c < n; ++c)
      {
        if (series.channel(c).size() == 0)
        {
          throw std::invalid_argument("Cannot join a channel without observations");
        }
        if (series.channel(c).origin() != series.channel(0).origin())
        {
          throw std::invalid_argument("Cannot join channels of different origins");
        }
      }
      const auto domain = series.domain();
      if (!(domain.first < domain.last || (domain.first == domain.last && domain.closedStart && domain.closedEnd)))
      {
        throw std::invalid_argument("Cannot join channels whose domains do not overlap");
      }

      std::vector<float64_t> times{domain.first};
      float64_t snapTolerance = series.channel(0).snap_tolerance();
      for (size_t c = 0; c < n; ++c)
      {
        const auto& channel = series.channel(c);
        snapTolerance = std::min(snapTolerance, channel.snap_tolerance());
        for (size_t k = 0; k < channel.size(); ++k)
        {
          if (domain.contains(channel.time(k)))
          {
            times.push_back(channel.time(k));
          }
        }
      }
      std::sort(times.begin(), times.end());
      times.erase(std::unique(times.begin(), times.end()), times.end());

      std::vector<size_t> order(times.size());
      std::iota(order.begin(), order.end(), size_t{0});
      std::vector<Tensor<T>> values;
      values.reserve(n);
      for (size_t c = 0; c < n; ++c)
      {
        const auto& channel = series.channel(c);
        std::vector<size_t> shape{times.size()};
        shape.insert(shape.end(), channel.shape().begin(), channel.shape().end());
        Tensor<T> out(shape);
        channel.lookup_sorted(times, order, out.data());
        values.push_back(std::move(out));
      }
      const auto joined = join(values);

      const auto origin = series.channel(0).origin();
      Channel<T> channel(Times(std::move(times)).with_origin(origin), joined, snapTolerance);
      channel = channel.with_start_closed(domain.closedStart).until(domain.last, domain.closedEnd);
      return TimeSeries<T>({std::move(channel)}, series.unit(), series.epoch());
    }
  }

  /// The channels of @p series stacked into one channel, as stack() stacks
  /// tensors: its observation at each time stacks the channels' observations
  /// in effect there along a new axis @p axis of length n_channels(), which
  /// counts from the end if negative. The times, domain, and snap tolerance
  /// are as for detail::join_channels(). Requires every channel to have
  /// observations, the same shape, and the same origin.
  template <typename T>
  TimeSeries<T> stack_channels(const TimeSeries<T>& series, std::ptrdiff_t axis = 0)
  {
    const auto rank = static_cast<std::ptrdiff_t>(series.channel(0).shape().size());
    const auto at = detail::channel_axis(axis, rank + 1, rank);
    // Axis 0 of each tensor is time, so the new axis comes one later.
    return detail::join_channels(series, [&](const std::vector<Tensor<T>>& values) {
      return stack(values, static_cast<std::ptrdiff_t>(at + 1));
    });
  }

  /// The channels of @p series concatenated into one channel, as
  /// concatenate() joins tensors: its observation at each time joins the
  /// channels' observations in effect there along their axis @p axis, which
  /// counts from the end if negative. The times, domain, and snap tolerance
  /// are as for detail::join_channels(). Requires every channel to have
  /// observations, the same origin, and the same shape except along
  /// @p axis, with at least one dimension.
  template <typename T>
  TimeSeries<T> concatenate_channels(const TimeSeries<T>& series, std::ptrdiff_t axis = 0)
  {
    const auto& shape = series.channel(0).shape();
    if (shape.empty())
    {
      throw std::invalid_argument("Cannot concatenate channels of scalars; stack them with stack_channels");
    }
    const auto rank = static_cast<std::ptrdiff_t>(shape.size());
    const auto at = detail::channel_axis(axis, rank, rank);
    for (size_t c = 1; c < series.n_channels(); ++c)
    {
      const auto& other = series.channel(c).shape();
      bool matches = other.size() == shape.size();
      for (size_t d = 0; matches && d < shape.size(); ++d)
      {
        matches = d == at || other[d] == shape[d];
      }
      if (!matches)
      {
        throw std::invalid_argument("Cannot concatenate channels whose shapes differ other than along axis "
                                    + std::to_string(axis));
      }
    }
    // Axis 0 of each tensor is time, so the joined axis comes one later.
    return detail::join_channels(series, [&](const std::vector<Tensor<T>>& values) {
      return concatenate(values, static_cast<std::ptrdiff_t>(at + 1));
    });
  }
}

#endif // STABLEBEAR_TIMESERIES_STACK_CHANNELS_H
