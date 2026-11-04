#ifndef STABLEBEAR_TIMESERIES_STACK_CHANNELS_H
#define STABLEBEAR_TIMESERIES_STACK_CHANNELS_H

#include "../tensor.hpp"
#include "../timeseries.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sb
{
  /// The channels of @p series stacked into one channel, as stack() stacks
  /// tensors: its observation at each time stacks the channels' observations
  /// in effect there along a new axis @p axis of length n_channels(), which
  /// counts from the end if negative. Its times are the start of the
  /// series' domain and every time in the domain at which a channel stores
  /// an observation, and its domain is the series'. The snap tolerance is
  /// the smallest of the channels'. Requires every channel to have
  /// observations, the same shape, and the same origin.
  template <typename T>
  TimeSeries<T> stack_channels(const TimeSeries<T>& series, std::ptrdiff_t axis = 0)
  {
    const auto n = series.n_channels();
    for (size_t c = 0; c < n; ++c)
    {
      if (series.channel(c).size() == 0)
      {
        throw std::invalid_argument("Cannot stack a channel without observations");
      }
      if (series.channel(c).origin() != series.channel(0).origin())
      {
        throw std::invalid_argument("Cannot stack channels of different origins");
      }
    }
    const auto rank = static_cast<std::ptrdiff_t>(series.channel(0).shape().size());
    if (axis < -(rank + 1) || axis > rank)
    {
      throw std::invalid_argument("axis " + std::to_string(axis) + " is out of range for observations of "
                                  + std::to_string(rank) + " dimensions");
    }
    const auto domain = series.domain();
    if (!(domain.first < domain.last || (domain.first == domain.last && domain.closedStart && domain.closedEnd)))
    {
      throw std::invalid_argument("Cannot stack channels whose domains do not overlap");
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
    // Axis 0 of each tensor is time, so the new axis comes one later.
    const auto stacked = stack(values, (axis < 0 ? axis + rank + 1 : axis) + 1);

    auto origin = series.channel(0).origin();
    Channel<T> channel(Times(std::move(times)).with_origin(origin), stacked, snapTolerance);
    channel = channel.with_start_closed(domain.closedStart).until(domain.last, domain.closedEnd);
    return TimeSeries<T>({std::move(channel)}, series.unit(), series.epoch());
  }
}

#endif // STABLEBEAR_TIMESERIES_STACK_CHANNELS_H
