#ifndef STABLEBEAR_TIMESERIES_WINDOWING_H
#define STABLEBEAR_TIMESERIES_WINDOWING_H

#include "../algorithms/gallop.hpp"
#include "../timeseries.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sb
{
  /// The windows of @p series of length @p size, in the series' unit: a
  /// series with one channel and one window per start time a_k = t_0 + k *
  /// @p step, while a_k + size is at most the last time at which every
  /// channel has an observation, t_0 being the first such time. Window k has
  /// the channels of @p series, each holding exactly its observations at
  /// times t with a_k <= t <= a_k + size, so a window can be empty, at the
  /// times t - a_k in the channel's unit if @p relative and at t otherwise.
  /// Each window channel is a view of its channel at those times, made by
  /// Channel::view_at, so it evaluates as its channel does there.
  template <typename T>
  TimeSeries<TimeSeries<T>> windows(const TimeSeries<T>& series, float64_t size, float64_t step, bool relative)
  {
    if (std::isnan(size) || std::isinf(size) || size <= 0)
    {
      throw std::invalid_argument("The window size must be finite and positive");
    }
    if (std::isnan(step) || std::isinf(step) || step <= 0)
    {
      throw std::invalid_argument("The window step must be finite and positive");
    }

    std::vector<float64_t> starts;
    std::vector<TimeSeries<T>> contents;
    bool observed = true;
    for (size_t c = 0; c < series.n_channels(); ++c)
    {
      observed = observed && series.channel(c).size() > 0;
    }
    if (observed)
    {
      const auto [first, last] = series.domain();
      // Each window's observations, from lo to hi - 1 per channel; windows
      // only move forward, so each search starts from the last window's.
      std::vector<size_t> lo(series.n_channels(), 0);
      std::vector<size_t> hi(series.n_channels(), 0);
      for (size_t k = 0;; ++k)
      {
        const auto start = first + static_cast<float64_t>(k) * step;
        if (start + size > last)
        {
          break;
        }
        std::vector<Channel<T>> channels;
        channels.reserve(series.n_channels());
        for (size_t c = 0; c < series.n_channels(); ++c)
        {
          const auto& channel = series.channel(c);
          const auto& times = channel.times().values();
          const auto begin = times.begin();
          const auto from = convert_time(start, series.unit(), channel.unit());
          const auto to = convert_time(start + size, series.unit(), channel.unit());
          // The first timestamp at or after from is the first one after the
          // float just below it.
          const auto below = std::nextafter(from, -std::numeric_limits<float64_t>::infinity());
          lo[c] = static_cast<size_t>(
              gallop_upper_bound(begin, times.end(), begin + static_cast<std::ptrdiff_t>(lo[c]), below) - begin);
          hi[c] = static_cast<size_t>(
              gallop_upper_bound(begin, times.end(), begin + static_cast<std::ptrdiff_t>(hi[c]), to) - begin);

          const auto shift = relative ? from : 0.0;
          std::vector<float64_t> at(begin + static_cast<std::ptrdiff_t>(lo[c]),
                                    begin + static_cast<std::ptrdiff_t>(hi[c]));
          for (auto& t : at)
          {
            t -= shift;
          }
          channels.push_back(channel.view_at(Times(std::move(at), channel.unit()), -shift));
        }
        starts.push_back(start);
        contents.emplace_back(std::move(channels), series.unit());
      }
    }

    Channel<TimeSeries<T>> outer(Times(std::move(starts), series.unit()), std::vector<size_t>{}, std::move(contents));
    return TimeSeries<TimeSeries<T>>({std::move(outer)}, series.unit());
  }
}

#endif // STABLEBEAR_TIMESERIES_WINDOWING_H
