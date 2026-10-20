#ifndef STABLEBEAR_TIMESERIES_WINDOWING_H
#define STABLEBEAR_TIMESERIES_WINDOWING_H

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
  /// times t - a_k in the channel's unit. A bound that falls just short of
  /// an observation reaches it, with the channel's snapping, as evaluation
  /// does: the observation is in the window, and a start that reaches one
  /// is that observation's time, so the window's first time is exactly 0.
  /// Each window channel is a view of its channel at its times, made by
  /// Channel::view_at, so it evaluates as its channel does there.
  template <typename T>
  TimeSeries<TimeSeries<T>> windows(const TimeSeries<T>& series, float64_t size, float64_t step)
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
      // Each channel's observations at the last window's start and end,
      // hints for the next window's, which are close by.
      std::vector<size_t> atFrom(series.n_channels(), 0);
      std::vector<size_t> atTo(series.n_channels(), 0);
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
          const auto from = convert_time(start, series.unit(), channel.unit());
          const auto to = convert_time(start + size, series.unit(), channel.unit());
          // The observations in effect at from and at to, with snapping. The
          // one at from is the first in the window if it is not before from,
          // that is, if from is at it or snaps forward onto it; the window
          // then starts at its time.
          atFrom[c] = channel.index_at(from, atFrom[c]);
          atTo[c] = channel.index_at(to, atTo[c]);
          const auto reached = channel.time(atFrom[c]) >= from;
          const auto lo = reached ? atFrom[c] : atFrom[c] + 1;
          const auto hi = atTo[c] + 1;
          const auto shift = reached ? channel.time(atFrom[c]) : from;

          // The view reads the channel at each window time plus the shift,
          // and t - shift + shift can round below t, which would read the
          // observation before; take the next float up until it reaches t.
          std::vector<float64_t> at;
          at.reserve(hi - lo);
          for (auto k = lo; k < hi; ++k)
          {
            const auto t = channel.time(k);
            auto relative = t - shift;
            while (relative + shift < t)
            {
              relative = std::nextafter(relative, std::numeric_limits<float64_t>::infinity());
            }
            at.push_back(relative);
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
