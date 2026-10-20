#ifndef STABLEBEAR_TIMESERIES_WINDOWING_H
#define STABLEBEAR_TIMESERIES_WINDOWING_H

#include "../timeseries.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sb
{
  /// The windows of @p series of length @p size, in the series' unit: a 1-D
  /// tensor with one window per start time a_k = t_0 + k * @p step, while
  /// a_k + size is at most the end of the series' domain, t_0 being its
  /// start, and a float tensor of shape (n_windows, 2) whose row k is
  /// (a_k, a_k + size) as computed, before any snapping, since each channel
  /// snaps on its own. Window k covers [a_k, a_k + size): it has the
  /// channels of @p series, each holding the observation in effect at a_k,
  /// at time 0, then its observations at times t with a_k < t < a_k + size,
  /// at the times t - a_k in the channel's unit, with the open end size
  /// (Channel::until). Every window channel thus can be evaluated from 0 up
  /// to its end but not at it, and an observation at a_k + size belongs only
  /// to the next window. A bound
  /// that falls just short of an observation reaches it, with the channel's
  /// snapping, as evaluation does: a start that reaches one is shifted by
  /// that observation's time, and an end that reaches one excludes it. Each
  /// window channel is a view of its channel made by Channel::take, so it
  /// copies no values. Throws std::invalid_argument if a channel has no
  /// observations or no window fits.
  template <typename T>
  std::pair<Tensor<TimeSeries<T>>, Tensor<float64_t>> windows(const TimeSeries<T>& series, float64_t size, float64_t step)
  {
    if (std::isnan(size) || std::isinf(size) || size <= 0)
    {
      throw std::invalid_argument("The window size must be finite and positive");
    }
    if (std::isnan(step) || std::isinf(step) || step <= 0)
    {
      throw std::invalid_argument("The window step must be finite and positive");
    }

    for (size_t c = 0; c < series.n_channels(); ++c)
    {
      if (series.channel(c).size() == 0)
      {
        throw std::invalid_argument("Cannot window a series with a channel without observations");
      }
    }
    // A window [a, a + size) fits when a + size is at most the last time,
    // whether that is a last timestamp or an open end.
    const auto [first, last, openEnd] = series.domain();
    if (first + size > last)
    {
      std::ostringstream ss;
      ss << "The window size " << size << " is longer than the series, from " << first << " to " << last;
      throw std::invalid_argument(ss.str());
    }

    std::vector<float64_t> starts;
    std::vector<TimeSeries<T>> contents;
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
        // one at from is the window's first, at time 0; if from is at it or
        // snaps forward onto it, the window is shifted by its time. The
        // window ends before to, so an observation at to, or that to snaps
        // onto, belongs to the next window.
        atFrom[c] = channel.index_at(from, atFrom[c]);
        atTo[c] = channel.index_at(to, atTo[c]);
        const auto lo = atFrom[c];
        const auto lastIn = (channel.time(atTo[c]) >= to && atTo[c] > lo) ? atTo[c] - 1 : atTo[c];
        const auto shift = std::max(channel.time(lo), from);

        // Subtracting the shift can round two close times together; keep
        // the times strictly increasing.
        std::vector<float64_t> at{0.0};
        std::vector<size_t> indices{lo};
        for (auto i = lo + 1; i <= lastIn; ++i)
        {
          at.push_back(std::max(channel.time(i) - shift, std::nextafter(at.back(), std::numeric_limits<float64_t>::infinity())));
          indices.push_back(i);
        }
        // The window can be evaluated up to its end but not at it.
        const auto end = std::max(to - shift, std::nextafter(at.back(), std::numeric_limits<float64_t>::infinity()));
        channels.push_back(channel.take(Times(std::move(at), channel.unit()), std::move(indices)).until(end));
      }
      starts.push_back(start);
      contents.emplace_back(std::move(channels), series.unit());
    }

    Tensor<TimeSeries<T>> windowTensor({contents.size()});
    Tensor<float64_t> bounds({contents.size(), 2});
    for (size_t k = 0; k < contents.size(); ++k)
    {
      windowTensor(k) = std::move(contents[k]);
      bounds({k, 0}) = starts[k];
      bounds({k, 1}) = starts[k] + size;
    }
    return {std::move(windowTensor), std::move(bounds)};
  }
}

#endif // STABLEBEAR_TIMESERIES_WINDOWING_H
