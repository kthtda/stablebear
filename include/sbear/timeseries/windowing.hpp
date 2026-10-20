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
  /// a_k + size is at most the last time at which every channel has an
  /// observation, t_0 being the first such time, and a float tensor of shape
  /// (n_windows, 2) whose row k is (a_k, a_k + size) as computed, before
  /// any snapping, since each channel snaps on its own. Window k has the
  /// channels of @p series, each holding its observations at times t with
  /// a_k < t < a_k + size, at the times t - a_k in the channel's unit,
  /// between the values in effect at a_k, at time 0, and at a_k + size, at
  /// time size. Every window channel thus starts at 0 and ends at size, with
  /// the channel's values there. A bound that falls just short of an
  /// observation reaches it, with the channel's snapping, as evaluation
  /// does: the observation is in the window, a start that reaches one is
  /// shifted by that observation's time, and an end that reaches one ends
  /// the window at it. Each
  /// window channel is a view of its channel at its times, made by
  /// Channel::view_at, so it evaluates as its channel does there. Throws
  /// std::invalid_argument if a channel has no observations or no window
  /// fits.
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
    const auto [first, last] = series.domain();
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
        // one at from is the window's first, at time 0. If from is at it or
        // snaps forward onto it, the window is shifted by its time.
        atFrom[c] = channel.index_at(from, atFrom[c]);
        atTo[c] = channel.index_at(to, atTo[c]);
        const auto lo = atFrom[c];
        const auto hi = atTo[c] + 1;
        const auto shift = std::max(channel.time(lo), from);

        // The view reads the channel at each window time plus the shift,
        // and t - shift + shift can round below t, which would read the
        // observation before; take the next float up until it reaches t.
        const auto relativeTo = [shift](float64_t t) {
          auto relative = t - shift;
          while (relative + shift < t)
          {
            relative = std::nextafter(relative, std::numeric_limits<float64_t>::infinity());
          }
          return relative;
        };
        std::vector<float64_t> at;
        at.reserve(hi - lo + 1);
        at.push_back(0.0);
        for (auto k = lo + 1; k < hi; ++k)
        {
          at.push_back(relativeTo(channel.time(k)));
        }
        // Unless an observation is at to or snaps onto it, the window ends
        // with the value in effect at to, at to's time.
        if (channel.time(atTo[c]) < to)
        {
          const auto end = relativeTo(to);
          if (end > at.back())
          {
            at.push_back(end);
          }
        }
        channels.push_back(channel.view_at(Times(std::move(at), channel.unit()), -shift));
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
