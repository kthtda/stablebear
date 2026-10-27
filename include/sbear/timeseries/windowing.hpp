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
  /// the window ends in the series' domain, t_0 being its start, and a float
  /// tensor of shape (n_windows, 2) whose row k is (a_k, a_k + size) as
  /// computed, before any snapping, since each channel snaps on its own.
  /// With @p anchorEnd, the windows instead end at b_k = t_1 - k * @p step,
  /// while they start in the domain, t_1 being its end, and come in
  /// chronological order, so the last one ends at t_1.
  /// Window k covers a_k to a_k + size, each end included when
  /// @p closedStart or @p closedEnd says so: it has the channels of
  /// @p series, each holding the observation in effect at a_k, at time 0,
  /// then its observations at times t with a_k < t < a_k + size, and one at
  /// a_k + size if the end is closed, at the times t - a_k in the channel's
  /// unit, with the end size (Channel::until). With @p partial, windows
  /// continue while their start, or with @p anchorEnd their end, is in the
  /// domain, and a window that would reach past the domain is cut where the
  /// series starts or ends instead, with its closedness, and its bounds row
  /// is cut there too. An open start leaves 0
  /// outside the domain, though the observation there stays in effect after
  /// it. With an open end, an observation at a_k + size belongs only to the
  /// next window. A bound
  /// that falls just short of an observation reaches it, with the channel's
  /// snapping, as evaluation does: a start that reaches one is shifted by
  /// that observation's time, and an end that reaches one excludes it, or
  /// includes it if closed. Each
  /// window channel is a view of its channel made by Channel::take, so it
  /// copies no values. Throws std::invalid_argument if a channel has no
  /// observations or no window fits.
  template <typename T>
  std::pair<Tensor<TimeSeries<T>>, Tensor<float64_t>> windows(const TimeSeries<T>& series, float64_t size, float64_t step,
                                                              bool closedStart = false, bool closedEnd = true,
                                                              bool partial = false, bool anchorEnd = true)
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

    // The bounds of each window and whether they are included.
    struct Bounds
    {
      float64_t start;
      float64_t stop;
      bool startClosed;
      bool stopClosed;
    };
    std::vector<Bounds> bounds;
    const auto [first, last, seriesClosedStart, seriesClosedEnd] = series.domain();
    if (!anchorEnd)
    {
      // A window fits when a + size is at most the last time, whether that
      // is a last timestamp or an end; a closed window end must come before
      // an open one of the series.
      const auto fits = [&](float64_t start) {
        return start + size < last || (start + size == last && (seriesClosedEnd || !closedEnd));
      };
      // The first window starts where the series does, so it cannot be
      // evaluated at 0 if the series cannot at its start.
      const auto startClosed = [&](size_t k) { return closedStart && (k > 0 || seriesClosedStart); };
      // A partial window needs its start in the domain.
      const auto startsIn = [&](float64_t start, size_t k) {
        return start < last || (start == last && startClosed(k) && seriesClosedEnd);
      };
      for (size_t k = 0;; ++k)
      {
        const auto start = first + static_cast<float64_t>(k) * step;
        const bool full = fits(start);
        if (!full && !(partial && startsIn(start, k)))
        {
          break;
        }
        // A partial window ends where the series does.
        bounds.push_back({start, full ? start + size : last, startClosed(k), full ? closedEnd : seriesClosedEnd});
      }
    }
    else
    {
      // The same, mirrored: a window fits when b - size is at least the first
      // time, the last window ends where the series does, and a partial
      // window needs its end in the domain and starts where the series does.
      const auto fits = [&](float64_t stop) {
        return stop - size > first || (stop - size == first && (seriesClosedStart || !closedStart));
      };
      const auto stopClosed = [&](size_t k) { return closedEnd && (k > 0 || seriesClosedEnd); };
      const auto stopsIn = [&](float64_t stop, size_t k) {
        return stop > first || (stop == first && stopClosed(k) && seriesClosedStart);
      };
      for (size_t k = 0;; ++k)
      {
        const auto stop = last - static_cast<float64_t>(k) * step;
        const bool full = fits(stop);
        if (!full && !(partial && stopsIn(stop, k)))
        {
          break;
        }
        bounds.push_back({full ? stop - size : first, stop, full ? closedStart : seriesClosedStart, stopClosed(k)});
      }
      std::reverse(bounds.begin(), bounds.end());
    }
    if (bounds.empty())
    {
      std::ostringstream ss;
      ss << "The window size " << size << " is longer than the series, from " << first << " to " << last;
      throw std::invalid_argument(ss.str());
    }

    std::vector<TimeSeries<T>> contents;
    // Each channel's observations at the last window's start and end,
    // hints for the next window's, which are close by.
    std::vector<size_t> atFrom(series.n_channels(), 0);
    std::vector<size_t> atTo(series.n_channels(), 0);
    for (const auto& [start, stop, startClosed, stopClosed] : bounds)
    {
      std::vector<Channel<T>> channels;
      channels.reserve(series.n_channels());
      for (size_t c = 0; c < series.n_channels(); ++c)
      {
        const auto& channel = series.channel(c);
        const auto from = convert_time(start, series.unit(), channel.unit());
        const auto to = convert_time(stop, series.unit(), channel.unit());
        // The observations in effect at from and at to, with snapping. The
        // one at from is the window's first, at time 0; if from is at it or
        // snaps forward onto it, the window is shifted by its time. An open
        // window end excludes an observation at to, or that to snaps onto,
        // and a closed one keeps it.
        atFrom[c] = channel.index_at(from, atFrom[c]);
        atTo[c] = channel.index_at(to, atTo[c]);
        const auto lo = atFrom[c];
        const bool atEnd = channel.time(atTo[c]) >= to;
        const auto lastIn = (atEnd && !stopClosed && atTo[c] > lo) ? atTo[c] - 1 : atTo[c];
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
        // The window can be evaluated up to its end, and at it if closed. A
        // closed end at or before the last time held needs no end: the
        // window ends at that timestamp.
        // Time 0 of the window is the shift in the channel; the origin
        // places it in the first series the channel was cut from.
        auto window = channel.take(channel.times_like(std::move(at)).with_origin(channel.origin() + shift),
                                   std::move(indices));
        const auto end = to - shift;
        const auto lastAt = window.time(window.size() - 1);
        if (!(stopClosed && end <= lastAt))
        {
          window = window.until(std::max(end, std::nextafter(lastAt, std::numeric_limits<float64_t>::infinity())),
                                stopClosed);
        }
        window = window.with_start_closed(startClosed);
        channels.push_back(std::move(window));
      }
      contents.emplace_back(std::move(channels), series.unit());
    }

    Tensor<TimeSeries<T>> windowTensor({contents.size()});
    Tensor<float64_t> boundsTensor({contents.size(), 2});
    for (size_t k = 0; k < contents.size(); ++k)
    {
      windowTensor(k) = std::move(contents[k]);
      boundsTensor({k, 0}) = bounds[k].start;
      boundsTensor({k, 1}) = bounds[k].stop;
    }
    return {std::move(windowTensor), std::move(boundsTensor)};
  }
}

#endif // STABLEBEAR_TIMESERIES_WINDOWING_H
