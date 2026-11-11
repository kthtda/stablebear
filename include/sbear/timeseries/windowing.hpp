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
  namespace detail
  {
    /// Whether observation @p i of @p channel lies within @p reach of @p t,
    /// a bound computed in floating point, and so is at it. The reach is
    /// capped at a quarter of the gaps around the observation, as snapping
    /// caps it, so that a bound nearer a neighbour stays with it.
    template <typename T>
    bool is_at(const Channel<T>& channel, size_t i, float64_t t, float64_t reach)
    {
      const auto at = channel.time(i);
      if (i > 0)
      {
        reach = std::min(reach, (at - channel.time(i - 1)) / 4);
      }
      if (i + 1 < channel.size())
      {
        reach = std::min(reach, (channel.time(i + 1) - at) / 4);
      }
      return std::abs(at - t) <= reach;
    }

    /// The index of the observation in effect at @p t, as Channel::index_at
    /// gives it, or of the next one if it is at @p t within @p reach.
    template <typename T>
    size_t index_near(const Channel<T>& channel, float64_t t, float64_t reach, size_t hint)
    {
      const auto i = channel.index_at(t, hint);
      return (i + 1 < channel.size() && is_at(channel, i + 1, t, reach)) ? i + 1 : i;
    }

    /// The part of @p channel from @p from to @p to, each end included when
    /// @p startClosed or @p stopClosed says so, at its times less @p zero,
    /// as windows() cuts each channel: the observation in effect at @p from,
    /// at from - zero, then the observations after it up to @p to, with the
    /// end to - zero (Channel::until). The origin moves by @p zero, so the
    /// times keep their place in the series the channel was cut from. A
    /// view made by Channel::view_at_indices, so no values are copied.
    /// @p atFrom and @p atTo are hints for the observations in effect at
    /// @p from and @p to, updated to them. @p fromScale and @p toScale are the larger
    /// magnitude of the operands that @p from and @p to were computed from,
    /// such as a start and a size; an observation within the channel's snap
    /// tolerance times them of a bound, on either side, is at it. They are 0
    /// for bounds that were not computed. Requires channel.size() > 0.
    template <typename T>
    Channel<T> cut_channel(const Channel<T>& channel, float64_t from, float64_t to, bool startClosed, bool stopClosed,
                           float64_t zero, size_t& atFrom, size_t& atTo, float64_t fromScale = 0,
                           float64_t toScale = 0)
    {
      // The observations in effect at from and at to, with snapping, and
      // the ones just after them that the bounds' rounding may have missed.
      // The one at from is the first. An open end excludes an observation
      // at to, and a closed one keeps it.
      const auto fromReach = channel.snap_tolerance() * fromScale;
      const auto toReach = channel.snap_tolerance() * toScale;
      atFrom = index_near(channel, from, fromReach, atFrom);
      atTo = index_near(channel, to, toReach, atTo);
      const auto lo = atFrom;
      const bool atEnd = channel.time(atTo) >= to || is_at(channel, atTo, to, toReach);
      const auto lastIn = (atEnd && !stopClosed && atTo > lo) ? atTo - 1 : atTo;

      // Subtracting zero can round two close times together, and the
      // observation after one that from snaps onto can lie at or before
      // from; keep the times strictly increasing.
      std::vector<float64_t> at{from - zero};
      std::vector<size_t> indices{lo};
      for (auto i = lo + 1; i <= lastIn; ++i)
      {
        at.push_back(std::max(channel.time(i) - zero, std::nextafter(at.back(), std::numeric_limits<float64_t>::infinity())));
        indices.push_back(i);
      }
      // The part can be evaluated up to its end, and at it if closed. A
      // closed end at or before the last time held needs no end: the part
      // ends at that timestamp.
      auto part = channel.view_at_indices(std::move(indices))
                      .with_times(channel.times_like(std::move(at)).with_origin(channel.origin() + zero));
      const auto end = to - zero;
      const auto lastAt = part.time(part.size() - 1);
      if (!(stopClosed && end <= lastAt))
      {
        part = part.until(std::max(end, std::nextafter(lastAt, std::numeric_limits<float64_t>::infinity())),
                          stopClosed);
      }
      return part.with_start_closed(startClosed);
    }
  }

  /// The windows of @p series of length @p size, in the series' unit: a 1-D
  /// tensor with one window per start time a_k = t_0 + k * @p step, while
  /// the window ends in the series' domain, t_0 being its start, and a float
  /// tensor of shape (n_windows, 2) whose row k is (a_k, a_k + size) as
  /// computed. Every channel of window k has its time 0 at a_k.
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
  /// is computed in floating point from terms such as a_k and size, and
  /// an observation within the channel's snap tolerance times the larger
  /// magnitude of them of it, on either side, or one it snaps to as
  /// evaluation does, is at it: a start at one holds it at time 0, and an
  /// end at one excludes it, or includes it if closed. Whether a window
  /// fits in the domain is decided alike, with the smallest snap tolerance
  /// of the channels. Each window channel is a view of its channel made by
  /// Channel::view_at_indices, so it copies no values. Throws std::invalid_argument if a channel has no
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

    // The bounds of each window, whether they are included, and the larger
    // magnitude of the operands each was computed from, which scales how
    // far it may have rounded from the time it means.
    struct Bounds
    {
      float64_t start;
      float64_t stop;
      bool startClosed;
      bool stopClosed;
      float64_t startScale;
      float64_t stopScale;
    };
    std::vector<Bounds> bounds;
    const auto [first, last, seriesClosedStart, seriesClosedEnd] = series.domain();
    // A computed bound within the smallest snap tolerance of the channels
    // times the magnitude of its operands of an end of the domain is at it,
    // so that whether a window fits does not depend on rounding.
    float64_t tolerance = series.channel(0).snap_tolerance();
    for (size_t c = 1; c < series.n_channels(); ++c)
    {
      tolerance = std::min(tolerance, series.channel(c).snap_tolerance());
    }
    const auto scale = [](float64_t a, float64_t b) { return std::max(std::abs(a), std::abs(b)); };
    // Whether a, computed from operands of magnitude up to aScale, lies
    // before b, or at it within that reach.
    const auto before = [&](float64_t a, float64_t b, float64_t aScale) { return a < b - tolerance * aScale; };
    const auto at = [&](float64_t a, float64_t b, float64_t aScale) { return std::abs(a - b) <= tolerance * aScale; };
    if (!anchorEnd)
    {
      // A window fits when a + size is at most the last time, whether that
      // is a last timestamp or an end; a closed window end must come before
      // an open one of the series.
      const auto fits = [&](float64_t start) {
        const auto stopScale = scale(start, size);
        return before(start + size, last, stopScale) ||
               (at(start + size, last, stopScale) && (seriesClosedEnd || !closedEnd));
      };
      // The first window starts where the series does, so it cannot be
      // evaluated at 0 if the series cannot at its start.
      const auto startClosed = [&](size_t k) { return closedStart && (k > 0 || seriesClosedStart); };
      // A partial window needs its start in the domain.
      const auto startsIn = [&](float64_t start, float64_t startScale, size_t k) {
        return before(start, last, startScale) ||
               (at(start, last, startScale) && startClosed(k) && seriesClosedEnd);
      };
      for (size_t k = 0;; ++k)
      {
        const auto offset = static_cast<float64_t>(k) * step;
        const auto start = first + offset;
        const auto startScale = scale(first, offset);
        const bool full = fits(start);
        if (!full && !(partial && startsIn(start, startScale, k)))
        {
          break;
        }
        // A partial window ends where the series does.
        bounds.push_back({start, full ? start + size : last, startClosed(k), full ? closedEnd : seriesClosedEnd,
                          startScale, full ? scale(start, size) : 0.0});
      }
    }
    else
    {
      // The same, mirrored: a window fits when b - size is at least the first
      // time, the last window ends where the series does, and a partial
      // window needs its end in the domain and starts where the series does.
      const auto fits = [&](float64_t stop) {
        const auto startScale = scale(stop, size);
        return before(first, stop - size, startScale) ||
               (at(stop - size, first, startScale) && (seriesClosedStart || !closedStart));
      };
      const auto stopClosed = [&](size_t k) { return closedEnd && (k > 0 || seriesClosedEnd); };
      const auto stopsIn = [&](float64_t stop, float64_t stopScale, size_t k) {
        return before(first, stop, stopScale) ||
               (at(stop, first, stopScale) && stopClosed(k) && seriesClosedStart);
      };
      for (size_t k = 0;; ++k)
      {
        const auto offset = static_cast<float64_t>(k) * step;
        const auto stop = last - offset;
        const auto stopScale = scale(last, offset);
        const bool full = fits(stop);
        if (!full && !(partial && stopsIn(stop, stopScale, k)))
        {
          break;
        }
        bounds.push_back({full ? stop - size : first, stop, full ? closedStart : seriesClosedStart, stopClosed(k),
                          full ? scale(stop, size) : 0.0, stopScale});
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
    for (const auto& [from, to, startClosed, stopClosed, fromScale, toScale] : bounds)
    {
      std::vector<Channel<T>> channels;
      channels.reserve(series.n_channels());
      for (size_t c = 0; c < series.n_channels(); ++c)
      {
        channels.push_back(detail::cut_channel(series.channel(c), from, to, startClosed, stopClosed, from,
                                               atFrom[c], atTo[c], fromScale, toScale));
      }
      contents.emplace_back(std::move(channels), series.unit(), series.epoch());
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
  /// @p series on the domain from @p from to @p to, each end included when
  /// @p startClosed or @p stopClosed says so, at the same times: each
  /// channel is cut as detail::cut_channel() cuts it, with nothing
  /// subtracted, so the unit, epoch, and origins stay. The caller checks
  /// that the domain lies within the series'. Requires every channel to
  /// have observations.
  template <typename T>
  TimeSeries<T> restrict(const TimeSeries<T>& series, float64_t from, float64_t to, bool startClosed,
                         bool stopClosed)
  {
    std::vector<Channel<T>> channels;
    channels.reserve(series.n_channels());
    for (size_t c = 0; c < series.n_channels(); ++c)
    {
      size_t atFrom = 0;
      size_t atTo = 0;
      channels.push_back(detail::cut_channel(series.channel(c), from, to, startClosed, stopClosed, 0.0, atFrom, atTo));
    }
    return TimeSeries<T>(std::move(channels), series.unit(), series.epoch());
  }
}

#endif // STABLEBEAR_TIMESERIES_WINDOWING_H
