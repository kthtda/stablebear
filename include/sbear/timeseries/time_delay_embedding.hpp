#ifndef STABLEBEAR_TIMESERIES_TIME_DELAY_EMBEDDING_H
#define STABLEBEAR_TIMESERIES_TIME_DELAY_EMBEDDING_H

#include "../timeseries.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sb
{
  /// The delay embedding of @p channel with @p dimension components spaced
  /// @p delay apart, in the channel's unit. Its value at t is
  /// (x(t - (dimension - 1) * delay), ..., x(t - delay), x(t)), oldest first
  /// as in Kantz and Schreiber's convention, for every t from
  /// t_0 + (dimension - 1) * delay to the end of the channel's domain, each
  /// end included when the channel's is, so each observation gains a leading
  /// axis of length @p dimension. Its times are the start of that domain and
  /// the times in it where a component reaches an observation,
  /// t_i + k * delay; times that round to the same point are recorded once,
  /// at the time they snap to. It is a view of the channel's storage.
  template <typename T>
  Channel<T> delay_embed(const Channel<T>& channel, size_t dimension, float64_t delay)
  {
    if (dimension == 0)
    {
      throw std::invalid_argument("The embedding dimension must be at least 1");
    }
    if (std::isnan(delay) || std::isinf(delay) || delay <= 0)
    {
      throw std::invalid_argument("The embedding delay must be finite and positive");
    }

    // Component j, oldest first, reads the channel lags[j] before t.
    std::vector<float64_t> lags(dimension);
    for (size_t j = 0; j < dimension; ++j)
    {
      lags[j] = static_cast<float64_t>(dimension - 1 - j) * delay;
    }

    const auto n = channel.size();
    std::vector<float64_t> at;
    if (n > 0)
    {
      const auto start = channel.time(0) + lags.front();
      // A channel with an end, such as a window, can be evaluated past its
      // last timestamp up to its end, so the embedding can too, and keeps
      // that end, and an open start, through stack_at.
      const auto end = channel.end().value_or(channel.time(n - 1));
      const auto closed = channel.end_closed();
      const auto past = [&](float64_t s) { return closed ? s > end : s >= end; };
      if (past(start))
      {
        throw std::invalid_argument("The series is too short for this embedding: no time has its full history");
      }
      // Component j changes where its lagged time reaches an observation: at
      // the timestamps shifted by lags[j]. A queue holds each component's next
      // shifted time, so the shifted copies are merged in one pass, from start
      // to end, without storing them.
      std::vector<size_t> shifted(dimension, 0);
      const auto pending = [&](size_t j) { return channel.time(shifted[j]) + lags[j]; };
      using Event = std::pair<float64_t, size_t>;
      std::priority_queue<Event, std::vector<Event>, std::greater<>> queue;
      for (size_t j = 0; j < dimension; ++j)
      {
        while (shifted[j] < n && pending(j) < start)
        {
          ++shifted[j];
        }
        if (shifted[j] < n)
        {
          queue.emplace(pending(j), j);
        }
      }

      // Shifted times can round to either side of the observation they mean,
      // so one point can come up at times an ulp apart. Two times give the
      // same point when every component reads the same observation at both;
      // index_at decides which observation, with the channel's
      // snap_tolerance and snap_eps, as evaluation does. A component's index
      // never decreases with time, so a run of the same point ends where some
      // component's index differs from the one at the time before. Only a
      // component whose next shifted time is at s, or within snapping
      // distance after it, can read a different observation at s. Snapping
      // moves a time by at most snap_eps machine epsilons times its
      // magnitude; a few more cover the rounding of the lags.
      //
      // Record each run at the time its members snap to: the channel's own
      // timestamp if the run has one, else its latest time, since snapping
      // moves a time forward onto the observation it falls just short of.
      // The first point stays at the start, so the domain keeps its start.
      // The last observation is a timestamp, so the domain keeps its end,
      // unless the first run reaches it.
      const auto reach = (channel.snap_eps() + 4) * std::numeric_limits<float64_t>::epsilon();
      std::vector<size_t> previous(dimension);
      for (size_t j = 0; j < dimension; ++j)
      {
        previous[j] = channel.index_at(start - lags[j]);
      }
      std::vector<size_t> reached;
      std::vector<Event> near;
      bool runHasTimestamp = false;
      while (!queue.empty() && !past(queue.top().first))
      {
        const auto s = queue.top().first;
        reached.clear();
        bool isTimestamp = false;
        while (!queue.empty() && queue.top().first == s)
        {
          const auto j = queue.top().second;
          queue.pop();
          while (shifted[j] < n && pending(j) == s)
          {
            ++shifted[j];
          }
          if (shifted[j] < n)
          {
            queue.emplace(pending(j), j);
          }
          reached.push_back(j);
          // The newest component has lag 0, so its shifted times are the
          // channel's own timestamps.
          isTimestamp = isTimestamp || j == dimension - 1;
        }
        near.clear();
        while (!queue.empty() && queue.top().first <= s + reach * (std::abs(s) + lags.front()))
        {
          near.push_back(queue.top());
          queue.pop();
          reached.push_back(near.back().second);
        }
        for (const auto& event : near)
        {
          queue.push(event);
        }

        bool changes = at.empty();
        for (const auto j : reached)
        {
          const auto index = channel.index_at(s - lags[j], previous[j]);
          if (index != previous[j])
          {
            changes = true;
          }
          previous[j] = index;
        }
        if (changes)
        {
          at.push_back(s);
          runHasTimestamp = isTimestamp;
        }
        else if (at.size() > 1 && (isTimestamp || !runHasTimestamp))
        {
          at.back() = s;
          runHasTimestamp = runHasTimestamp || isTimestamp;
        }
      }
      if (!channel.end() && at.back() != end)
      {
        at.push_back(end);
      }
    }

    return channel.stack_at(Times(std::move(at), channel.unit()), lags);
  }

  /// The delay embedding of every channel of @p series, with @p delay given
  /// in the series' unit and converted into each channel's unit.
  template <typename T>
  TimeSeries<T> delay_embed(const TimeSeries<T>& series, size_t dimension, float64_t delay)
  {
    std::vector<Channel<T>> channels;
    channels.reserve(series.n_channels());
    for (size_t c = 0; c < series.n_channels(); ++c)
    {
      const auto& channel = series.channel(c);
      channels.push_back(delay_embed(channel, dimension, convert_time(delay, series.unit(), channel.unit())));
    }
    return TimeSeries<T>(std::move(channels), series.unit());
  }
}

#endif // STABLEBEAR_TIMESERIES_TIME_DELAY_EMBEDDING_H
