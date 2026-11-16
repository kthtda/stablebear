#ifndef STABLEBEAR_TIMESERIES_TIME_DELAY_EMBEDDING_H
#define STABLEBEAR_TIMESERIES_TIME_DELAY_EMBEDDING_H

#include "../errors.hpp"
#include "../tensor.hpp"
#include "../timeseries.hpp"
#include "../ufunc.hpp"
#include "../walk.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sb
{
  /// The delay embedding of @p channel with @p dimension components spaced
  /// @p delay apart, in the channel's unit, with w = (dimension - 1) * delay.
  /// Its value at t is (x(t - w), ..., x(t - delay), x(t)), oldest first as
  /// in Kantz and Schreiber's convention, for every t from t_0 + w to the end
  /// of the channel's domain. If @p forward, it is instead
  /// (x(t), x(t + delay), ..., x(t + w)), also oldest first, for every t from
  /// t_0 to the end of the domain minus w. Each end is included when the
  /// channel's is, and each observation gains a leading axis of length
  /// @p dimension. Its times are the start of its domain and the times in it
  /// where a component reaches an observation, t_i + k * delay, or
  /// t_i - k * delay if @p forward; times that round to the same point are
  /// recorded once, at the time they snap to. It is a view of the channel's
  /// storage.
  namespace detail
  {
    /// Throw unless @p dimension and @p delay make an embedding.
    inline void check_embedding_arguments(size_t dimension, float64_t delay)
    {
      if (dimension == 0)
      {
        throw std::invalid_argument("The embedding dimension must be at least 1");
      }
      if (std::isnan(delay) || std::isinf(delay) || delay <= 0)
      {
        throw std::invalid_argument("The embedding delay must be finite and positive");
      }
    }

    /// The lag of each component of an embedding, oldest first: component j
    /// reads the channel lags[j] before t. Looking forward, the lags are
    /// negative: component j reads j * delay after t.
    [[nodiscard]] inline std::vector<float64_t> embedding_lags(size_t dimension, float64_t delay, bool forward)
    {
      std::vector<float64_t> lags(dimension);
      for (size_t j = 0; j < dimension; ++j)
      {
        lags[j] = forward ? -(static_cast<float64_t>(j) * delay) : static_cast<float64_t>(dimension - 1 - j) * delay;
      }
      return lags;
    }

    /// The first and last times of the embedding of @p channel, which has
    /// observations, with lags from @p lowest to @p highest. A channel with
    /// an end, such as a window, can be evaluated past its last timestamp up
    /// to its end, so the embedding can too, up to that end plus the lowest
    /// lag. Throws if no time has all of the components.
    template <typename T>
    std::pair<float64_t, float64_t> embedding_bounds(const Channel<T>& channel, float64_t lowest, float64_t highest)
    {
      const auto start = channel.time(0) + highest;
      const auto end = channel.end().value_or(channel.time(channel.size() - 1)) + lowest;
      if (channel.end_closed() ? start > end : start >= end)
      {
        throw std::invalid_argument("The series is too short for this embedding: no time has all of its components");
      }
      return {start, end};
    }

    /// Throw unless every channel of @p series has an embedding with
    /// @p delay in the series' unit, as delay_embed() checks it.
    template <typename T>
    void check_embedding_fits(const TimeSeries<T>& series, size_t dimension, float64_t delay, bool forward)
    {
      const auto lags = embedding_lags(dimension, delay, forward);
      const auto [lowest, highest] = std::minmax_element(lags.begin(), lags.end());
      for (size_t c = 0; c < series.n_channels(); ++c)
      {
        if (series.channel(c).size() > 0)
        {
          (void)embedding_bounds(series.channel(c), *lowest, *highest);
        }
      }
    }
  }

  template <typename T>
  Channel<T> delay_embed(const Channel<T>& channel, size_t dimension, float64_t delay, bool forward = false)
  {
    detail::check_embedding_arguments(dimension, delay);

    const auto lags = detail::embedding_lags(dimension, delay, forward);
    const auto [lowest, highest] = std::minmax_element(lags.begin(), lags.end());

    const auto n = channel.size();
    std::vector<float64_t> at;
    if (n > 0)
    {
      // The embedding keeps an open start through view_lagged_at.
      const auto [start, end] = detail::embedding_bounds(channel, *lowest, *highest);
      const auto closed = channel.end_closed();
      const auto past = [&](float64_t s) { return closed ? s > end : s >= end; };
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
      // snap_tolerance, as evaluation does. A component's index
      // never decreases with time, so a run of the same point ends where some
      // component's index differs from the one at the time before. Only a
      // component whose next shifted time is at s, or within snapping
      // distance after it, can read a different observation at s. Snapping
      // moves a time by at most snap_tolerance times its magnitude; a few
      // machine epsilons more cover the rounding of the lags.
      //
      // Record each run at the time its members snap to: the channel's own
      // timestamp if the run has one, else its latest time, since snapping
      // moves a time forward onto the observation it falls just short of.
      // The first point stays at the start, so the domain keeps its start.
      const auto reach = channel.snap_tolerance() + 4 * std::numeric_limits<float64_t>::epsilon();
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
          // The component with lag 0 has the channel's own timestamps as its
          // shifted times.
          isTimestamp = isTimestamp || lags[j] == 0;
        }
        near.clear();
        while (!queue.empty() && queue.top().first <= s + reach * (std::abs(s) + *highest - *lowest))
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
    }

    auto embedding = channel.view_lagged_at(channel.times_like(std::move(at)), lags);
    if (n == 0)
    {
      return embedding;
    }
    // The domain ends at the channel's end plus the lowest lag, which a run
    // recorded at an earlier time may not reach.
    return embedding.until(channel.end().value_or(channel.time(n - 1)) + *lowest, channel.end_closed());
  }

  /// The delay embedding of every channel of @p series, with @p delay given
  /// in the series' unit, in which its channels hold their times.
  template <typename T>
  TimeSeries<T> delay_embed(const TimeSeries<T>& series, size_t dimension, float64_t delay, bool forward = false)
  {
    std::vector<Channel<T>> channels;
    channels.reserve(series.n_channels());
    for (size_t c = 0; c < series.n_channels(); ++c)
    {
      channels.push_back(delay_embed(series.channel(c), dimension, delay, forward));
    }
    return TimeSeries<T>(std::move(channels), series.unit(), series.epoch());
  }

  /// The delay embedding of a time series as an operation without extra
  /// dimensions: each series maps to its embedding. @p delay is a count of
  /// @p delayUnit, converted into the unit of each series, or a plain number
  /// for series without a unit.
  template <typename T>
  struct DelayEmbedOperation
  {
    using input_type = TimeSeries<T>;
    using output_type = TimeSeries<T>;

    size_t dimension = 1;
    float64_t delay = 1;
    TimeUnit delayUnit = TimeUnit::Unitless;
    bool forward = false;

    [[nodiscard]] std::vector<size_t> extra_dims() const { return {}; }

    /// The delay in the unit of @p series, throwing TypeError if a duration
    /// is given for a series without a unit or a number for one with a unit.
    [[nodiscard]] float64_t delay_in(const TimeSeries<T>& series) const
    {
      check_query_unit(delayUnit, series.unit());
      return convert_time(delay, delayUnit, series.unit());
    }

    void operator()(const TimeSeries<T>& series, std::span<TimeSeries<T>> out) const
    {
      out[0] = delay_embed(series, dimension, delay_in(series), forward);
    }
  };

  template <typename T>
  using DelayEmbedTask = UfuncTask<DelayEmbedOperation<T>, Tensor<TimeSeries<T>>>;

  /// A task embedding every series of @p seriesTensor into a tensor of the
  /// same shape, as delay_embed() embeds each, with the delay as
  /// DelayEmbedOperation takes it. Every series is checked before the task
  /// is made, so that a series that cannot be embedded throws, for the first
  /// such series, an error of the same type whose message names its
  /// position.
  template <typename T>
  std::unique_ptr<DelayEmbedTask<T>> make_delay_embed_task(const Tensor<TimeSeries<T>>& seriesTensor,
                                                           size_t dimension, float64_t delay, TimeUnit delayUnit,
                                                           bool forward = false)
  {
    detail::check_embedding_arguments(dimension, delay);
    const DelayEmbedOperation<T> op{.dimension = dimension, .delay = delay, .delayUnit = delayUnit,
                                    .forward = forward};
    walk(seriesTensor, [&](const std::vector<size_t>& index) {
      const auto& series = seriesTensor(index);
      try
      {
        detail::check_embedding_fits(series, dimension, op.delay_in(series), forward);
      }
      catch (const TypeError& e)
      {
        throw TypeError(detail::position_text(index) + ": " + e.what());
      }
      catch (const std::invalid_argument& e)
      {
        throw std::invalid_argument(detail::position_text(index) + ": " + e.what());
      }
    });
    return make_ufunc_task(seriesTensor, op, "Embedding time series", "series");
  }
}

#endif // STABLEBEAR_TIMESERIES_TIME_DELAY_EMBEDDING_H
