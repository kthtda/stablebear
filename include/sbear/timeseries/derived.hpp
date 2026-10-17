#ifndef STABLEBEAR_TIMESERIES_DERIVED_H
#define STABLEBEAR_TIMESERIES_DERIVED_H

#include "../timeseries.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace sb
{
  /// The delay embedding of @p channel with @p dimension components spaced
  /// @p delay apart, in the channel's unit. Its times are the channel's own
  /// timestamps from t_0 + (dimension - 1) * delay on, where every component
  /// has a value, and its observation at t is
  /// (x(t - (dimension - 1) * delay), ..., x(t - delay), x(t)), oldest first
  /// as in Kantz and Schreiber's convention, so each observation gains a
  /// leading axis of length @p dimension. It is a view of the channel's
  /// storage.
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

    const auto span = static_cast<float64_t>(dimension - 1) * delay;
    const auto& times = channel.times().values();
    std::vector<float64_t> at;
    if (!times.empty())
    {
      const auto first = std::lower_bound(times.begin(), times.end(), times.front() + span);
      if (first == times.end())
      {
        throw std::invalid_argument("The series is too short for this embedding: no time has its full history");
      }
      at.assign(first, times.end());
    }

    std::vector<Channel<T>> components;
    components.reserve(dimension);
    for (size_t j = 0; j < dimension; ++j)
    {
      const auto lag = -static_cast<float64_t>(dimension - 1 - j) * delay;
      std::vector<float64_t> shifted(at.size());
      std::transform(at.begin(), at.end(), shifted.begin(), [lag](float64_t t) { return t + lag; });
      components.push_back(channel.view_at(std::move(shifted)));
    }
    return Channel<T>::stack(Times(std::move(at), channel.unit()), components);
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

#endif // STABLEBEAR_TIMESERIES_DERIVED_H
