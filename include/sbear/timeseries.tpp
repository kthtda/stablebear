#ifndef STABLEBEAR_TIMESERIES_TPP
#define STABLEBEAR_TIMESERIES_TPP

// Definitions of the Channel and TimeSeries members declared in
// timeseries.hpp, which includes this file.

namespace sb
{
  template <typename T>
  Channel<T>::Channel(Times times, const Tensor<T>& values, float64_t snapTolerance, float64_t snapEps)
  {
    set_snap_tolerance(snapTolerance);
    set_snap_eps(snapEps);
    if (values.rank() == 0)
    {
      throw std::invalid_argument("Values must have a time axis");
    }
    if (values.shape(0) != times.size())
    {
      throw std::invalid_argument(
          "Expected one timestamp per observation: got " + std::to_string(times.size())
          + " timestamps for " + std::to_string(values.shape(0)) + " observations");
    }

    auto storage = std::make_shared<ChannelStorage<T>>(ChannelStorage<T>{
        std::move(times),
        std::vector<size_t>(values.shape().begin() + 1, values.shape().end()),
        {}});
    // ponytail: copies on construction; adopt the caller's storage if this shows up in profiles
    auto owned = values.copy();
    storage->values.reserve(owned.size());
    walk(owned, [&](const std::vector<size_t>& index) {
      storage->values.push_back(owned(index));
    });
    m_storage = std::move(storage);
    m_shape = m_storage->shape;
  }

  template <typename T>
  Channel<T>::Channel(Times times, std::vector<size_t> shape, std::vector<T> values, float64_t snapTolerance,
                      float64_t snapEps)
  {
    set_snap_tolerance(snapTolerance);
    set_snap_eps(snapEps);
    const size_t blockSize = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
    if (values.size() != times.size() * blockSize)
    {
      throw std::invalid_argument(
          "Expected " + std::to_string(times.size() * blockSize) + " values for " + std::to_string(times.size())
          + " observations, got " + std::to_string(values.size()));
    }
    m_storage = std::make_shared<ChannelStorage<T>>(
        ChannelStorage<T>{std::move(times), std::move(shape), std::move(values)});
    m_shape = m_storage->shape;
  }

  template <typename T>
  void Channel<T>::set_snap_tolerance(float64_t tolerance)
  {
    if (std::isnan(tolerance) || tolerance < 0 || tolerance >= 1)
    {
      throw std::invalid_argument("The snap tolerance must be in [0, 1)");
    }
    m_snapTolerance = tolerance;
  }

  template <typename T>
  void Channel<T>::set_snap_eps(float64_t eps)
  {
    if (std::isnan(eps) || std::isinf(eps) || eps < 0)
    {
      throw std::invalid_argument("snap_eps must be finite and nonnegative");
    }
    m_snapEps = eps;
  }

  template <typename T>
  Channel<T> Channel<T>::stack_at(Times times, std::span<const float64_t> lags) const
  {
    if (lags.empty())
    {
      throw std::invalid_argument("Cannot stack zero lags");
    }
    const auto& at = times.values();
    if (!at.empty() && size() == 0)
    {
      throw std::invalid_argument("Cannot view a channel without observations at a time");
    }
    // Every component of observation k reads this channel at or before its
    // observation at times[k] minus the smallest lag, the base.
    const auto smallest = *std::min_element(lags.begin(), lags.end());
    auto base = std::make_shared<std::vector<size_t>>();
    base->reserve(at.size());
    size_t next = 0;
    for (const auto t : at)
    {
      base->push_back(snapped_index(next, t - smallest));
    }

    Channel view = *this;
    view.m_storage = nullptr;
    view.m_source = std::make_shared<const Channel>(*this);
    view.m_times = std::make_shared<const Times>(std::move(times));
    view.m_base = std::move(base);
    view.m_lags.assign(lags.begin(), lags.end());
    view.m_shape.insert(view.m_shape.begin(), lags.size());
    return view;
  }

  template <typename T>
  Channel<T> Channel<T>::view_at(Times times, float64_t lag) const
  {
    const float64_t lags[] = {lag};
    auto view = stack_at(std::move(times), lags);
    view.m_shape.erase(view.m_shape.begin());
    return view;
  }

  template <typename T>
  bool Channel<T>::operator==(const Channel& other) const
  {
    return m_snapTolerance == other.m_snapTolerance && m_snapEps == other.m_snapEps && m_shape == other.m_shape
           && times() == other.times() && values() == other.values();
  }

  template <typename T>
  Tensor<T> Channel<T>::lookup(float64_t t) const
  {
    Tensor<T> out(m_shape);
    const float64_t times[] = {t};
    const size_t order[] = {0};
    lookup_sorted(times, order, out.data());
    return out;
  }

  template <typename T>
  template <std::random_access_iterator Out>
  void Channel<T>::lookup_sorted(std::span<const float64_t> sortedTimes, std::span<const size_t> order, Out out) const
  {
    if (m_source)
    {
      // Look up the source at each lagged time, so a view snaps as its
      // source does. Component j of the query for block order[k] fills
      // source block order[k] * m + j.
      const auto m = m_lags.size();
      std::vector<float64_t> lagged(sortedTimes.size());
      std::vector<size_t> blocks(sortedTimes.size());
      for (size_t j = 0; j < m; ++j)
      {
        for (size_t k = 0; k < sortedTimes.size(); ++k)
        {
          lagged[k] = sortedTimes[k] - m_lags[j];
          blocks[k] = order[k] * m + j;
        }
        m_source->lookup_sorted(lagged, blocks, out);
      }
      return;
    }
    const auto blockSize = block_size();
    const auto indices = indices_sorted(sortedTimes);
    for (size_t k = 0; k < sortedTimes.size(); ++k)
    {
      copy_observation(indices[k], out + static_cast<std::ptrdiff_t>(order[k] * blockSize));
    }
  }

  template <typename T>
  typename std::vector<T>::const_reference Channel<T>::flat_value(size_t i) const
  {
    if (!m_source)
    {
      return m_storage->values[i];
    }
    const auto blockSize = block_size();
    const auto sourceBlock = blockSize / m_lags.size();
    const auto r = i % blockSize;
    return m_source->flat_value(source_index(i / blockSize, r / sourceBlock) * sourceBlock + r % sourceBlock);
  }

  template <typename T>
  Tensor<T> Channel<T>::values() const
  {
    std::vector<size_t> fullShape{size()};
    fullShape.insert(fullShape.end(), shape().begin(), shape().end());
    Tensor<T> out(fullShape);
    if (size() > 0)
    {
      const auto blockSize = block_size();
      for (size_t k = 0; k < size(); ++k)
      {
        copy_observation(k, out.data() + static_cast<std::ptrdiff_t>(k * blockSize));
      }
    }
    return out;
  }

  template <typename T>
  Tensor<T> Channel<T>::observation(std::ptrdiff_t i) const
  {
    const auto n = static_cast<std::ptrdiff_t>(size());
    if (i < -n || i >= n)
    {
      throw std::out_of_range(
          "Observation index " + std::to_string(i) + " is out of range for "
          + std::to_string(n) + " observations");
    }
    Tensor<T> out(shape());
    copy_observation(static_cast<size_t>(i < 0 ? i + n : i), out.data());
    return out;
  }

  template <typename T>
  template <std::random_access_iterator Out>
  void Channel<T>::copy_observation(size_t k, Out out) const
  {
    if (!m_source)
    {
      const auto blockSize = block_size();
      const auto& stored = m_storage->values;
      for (size_t v = 0; v < blockSize; ++v)
      {
        out[static_cast<std::ptrdiff_t>(v)] = stored[k * blockSize + v];
      }
      return;
    }
    const auto sourceBlock = m_source->block_size();
    for (size_t j = 0; j < m_lags.size(); ++j)
    {
      m_source->copy_observation(source_index(k, j), out + static_cast<std::ptrdiff_t>(j * sourceBlock));
    }
  }

  template <typename T>
  size_t Channel<T>::source_index(size_t k, size_t j) const
  {
    // The first timestamp after the lagged time is at or before the one
    // after the base, usually a few observations back.
    const auto& times = m_source->times().values();
    const auto t = this->times().values()[k] - m_lags[j];
    const auto afterBase = times.begin() + static_cast<std::ptrdiff_t>((*m_base)[k] + 1);
    const auto next = gallop_upper_bound(times.begin(), times.end(), afterBase, t);
    return m_source->snap(static_cast<size_t>(next - times.begin()), t);
  }

  template <typename T>
  std::vector<size_t> Channel<T>::indices_sorted(std::span<const float64_t> sortedTimes) const
  {
    std::vector<size_t> indices;
    indices.reserve(sortedTimes.size());
    size_t next = 0;
    for (const auto t : sortedTimes)
    {
      indices.push_back(snapped_index(next, t));
    }
    return indices;
  }

  template <typename T>
  size_t Channel<T>::snapped_index(size_t& next, float64_t t) const
  {
    // The first timestamp after t is at or after the cursor. Gallop to it,
    // so a query close ahead costs a comparison or two and a distant one
    // O(log n).
    const auto& times = this->times().values();
    const auto from = times.begin() + static_cast<std::ptrdiff_t>(next);
    next = static_cast<size_t>(gallop_upper_bound(from, times.end(), from, t) - times.begin());
    return snap(next, t);
  }

  template <typename T>
  size_t Channel<T>::snap(size_t next, float64_t t) const
  {
    const auto& times = this->times().values();
    if (next == 0)
    {
      return 0;
    }
    // A query computed in floating point, such as 0.3 - 0.1, can land just
    // below the observation it means; snap it forward. Measuring against
    // the gap keeps the rule independent of the unit and of how large the
    // timestamps are, and a tolerance below 1 never skips an observation.
    // The epsilon cap keeps widely spaced observations from snapping
    // queries that are far off in absolute terms but small next to the gap.
    if (next < times.size())
    {
      const auto prev = times[next - 1];
      const auto magnitude = std::max(std::abs(prev), std::abs(times[next]));
      const auto cap = m_snapEps * std::numeric_limits<float64_t>::epsilon() * magnitude;
      if (times[next] - t <= std::min(m_snapTolerance * (times[next] - prev), cap))
      {
        return next;
      }
    }
    return next - 1;
  }

  template <typename T>
  TimeSeries<T>::TimeSeries(std::vector<Channel<T>> channels, TimeUnit unit)
    : m_channels(std::move(channels)), m_unit(unit)
  {
    if (m_channels.empty())
    {
      throw std::invalid_argument("A time series needs at least one channel");
    }
  }

  template <typename T>
  TimeSeries<T> TimeSeries<T>::from_channels(const std::vector<TimeSeries>& parts, std::optional<TimeUnit> unit)
  {
    if (parts.empty())
    {
      throw std::invalid_argument("A time series needs at least one channel");
    }
    std::vector<Channel<T>> channels;
    channels.reserve(parts.size());
    for (const auto& part : parts)
    {
      if (part.n_channels() != 1)
      {
        throw std::invalid_argument("Each channel must be a one-channel series; select channels with series[i]");
      }
      channels.push_back(part.m_channels.front());
      if (!unit && part.unit() != parts.front().unit())
      {
        throw std::invalid_argument("The channels differ in unit; pass unit= to choose one");
      }
    }
    return TimeSeries(std::move(channels), unit.value_or(parts.front().unit()));
  }

  template <typename T>
  TimeSeries<T> TimeSeries<T>::select(const std::vector<std::ptrdiff_t>& indices) const
  {
    const auto n = static_cast<std::ptrdiff_t>(m_channels.size());
    std::vector<Channel<T>> channels;
    channels.reserve(indices.size());
    for (const auto i : indices)
    {
      if (i < -n || i >= n)
      {
        throw std::out_of_range(
            "Channel index " + std::to_string(i) + " is out of range for " + std::to_string(n) + " channels");
      }
      channels.push_back(m_channels[static_cast<size_t>(i < 0 ? i + n : i)]);
    }
    return TimeSeries(std::move(channels), m_unit);
  }

  template <typename T>
  TimeSeries<T> TimeSeries<T>::to(TimeUnit unit) const
  {
    auto series = *this;
    series.m_unit = unit;
    return series;
  }

  template <typename T>
  NestedTensor<T> TimeSeries<T>::evaluate(float64_t t) const
  {
    check_observed(t);
    Tensor<NestedTensor<T>> values({m_channels.size()});
    for (size_t i = 0; i < m_channels.size(); ++i)
    {
      const auto& c = m_channels[i];
      values(i) = NestedTensor<T>::from_leaf_view(c.lookup(convert_time(t, m_unit, c.unit())));
    }
    return NestedTensor<T>::from_outer_view(std::move(values), 1);
  }

  template <typename T>
  NestedTensor<T> TimeSeries<T>::evaluate(const Tensor<float64_t>& t) const
  {
    std::vector<float64_t> queries;
    walk(t, [&](const std::vector<size_t>& index) { queries.push_back(t(index)); });
    if (!queries.empty())
    {
      const auto [lo, hi] = std::minmax_element(queries.begin(), queries.end());
      check_observed(*lo);
      check_observed(*hi);
    }
    std::vector<size_t> order(queries.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return queries[a] < queries[b]; });

    Tensor<NestedTensor<T>> values({m_channels.size()});
    std::vector<float64_t> sorted(queries.size());
    for (size_t c = 0; c < m_channels.size(); ++c)
    {
      const auto& channel = m_channels[c];
      for (size_t k = 0; k < order.size(); ++k)
      {
        sorted[k] = convert_time(queries[order[k]], m_unit, channel.unit());
      }
      std::vector<size_t> shape(t.shape().begin(), t.shape().end());
      shape.insert(shape.end(), channel.shape().begin(), channel.shape().end());
      Tensor<T> out(shape);
      if (!order.empty())
      {
        channel.lookup_sorted(sorted, order, out.data());
      }
      values(c) = NestedTensor<T>::from_leaf_view(std::move(out));
    }
    return NestedTensor<T>::from_outer_view(std::move(values), 1);
  }

  template <typename T>
  std::pair<float64_t, float64_t> TimeSeries<T>::domain() const
  {
    auto first = -std::numeric_limits<float64_t>::infinity();
    auto last = std::numeric_limits<float64_t>::infinity();
    for (const auto& c : m_channels)
    {
      if (c.size() == 0)
      {
        throw std::domain_error("Cannot evaluate a channel without observations");
      }
      const auto& recorded = c.times().values();
      first = std::max(first, convert_time(recorded.front(), c.unit(), m_unit));
      last = std::min(last, convert_time(recorded.back(), c.unit(), m_unit));
    }
    return {first, last};
  }

  template <typename T>
  void TimeSeries<T>::check_observed(float64_t t) const
  {
    const auto [first, last] = domain();
    if (t < first || t > last)
    {
      std::ostringstream ss;
      ss << "Time " << t << " is outside the domain [" << first << ", " << last << "]";
      throw std::domain_error(ss.str());
    }
  }
}

#endif // STABLEBEAR_TIMESERIES_TPP
