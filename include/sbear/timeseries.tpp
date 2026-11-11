#ifndef STABLEBEAR_TIMESERIES_TPP
#define STABLEBEAR_TIMESERIES_TPP

// Definitions of the Channel and TimeSeries members declared in
// timeseries.hpp, which includes this file.

namespace sb
{
  namespace detail
  {
    /// The shortest text that reads back as @p x, so a bound such as
    /// 0.8999999999999999 does not show as 0.9.
    inline std::string shortest_text(float64_t x)
    {
      // std::to_chars for floating point needs macOS 13.3, and wheels target
      // macOS 11, so try increasing precision until the text reads back.
      std::array<char, 32> buffer;
      for (int precision = 1; precision <= 17; ++precision)
      {
        std::snprintf(buffer.data(), buffer.size(), "%.*g", precision, x);
        if (std::strtod(buffer.data(), nullptr) == x)
        {
          break;
        }
      }

      // %g turns to scientific notation once the exponent reaches the
      // precision, so 906680 shows as 9.0668e+05. Like std::to_chars, use
      // fixed notation when it is no longer.
      std::string text = buffer.data();
      const auto e = text.find('e');
      if (e != std::string::npos)
      {
        const int exponent = std::atoi(text.c_str() + e + 1);
        if (exponent >= 0 && (x < 0) + exponent + 1 <= static_cast<int>(text.size()))
        {
          std::snprintf(buffer.data(), buffer.size(), "%.0f", x);
          text = buffer.data();
        }
      }
      return text;
    }

    /// The queries in @p t in row-major order, and their positions sorted by
    /// time. A NaN sorts first, so checking the first and last sorted queries
    /// against a domain rejects it.
    template <typename Q>
    struct SortedQueries
    {
      std::vector<Q> values;
      std::vector<size_t> order;
    };

    template <typename Q>
    SortedQueries<Q> sort_queries(const Tensor<Q>& t)
    {
      SortedQueries<Q> queries;
      walk(t, [&](const std::vector<size_t>& index) { queries.values.push_back(t(index)); });
      queries.order.resize(queries.values.size());
      std::iota(queries.order.begin(), queries.order.end(), size_t{0});
      const auto& v = queries.values;
      std::sort(queries.order.begin(), queries.order.end(), [&](size_t a, size_t b) {
        if constexpr (std::is_floating_point_v<Q>)
        {
          return std::isnan(v[a]) ? !std::isnan(v[b]) : v[a] < v[b];
        }
        else
        {
          return v[a] < v[b];
        }
      });
      return queries;
    }
  }

  template <typename T>
  Channel<T>::Channel(Times times, const Tensor<T>& values, float64_t snapTolerance)
  {
    set_snap_tolerance(snapTolerance);
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
    storage->values.reserve(values.size());
    walk(values, [&](const std::vector<size_t>& index) {
      storage->values.push_back(detail::store_copy(values(index)));
    });
    m_storage = std::move(storage);
    m_shape = m_storage->shape;
  }

  template <typename T>
  Channel<T>::Channel(Times times, std::vector<size_t> shape, std::vector<T> values, float64_t snapTolerance)
  {
    set_snap_tolerance(snapTolerance);
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
  Channel<T> Channel<T>::view_lagged_at(Times times, std::span<const float64_t> lags) const
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
    size_t hint = 0;
    for (const auto t : at)
    {
      hint = index_at(t - smallest, hint);
      base->push_back(hint);
    }

    Channel view = *this;
    view.m_storage = nullptr;
    view.m_source = std::make_shared<const Channel>(*this);
    view.m_times = std::make_shared<const Times>(std::move(times));
    view.m_base = std::move(base);
    view.m_lags.assign(lags.begin(), lags.end());
    view.m_shape.insert(view.m_shape.begin(), lags.size());
    // A closed end at the new last timestamp is dropped, as until drops it.
    return view.m_end ? view.until(*view.m_end, view.m_endClosed) : view;
  }

  template <typename T>
  Channel<T> Channel<T>::view_at_indices(std::vector<size_t> indices) const
  {
    std::vector<float64_t> at;
    at.reserve(indices.size());
    for (size_t k = 0; k < indices.size(); ++k)
    {
      if (indices[k] >= size())
      {
        throw std::out_of_range(
            "Observation index " + std::to_string(indices[k]) + " is out of range for " + std::to_string(size())
            + " observations");
      }
      if (k > 0 && indices[k] <= indices[k - 1])
      {
        throw std::invalid_argument("Observation indices must be strictly increasing");
      }
      at.push_back(time(indices[k]));
    }

    // One component at lag 0, whose base is the given index. A view with
    // one component reads observation i of its source at its base i, so a
    // view of it takes from that source directly: chains of views stay one
    // level deep, and the views between are freed.
    Channel view = *this;
    view.m_storage = nullptr;
    if (m_source && m_lags.size() == 1)
    {
      for (auto& i : indices)
      {
        i = (*m_base)[i];
      }
      view.m_source = m_source;
    }
    else
    {
      view.m_source = std::make_shared<const Channel>(*this);
    }
    view.m_times = std::make_shared<const Times>(times_like(std::move(at)));
    view.m_base = std::make_shared<const std::vector<size_t>>(std::move(indices));
    view.m_lags = {0.0};
    view.m_end = std::nullopt;
    view.m_endClosed = false;
    view.m_startClosed = true;
    return view;
  }

  template <typename T>
  Channel<T> Channel<T>::with_times(Times times) const
  {
    if (m_lags.size() > 1)
    {
      throw std::logic_error("A view with several lags finds its observations from its times; it cannot move them");
    }
    if (times.size() != size())
    {
      throw std::invalid_argument("Expected one time per observation: got " + std::to_string(times.size())
                                  + " times for " + std::to_string(size()) + " observations");
    }
    auto channel = *this;
    channel.m_times = std::make_shared<const Times>(std::move(times));
    channel.m_end = std::nullopt;
    channel.m_endClosed = false;
    return channel;
  }

  template <typename T>
  Channel<T> Channel<T>::until(float64_t end, bool closed) const
  {
    const auto atLast = size() > 0 && end == time(size() - 1);
    if (std::isnan(end) || (size() > 0 && end < time(size() - 1)) || (atLast && !closed))
    {
      throw std::invalid_argument("The end of a channel must come after its last timestamp");
    }
    auto channel = *this;
    channel.m_end = atLast ? std::nullopt : std::optional(end);
    channel.m_endClosed = closed;
    return channel;
  }

  template <typename T>
  Channel<T> Channel<T>::to(TimeUnit from, TimeUnit to) const
  {
    if (from == to)
    {
      return *this;
    }
    auto channel = *this;
    channel.m_times = std::make_shared<const Times>(held_times().to(from, to));
    if (m_end)
    {
      channel.m_end = convert_time(*m_end, from, to);
    }
    for (auto& lag : channel.m_lags)
    {
      lag = convert_time(lag, from, to);
    }
    // A view with several lags reads its source at its own times less its
    // lags, so the source moves to the same unit. One with one lag reads it
    // by index only, so it stays as it is.
    if (m_source && m_lags.size() > 1)
    {
      channel.m_source = std::make_shared<const Channel>(m_source->to(from, to));
    }
    return channel.with_end_after_last();
  }

  template <typename T>
  Channel<T> Channel<T>::shifted(float64_t delta) const
  {
    auto channel = *this;
    auto values = held_times().values();
    for (auto& value : values)
    {
      value += delta;
    }
    detail::keep_increasing(values);
    channel.m_times = std::make_shared<const Times>(held_times().with_values(std::move(values)));
    if (m_end)
    {
      channel.m_end = *m_end + delta;
    }
    // A view with several lags reads its source at its own times less its
    // lags, so the source shifts alike. One with one lag reads it by index
    // only, so it stays as it is.
    if (m_source && m_lags.size() > 1)
    {
      channel.m_source = std::make_shared<const Channel>(m_source->shifted(delta));
    }
    return channel.with_end_after_last();
  }

  template <typename T>
  Channel<T> Channel<T>::with_end_after_last() const
  {
    if (!m_end || size() == 0 || *m_end > time(size() - 1))
    {
      return *this;
    }
    auto channel = *this;
    channel.m_end = m_endClosed ? std::nullopt
                                : std::optional(std::nextafter(time(size() - 1),
                                                               std::numeric_limits<float64_t>::infinity()));
    return channel;
  }

  template <typename T>
  Channel<T> Channel<T>::with_start_closed(bool closed) const
  {
    auto channel = *this;
    channel.m_startClosed = closed;
    return channel;
  }

  template <typename T>
  bool Channel<T>::operator==(const Channel& other) const
  {
    return same_observations(other) && held_times() == other.held_times();
  }

  template <typename T>
  bool Channel<T>::same_observations(const Channel& other) const
  {
    return m_snapTolerance == other.m_snapTolerance && m_shape == other.m_shape
           && m_end == other.m_end && end_closed() == other.end_closed() && m_startClosed == other.m_startClosed
           && held_times().values() == other.held_times().values() && values() == other.values();
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
        out[static_cast<std::ptrdiff_t>(v)] = detail::store_copy(stored[k * blockSize + v]);
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
    // A single component is the base, given by view_at_indices or found by
    // view_lagged_at.
    if (m_lags.size() == 1)
    {
      return (*m_base)[k];
    }
    // The first timestamp after the lagged time is at or before the one
    // after the base, usually a few observations back.
    const auto& times = m_source->held_times().values();
    const auto t = time(k) - m_lags[j];
    const auto afterBase = times.begin() + static_cast<std::ptrdiff_t>((*m_base)[k] + 1);
    const auto next = gallop_upper_bound(times.begin(), times.end(), afterBase, t);
    return m_source->snap(static_cast<size_t>(next - times.begin()), t);
  }

  template <typename T>
  std::vector<size_t> Channel<T>::indices_sorted(std::span<const float64_t> sortedTimes) const
  {
    std::vector<size_t> indices;
    indices.reserve(sortedTimes.size());
    size_t hint = 0;
    for (const auto t : sortedTimes)
    {
      hint = index_at(t, hint);
      indices.push_back(hint);
    }
    return indices;
  }

  template <typename T>
  size_t Channel<T>::index_at(float64_t t, size_t hint) const
  {
    // Gallop from the hint to the first timestamp after t, so a query near
    // the hint costs a comparison or two and a distant one O(log n).
    const auto& times = held_times().values();
    const auto begin = times.begin();
    const auto from = begin + static_cast<std::ptrdiff_t>(std::min(hint, times.size()));
    return snap(static_cast<size_t>(gallop_upper_bound(begin, times.end(), from, t) - begin), t);
  }

  template <typename T>
  size_t Channel<T>::snap(size_t next, float64_t t) const
  {
    const auto& times = held_times().values();
    if (next == 0)
    {
      return 0;
    }
    // A query computed in floating point, such as 0.3 - 0.1, can land just
    // below the observation it means; snap it forward. As in the tolerant
    // comparison of APL, the distance is relative to the magnitude of the
    // timestamps, so it scales with their rounding errors and does not
    // depend on the unit. On large timestamps close together, it can exceed
    // the gap, so that even a query at the previous timestamp would snap past
    // the observation recorded there; capping it at a quarter of the gap
    // keeps a query nearer the previous observation where it is.
    if (next < times.size() && m_snapTolerance > 0)
    {
      const auto prev = times[next - 1];
      const auto magnitude = std::max(std::abs(prev), std::abs(times[next]));
      const auto reach = std::min(m_snapTolerance * magnitude, (times[next] - prev) / 4);
      if (times[next] - t <= reach)
      {
        return next;
      }
    }
    return next - 1;
  }

  template <typename T>
  TimeSeries<T>::TimeSeries(std::vector<Channel<T>> channels, TimeUnit unit, std::optional<DateTime> epoch)
    : m_channels(std::move(channels)), m_unit(unit), m_epoch(epoch)
  {
    if (m_channels.empty())
    {
      throw std::invalid_argument("A time series needs at least one channel");
    }
    check_epoch(m_unit, m_epoch);
  }

  template <typename T>
  TimeSeries<T> TimeSeries<T>::from_channels(const std::vector<TimeSeries>& parts, std::optional<TimeUnit> unit)
  {
    if (parts.empty())
    {
      throw std::invalid_argument("A time series needs at least one channel");
    }
    const auto& front = parts.front();
    const auto target = unit.value_or(front.unit());
    std::vector<Channel<T>> channels;
    channels.reserve(parts.size());
    for (const auto& part : parts)
    {
      if (part.n_channels() != 1)
      {
        throw std::invalid_argument("Each channel must be a one-channel series; select channels with series[i]");
      }
      check_same_kind_of_unit(part.unit(), front.unit());
      if (!unit && part.unit() != front.unit())
      {
        throw std::invalid_argument("The channels differ in unit; pass unit= to choose one");
      }
      check_same_kind_of_unit(part.unit(), target);
      // One epoch for the series: every channel has the same one, or none.
      // Python counts dated channels from one epoch before combining them.
      if (part.epoch() != front.epoch())
      {
        throw std::invalid_argument("The channels have different epochs; give every channel an epoch, or none");
      }
      channels.push_back(part.m_channels.front().to(part.unit(), target));
    }
    return TimeSeries(std::move(channels), target, front.epoch());
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
    return TimeSeries(std::move(channels), m_unit, m_epoch);
  }

  template <typename T>
  TimeSeries<T> TimeSeries<T>::to(TimeUnit unit) const
  {
    if (unit == TimeUnit::Unitless && m_unit != TimeUnit::Unitless)
    {
      throw std::invalid_argument("Times in a unit cannot become plain numbers");
    }
    auto series = *this;
    for (auto& channel : series.m_channels)
    {
      channel = channel.to(m_unit, unit);
    }
    series.m_unit = unit;
    return series;
  }

  template <typename T>
  TimeSeries<T> TimeSeries<T>::with_epoch(std::optional<DateTime> epoch) const
  {
    check_epoch(m_unit, epoch);
    auto series = *this;
    series.m_epoch = epoch;
    return series;
  }

  template <typename T>
  TimeSeries<T> TimeSeries<T>::shifted(float64_t delta, std::optional<DateTime> epoch) const
  {
    check_epoch(m_unit, epoch);
    auto series = *this;
    for (auto& channel : series.m_channels)
    {
      channel = channel.shifted(delta);
    }
    series.m_epoch = epoch;
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
      values(i) = NestedTensor<T>::from_leaf_view(c.lookup(t));
    }
    return NestedTensor<T>::from_outer_view(std::move(values), 1);
  }

  template <typename T>
  NestedTensor<T> TimeSeries<T>::evaluate(const Tensor<float64_t>& t) const
  {
    const auto [queries, order] = detail::sort_queries(t);
    if (!order.empty())
    {
      check_observed(queries[order.front()]);
      check_observed(queries[order.back()]);
    }

    Tensor<NestedTensor<T>> values({m_channels.size()});
    std::vector<float64_t> sorted(queries.size());
    for (size_t c = 0; c < m_channels.size(); ++c)
    {
      const auto& channel = m_channels[c];
      for (size_t k = 0; k < order.size(); ++k)
      {
        sorted[k] = queries[order[k]];
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
  typename TimeSeries<T>::Domain TimeSeries<T>::domain() const
  {
    Domain domain{-std::numeric_limits<float64_t>::infinity(), std::numeric_limits<float64_t>::infinity(), true,
                  true};
    for (const auto& c : m_channels)
    {
      if (c.size() == 0)
      {
        throw std::domain_error("Cannot evaluate a channel without observations");
      }
      // An open end at the same time as another channel's closed one still
      // excludes that time, and likewise for starts.
      const auto first = c.time(0);
      if (first > domain.first || (first == domain.first && !c.start_closed()))
      {
        domain.first = first;
        domain.closedStart = c.start_closed();
      }
      const auto last = c.end().value_or(c.time(c.size() - 1));
      if (last < domain.last || (last == domain.last && !c.end_closed()))
      {
        domain.last = last;
        domain.closedEnd = c.end_closed();
      }
    }
    return domain;
  }

  template <typename T>
  void TimeSeries<T>::check_observed(float64_t t) const
  {
    const auto domain = this->domain();
    if (!domain.contains(t))
    {
      throw TimeSeriesDomainError("Time " + detail::shortest_text(t) + " is outside the domain " + domain.to_string(),
                                  t, domain);
    }
  }

  inline std::string TimeDomain::to_string() const
  {
    return (closedStart ? "[" : "(") + detail::shortest_text(first) + ", " + detail::shortest_text(last)
           + (closedEnd ? "]" : ")");
  }

  namespace detail
  {
    /// The position of a series in a tensor, as "TimeSeries at position
    /// (1, 2)".
    inline std::string position_text(const std::vector<size_t>& index)
    {
      std::string position;
      for (const auto i : index)
      {
        position += (position.empty() ? "" : ", ") + std::to_string(i);
      }
      return "TimeSeries at position (" + position + ")";
    }

    /// Every series of @p series evaluated at the queries @p t, where
    /// time_of(one, c, q) is the time of query q in channel c of the series
    /// one, increasing in q. check(one) throws for a series that cannot take
    /// the queries. See evaluate() for the result and errors.
    template <typename T, typename Q, typename TimeOf, typename Check>
    NestedTensor<T> evaluate_each(const Tensor<TimeSeries<T>>& series, const Tensor<Q>& t, TimeOf time_of,
                                  Check check, Executor& exec)
    {
      if (series.size() == 0)
      {
        throw std::invalid_argument("Cannot evaluate an empty tensor of series");
      }

      // The queries are sorted once for every series, as TimeSeries::evaluate
      // sorts them for one.
      const auto [queries, order] = sort_queries(t);

      // Leaf c holds, in block k, the values of channel c of the k-th series
      // in row-major order.
      const auto& first = series(std::vector<size_t>(series.rank(), 0));
      std::vector<Tensor<T>> leaves;
      for (size_t c = 0; c < first.n_channels(); ++c)
      {
        std::vector<size_t> shape(series.shape().begin(), series.shape().end());
        shape.insert(shape.end(), t.shape().begin(), t.shape().end());
        shape.insert(shape.end(), first.channel(c).shape().begin(), first.channel(c).shape().end());
        leaves.emplace_back(shape);
      }

      // Check every series in order first, so that an error names the first
      // series that cannot be evaluated, whichever worker would reach it.
      walk(series, [&](const std::vector<size_t>& index) {
        const auto& one = series(index);
        check(one);
        if (one.n_channels() != first.n_channels())
        {
          throw std::invalid_argument("Every series must have the same number of channels");
        }
        for (size_t c = 0; c < one.n_channels(); ++c)
        {
          if (one.channel(c).shape() != first.channel(c).shape())
          {
            throw std::invalid_argument("Every series must have the same channel shapes");
          }
        }
        if (order.empty())
        {
          return;
        }
        for (size_t c = 0; c < one.n_channels(); ++c)
        {
          // The domain of the series, or of channel c for queries that
          // depend on the channel.
          const auto domain = one.n_channels() == 1 ? one.domain() : one.select({static_cast<std::ptrdiff_t>(c)}).domain();
          for (const auto query : {queries[order.front()], queries[order.back()]})
          {
            const auto x = time_of(one, c, query);
            if (!domain.contains(x))
            {
              throw TimeSeriesDomainError(position_text(index) + ": Time " + shortest_text(x)
                                          + " is outside the domain " + domain.to_string(), x, domain, index, c);
            }
          }
        }
      });

      if (!order.empty())
      {
        parallel_walk(series, [&](const std::vector<size_t>& index) {
          size_t k = 0;
          for (size_t i = 0; i < index.size(); ++i)
          {
            k = k * series.shape(i) + index[i];
          }
          const auto& one = series(index);
          thread_local std::vector<float64_t> sorted;
          sorted.resize(order.size());
          for (size_t c = 0; c < leaves.size(); ++c)
          {
            for (size_t q = 0; q < order.size(); ++q)
            {
              sorted[q] = time_of(one, c, queries[order[q]]);
            }
            const auto blockSize = leaves[c].size() / series.size();
            one.channel(c).lookup_sorted(sorted, order, leaves[c].data() + static_cast<std::ptrdiff_t>(k * blockSize));
          }
        }, exec);
      }

      Tensor<NestedTensor<T>> out({leaves.size()});
      for (size_t c = 0; c < leaves.size(); ++c)
      {
        out(c) = NestedTensor<T>::from_leaf_view(std::move(leaves[c]));
      }
      return NestedTensor<T>::from_outer_view(std::move(out), 1);
    }
  }

  template <typename T>
  NestedTensor<T> evaluate(const Tensor<TimeSeries<T>>& series, const Tensor<float64_t>& t,
                           std::optional<TimeUnit> unit, Executor& exec)
  {
    const auto queryUnit = unit.value_or(TimeUnit::Unitless);
    return detail::evaluate_each(
        series, t,
        [&](const TimeSeries<T>& one, size_t, float64_t query) { return convert_time(query, queryUnit, one.unit()); },
        [&](const TimeSeries<T>& one) { check_query_unit(queryUnit, one.unit()); }, exec);
  }

  template <typename T>
  float64_t time_of_date(const TimeSeries<T>& series, size_t c, const DateTime& date)
  {
    const auto& channel = series.channel(c);
    const auto offset = time_since(date, *series.epoch(), series.unit());
    const auto origin = channel.origin();
    const auto t = offset - origin;
    if (channel.size() == 0)
    {
      return t;
    }
    const auto reach = channel.snap_tolerance() * std::max(std::abs(offset), std::abs(origin));
    const auto start = channel.time(0);
    if (channel.start_closed() && t < start && start - t <= reach)
    {
      return start;
    }
    const auto end = channel.end().value_or(channel.time(channel.size() - 1));
    if (channel.end_closed() && t > end && t - end <= reach)
    {
      return end;
    }
    return t;
  }

  template <typename T>
  NestedTensor<T> evaluate_dates(const Tensor<TimeSeries<T>>& series, const Tensor<int64_t>& dates,
                                 TimeUnit dateUnit, Executor& exec)
  {
    return detail::evaluate_each(
        series, dates,
        [&](const TimeSeries<T>& one, size_t c, int64_t date) {
          return time_of_date(one, c, DateTime(date, dateUnit));
        },
        [&](const TimeSeries<T>& one) {
          if (!one.epoch())
          {
            throw std::invalid_argument("Every series needs an epoch to place dates and times; set one with with_epoch");
          }
        },
        exec);
  }
}

#endif // STABLEBEAR_TIMESERIES_TPP
