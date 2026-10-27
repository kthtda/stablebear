#ifndef STABLEBEAR_TIMESERIES_IO_H
#define STABLEBEAR_TIMESERIES_IO_H

#include "io_stream_base.hpp"
#include "tensor_io.hpp"
#include "../timeseries.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace sb::io::detail
{
  /// Units are stored as their length in seconds, mult * 10^exp, so neither
  /// renaming nor reordering units changes what a saved file means.
  inline void write_time_unit(std::ostream& os, TimeUnit unit)
  {
    const auto& info = sb::detail::time_unit_info(unit);
    write_bytes<int32_t>(os, info.exp);
    write_bytes<int64_t>(os, info.mult);
  }

  inline TimeUnit read_time_unit(std::istream& is)
  {
    const auto exp = read_bytes<int32_t>(is);
    const auto mult = read_bytes<int64_t>(is);
    for (const auto& info : sb::detail::TimeUnitTable)
    {
      if (info.exp == exp && info.mult == mult)
      {
        return info.unit;
      }
    }
    throw std::runtime_error(
        "Unknown time unit of " + std::to_string(mult) + " * 10^" + std::to_string(exp) + " seconds in saved time series");
  }

  /// A series is written as its query unit and one record per channel: the
  /// channel's unit, timestamps, values as a tensor with time along axis 0,
  /// snap tolerance, whether its start and end are closed, its end, if set,
  /// its origin, and the epoch of its timestamps, if set, as a count and its
  /// unit. The end is always closed without one, but its flag is written
  /// anyway so that an open end at the last timestamp needs no new format.
  /// Each channel is written on its own, as the
  /// values it shows, so a file holds no views or shared storage.
  /// read_timeseries reads it back.
  template <typename T>
  void write_value(std::ostream& os, const TimeSeries<T>& series)
  {
    write_time_unit(os, series.unit());
    write_bytes<uint64_t>(os, series.n_channels());
    for (size_t i = 0; i < series.n_channels(); ++i)
    {
      const auto& channel = series.channel(i);
      const auto& times = channel.times();
      write_time_unit(os, times.unit());
      write_elements(os, times.values().begin(), times.values().end());
      write_tensor(os, channel.values());
      write_bytes<float64_t>(os, channel.snap_tolerance());
      write_bytes<bool>(os, channel.start_closed());
      write_bytes<bool>(os, channel.end_closed());
      write_bytes<std::optional<float64_t>>(os, channel.end());
      write_bytes<float64_t>(os, channel.origin());
      const auto& epoch = times.epoch();
      write_bytes<bool>(os, epoch.has_value());
      if (epoch)
      {
        write_bytes<int64_t>(os, epoch->count);
        write_time_unit(os, epoch->unit);
      }
    }
  }

  /// Read a series written by write_value, rebuilding every channel through
  /// the validating constructors.
  template <typename T>
  TimeSeries<T> read_timeseries(BinaryReader& reader)
  {
    auto& is = reader.stream();
    try
    {
      const auto unit = read_time_unit(is);
      const auto nChannels = read_bytes<uint64_t>(is);
      std::vector<Channel<T>> channels;
      for (uint64_t i = 0; i < nChannels; ++i)
      {
        const auto channelUnit = read_time_unit(is);
        auto times = read_vector<float64_t>(is);
        auto values = read_element<Tensor<T>>(reader);
        const auto snapTolerance = read_bytes<float64_t>(is);
        const auto startClosed = read_bytes<bool>(is);
        const auto endClosed = read_bytes<bool>(is);
        const auto end = read_bytes<std::optional<float64_t>>(is);
        const auto origin = read_bytes<float64_t>(is);
        std::optional<Epoch> epoch;
        if (read_bytes<bool>(is))
        {
          const auto count = read_bytes<int64_t>(is);
          epoch = Epoch(count, read_time_unit(is));
        }

        auto channel = Channel<T>(Times(std::move(times), channelUnit, epoch).with_origin(origin), values,
                                  snapTolerance)
                           .with_start_closed(startClosed);
        if (end)
        {
          channel = channel.until(*end, endClosed);
        }
        else if (!endClosed)
        {
          throw std::invalid_argument("An open end at the last timestamp is not supported");
        }
        channels.push_back(std::move(channel));
      }
      return TimeSeries<T>(std::move(channels), unit);
    }
    catch (const std::invalid_argument& error)
    {
      throw std::runtime_error("Invalid time series in saved data: " + std::string(error.what()));
    }
  }

  /// Read a series of whichever element type @p format names: step through
  /// the alternatives of StreamableObject from index I on and read the first
  /// time series whose format matches. Nothing if none does.
  template <size_t I = 0>
  std::optional<StreamableObject> read_any_timeseries(BinaryReader& reader, TensorFormat format)
  {
    if constexpr (I == std::variant_size_v<StreamableObject>)
    {
      return std::nullopt;
    }
    else
    {
      using Alternative = std::variant_alternative_t<I, StreamableObject>;
      if constexpr (is_timeseries_v<Alternative>)
      {
        if (format.matches_type<Alternative>(reader.format_version()))
        {
          return read_timeseries<typename Alternative::value_type>(reader);
        }
      }
      return read_any_timeseries<I + 1>(reader, format);
    }
  }

  /// Read a tensor of series of whichever element type @p format names, as
  /// read_any_timeseries reads one series. Nothing if none matches.
  template <size_t I = 0>
  std::optional<StreamableTensor> read_any_timeseries_tensor(BinaryReader& reader, TensorFormat format)
  {
    if constexpr (I == std::variant_size_v<StreamableTensor>)
    {
      return std::nullopt;
    }
    else
    {
      using Alternative = std::variant_alternative_t<I, StreamableTensor>;
      if constexpr (is_timeseries_v<typename Alternative::value_type>)
      {
        if (reader.format_version() >= 3 && format.matches_type<Alternative>(reader.format_version()))
        {
          return read_tensor<typename Alternative::value_type>(reader);
        }
      }
      return read_any_timeseries_tensor<I + 1>(reader, format);
    }
  }
}

#endif // STABLEBEAR_TIMESERIES_IO_H
