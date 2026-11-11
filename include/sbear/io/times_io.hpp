#ifndef STABLEBEAR_TIMES_IO_H
#define STABLEBEAR_TIMES_IO_H

#include "io_stream_base.hpp"
#include "../timeseries.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

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

  /// A date and time is written as its count and its unit.
  template <>
  inline void write_bytes<DateTime>(std::ostream& os, const DateTime& v)
  {
    write_bytes<int64_t>(os, v.count);
    write_time_unit(os, v.unit);
  }

  template <>
  inline DateTime read_bytes<DateTime>(std::istream& is)
  {
    const auto count = read_bytes<int64_t>(is);
    return DateTime(count, read_time_unit(is));
  }
}

#endif // STABLEBEAR_TIMES_IO_H
