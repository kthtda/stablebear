#ifndef STABLEBEAR_PCF_IO_H
#define STABLEBEAR_PCF_IO_H

#include "io_stream_base.hpp"
#include "../functional/pcf.hpp"

namespace sb::io::detail
{
  template <PcfLike PcfT>
  void write_element(std::ostream& os, const PcfT& pcf)
  {
    write_elements(os, pcf.points().begin(), pcf.points().end());
  }

  template <PcfLike PcfT>
  PcfT read_element(std::istream& is)
  {
    using point_type = PcfT::point_type;
    auto pts = read_vector<point_type>(is);
    return PcfT(std::move(pts));
  }
}

#endif // STABLEBEAR_PCF_IO_H
