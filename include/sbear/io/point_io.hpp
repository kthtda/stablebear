#ifndef STABLEBEAR_POINT_IO_H
#define STABLEBEAR_POINT_IO_H

#include "io_stream_base.hpp"
#include "../functional/time_point.hpp"

namespace sb::io::detail
{
  template <TimePointLike PointT>
  void write_element(std::ostream& os, const PointT& pt)
  {
    write_bytes<typename PointT::time_type>(os, pt.t);
    write_bytes<typename PointT::value_type>(os, pt.v);
  }

  template <TimePointLike PointT>
  PointT read_element(std::istream& is)
  {
    PointT ret;
    ret.t = read_bytes<typename PointT::time_type>(is);
    ret.v = read_bytes<typename PointT::value_type>(is);
    return ret;
  }
}

#endif // STABLEBEAR_POINT_IO_H
