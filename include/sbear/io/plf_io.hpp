#ifndef STABLEBEAR_PLF_IO_H
#define STABLEBEAR_PLF_IO_H

#include "io_stream_base.hpp"
#include "../functional/plf.hpp"

#include <type_traits>

namespace sb::io::detail
{
  template <typename T>
  struct is_plf : std::false_type {};

  template <typename Tt, typename Tv>
  struct is_plf<Plf<Tt, Tv>> : std::true_type {};

  template <typename Tt, typename Tv>
  void write_element(std::ostream& os, const Plf<Tt, Tv>& plf)
  {
    write_elements(os, plf.points().begin(), plf.points().end());
  }

  /// Read a PLF through its validating constructor, so that invalid
  /// breakpoints throw std::invalid_argument.
  template <typename PlfT>
    requires is_plf<PlfT>::value
  PlfT read_element(std::istream& is)
  {
    using point_type = typename PlfT::point_type;
    auto pts = read_vector<point_type>(is);
    return PlfT(std::move(pts));
  }
}

#endif // STABLEBEAR_PLF_IO_H
