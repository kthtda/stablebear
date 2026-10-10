#ifndef STABLEBEAR_IO_STREAM_H
#define STABLEBEAR_IO_STREAM_H

#include "../config.hpp"

#include "io_stream_base.hpp"
#include "barcode_io.hpp"
#include "pcf_io.hpp"
#include "point_io.hpp"
#include "point_cloud_io.hpp"
#include "compressed_matrix_io.hpp"

#include <cstddef>
#include <vector>
#include <iostream>

namespace sb::io::detail
{
  template <std::forward_iterator FwdIt>
  void write_elements(std::ostream& os, FwdIt begin, FwdIt end)
  {
    write_length(os, begin, end);
    for (auto it = begin; it != end; ++it)
    {
      write_element(os, *it);
    }
  }

  template <typename T, typename AT>
  std::vector<T, AT> read_vector(std::istream& is)
  {
    auto len = read_bytes<uint64_t>(is);
    std::vector<T, AT> ret;
    ret.reserve(len);
    uint64_t nRead = 0;
    for (; nRead < len; ++nRead)
    {
      ret.emplace_back(std::move(read_element<T>(is)));
    }
    return ret;
  }

}

#endif //STABLEBEAR_IO_STREAM_H
