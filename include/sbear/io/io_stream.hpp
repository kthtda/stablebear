// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

  // For new types, make sure to add io::detail::read/write_element in their corresponding headers

  template <ArithmeticType T>
  T read_element(std::istream& is)
  {
    return read_bytes<T>(is);
  }

  template <ArithmeticType T>
    void write_element(std::ostream& os, T elem)
  {
    write_bytes<T>(os, elem);
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
