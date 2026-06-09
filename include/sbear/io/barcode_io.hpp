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

#ifndef STABLEBEAR_BARCODE_IO_H
#define STABLEBEAR_BARCODE_IO_H

#include "../persistence/barcode.hpp"
#include "io_stream_base.hpp"

namespace sb::io::detail
{
  template <typename T>
  void write_element(std::ostream& os, const sb::ph::Barcode<T>& barcode)
  {
    write_length(os, barcode.bars().begin(), barcode.bars().end());
    for (auto const & bar : barcode.bars())
    {
      write_bytes<T>(os, bar.birth);
      write_bytes<T>(os, bar.death);
    }

  }

  template <typename T>
  sb::ph::Barcode<T> read_barcode(std::istream& is)
  {
    auto len = read_length(is);
    std::vector<sb::ph::PersistencePair<T>> bars;
    bars.reserve(len);
    for (auto i = 0_uz; i < len; ++i)
    {
      auto birth = read_bytes<T>(is);
      auto death = read_bytes<T>(is);

      bars.emplace_back(birth, death);
    }

    return sb::ph::Barcode<T>(std::move(bars));
  }

}

#endif //STABLEBEAR_BARCODE_IO_H
