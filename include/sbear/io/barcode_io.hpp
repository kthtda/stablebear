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
