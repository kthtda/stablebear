#ifndef STABLEBEAR_FIXED_RANK_TENSOR_IO_H
#define STABLEBEAR_FIXED_RANK_TENSOR_IO_H

#include "io_stream_base.hpp"
#include "../fixed_rank_tensor.hpp"

#include <limits>

namespace sb::io::detail
{
  // Store logical extents and row-major values independently of the in-memory
  // layout. The enclosing format supplies the rank and scalar type.
  template <typename T, size_t Rank, typename Layout>
  void write_fixed_rank_tensor(std::ostream& os, const FixedRankTensor<T, Rank, Layout>& tensor)
  {
    write_array<uint64_t>(os, tensor.shape());
    const auto values = tensor.flat_view();
    for (const T& value : values)
      write_bytes<T>(os, value);
  }

  template <typename T, size_t Rank, typename Layout = ContiguousFixedRankLayout<Rank>>
  FixedRankTensor<T, Rank, Layout> read_fixed_rank_tensor(std::istream& is)
  {
    const auto storedShape = read_array<uint64_t, Rank>(is);
    size_t count = 1;
    for (const auto extent : storedShape)
    {
      if (extent > std::numeric_limits<size_t>::max()
          || (extent != 0 && count > std::numeric_limits<size_t>::max() / extent))
        throw std::runtime_error("Fixed-rank tensor shape overflows storage size");
      count *= static_cast<size_t>(extent);
    }
    if (count > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::runtime_error("Fixed-rank tensor allocation size overflows");

    typename FixedRankTensor<T, Rank, Layout>::shape_type shape;
    std::copy(storedShape.begin(), storedShape.end(), shape.begin());
    FixedRankTensor<T, Rank, Layout> tensor(shape);
    // The shape constructor supplies dense row-major storage for this reader.
    auto* values = tensor.storage_data();
    for (size_t i = 0; i < count; ++i)
      values[i] = read_bytes<T>(is);
    return tensor;
  }
}

#endif // STABLEBEAR_FIXED_RANK_TENSOR_IO_H
