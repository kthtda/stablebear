#ifndef STABLEBEAR_COMPRESSED_MATRIX_IO_H
#define STABLEBEAR_COMPRESSED_MATRIX_IO_H

#include "io_stream_base.hpp"
#include "../symmetric_matrix.hpp"
#include "../distance_matrix.hpp"

#include <stdexcept>

namespace sb::io::detail
{
  template <typename MatT>
  void write_element(std::ostream& os, const MatT& mat)
    requires requires { mat.size(); mat.storage_count(); mat.data(); }
  {
    // An indexed view's size() (selected points) disagrees with its
    // storage_count()/data() (the full shared source buffer): writing it raw
    // would desynchronize the stream. Views inside tensors are handled by the
    // shared-source format in tensor_io.hpp; a standalone view is written as
    // its materialization (indistinguishable on read).
    if constexpr (requires { mat.is_indexed(); mat.materialize(); })
    {
      if (mat.is_indexed())
      {
        write_element(os, mat.materialize());
        return;
      }
    }
    write_bytes<uint64_t>(os, mat.size());
    for (size_t i = 0; i < mat.storage_count(); ++i)
    {
      write_bytes<typename MatT::value_type>(os, mat.data()[i]);
    }
  }

  template <typename MatT>
  MatT read_compressed_matrix(std::istream& is)
  {
    auto n = read_bytes<uint64_t>(is);
    MatT mat(n);
    auto* ptr = mat.mutable_data();
    for (size_t i = 0; i < mat.storage_count(); ++i)
    {
      ptr[i] = read_bytes<typename MatT::value_type>(is);
    }
    return mat;
  }
}

#endif // STABLEBEAR_COMPRESSED_MATRIX_IO_H
