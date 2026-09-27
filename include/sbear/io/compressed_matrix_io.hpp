#ifndef STABLEBEAR_COMPRESSED_MATRIX_IO_H
#define STABLEBEAR_COMPRESSED_MATRIX_IO_H

#include "io_stream_base.hpp"
#include "../symmetric_matrix.hpp"
#include "../distance_matrix.hpp"

namespace sb::io::detail
{
  // The file format uses row-major lower-triangle order, independently of
  // physical storage. Only symmetric matrices include diagonal entries.
  template <typename MatT>
  struct CompressedMatrixFormat;

  template <typename T>
  struct CompressedMatrixFormat<DistanceMatrix<T>>
  {
    static constexpr bool include_diagonal = false;
  };

  template <typename T>
  struct CompressedMatrixFormat<SymmetricMatrix<T>>
  {
    static constexpr bool include_diagonal = true;
  };

  template <typename MatT>
  void write_element(std::ostream& os, const MatT& mat)
    requires is_compressed_matrix_v<MatT>
  {
    write_bytes<uint64_t>(os, mat.size());
    for (size_t i = 0; i < mat.size(); ++i)
    {
      for (size_t j = 0; j < i + CompressedMatrixFormat<MatT>::include_diagonal; ++j)
        write_bytes<typename MatT::value_type>(os, mat(i, j));
    }
  }

  template <typename MatT>
  MatT read_compressed_matrix(std::istream& is)
  {
    auto n = read_bytes<uint64_t>(is);
    MatT mat(n);
    for (size_t i = 0; i < n; ++i)
    {
      for (size_t j = 0; j < i + CompressedMatrixFormat<MatT>::include_diagonal; ++j)
        mat(i, j) = read_bytes<typename MatT::value_type>(is);
    }
    return mat;
  }
}

#endif // STABLEBEAR_COMPRESSED_MATRIX_IO_H
