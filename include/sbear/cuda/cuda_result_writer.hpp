#ifndef SB_CUDA_RESULT_WRITER_HPP
#define SB_CUDA_RESULT_WRITER_HPP

#include "../distance_matrix.hpp"
#include "../symmetric_matrix.hpp"
#include "../tensor.hpp"
#include "cuda_block_scheduler.hpp"
#include "triangle_skip_mode.hpp"

#include <cstddef>
#include <stdexcept>

namespace sb
{
  /// Row-major 2D view over a flat Tensor, providing operator()(i,j).
  /// Holds a copy of the Tensor (which shares data via shared_ptr).
  template <typename Tv>
  class DenseMatrixView
  {
  public:
    DenseMatrixView(Tensor<Tv> tensor, size_t nCols)
      : m_tensor(std::move(tensor))
      , m_nCols(nCols)
    { }

    Tv& operator()(size_t i, size_t j)
    {
      return m_tensor.data()[i * m_nCols + j];
    }

  private:
    Tensor<Tv> m_tensor;
    size_t m_nCols;
  };

  /// Generic block result writer that scatters CUDA block output into any
  /// matrix-like type supporting operator()(i,j).
  ///
  /// The TriangleSkipMode controls which elements are written:
  ///   None                 — write all elements (dense / cdist)
  ///   LowerTriangleSkipDiag — write only i > j  (DistanceMatrix)
  ///   LowerTriangle         — write only i >= j (SymmetricMatrix)
  ///
  /// Holds a copy of the matrix (which typically shares data via shared_ptr),
  /// so it is safe to outlive the original.
  template <typename MatrixT, TriangleSkipMode Mode>
  class BlockResultWriter
  {
  public:
    explicit BlockResultWriter(MatrixT mat)
      : m_mat(std::move(mat))
    { }

    template <typename Tv>
    void scatter(const Tv* hostBlock, const BlockInfo& block)
    {
      for (size_t iLocal = 0; iLocal < block.rowHeight; ++iLocal)
      {
        size_t iGlobal = block.rowStart + iLocal;
        for (size_t jLocal = 0; jLocal < block.colWidth; ++jLocal)
        {
          size_t jGlobal = block.colStart + jLocal;
          if constexpr (Mode == TriangleSkipMode::LowerTriangleSkipDiag)
          {
            if (iGlobal <= jGlobal)
            {
              Tv val = hostBlock[iLocal * block.colWidth + jLocal];
              if (val != Tv(0))
              {
                throw std::logic_error(
                  "Non-zero value at skipped position (" + std::to_string(iGlobal)
                  + ", " + std::to_string(jGlobal) + ") in distance matrix scatter");
              }
              continue;
            }
          }
          else if constexpr (Mode == TriangleSkipMode::LowerTriangle)
          {
            if (iGlobal < jGlobal)
            {
              Tv val = hostBlock[iLocal * block.colWidth + jLocal];
              if (val != Tv(0))
              {
                throw std::logic_error(
                  "Non-zero value at skipped position (" + std::to_string(iGlobal)
                  + ", " + std::to_string(jGlobal) + ") in symmetric matrix scatter");
              }
              continue;
            }
          }
          m_mat(iGlobal, jGlobal) = hostBlock[iLocal * block.colWidth + jLocal];
        }
      }
    }

  private:
    MatrixT m_mat;
  };

  // Convenience aliases preserving the original names.
  template <typename Tv>
  using DistanceMatrixResultWriter = BlockResultWriter<DistanceMatrix<Tv>, TriangleSkipMode::LowerTriangleSkipDiag>;

  template <typename Tv>
  using SymmetricMatrixResultWriter = BlockResultWriter<SymmetricMatrix<Tv>, TriangleSkipMode::LowerTriangle>;

  template <typename Tv>
  using DenseResultWriter = BlockResultWriter<DenseMatrixView<Tv>, TriangleSkipMode::None>;
}

#endif
