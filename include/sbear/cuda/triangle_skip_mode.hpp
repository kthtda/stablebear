#ifndef SB_CUDA_TRIANGLE_SKIP_MODE_HPP
#define SB_CUDA_TRIANGLE_SKIP_MODE_HPP

namespace sb
{
  /// Controls which (i,j) pairs are computed or written.
  enum class TriangleSkipMode : int
  {
    None = 0,                  ///< All pairs (cdist / dense)
    LowerTriangleSkipDiag = 1, ///< i > j only (DistanceMatrix)
    LowerTriangle = 2          ///< i >= j (SymmetricMatrix)
  };
}

#endif
