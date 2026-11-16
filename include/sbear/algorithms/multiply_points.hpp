#ifndef STABLEBEAR_MULTIPLY_POINTS_H
#define STABLEBEAR_MULTIPLY_POINTS_H

#include "../point_cloud.hpp"

#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace sb
{
  /// The points of a cloud multiplied by a matrix: point x, as a row,
  /// becomes x M, for the (rows, cols) matrix M given in row-major order.
  /// The cloud must have dimension rows, unless it has no points; the
  /// result has dimension cols.
  template <typename T>
  PointCloud<T> multiply_points(const PointCloud<T>& cloud, std::span<const T> matrix, size_t rows, size_t cols)
  {
    if (matrix.size() != rows * cols)
    {
      throw std::invalid_argument("multiply_points: the matrix does not have rows * cols entries");
    }
    const size_t n = cloud.n_points();
    if (n > 0 && cloud.dim() != rows)
    {
      throw std::invalid_argument(
          "The points have dimension " + std::to_string(cloud.dim()) + ", but the matrix has " +
          std::to_string(rows) + " rows.");
    }
    PointCloud<T> result(std::vector<size_t>{n, cols});
    for (size_t i = 0; i < n; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        T total = 0;
        for (size_t k = 0; k < rows; ++k)
        {
          total += cloud(i, k) * matrix[k * cols + j];
        }
        result(i, j) = total;
      }
    }
    return result;
  }

  /// multiply_points as a ufunc operation: applied to a tensor or a time
  /// series of point clouds, it gives one of the multiplied point clouds.
  template <typename T>
  struct MultiplyPointsOperation
  {
    using input_type = PointCloud<T>;
    using output_type = PointCloud<T>;

    std::vector<T> matrix;
    size_t rows;
    size_t cols;

    MultiplyPointsOperation(std::span<const T> entries, size_t nRows, size_t nCols)
      : matrix(entries.begin(), entries.end()), rows(nRows), cols(nCols)
    {
      if (matrix.size() != rows * cols)
      {
        throw std::invalid_argument("multiply_points: the matrix does not have rows * cols entries");
      }
    }

    [[nodiscard]] std::vector<size_t> extra_dims() const { return {}; }

    void operator()(const PointCloud<T>& cloud, std::span<PointCloud<T>> out) const
    {
      out[0] = multiply_points(cloud, std::span<const T>(matrix), rows, cols);
    }
  };
}

#endif // STABLEBEAR_MULTIPLY_POINTS_H
