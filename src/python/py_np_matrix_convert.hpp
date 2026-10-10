#pragma once

#ifndef STABLEBEAR_PY_NP_MATRIX_CONVERT_H
#define STABLEBEAR_PY_NP_MATRIX_CONVERT_H

#include <pybind11/numpy.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace sb_py::detail
{
  template <typename MatT>
  struct NumpyMatrixTraits;

  template <typename MatT>
  MatT matrix_from_numpy_compact(pybind11::array_t<typename MatT::value_type> array)
  {
    using Traits = NumpyMatrixTraits<MatT>;
    using T = MatT::value_type;

    if (array.ndim() != 1)
      throw std::invalid_argument("Expected a 1-D compact array");

    const auto storageCount = static_cast<size_t>(array.shape(0));
    const size_t n = Traits::compact_matrix_size(storageCount);
    auto values = array.template unchecked<1>();

    MatT result(n);
    size_t k = 0;
    Traits::for_each_compact_index(n, [&](size_t i, size_t j) {
      const T value = values(k++);
      Traits::validate_value(value);
      result(i, j) = value;
    });
    return result;
  }

  template <typename MatT>
  MatT matrix_from_numpy_dense(pybind11::array_t<typename MatT::value_type> array)
  {
    using Traits = NumpyMatrixTraits<MatT>;
    using T = MatT::value_type;

    if (array.ndim() != 2)
      throw std::invalid_argument("Expected a 2-D dense array");

    const auto n = static_cast<size_t>(array.shape(0));
    if (static_cast<size_t>(array.shape(1)) != n)
      throw std::invalid_argument("Expected a square array");

    auto values = array.template unchecked<2>();
    MatT result(n);
    for (size_t i = 0; i < n; ++i)
    {
      const T diagonal = values(i, i);
      Traits::validate_diagonal(diagonal);
      result(i, i) = diagonal;

      for (size_t j = i + 1; j < n; ++j)
      {
        const T value = values(i, j);
        const T transposeValue = values(j, i);
        Traits::validate_value(value);
        Traits::validate_value(transposeValue);
        if (value != transposeValue)
          throw std::invalid_argument("Matrix must be symmetric");
        result(i, j) = value;
      }
    }
    return result;
  }

  template <typename MatT>
  MatT matrix_from_numpy(pybind11::array_t<typename MatT::value_type> array)
  {
    if (array.ndim() == 1)
      return matrix_from_numpy_compact<MatT>(array);
    if (array.ndim() == 2)
      return matrix_from_numpy_dense<MatT>(array);
    throw std::invalid_argument(
      "Expected a one-dimensional compact array or a two-dimensional square array; got "
      + std::to_string(array.ndim()) + " dimensions");
  }
}

#endif // STABLEBEAR_PY_NP_MATRIX_CONVERT_H
