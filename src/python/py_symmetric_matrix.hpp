#pragma once

#ifndef STABLEBEAR_PY_SYMMETRIC_MATRIX_H
#define STABLEBEAR_PY_SYMMETRIC_MATRIX_H

#include "pybind.hpp"
#include <pybind11/numpy.h>

#include <sbear/symmetric_matrix.hpp>
#include <sbear/tensor.hpp>

#include "py_np_matrix_convert.hpp"
#include "py_np_support.hpp"

#include <cmath>
#include <sstream>

namespace sb_py
{

  namespace detail
  {
    template <typename T>
    struct NumpyMatrixTraits<sb::SymmetricMatrix<T>>
    {
      static size_t compact_matrix_size(size_t storageCount)
      {
        size_t n = 0;
        size_t nextRowSize = 1;
        size_t remaining = storageCount;
        while (remaining >= nextRowSize)
        {
          remaining -= nextRowSize;
          ++n;
          ++nextRowSize;
        }
        if (remaining != 0)
        {
          throw std::invalid_argument(
            "Compact symmetric matrix length must equal n*(n+1)/2 for some n; got "
            + std::to_string(storageCount));
        }
        return n;
      }

      static void validate_value(T value)
      {
        if (std::isnan(value))
          throw std::invalid_argument("Symmetric matrix entries must not be NaN");
      }

      static void validate_diagonal(T value)
      {
        validate_value(value);
      }

      template <typename Fn>
      static void for_each_compact_index(size_t n, Fn&& fn)
      {
        for (size_t i = 0; i < n; ++i)
        {
          for (size_t j = i; j < n; ++j)
            fn(i, j);
        }
      }
    };
  }

  void register_symmetric_matrix(pybind11::module_& m);

  template <typename T>
  void register_symmetric_matrix_bindings(pybind11::module_& m, const std::string& suffix)
  {
    namespace py = pybind11;
    using MatT = sb::SymmetricMatrix<T>;

    py::class_<MatT>(m, ("SymmetricMatrix" + suffix).c_str(), py::buffer_protocol())
      .def(py::init<size_t>(), py::arg("n"))
      .def(py::init<size_t, T>(), py::arg("n"), py::arg("init"))
      .def_property_readonly("size", &MatT::size)
      .def_property_readonly("storage_count", &MatT::storage_count)
      .def("__getitem__", [](const MatT& self, std::pair<size_t, size_t> ij) {
        return self(ij.first, ij.second);
      })
      .def("__setitem__", [](MatT& self, std::pair<size_t, size_t> ij, T val) {
        self(ij.first, ij.second) = val;
      })
      .def_buffer([](const MatT& self) -> py::buffer_info {
        return py::buffer_info(
          const_cast<T*>(self.data()),
          sizeof(T),
          py::format_descriptor<T>::format(),
          1,
          { self.storage_count() },
          { sizeof(T) },
          true  // readonly
        );
      })
      .def("to_dense", [](const MatT& self) {
        auto n = self.size();
        py::array_t<T> out({n, n});
        NumpyTensor<T> buf(out);
        for (size_t i = 0; i < n; ++i)
        {
          for (size_t j = 0; j < n; ++j)
          {
            buf(i, j) = self(i, j);
          }
        }
        return out;
      })
      .def_static("from_numpy", &detail::matrix_from_numpy<MatT>)
      .def_static("from_dense", &detail::matrix_from_numpy_dense<MatT>)
      .def("allclose", [](const MatT& self, const MatT& rhs, double atol, double rtol){
        return sb::allclose(self, rhs, T(atol), T(rtol));
      }, py::arg("other"), py::arg("atol") = 1e-8, py::arg("rtol") = 1e-5)
      .def("__repr__", [suffix](const MatT& self) {
        std::ostringstream oss;
        oss << "SymmetricMatrix" << suffix << "(size=" << self.size() << ")";
        return oss.str();
      })
    ;
  }

}

#endif // STABLEBEAR_PY_SYMMETRIC_MATRIX_H
