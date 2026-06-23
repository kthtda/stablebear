#pragma once

#ifndef STABLEBEAR_PY_DISTANCE_MATRIX_H
#define STABLEBEAR_PY_DISTANCE_MATRIX_H

#include "pybind.hpp"
#include <pybind11/numpy.h>

#include <sbear/distance_matrix.hpp>
#include <sbear/tensor.hpp>

#include "py_np_support.hpp"

#include <sstream>

namespace sb_py
{

  void register_distance_matrix(pybind11::module_& m);

  template <typename T>
  void register_distance_matrix_bindings(pybind11::module_& m, const std::string& suffix)
  {
    namespace py = pybind11;
    using MatT = sb::DistanceMatrix<T>;

    py::class_<MatT>(m, ("DistanceMatrix" + suffix).c_str())
      .def(py::init<size_t>(), py::arg("n"))
      .def_property_readonly("size", &MatT::size)
      .def_property_readonly("storage_count", &MatT::storage_count)
      .def("__getitem__", [](const MatT& self, std::pair<size_t, size_t> ij) {
        return self(ij.first, ij.second);
      })
      .def("__setitem__", [](MatT& self, std::pair<size_t, size_t> ij, T val) {
        self(ij.first, ij.second) = val;
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
      .def_static("from_dense", [](py::array_t<T> dense) {
        if (dense.ndim() != 2)
          throw std::invalid_argument("Expected a 2-D array");
        auto n = static_cast<size_t>(dense.shape(0));
        if (static_cast<size_t>(dense.shape(1)) != n)
          throw std::invalid_argument("Expected a square array");
        auto r = dense.template unchecked<2>();
        for (size_t i = 0; i < n; ++i)
        {
          if (r(i, i) != T{})
            throw std::invalid_argument("Diagonal entries must be zero");
        }
        MatT dm(n);
        for (size_t i = 0; i < n; ++i)
        {
          for (size_t j = i + 1; j < n; ++j)
          {
            if (r(i, j) < T{})
              throw std::invalid_argument("Entries must be nonnegative");
            if (r(i, j) != r(j, i))
              throw std::invalid_argument("Matrix must be symmetric");
            dm(i, j) = r(i, j);
          }
        }
        return dm;
      })
      .def("allclose", [](const MatT& self, const MatT& rhs, double atol, double rtol){
        return sb::allclose(self, rhs, T(atol), T(rtol));
      }, py::arg("other"), py::arg("atol") = 1e-8, py::arg("rtol") = 1e-5)
      .def("__repr__", [suffix](const MatT& self) {
        std::ostringstream oss;
        oss << "DistanceMatrix" << suffix << "(size=" << self.size() << ")";
        return oss.str();
      })
    ;
  }

}

#endif // STABLEBEAR_PY_DISTANCE_MATRIX_H
