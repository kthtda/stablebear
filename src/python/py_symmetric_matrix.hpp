/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#ifndef STABLEBEAR_PY_SYMMETRIC_MATRIX_H
#define STABLEBEAR_PY_SYMMETRIC_MATRIX_H

#include "pybind.hpp"
#include <pybind11/numpy.h>

#include <sbear/symmetric_matrix.hpp>
#include <sbear/tensor.hpp>

#include "py_np_support.hpp"

#include <sstream>

namespace sb_py
{

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
      .def_static("from_dense", [](py::array_t<T> dense) {
        if (dense.ndim() != 2)
          throw std::invalid_argument("Expected a 2-D array");
        auto n = static_cast<size_t>(dense.shape(0));
        if (static_cast<size_t>(dense.shape(1)) != n)
          throw std::invalid_argument("Expected a square array");
        auto r = dense.template unchecked<2>();
        MatT sm(n);
        for (size_t i = 0; i < n; ++i)
        {
          sm(i, i) = r(i, i);
          for (size_t j = i + 1; j < n; ++j)
          {
            if (r(i, j) != r(j, i))
              throw std::invalid_argument("Matrix must be symmetric");
            sm(i, j) = r(i, j);
          }
        }
        return sm;
      })
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
