#include "config.hpp"
#include "py_linkage.hpp"
#include "../py_np_support.hpp"

#include <sbear/persistence/linkage.hpp>

#include <pybind11/numpy.h>
#include <utility>

namespace py = pybind11;

namespace
{
  template <typename T>
  class PyLinkageBindings
  {
  public:
    static py::tuple linkage_to_barcode(py::array_t<T, 0> input, bool reduced)
    {
      const NumpyTensor<T> matrix(input);
      bool hasNonMonotoneHeights = false;
      auto barcode = [&] {
        py::gil_scoped_release release;
        return sb::ph::linkage_to_barcode(matrix, reduced, &hasNonMonotoneHeights);
      }();
      return py::make_tuple(std::move(barcode), hasNonMonotoneHeights);
    }

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<PyLinkageBindings>(m, ("Linkage" + suffix).c_str())
        .def_static("linkage_to_barcode", &PyLinkageBindings::linkage_to_barcode,
                    py::arg("Z").noconvert(), py::arg("reduced") = false);
    }
  };
}

namespace sb_py
{
  void register_persistence_linkage(py::module_& m)
  {
    PyLinkageBindings<sb::float32_t>::register_bindings(m, "32");
    PyLinkageBindings<sb::float64_t>::register_bindings(m, "64");
  }
}
