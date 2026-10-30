#include "config.hpp"
#include "py_linkage.hpp"
#include "../py_np_support.hpp"

#include <sbear/persistence/linkage.hpp>

#include <nanobind/ndarray.h>
#include <utility>

namespace nb = nanobind;

namespace
{
  template <typename T>
  class PyLinkageBindings
  {
  public:
    static nb::tuple linkage_to_barcode(sb_py::NpIn<T> input, bool reduced)
    {
      const NumpyTensor<const T> matrix(input);
      bool hasNonMonotoneHeights = false;
      auto barcode = [&] {
        nb::gil_scoped_release release;
        return sb::ph::linkage_to_barcode(matrix, reduced, &hasNonMonotoneHeights);
      }();
      return nb::make_tuple(std::move(barcode), hasNonMonotoneHeights);
    }

    static void register_bindings(nb::module_& m, const std::string& suffix)
    {
      nb::class_<PyLinkageBindings>(m, ("Linkage" + suffix).c_str())
        .def_static("linkage_to_barcode", &PyLinkageBindings::linkage_to_barcode,
                    nb::arg("Z").noconvert(), nb::arg("reduced") = false);
    }
  };
}

namespace sb_py
{
  void register_persistence_linkage(nb::module_& m)
  {
    PyLinkageBindings<sb::float32_t>::register_bindings(m, "32");
    PyLinkageBindings<sb::float64_t>::register_bindings(m, "64");
  }
}
