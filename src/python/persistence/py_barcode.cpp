#include "config.hpp"
#include "py_barcode.hpp"
#include "../py_tensor.hpp"
#include "../py_np_support.hpp"

#include <sbear/persistence/barcode.hpp>

#include <pybind11/numpy.h>

#include <sstream>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PyPersistenceBarcodeBindings
  {
  public:
    using BcT = sb::ph::Barcode<T>;

    static BcT construct(py::array_t<T> in)
    {
      NumpyTensor<T> inTensor(in);

      if (inTensor.rank() != 2)
      {
        throw py::value_error("Input should have 2 dimensions (supplied input has " + std::to_string(inTensor.rank()) + " dimension(s)).");
      }

      if (inTensor.shape(1) != 2)
      {
        throw py::value_error("Input should have shape (n, 2). Supplied input has shape " + sb::shape_to_string(inTensor.shape()) + ".");
      }

      std::vector<sb::ph::PersistencePair<T>> bars;
      bars.reserve(inTensor.shape(0));

      for (auto i = 0_uz; i < inTensor.shape(0); ++i)
      {
        bars.emplace_back(inTensor(i, 0), inTensor(i, 1));
      }

      return BcT(std::move(bars));
    }

    [[nodiscard]] static std::string dunder_repr(const BcT& self)
    {
      std::stringstream ss;
      ss << self;
      return ss.str();
    }

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<sb::ph::Barcode<T>>(m, ("Barcode" + suffix).c_str(), py::buffer_protocol())
        .def(py::init([](py::array_t<T> arr) { return construct(arr); }))

        .def_buffer([](const BcT& self) -> py::buffer_info {
          auto& bars = self.bars();
          return py::buffer_info(
              const_cast<sb::ph::PersistencePair<T>*>(bars.data()),
              sizeof(T),
              py::format_descriptor<T>::format(),
              2,
              { static_cast<py::ssize_t>(bars.size()), py::ssize_t{2} },
              { static_cast<py::ssize_t>(sizeof(sb::ph::PersistencePair<T>)),
                static_cast<py::ssize_t>(sizeof(T)) }
          );
        })

        .def("__len__", [](const BcT& self) { return self.bars().size(); })

        .def("__str__", [](const BcT& self) -> std::string{
          return "Barcode(" + PyPersistenceBarcodeBindings<T>::dunder_repr(self) + ")";
        })

        .def("__repr__", [](const BcT& self) -> std::string{
          return PyPersistenceBarcodeBindings<T>::dunder_repr(self);
        })

        .def("is_isomorphic_to", &BcT::is_isomorphic_to,
             py::arg("other"), py::arg("atol") = 1e-8, py::arg("rtol") = 1e-5)
      ;
    }
  };
}

namespace sb_py
{

  void register_persistence_barcode_tensor(pybind11::module_ &m)
  {
    PyPersistenceBarcodeBindings<sb::float32_t>::register_bindings(m, "32");
    PyPersistenceBarcodeBindings<sb::float64_t>::register_bindings(m, "64");

    register_typed_tensor_bindings<sb::ph::Barcode<sb::float32_t>>(m, "Barcode32", "");
    register_typed_tensor_bindings<sb::ph::Barcode<sb::float64_t>>(m, "Barcode64", "");
  }
}
