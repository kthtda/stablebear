#include "config.hpp"
#include "py_barcode.hpp"
#include "../py_tensor.hpp"
#include "../py_np_support.hpp"

#include <sbear/persistence/barcode.hpp>
#include <sbear/persistence/barcode_tensor.hpp>

#include <nanobind/ndarray.h>

#include <sstream>

namespace nb = nanobind;

namespace
{

  template <typename T>
  void register_barcode_tensor_isomorphism(nb::module_& m)
  {
    using BarcodeTensor = sb::Tensor<sb::ph::Barcode<T>>;
    m.def("barcode_tensors_are_isomorphic",
      [](const BarcodeTensor& lhs, const BarcodeTensor& rhs,
          double atol, double rtol) {
        return sb::ph::are_isomorphic(lhs, rhs, atol, rtol);
      }, nb::arg("lhs"), nb::arg("rhs"),
      nb::arg("atol") = 1e-8, nb::arg("rtol") = 1e-5);
  }

  template <typename T>
  class PyPersistenceBarcodeBindings
  {
  public:
    using BcT = sb::ph::Barcode<T>;

    static BcT construct(sb_py::NpIn<T> in)
    {
      NumpyTensor<const T> inTensor(in);

      if (inTensor.rank() != 2)
      {
        throw std::invalid_argument("Input should have 2 dimensions (supplied input has " + std::to_string(inTensor.rank()) + " dimension(s)).");
      }

      if (inTensor.shape(1) != 2)
      {
        throw std::invalid_argument("Input should have shape (n, 2). Supplied input has shape " + sb::shape_to_string(inTensor.shape()) + ".");
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

    static void register_bindings(nb::module_& m, const std::string& suffix)
    {
      nb::class_<BcT> cls(m, ("Barcode" + suffix).c_str());
      sb_py::bind_array_view(cls, [](const BcT& self, nb::handle owner) {
        auto& bars = self.bars();
        constexpr auto pairStride = static_cast<int64_t>(sizeof(sb::ph::PersistencePair<T>) / sizeof(T));
        return sb_py::array_view<T>(reinterpret_cast<T*>(const_cast<sb::ph::PersistencePair<T>*>(bars.data())),
                                    {bars.size(), 2}, {pairStride, 1}, owner);
      });
      cls
        .def("__init__", [](BcT* self, sb_py::NpIn<T> arr) { new (self) BcT(construct(arr)); })

        .def("__len__", [](const BcT& self) { return self.bars().size(); })

        .def("__str__", [](const BcT& self) -> std::string{
          return "Barcode(" + PyPersistenceBarcodeBindings<T>::dunder_repr(self) + ")";
        })

        .def("__repr__", [](const BcT& self) -> std::string{
          return PyPersistenceBarcodeBindings<T>::dunder_repr(self);
        })

        .def("is_isomorphic_to", &BcT::is_isomorphic_to,
             nb::arg("other"), nb::arg("atol") = 1e-8, nb::arg("rtol") = 1e-5)
      ;
    }
  };
}

namespace sb_py
{

  void register_persistence_barcode_tensor(nb::module_ &m)
  {
    PyPersistenceBarcodeBindings<sb::float32_t>::register_bindings(m, "32");
    PyPersistenceBarcodeBindings<sb::float64_t>::register_bindings(m, "64");

    register_tensor_element<sb::ph::Barcode<sb::float32_t>>(m, "Barcode32");
    register_tensor_element<sb::ph::Barcode<sb::float64_t>>(m, "Barcode64");
    register_barcode_tensor_isomorphism<sb::float32_t>(m);
    register_barcode_tensor_isomorphism<sb::float64_t>(m);
  }
}
