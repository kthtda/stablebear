#include "py_barcode_summary.hpp"

#include "../py_async_support.hpp"

#include <sbear/functional/pcf.hpp>
#include <sbear/tensor.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/persistence/accumulated_persistence.hpp>
#include <sbear/persistence/stable_rank.hpp>
#include <sbear/persistence/betti_curve.hpp>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PyBarcodeSummaryBindings
  {
  public:
    using PcfT = sb::Pcf<T, T>;
    using BarcodeT = sb::ph::Barcode<T>;

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<PyBarcodeSummaryBindings>(m, ("PersistenceBarcodeSummary" + suffix).c_str())
          .def_static("barcode_to_stable_rank", [](const BarcodeT& bc) {
            return sb::ph::barcode_to_stable_rank(bc);
          })
          .def_static("spawn_barcode_to_stable_rank_task", [](const sb::Tensor<BarcodeT>& bcs, sb::Tensor<PcfT>& out)
              -> std::unique_ptr<sb::StoppableTask<void>> {
            auto task = sb::ph::make_stable_rank_task(bcs, out);
            task->start_async(sb::default_executor());
            return task;
          })
          .def_static("barcode_to_betti_curve", [](const BarcodeT& bc) {
            return sb::ph::barcode_to_betti_curve(bc);
          })
          .def_static("spawn_barcode_to_betti_curve_task", [](const sb::Tensor<BarcodeT>& bcs, sb::Tensor<PcfT>& out)
              -> std::unique_ptr<sb::StoppableTask<void>> {
            auto task = sb::ph::make_betti_curve_task(bcs, out);
            task->start_async(sb::default_executor());
            return task;
          })
          .def_static("barcode_to_accumulated_persistence", [](const BarcodeT& bc, T max_death) {
            return sb::ph::barcode_to_accumulated_persistence(bc, max_death);
          }, py::arg("barcode"), py::arg("max_death") = std::numeric_limits<T>::infinity())
          .def_static("spawn_barcode_to_accumulated_persistence_task", [](const sb::Tensor<BarcodeT>& bcs, sb::Tensor<PcfT>& out, T max_death)
              -> std::unique_ptr<sb::StoppableTask<void>> {
            auto task = sb::ph::make_accumulated_persistence_task(bcs, out, max_death);
            task->start_async(sb::default_executor());
            return task;
          }, py::arg("barcodes"), py::arg("out"), py::arg("max_death") = std::numeric_limits<T>::infinity())
          ;
    }
  };

}

namespace sb_py
{
  void register_persistence_barcode_summary(pybind11::module_ &m)
  {
    PyBarcodeSummaryBindings<sb::float32_t>::register_bindings(m, "32");
    PyBarcodeSummaryBindings<sb::float64_t>::register_bindings(m, "64");
  }
}
