#include "py_barcode_summary.hpp"

#include "../py_async_support.hpp"
#include "../py_ufunc_containers.hpp"

#include <sbear/functional/pcf.hpp>
#include <sbear/tensor.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/persistence/accumulated_persistence.hpp>
#include <sbear/persistence/stable_rank.hpp>
#include <sbear/persistence/betti_curve.hpp>

#include <limits>
#include <utility>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PyBarcodeSummaryBindings
  {
  public:
    using PcfT = sb::Pcf<T, T>;
    using BarcodeT = sb::ph::Barcode<T>;

    static sb::Tensor<BarcodeT> as_tensor(const BarcodeT& barcode)
    {
      return sb::Tensor<BarcodeT>(std::vector<size_t>{}, barcode);
    }

    template <typename TaskPtr>
    static TaskPtr start(TaskPtr task)
    {
      task->start_async(sb::default_executor());
      return task;
    }

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<PyBarcodeSummaryBindings> cls(m, ("PersistenceBarcodeSummary" + suffix).c_str());

      // Each summary takes any container of barcodes or a single barcode; a
      // single barcode runs as a 0-d tensor.
      sb_py::bind_ufunc_containers<BarcodeT>([&]<typename Container>(const std::string& variant) {
        using StableRankTask = typename decltype(sb::ph::make_stable_rank_task(std::declval<Container>()))::element_type;
        using BettiCurveTask = typename decltype(sb::ph::make_betti_curve_task(std::declval<Container>()))::element_type;
        using AccumulatedPersistenceTask =
            typename decltype(sb::ph::make_accumulated_persistence_task(std::declval<Container>()))::element_type;
        sb_py::register_result_task<StableRankTask>(m, "_StableRankTask" + suffix + variant);
        sb_py::register_result_task<BettiCurveTask>(m, "_BettiCurveTask" + suffix + variant);
        sb_py::register_result_task<AccumulatedPersistenceTask>(
            m, "_AccumulatedPersistenceTask" + suffix + variant);

        cls.def_static("spawn_barcode_to_stable_rank_task", [](const Container& bcs) {
          return start(sb::ph::make_stable_rank_task(bcs));
        });
        cls.def_static("spawn_barcode_to_betti_curve_task", [](const Container& bcs) {
          return start(sb::ph::make_betti_curve_task(bcs));
        });
        cls.def_static("spawn_barcode_to_accumulated_persistence_task", [](const Container& bcs, T max_death) {
          return start(sb::ph::make_accumulated_persistence_task(bcs, max_death));
        }, py::arg("barcodes"), py::arg("max_death") = std::numeric_limits<T>::infinity());
      });

      cls.def_static("spawn_barcode_to_stable_rank_task", [](const BarcodeT& bc) {
        return start(sb::ph::make_stable_rank_task(as_tensor(bc)));
      });
      cls.def_static("spawn_barcode_to_betti_curve_task", [](const BarcodeT& bc) {
        return start(sb::ph::make_betti_curve_task(as_tensor(bc)));
      });
      cls.def_static("spawn_barcode_to_accumulated_persistence_task", [](const BarcodeT& bc, T max_death) {
        return start(sb::ph::make_accumulated_persistence_task(as_tensor(bc), max_death));
      }, py::arg("barcode"), py::arg("max_death") = std::numeric_limits<T>::infinity());
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
