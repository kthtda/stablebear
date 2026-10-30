#include "py_poisson.hpp"

#include <sbear/point_process/poisson.hpp>

#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace
{

  template <typename T>
  class PyPoissonBindings
  {
  public:
    using TensorT = sb::Tensor<sb::PointCloud<T>>;

    static void poisson_pp(TensorT& out, size_t dim, T rate,
                           std::vector<T> lo, std::vector<T> hi,
                           sb::DefaultRandomGenerator* gen)
    {
      if (gen)
        sb::pp::sample_poisson(out, dim, rate, lo, hi, *gen, sb::default_executor());
      else
        sb::pp::sample_poisson(out, dim, rate, lo, hi, sb::default_generator(), sb::default_executor());
    }

    static void register_bindings(nb::handle m, const std::string& suffix)
    {
      nb::class_<PyPoissonBindings> cls(m, ("Poisson" + suffix).c_str());

      cls
          .def_static("sample_poisson", &PyPoissonBindings::poisson_pp,
                       nb::arg("out"), nb::arg("dim"), nb::arg("rate"),
                       nb::arg("lo"), nb::arg("hi"),
                       nb::arg("generator").none() = nb::none())
          ;
    }
  };

}

void sb_py::register_point_process_poisson(nb::module_& m)
{
  PyPoissonBindings<sb::float32_t>::register_bindings(m, "32");
  PyPoissonBindings<sb::float64_t>::register_bindings(m, "64");
}
