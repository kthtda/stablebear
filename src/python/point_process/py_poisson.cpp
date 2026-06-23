#include "py_poisson.hpp"

#include <sbear/point_process/poisson.hpp>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyPoissonBindings> cls(m, ("Poisson" + suffix).c_str());

      cls
          .def_static("sample_poisson", &PyPoissonBindings::poisson_pp,
                       py::arg("out"), py::arg("dim"), py::arg("rate"),
                       py::arg("lo"), py::arg("hi"),
                       py::arg("generator").none(true) = py::none())
          ;
    }
  };

}

void sb_py::register_point_process_poisson(py::module_& m)
{
  PyPoissonBindings<sb::float32_t>::register_bindings(m, "32");
  PyPoissonBindings<sb::float64_t>::register_bindings(m, "64");
}
