#include "py_distributions.hpp"

#include <pybind11/stl.h>
#include <sbear/sampling/weighting.hpp>

namespace py = pybind11;

void sb_py::register_distributions(py::module_& m)
{
  using Distribution = sb::sampling::Distribution;
  py::class_<Distribution>(m, "_Distribution")
    .def_static("gaussian", &Distribution::gaussian)
    .def_static("uniform", &Distribution::uniform)
    .def_static("mixture", &Distribution::mixture);
}
