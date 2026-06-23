#include "pymodule_point_process.hpp"

#include "py_poisson.hpp"

namespace py = pybind11;

namespace sb_py
{
  void register_module_point_process(py::module_& m)
  {
    auto sm = m.def_submodule("point_process");

    register_point_process_poisson(sm);
  }
}
