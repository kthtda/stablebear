#include "pymodule_sampling.hpp"

#include "py_subsample.hpp"

namespace py = pybind11;

namespace sb_py
{
  void register_module_sampling(py::module_& m)
  {
    auto sm = m.def_submodule("sampling");

    register_sampling_subsample(sm);
  }
}
