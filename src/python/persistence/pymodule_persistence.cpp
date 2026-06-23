#include "pymodule_persistence.hpp"

#include "py_barcode.hpp"
#include "py_barcode_summary.hpp"
#include "py_persistence_pair.hpp"
#include "py_ripser.hpp"

namespace py = pybind11;

namespace sb_py
{
  void register_module_persistence(py::module_ &m)
  {
    auto sm = m.def_submodule("persistence");

    register_persistence_barcode_tensor(sm);
    register_persistence_barcode_summary(sm);
    register_persistence_persistence_pair(sm);
    register_persistence_ripser(sm);
  }
}
