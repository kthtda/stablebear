#include "config.hpp"
#include "py_persistence_pair.hpp"
#include <sbear/persistence/persistence_pair.hpp>

namespace py = pybind11;

namespace
{
  template <typename T>
  void register_bindings_persistence_pair(pybind11::module_ &m, const std::string& suffix)
  {
    using PPairT = sb::ph::PersistencePair<T>;

    py::class_<PPairT>(m, ("PersistencePair" + suffix).c_str())

    ;

  }
}

namespace sb_py
{
  void register_persistence_persistence_pair(pybind11::module_ &m)
  {
    register_bindings_persistence_pair<sb::float32_t>(m, "32");
    register_bindings_persistence_pair<sb::float64_t>(m, "64");
  }
}
