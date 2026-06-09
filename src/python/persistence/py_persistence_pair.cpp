// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
