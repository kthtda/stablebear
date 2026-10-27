#include "py_timeseries_bind.hpp"

#include <sbear/point_cloud.hpp>

namespace sb_py
{
  void register_timeseries_point_cloud(pybind11::module_& m)
  {
    register_typed_timeseries<sb::PointCloud<sb::float32_t>>(m, "PointCloud32");
    register_typed_timeseries<sb::PointCloud<sb::float64_t>>(m, "PointCloud64");
  }
}
