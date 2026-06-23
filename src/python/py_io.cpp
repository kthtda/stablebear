#include "py_io.hpp"

#include <string_view>

#include <sbear/tensor.hpp>
#include <sbear/io.hpp>

#include <pybind11/stl.h>

namespace py = pybind11;

namespace
{
  class IoOps
  {
  public:
    template <typename T>
    static void save_tensor_to_file(const sb::Tensor<T>& tensor, py::object file)
    {
      sb_py::PythonOStreamBuf buf(file);
      std::ostream os(&buf);
      sb::write(tensor, os);
    }

    static sb::io::detail::StreamableTensor load_tensor_from_file(py::object file)
    {
      sb_py::PythonIStreamBuf buf(file);
      std::istream is(&buf);
      return sb::read_any_tensor(is);
    }

    template <typename T>
    static void save_object_to_file(const T& obj, py::object file)
    {
      sb_py::PythonOStreamBuf buf(file);
      std::ostream os(&buf);
      sb::write_object(obj, os);
    }

    static sb::io::detail::StreamableObject load_object_from_file(py::object file)
    {
      sb_py::PythonIStreamBuf buf(file);
      std::istream is(&buf);
      return sb::read_any_object(is);
    }
  };

}

namespace sb_py
{

  void register_io(py::module_& m)
  {
    py::class_<IoOps>(m, "IoOps")
        .def_static("save_float32_tensor",       &IoOps::save_tensor_to_file<sb::float32_t>)
        .def_static("save_float64_tensor",       &IoOps::save_tensor_to_file<sb::float64_t>)

        .def_static("save_int32_tensor",         &IoOps::save_tensor_to_file<sb::int32_t>)
        .def_static("save_int64_tensor",         &IoOps::save_tensor_to_file<sb::int64_t>)
        .def_static("save_uint32_tensor",        &IoOps::save_tensor_to_file<sb::uint32_t>)
        .def_static("save_uint64_tensor",        &IoOps::save_tensor_to_file<sb::uint64_t>)
        .def_static("save_bool_tensor",          &IoOps::save_tensor_to_file<bool>)

        .def_static("save_pcf32_tensor",         &IoOps::save_tensor_to_file<sb::Pcf<sb::float32_t, sb::float32_t>>)
        .def_static("save_pcf64_tensor",         &IoOps::save_tensor_to_file<sb::Pcf<sb::float64_t, sb::float64_t>>)

        .def_static("save_pcf32i_tensor",        &IoOps::save_tensor_to_file<sb::Pcf<sb::int32_t, sb::int32_t>>)
        .def_static("save_pcf64i_tensor",        &IoOps::save_tensor_to_file<sb::Pcf<sb::int64_t, sb::int64_t>>)

        .def_static("save_point_cloud32_tensor", &IoOps::save_tensor_to_file<sb::PointCloud<sb::float32_t>>)
        .def_static("save_point_cloud64_tensor", &IoOps::save_tensor_to_file<sb::PointCloud<sb::float64_t>>)

        .def_static("save_barcode32_tensor",     &IoOps::save_tensor_to_file<sb::ph::Barcode<sb::float32_t>>)
        .def_static("save_barcode64_tensor",     &IoOps::save_tensor_to_file<sb::ph::Barcode<sb::float64_t>>)

        .def_static("save_symmetric_matrix32_tensor", &IoOps::save_tensor_to_file<sb::SymmetricMatrix<sb::float32_t>>)
        .def_static("save_symmetric_matrix64_tensor", &IoOps::save_tensor_to_file<sb::SymmetricMatrix<sb::float64_t>>)

        .def_static("save_distance_matrix32_tensor", &IoOps::save_tensor_to_file<sb::DistanceMatrix<sb::float32_t>>)
        .def_static("save_distance_matrix64_tensor", &IoOps::save_tensor_to_file<sb::DistanceMatrix<sb::float64_t>>)

        .def_static("load_tensor_from_file", &IoOps::load_tensor_from_file)

        .def_static("save_pcf32_object",         &IoOps::save_object_to_file<sb::Pcf<sb::float32_t, sb::float32_t>>)
        .def_static("save_pcf64_object",         &IoOps::save_object_to_file<sb::Pcf<sb::float64_t, sb::float64_t>>)
        .def_static("save_pcf32i_object",        &IoOps::save_object_to_file<sb::Pcf<sb::int32_t, sb::int32_t>>)
        .def_static("save_pcf64i_object",        &IoOps::save_object_to_file<sb::Pcf<sb::int64_t, sb::int64_t>>)

        .def_static("save_barcode32_object",     &IoOps::save_object_to_file<sb::ph::Barcode<sb::float32_t>>)
        .def_static("save_barcode64_object",     &IoOps::save_object_to_file<sb::ph::Barcode<sb::float64_t>>)

        .def_static("save_symmetric_matrix32_object", &IoOps::save_object_to_file<sb::SymmetricMatrix<sb::float32_t>>)
        .def_static("save_symmetric_matrix64_object", &IoOps::save_object_to_file<sb::SymmetricMatrix<sb::float64_t>>)

        .def_static("save_distance_matrix32_object", &IoOps::save_object_to_file<sb::DistanceMatrix<sb::float32_t>>)
        .def_static("save_distance_matrix64_object", &IoOps::save_object_to_file<sb::DistanceMatrix<sb::float64_t>>)

        .def_static("load_object_from_file", &IoOps::load_object_from_file)

        ;
  }

}