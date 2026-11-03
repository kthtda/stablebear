#include "../include/sbear/io.hpp"

// The readers of any saved type live here, not in io.hpp, so that only
// this file instantiates the readers for every element type.

namespace sb
{
  io::detail::StreamableTensor read_any_tensor(std::istream& is)
  {
    auto reader = io::detail::read_header(is);

    auto formatType = io::detail::read_bytes<uint32_t>(is);
    if (formatType != static_cast<uint32_t>(FormatType::SingleTensor))
    {
      throw std::runtime_error("Expected format type " + formatName(static_cast<uint32_t>(FormatType::SingleTensor))
                               + " for this operation but got format type " + formatName(formatType));
    }

    auto format = io::detail::read_type_format(reader);

    if      (format.matches_type<float32_t>(reader.format_version())) { return io::detail::read_tensor<float32_t>(reader); }
    else if (format.matches_type<float64_t>(reader.format_version())) { return io::detail::read_tensor<float64_t>(reader); }

    else if (format.matches_type<int32_t>(reader.format_version()))  { return io::detail::read_tensor<int32_t>(reader); }
    else if (format.matches_type<int64_t>(reader.format_version()))  { return io::detail::read_tensor<int64_t>(reader); }

    else if (format.matches_type<uint32_t>(reader.format_version())) { return io::detail::read_tensor<uint32_t>(reader); }
    else if (format.matches_type<uint64_t>(reader.format_version())) { return io::detail::read_tensor<uint64_t>(reader); }

    else if (format.matches_type<bool>(reader.format_version()))     { return io::detail::read_tensor<bool>(reader); }

    else if (auto nested = io::detail::read_any_nested_tensor(reader, format)) { return std::move(*nested); }

    else if (format.matches_type<Pcf<float32_t, float32_t>>(reader.format_version())) { return io::detail::read_tensor<Pcf<float32_t, float32_t>>(reader); }
    else if (format.matches_type<Pcf<float64_t, float64_t>>(reader.format_version())) { return io::detail::read_tensor<Pcf<float64_t, float64_t>>(reader); }

    else if (format.matches_type<Pcf<int32_t, int32_t>>(reader.format_version())) { return io::detail::read_tensor<Pcf<int32_t, int32_t>>(reader); }
    else if (format.matches_type<Pcf<int64_t, int64_t>>(reader.format_version())) { return io::detail::read_tensor<Pcf<int64_t, int64_t>>(reader); }

    else if (io::detail::is_tensor_format<Tensor<PointCloud<float32_t>>>(reader, format)) { return io::detail::read_tensor_for_format<Tensor<PointCloud<float32_t>>>(reader, format); }
    else if (io::detail::is_tensor_format<Tensor<PointCloud<float64_t>>>(reader, format)) { return io::detail::read_tensor_for_format<Tensor<PointCloud<float64_t>>>(reader, format); }
    else if (io::detail::is_tensor_format<Tensor<PointCloud<float32_t>, TensorProperty::Indexed>>(reader, format)) { return io::detail::read_tensor_for_format<Tensor<PointCloud<float32_t>, TensorProperty::Indexed>>(reader, format); }
    else if (io::detail::is_tensor_format<Tensor<PointCloud<float64_t>, TensorProperty::Indexed>>(reader, format)) { return io::detail::read_tensor_for_format<Tensor<PointCloud<float64_t>, TensorProperty::Indexed>>(reader, format); }

    else if (format.matches_type<SymmetricMatrix<float32_t>>(reader.format_version())) { return io::detail::read_tensor<SymmetricMatrix<float32_t>>(reader); }
    else if (format.matches_type<SymmetricMatrix<float64_t>>(reader.format_version())) { return io::detail::read_tensor<SymmetricMatrix<float64_t>>(reader); }

    else if (io::detail::is_tensor_format<Tensor<DistanceMatrix<float32_t>>>(reader, format)) { return io::detail::read_tensor_for_format<Tensor<DistanceMatrix<float32_t>>>(reader, format); }
    else if (io::detail::is_tensor_format<Tensor<DistanceMatrix<float64_t>>>(reader, format)) { return io::detail::read_tensor_for_format<Tensor<DistanceMatrix<float64_t>>>(reader, format); }
    else if (io::detail::is_tensor_format<Tensor<DistanceMatrix<float32_t>, TensorProperty::Indexed>>(reader, format)) { return io::detail::read_tensor_for_format<Tensor<DistanceMatrix<float32_t>, TensorProperty::Indexed>>(reader, format); }
    else if (io::detail::is_tensor_format<Tensor<DistanceMatrix<float64_t>, TensorProperty::Indexed>>(reader, format)) { return io::detail::read_tensor_for_format<Tensor<DistanceMatrix<float64_t>, TensorProperty::Indexed>>(reader, format); }

    else if (format.matches_type<ph::Barcode<float32_t>>(reader.format_version())) { return io::detail::read_tensor<ph::Barcode<float32_t>>(reader); }
    else if (format.matches_type<ph::Barcode<float64_t>>(reader.format_version())) { return io::detail::read_tensor<ph::Barcode<float64_t>>(reader); }

    else if (auto series = io::detail::read_any_timeseries_tensor(reader, format)) { return std::move(*series); }

    else
    {
      throw std::runtime_error("Unhandled tensor type (" + std::to_string(format.baseFormat) + ", " + std::to_string(format.subFormat) + ")");
    }

  }

  io::detail::StreamableObject read_any_object(std::istream& is)
  {
    auto reader = io::detail::read_header(is);

    auto formatType = io::detail::read_bytes<uint32_t>(is);
    if (formatType != static_cast<uint32_t>(FormatType::SingleObject))
    {
      throw std::runtime_error("Expected format type " + formatName(static_cast<uint32_t>(FormatType::SingleObject))
                               + " for this operation but got format type " + formatName(formatType));
    }

    auto format = io::detail::read_type_format(reader);

    if      (format.matches_type<Pcf<float32_t, float32_t>>(reader.format_version())) { return io::detail::read_element<Pcf<float32_t, float32_t>>(is); }
    else if (format.matches_type<Pcf<float64_t, float64_t>>(reader.format_version())) { return io::detail::read_element<Pcf<float64_t, float64_t>>(is); }

    else if (format.matches_type<Pcf<int32_t, int32_t>>(reader.format_version())) { return io::detail::read_element<Pcf<int32_t, int32_t>>(is); }
    else if (format.matches_type<Pcf<int64_t, int64_t>>(reader.format_version())) { return io::detail::read_element<Pcf<int64_t, int64_t>>(is); }

    else if (format.matches_type<PointCloud<float32_t>>(reader.format_version())) { return io::detail::read_point_cloud<float32_t>(reader); }
    else if (format.matches_type<PointCloud<float64_t>>(reader.format_version())) { return io::detail::read_point_cloud<float64_t>(reader); }

    else if (format.matches_type<ph::Barcode<float32_t>>(reader.format_version())) { return io::detail::read_barcode<float32_t>(is); }
    else if (format.matches_type<ph::Barcode<float64_t>>(reader.format_version())) { return io::detail::read_barcode<float64_t>(is); }

    else if (format.matches_type<SymmetricMatrix<float32_t>>(reader.format_version())) { return io::detail::read_compressed_matrix<SymmetricMatrix<float32_t>>(is); }
    else if (format.matches_type<SymmetricMatrix<float64_t>>(reader.format_version())) { return io::detail::read_compressed_matrix<SymmetricMatrix<float64_t>>(is); }

    else if (format.matches_type<DistanceMatrix<float32_t>>(reader.format_version())) { return io::detail::read_compressed_matrix<DistanceMatrix<float32_t>>(is); }
    else if (format.matches_type<DistanceMatrix<float64_t>>(reader.format_version())) { return io::detail::read_compressed_matrix<DistanceMatrix<float64_t>>(is); }

    else if (auto series = io::detail::read_any_timeseries(reader, format)) { return std::move(*series); }

    else
    {
      throw std::runtime_error("Unhandled object type (" + std::to_string(format.baseFormat) + ", " + std::to_string(format.subFormat) + ")");
    }

  }
}
