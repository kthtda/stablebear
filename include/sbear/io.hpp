#ifndef STABLEBEAR_IO_H
#define STABLEBEAR_IO_H

#include "io/io_stream.hpp"
#include "tensor.hpp"
#include "version.hpp"

#include "functional/pcf.hpp"
#include "point_cloud.hpp"
#include "symmetric_matrix.hpp"
#include "distance_matrix.hpp"
#include "persistence/barcode.hpp"

#include <variant>
#include <cstdint>
#include <bit>
#include <string_view>

#ifdef __APPLE__
#include "TargetConditionals.h"
#endif

namespace sb
{
  enum class FormatType : std::uint32_t
  {
    SingleTensor = 1,
    SingleObject = 2
  };

  namespace io::detail
  {
    // This should never change! It spells out the legacy project name (masspcf) and is kept as-is so that
    // files written before the rename to stablebear remain readable.
    constexpr const std::string_view HeaderIdBytes = "\1MPCF";

    inline void write_endianness(std::ostream& os)
    {
      if constexpr (std::endian::native == std::endian::little)
      {
        write_binary_string(os, "e");
      }
      else if constexpr (std::endian::native == std::endian::big)
      {
        write_binary_string(os, "E");
      }
      else
      {
        throw std::runtime_error("System not supported (unknown endianness).");
      }
    }

    inline void read_endianness(std::istream& is)
    {
      auto endiannessString = read_binary_string(is, 1);
      if constexpr (std::endian::native == std::endian::little)
      {
        if (endiannessString == "e")
        {
          return;
        }
      }
      else if constexpr (std::endian::native == std::endian::big)
      {
        if (endiannessString == "E")
        {
          return;
        }
      }
      else
      {
        throw std::runtime_error("System not supported (unknown endianness).");
      }
      throw std::runtime_error("Data were saved on a platform with different endianness and cannot be read on this platform. If you need to move data in this way, please file an issue.");
    }

    inline void write_format_version(std::ostream& os)
    {
      write_bytes<int>(os, FormatVersion);
    }

    inline int read_format_version(std::istream& is)
    {
      return read_bytes<int>(is);
    }

    inline void write_platform(std::ostream& os)
    {
      // This is just for record keeping at the moment. We don't really use the information. The main purpose is if we
      // get a bug, having this information could help us track down the problem if it's related to a specific platform.

      std::string system =
#if defined(_WIN32) || defined(_WIN64)
      "windows";
#elif defined(__linux__)
      "linux";
#elif __APPLE__
#if defined(TARGET_OS_MAC)
      "osx";
#elif defined(TARGET_OS_IPHONE) or defined(TARGET_IPHONE_SIMULATOR)
      "ios";
#else
      "apple_other";
#endif
#elif defined(ANDROID)
      "android";
#else
      "other";
#endif

      std::string arch =
#if defined(__x86_64__) || defined(_M_X64)
      "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
      "x86";
#elif defined(__aarch64__) || defined(_M_ARM64)
      "arm64";
#elif defined(__arm__) || defined(_M_ARM)
      "arm";
#elif defined(__riscv) && (__riscv_xlen == 64)
      "riscv64";
#else
      "unknown-arch";
#endif

      write_string(os, system + "-" + arch);
    }



    inline void write_header(std::ostream& os, FormatType formatType)
    {
      write_binary_string(os, HeaderIdBytes);
      write_endianness(os); // Will likely be little-endian until the end of time, but just to be sure!
      write_format_version(os);
      write_string(os, PROJECT_VERSION_FULL);
      write_string(os, PROJECT_BUILD_DATE);
      write_platform(os); // Added in format version 2
      write_bytes<std::uint32_t>(
        os, static_cast<std::uint32_t>(formatType));
    }

    inline BinaryReader read_header(std::istream& is)
    {
      auto idBytes = read_binary_string(is, HeaderIdBytes.length());
      if (idBytes != HeaderIdBytes)
      {
        throw std::runtime_error("Unrecognized file format.");
      }

      read_endianness(is);

      auto formatVersion = read_format_version(is);
      if (formatVersion < 1 || formatVersion > FormatVersion)
      {
        throw std::runtime_error("Input file has format version " + std::to_string(formatVersion) + ". This version of stablebear reads format versions 1 through " + std::to_string(FormatVersion) + ".");
      }

      read_string(is); // PROJECT_VERSION_FULL
      read_string(is); // PROJECT_BUILD_DATE

      if (formatVersion >= 2)
      {
        read_string(is); // platform (added in format version 2)
      }
      return BinaryReader(is, formatVersion);
    }

  }

  inline std::string formatName(uint32_t tp)
  {
    // C++26 reflection, please!
    if (tp == static_cast<uint32_t>(FormatType::SingleTensor))
    {
      return "SingleTensor";
    }
    else if (tp == static_cast<uint32_t>(FormatType::SingleObject))
    {
      return "SingleObject";
    }
    else
    {
      return "Unknown(" + std::to_string(tp) + ")";
    }
  }

  namespace io::detail
  {
    template <typename TensorT>
    void write_single_tensor(std::ostream& os, const TensorT& tensor)
    {
      write_header(os, FormatType::SingleTensor);
      write_tensor(os, tensor);
    }
  }

  template <IsTensor TensorT>
    requires (!is_nested_tensor_v<typename TensorT::value_type>)
  void write(const TensorT& tensor, std::ostream& os)
  {
    io::detail::write_single_tensor(os, tensor);
  }

  template <IsTensor TensorT>
    requires (!is_nested_tensor_v<TensorT>
              && !is_nested_tensor_v<typename TensorT::value_type>)
  TensorT read(std::istream& is)
  {
    auto reader = io::detail::read_header(is);

    auto formatType = io::detail::read_bytes<uint32_t>(is);
    if (formatType != static_cast<uint32_t>(FormatType::SingleTensor))
    {
      throw std::runtime_error("Expected format type " + formatName(static_cast<uint32_t>(FormatType::SingleTensor))
          + " for this operation but got format type " + formatName(formatType));
    }

    const auto format = io::detail::read_type_format(reader);
    return io::detail::read_tensor_for_format<TensorT>(reader, format);
  }

  template <typename NestedT>
    requires is_nested_tensor_v<NestedT>
  NestedT read(std::istream& is)
  {
    auto reader = io::detail::read_header(is);

    auto formatType = io::detail::read_bytes<uint32_t>(is);
    if (formatType != static_cast<uint32_t>(FormatType::SingleTensor))
    {
      throw std::runtime_error("Expected format type " + formatName(static_cast<uint32_t>(FormatType::SingleTensor))
          + " for this operation but got format type " + formatName(formatType));
    }

    const auto format = io::detail::read_type_format(reader);
    const auto expectedFormat = io::detail::tensorFormatV3<NestedT>();
    if (!io::detail::matches_nested_tensor_format<NestedT>(reader, format))
    {
      throw std::runtime_error("Unexpected tensor format " + format.toString() + " where " + expectedFormat.toString() + " was expected.");
    }

    return io::detail::read_nested_tensor_element<typename is_nested_tensor<NestedT>::leaf_type>(reader);
  }

  inline io::detail::StreamableTensor read_any_tensor(std::istream& is)
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

    else if (io::detail::matches_nested_tensor_format<NestedTensor<float32_t>>(reader, format)) { return io::detail::read_nested_tensor_element<float32_t>(reader); }
    else if (io::detail::matches_nested_tensor_format<NestedTensor<float64_t>>(reader, format)) { return io::detail::read_nested_tensor_element<float64_t>(reader); }
    else if (io::detail::matches_nested_tensor_format<NestedTensor<int32_t>>(reader, format)) { return io::detail::read_nested_tensor_element<int32_t>(reader); }
    else if (io::detail::matches_nested_tensor_format<NestedTensor<int64_t>>(reader, format)) { return io::detail::read_nested_tensor_element<int64_t>(reader); }
    else if (io::detail::matches_nested_tensor_format<NestedTensor<uint32_t>>(reader, format)) { return io::detail::read_nested_tensor_element<uint32_t>(reader); }
    else if (io::detail::matches_nested_tensor_format<NestedTensor<uint64_t>>(reader, format)) { return io::detail::read_nested_tensor_element<uint64_t>(reader); }

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

    else
    {
      throw std::runtime_error("Unhandled tensor type (" + std::to_string(format.baseFormat) + ", " + std::to_string(format.subFormat) + ")");
    }

  }

  template <typename T>
  void write_object(const T& obj, std::ostream& os)
  {
    io::detail::write_header(os, FormatType::SingleObject);
    io::detail::write_type_format<T>(os);
    io::detail::write_value(os, obj);
  }

  inline io::detail::StreamableObject read_any_object(std::istream& is)
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

    else
    {
      throw std::runtime_error("Unhandled object type (" + std::to_string(format.baseFormat) + ", " + std::to_string(format.subFormat) + ")");
    }

  }

}

#endif //STABLEBEAR_IO_H
