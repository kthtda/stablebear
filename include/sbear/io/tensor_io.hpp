#ifndef STABLEBEAR_TENSOR_IO_H
#define STABLEBEAR_TENSOR_IO_H

#include "io_stream_base.hpp"
#include "barcode_io.hpp"
#include "compressed_matrix_io.hpp"
#include "../tensor.hpp"
#include "../functional/pcf.hpp"
#include "../persistence/barcode.hpp"

namespace sb::io::detail
{
  template <typename T>
  struct is_barcode : std::false_type {};

  template <typename T>
  struct is_barcode<ph::Barcode<T>> : std::true_type { using scalar_type = T; };

  template <typename T>
  inline constexpr bool is_barcode_v = is_barcode<T>::value;

  template <typename T>
  struct is_compressed_matrix : std::false_type {};

  template <typename T>
  struct is_compressed_matrix<SymmetricMatrix<T>> : std::true_type {};

  template <typename T>
  struct is_compressed_matrix<DistanceMatrix<T>> : std::true_type {};

  template <typename T>
  inline constexpr bool is_compressed_matrix_v = is_compressed_matrix<T>::value;

  using StreamableTensor = std::variant<
      Tensor<float32_t>,
      Tensor<float64_t>,

      Tensor<int32_t>,
      Tensor<int64_t>,
      Tensor<uint32_t>,
      Tensor<uint64_t>,
      Tensor<bool>,

      Tensor<Pcf<float32_t, float32_t>>,
      Tensor<Pcf<float64_t, float64_t>>,

      Tensor<Pcf<int32_t, int32_t>>,
      Tensor<Pcf<int64_t, int64_t>>,

      Tensor<PointCloud<float32_t>>,
      Tensor<PointCloud<float64_t>>,

      Tensor<ph::Barcode<float32_t>>,
      Tensor<ph::Barcode<float64_t>>,

      Tensor<SymmetricMatrix<float32_t>>,
      Tensor<SymmetricMatrix<float64_t>>,

      Tensor<DistanceMatrix<float32_t>>,
      Tensor<DistanceMatrix<float64_t>>
      >;

  using StreamableObject = std::variant<
      Pcf<float32_t, float32_t>,
      Pcf<float64_t, float64_t>,

      Pcf<int32_t, int32_t>,
      Pcf<int64_t, int64_t>,

      ph::Barcode<float32_t>,
      ph::Barcode<float64_t>,

      SymmetricMatrix<float32_t>,
      SymmetricMatrix<float64_t>,

      DistanceMatrix<float32_t>,
      DistanceMatrix<float64_t>
      >;

  struct TensorFormat
  {
    std::int32_t baseFormat;
    std::int32_t subFormat;

    std::string toString() const
    {
      return "(" + std::to_string(baseFormat) + ", " + std::to_string(subFormat) + ")";
    }

    bool operator==(const TensorFormat&) const = default;
    bool operator!=(const TensorFormat&) const = default;
  };

  template <typename U>
  TensorFormat tensorFormat()
  {
    using namespace std::string_literals;
    using T = std::decay_t<U>;

    if      constexpr (std::is_same_v<T, float32_t>) { return TensorFormat{ .baseFormat = 1, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, float64_t>) { return TensorFormat{ .baseFormat = 1, .subFormat = 64 }; }

    else if constexpr (std::is_same_v<T, int32_t>)  { return TensorFormat{ .baseFormat = 2, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, int64_t>)  { return TensorFormat{ .baseFormat = 2, .subFormat = 64 }; }

    else if constexpr (std::is_same_v<T, uint32_t>) { return TensorFormat{ .baseFormat = 3, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, uint64_t>) { return TensorFormat{ .baseFormat = 3, .subFormat = 64 }; }

    else if constexpr (std::is_same_v<T, bool>)     { return TensorFormat{ .baseFormat = 4, .subFormat = 8 }; }

    else if constexpr (std::is_same_v<T, Pcf<float32_t, float32_t>>) { return TensorFormat{ .baseFormat = 100, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, Pcf<float64_t, float64_t>>) { return TensorFormat{ .baseFormat = 100, .subFormat = 64 }; }

    else if constexpr (std::is_same_v<T, Pcf<int32_t, int32_t>>) { return TensorFormat{ .baseFormat = 101, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, Pcf<int64_t, int64_t>>) { return TensorFormat{ .baseFormat = 101, .subFormat = 64 }; }

    else if constexpr (std::is_same_v<T, PointCloud<float32_t>>) { return TensorFormat{ .baseFormat = 1000, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, PointCloud<float64_t>>) { return TensorFormat{ .baseFormat = 1000, .subFormat = 64 }; }

    else if constexpr (std::is_same_v<T, SymmetricMatrix<float32_t>>) { return TensorFormat{ .baseFormat = 1100, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, SymmetricMatrix<float64_t>>) { return TensorFormat{ .baseFormat = 1100, .subFormat = 64 }; }

    else if constexpr (std::is_same_v<T, DistanceMatrix<float32_t>>) { return TensorFormat{ .baseFormat = 1120, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, DistanceMatrix<float64_t>>) { return TensorFormat{ .baseFormat = 1120, .subFormat = 64 }; }

    else if constexpr (std::is_same_v<T, ph::Barcode<float32_t>>) { return TensorFormat{ .baseFormat = 10000, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, ph::Barcode<float64_t>>) { return TensorFormat{ .baseFormat = 10000, .subFormat = 64 }; }

    throw std::runtime_error("Tensor type "s + sb::detail::unmangled_typename<T>() +  " not supported.");
  }

  inline TensorFormat getTensorFormat(const StreamableTensor& tensor)
  {
    return std::visit([](auto&& arg) -> TensorFormat {
      using TensorT = std::decay_t<decltype(arg)>;
      using T = typename TensorT::value_type;
      return tensorFormat<T>();
    }, tensor);
  }

  template <IsTensor TensorT>
  void write_tensor(std::ostream& os, const TensorT& tensor);

  template <typename T>
  void write_element(std::ostream& os, const sb::Tensor<T>& t)
  {
    io::detail::write_tensor(os, t);
  }

  template <typename T>
  Tensor<T> read_tensor(std::istream& is);

  inline TensorFormat read_tensor_format(std::istream& is)
  {
    TensorFormat format;
    format.baseFormat = read_bytes<std::int32_t>(is);
    format.subFormat  = read_bytes<std::int32_t>(is);
    return format;
  }

  template <IsTensor TensorT>
  TensorT read_element(std::istream& is)
  {
    auto format = read_tensor_format(is);
    auto expectedFormat = tensorFormat<typename TensorT::value_type>();
    if (format != expectedFormat)
    {
      throw std::runtime_error("Unexpected tensor of type " + format.toString() + " where " + expectedFormat.toString() + " was expected.");
    }
    return io::detail::read_tensor<typename TensorT::value_type>(is);
  }

  template <IsTensor TensorT>
    void write_contiguous_tensor(std::ostream& os, const TensorT& tensor)
  {
    auto format = getTensorFormat(tensor);
    write_bytes<std::int32_t>(os, format.baseFormat);
    write_bytes<std::int32_t>(os, format.subFormat);

    write_bytes<std::uint64_t>(os, tensor.shape().size());
    for (auto i = 0_uz; i < tensor.shape().size(); ++i)
    {
      write_bytes<std::uint64_t>(os, tensor.shape()[i]);
      // Safe: write_tensor() guarantees contiguous input, so strides are always positive
      write_bytes<std::uint64_t>(os, static_cast<uint64_t>(tensor.strides()[i]));
    }

    auto sz = tensor.size();
    for (auto const * elem = tensor.data(); elem != tensor.data() + sz; ++elem)
    {
      //using value_type = typename TensorT::value_type;
      write_element(os, *elem);
    }
  }

  template <IsTensor TensorT>
  void write_tensor(std::ostream& os, const TensorT& tensor)
  {
    if (!tensor.is_contiguous())
    {
      auto copy = tensor.copy();
      if (!copy.is_contiguous())
      {
        // To avoid infinite loop
        throw std::runtime_error("Tensor copy is non-contiguous/non-zero-offset (this is a bug, please report it!).");
      }
      write_tensor(os, copy);
      return;
    }
    write_contiguous_tensor(os, tensor);
  }



  template <typename T>
  Tensor<T> read_tensor(std::istream& is)
  {
    auto shapeSz = read_bytes<std::uint64_t>(is);
    std::vector<size_t> shape(shapeSz);
    std::vector<ptrdiff_t> strides(shapeSz);
    for (auto i = 0_uz; i < shapeSz; ++i)
    {
      shape[i] = read_bytes<std::uint64_t>(is);
      strides[i] = static_cast<ptrdiff_t>(read_bytes<std::uint64_t>(is));
    }

    Tensor<T> ret(shape);
    if (ret.strides() != strides)
    {
      throw std::runtime_error("Incorrect strides in saved data (expected " + index_to_string(ret.strides()) + " but got " + index_to_string(strides) + ")");
    }

    auto sz = ret.size();
    for (auto * elem = ret.data(); elem != ret.data() + sz; ++elem)
    {
      if constexpr (is_barcode_v<T>)
        *elem = read_barcode<typename is_barcode<T>::scalar_type>(is);
      else if constexpr (is_compressed_matrix_v<T>)
        *elem = read_compressed_matrix<T>(is);
      else
        *elem = read_element<T>(is);
    }

    return ret;
  }
}

#endif // STABLEBEAR_TENSOR_IO_H
