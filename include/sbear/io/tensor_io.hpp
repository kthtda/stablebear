#ifndef STABLEBEAR_TENSOR_IO_H
#define STABLEBEAR_TENSOR_IO_H

#include "io_stream_base.hpp"
#include "barcode_io.hpp"
#include "compressed_matrix_io.hpp"
#include "../tensor.hpp"
#include "../point_cloud.hpp"
#include "../functional/pcf.hpp"
#include "../persistence/barcode.hpp"

#include <unordered_map>
#include <vector>

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

  // Point clouds are identified via sb::is_point_cloud (point_cloud.hpp).

  template <typename T>
  struct is_distance_matrix : std::false_type {};

  template <typename T>
  struct is_distance_matrix<DistanceMatrix<T>> : std::true_type { using scalar_type = T; };

  template <typename T>
  inline constexpr bool is_distance_matrix_v = is_distance_matrix<T>::value;

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

    // baseFormat 1000 is the legacy point cloud format (every element stored as a
    // full nested tensor); 1001 is the current format that stores each distinct
    // source coordinate buffer once plus per-element (source id, indices).
    else if constexpr (std::is_same_v<T, PointCloud<float32_t>>) { return TensorFormat{ .baseFormat = 1001, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, PointCloud<float64_t>>) { return TensorFormat{ .baseFormat = 1001, .subFormat = 64 }; }

    else if constexpr (std::is_same_v<T, SymmetricMatrix<float32_t>>) { return TensorFormat{ .baseFormat = 1100, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, SymmetricMatrix<float64_t>>) { return TensorFormat{ .baseFormat = 1100, .subFormat = 64 }; }

    // baseFormat 1120 is the legacy distance-matrix format (every tensor element
    // a full compressed matrix); 1121 is the current format that stores each
    // distinct source buffer once plus per-element (source id, indices) — the
    // distance-matrix analogue of the 1000 -> 1001 point cloud change above.
    else if constexpr (std::is_same_v<T, DistanceMatrix<float32_t>>) { return TensorFormat{ .baseFormat = 1121, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, DistanceMatrix<float64_t>>) { return TensorFormat{ .baseFormat = 1121, .subFormat = 64 }; }

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

  // Shared writer for tensors whose elements may be indexed views over a
  // source buffer (PointCloud / DistanceMatrix): each distinct source is
  // stored once (deduplicated by buffer address — elements sharing a source,
  // e.g. the indexed subsamples from stablebear.sampling, are written once),
  // then every element as its source id plus, for indexed views, its index
  // array. @p sourceKey maps an element to its source buffer address;
  // @p writeSource writes one element's source.
  template <typename ElemT, typename SourceKeyF, typename WriteSourceF>
  void write_shared_source_elements(
      std::ostream& os, const Tensor<ElemT>& tensor, SourceKeyF sourceKey, WriteSourceF writeSource)
  {
    using KeyT = decltype(sourceKey(std::declval<const ElemT&>()));
    std::unordered_map<KeyT, uint64_t> idOf;
    std::vector<const ElemT*> sources;

    auto sz = tensor.size();
    const auto* data = tensor.data();
    std::vector<uint64_t> sourceId(sz);
    for (auto k = 0_uz; k < sz; ++k)
    {
      auto [it, inserted] = idOf.try_emplace(sourceKey(data[k]), static_cast<uint64_t>(sources.size()));
      if (inserted)
      {
        sources.push_back(&data[k]);
      }
      sourceId[k] = it->second;
    }

    write_bytes<uint64_t>(os, static_cast<uint64_t>(sources.size()));
    for (const ElemT* src : sources)
    {
      writeSource(os, *src);
    }

    for (auto k = 0_uz; k < sz; ++k)
    {
      write_bytes<uint64_t>(os, sourceId[k]);
      write_bytes<bool>(os, data[k].is_indexed());
      if (data[k].is_indexed())
      {
        write_tensor(os, data[k].indices());
      }
    }
  }

  // Point cloud sources are their coordinate tensors.
  template <typename ScalarT>
  void write_point_cloud_elements(std::ostream& os, const Tensor<PointCloud<ScalarT>>& tensor)
  {
    write_shared_source_elements(
        os, tensor,
        [](const PointCloud<ScalarT>& elem) { return elem.coords().data(); },
        [](std::ostream& o, const PointCloud<ScalarT>& src) { write_tensor(o, src.coords()); });
  }

  // Distance matrix sources are full compressed matrices (uint64 size +
  // entries, the read_compressed_matrix layout). This is what lets subsampled
  // sub-matrices be saved without either duplicating the source per element
  // or desynchronizing on size()/storage_count().
  template <typename ScalarT>
  void write_distance_matrix_elements(std::ostream& os, const Tensor<DistanceMatrix<ScalarT>>& tensor)
  {
    write_shared_source_elements(
        os, tensor,
        [](const DistanceMatrix<ScalarT>& elem) { return elem.source_data(); },
        [](std::ostream& o, const DistanceMatrix<ScalarT>& src) {
          // The full shared buffer: source_size(), not size(), which for an
          // indexed view reports the selected submatrix instead.
          const uint64_t n = src.source_size();
          write_bytes<uint64_t>(o, n);
          for (size_t i = 0; i < DistanceMatrix<ScalarT>::storage_size(n); ++i)
          {
            write_bytes<ScalarT>(o, src.source_data()[i]);
          }
        });
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

    using value_type = typename TensorT::value_type;
    if constexpr (is_point_cloud_v<value_type>)
    {
      write_point_cloud_elements<typename is_point_cloud<value_type>::scalar_type>(os, tensor);
    }
    else if constexpr (is_distance_matrix_v<value_type>)
    {
      write_distance_matrix_elements<typename is_distance_matrix<value_type>::scalar_type>(os, tensor);
    }
    else
    {
      auto sz = tensor.size();
      for (auto const * elem = tensor.data(); elem != tensor.data() + sz; ++elem)
      {
        write_element(os, *elem);
      }
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
      else if constexpr (is_point_cloud_v<T>)
        // Legacy (baseFormat 1000) point cloud tensors: every element is a full
        // nested coordinate tensor.
        *elem = T(read_element<Tensor<typename is_point_cloud<T>::scalar_type>>(is));
      else
        *elem = read_element<T>(is);
    }

    return ret;
  }

  // Shared reader for the shared-source tensor formats (see
  // write_shared_source_elements): distinct sources stored once, then
  // per-element (source id, indexed flag, optional indices). Elements that
  // reference the same source share its buffer, as before saving.
  // @p readSource reads one source of type SourceT; elements are built as
  // ElemT(source) or ElemT(source, indices).
  template <typename ElemT, typename SourceT, typename ReadSourceF>
  Tensor<ElemT> read_shared_source_tensor(std::istream& is, ReadSourceF readSource)
  {
    auto shapeSz = read_bytes<std::uint64_t>(is);
    std::vector<size_t> shape(shapeSz);
    std::vector<ptrdiff_t> strides(shapeSz);
    for (auto i = 0_uz; i < shapeSz; ++i)
    {
      shape[i] = read_bytes<std::uint64_t>(is);
      strides[i] = static_cast<ptrdiff_t>(read_bytes<std::uint64_t>(is));
    }

    Tensor<ElemT> ret(shape);
    if (ret.strides() != strides)
    {
      throw std::runtime_error("Incorrect strides in saved data (expected " + index_to_string(ret.strides()) + " but got " + index_to_string(strides) + ")");
    }

    auto numSources = read_bytes<std::uint64_t>(is);
    std::vector<SourceT> sources;
    sources.reserve(numSources);
    for (auto i = 0_uz; i < numSources; ++i)
    {
      sources.push_back(readSource(is));
    }

    auto sz = ret.size();
    for (auto* elem = ret.data(); elem != ret.data() + sz; ++elem)
    {
      auto id = read_bytes<std::uint64_t>(is);
      const bool indexed = read_bytes<bool>(is);
      if (indexed)
      {
        *elem = ElemT(sources[id], read_element<Tensor<uint64_t>>(is));
      }
      else
      {
        // Sharing, not copying: PointCloud wraps the coordinate tensor,
        // DistanceMatrix's copy shares the source buffer (shared_ptr).
        *elem = ElemT(sources[id]);
      }
    }

    return ret;
  }

  // Read the current (baseFormat 1001) point cloud tensor format.
  template <typename ScalarT>
  Tensor<PointCloud<ScalarT>> read_indexed_point_cloud_tensor(std::istream& is)
  {
    return read_shared_source_tensor<PointCloud<ScalarT>, Tensor<ScalarT>>(
        is, [](std::istream& s) { return read_element<Tensor<ScalarT>>(s); });
  }

  // Read the current (baseFormat 1121) distance-matrix tensor format.
  template <typename ScalarT>
  Tensor<DistanceMatrix<ScalarT>> read_indexed_distance_matrix_tensor(std::istream& is)
  {
    return read_shared_source_tensor<DistanceMatrix<ScalarT>, DistanceMatrix<ScalarT>>(
        is, [](std::istream& s) { return read_compressed_matrix<DistanceMatrix<ScalarT>>(s); });
  }
}

#endif // STABLEBEAR_TENSOR_IO_H
