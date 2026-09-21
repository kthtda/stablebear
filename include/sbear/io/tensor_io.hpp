#ifndef STABLEBEAR_TENSOR_IO_H
#define STABLEBEAR_TENSOR_IO_H

#include "io_stream_base.hpp"
#include "barcode_io.hpp"
#include "compressed_matrix_io.hpp"
#include "../tensor.hpp"
#include "../nested_tensor.hpp"
#include "../point_cloud.hpp"
#include "../functional/pcf.hpp"
#include "../persistence/barcode.hpp"

#include <map>
#include <vector>

namespace sb::io::detail
{
  constexpr std::int32_t NestedTensorSubFormatBase = 100'000'000;

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

  using StreamableTensor = std::variant<
      Tensor<float32_t>,
      Tensor<float64_t>,

      Tensor<int32_t>,
      Tensor<int64_t>,
      Tensor<uint32_t>,
      Tensor<uint64_t>,
      Tensor<bool>,

      NestedTensor<float32_t>,
      NestedTensor<float64_t>,
      NestedTensor<int32_t>,
      NestedTensor<int64_t>,
      NestedTensor<uint32_t>,
      NestedTensor<uint64_t>,

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

      PointCloud<float32_t>,
      PointCloud<float64_t>,

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

    // Nested tensors retain the leaf tensor's base format and occupy their own
    // subformat series: nested float32 is (1, 100000032), signed int64 is
    // (2, 100000064), and unsigned int32 is (3, 100000032).
    else if constexpr (is_nested_tensor_v<T>)
    {
      const auto leafFormat = tensorFormat<typename is_nested_tensor<T>::leaf_type>();
      return TensorFormat{
        .baseFormat = leafFormat.baseFormat,
        .subFormat = NestedTensorSubFormatBase + leafFormat.subFormat
      };
    }

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

    else if constexpr (std::is_same_v<T, DistanceMatrix<float32_t>>) { return TensorFormat{ .baseFormat = 1120, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, DistanceMatrix<float64_t>>) { return TensorFormat{ .baseFormat = 1120, .subFormat = 64 }; }

    else if constexpr (std::is_same_v<T, ph::Barcode<float32_t>>) { return TensorFormat{ .baseFormat = 10000, .subFormat = 32 }; }
    else if constexpr (std::is_same_v<T, ph::Barcode<float64_t>>) { return TensorFormat{ .baseFormat = 10000, .subFormat = 64 }; }

    throw std::runtime_error("Tensor type "s + sb::detail::unmangled_typename<T>() +  " not supported.");
  }

  template <IsTensor TensorT>
  TensorFormat getTensorFormat(const TensorT&)
  {
    return tensorFormat<typename TensorT::value_type>();
  }

  template <typename T>
  TensorFormat getTensorFormat(const NestedTensor<T>&)
  {
    return tensorFormat<NestedTensor<T>>();
  }

  inline TensorFormat getTensorFormat(const StreamableTensor& tensor)
  {
    return std::visit([](auto&& arg) -> TensorFormat {
      return getTensorFormat(arg);
    }, tensor);
  }

  template <IsTensor TensorT>
  void write_tensor(std::ostream& os, const TensorT& tensor);

  // A standalone cloud is a logical value, not a view into its owner. Store
  // only its selected coordinates so owner-backed clouds cannot pull their
  // parent tensor (or unrelated cells) into a pickle.
  template <typename T>
  void write_element(std::ostream& os, const PointCloud<T>& cloud)
  {
    write_tensor(os, cloud.materialize());
  }

  template <typename T>
  void write_element(std::ostream& os, const sb::Tensor<T>& t)
  {
    io::detail::write_tensor(os, t);
  }

  template <typename T>
  Tensor<T> read_tensor(std::istream& is);

  template <typename T>
  NestedTensor<T> read_nested_tensor_element(std::istream& is);

  template <typename T>
  void write_element(std::ostream& os, const NestedTensor<T>& value)
  {
    write_bytes<uint64_t>(os, static_cast<uint64_t>(value.depth()));
    write_bytes<bool>(os, value.is_leaf());
    if (value.is_leaf())
    {
      write_tensor(os, value.leaf());
    }
    else
    {
      write_tensor(os, value.nested());
    }
  }

  inline void write_tensor_format(std::ostream& os, TensorFormat format)
  {
    write_bytes<std::int32_t>(os, format.baseFormat);
    write_bytes<std::int32_t>(os, format.subFormat);
  }

  template <typename T>
  void write_tensor_format(std::ostream& os)
  {
    write_tensor_format(os, tensorFormat<T>());
  }

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

  template <typename T>
  PointCloud<T> read_point_cloud(std::istream& is)
  {
    auto coords = read_element<Tensor<T>>(is);
    if (coords.rank() != 2)
    {
      throw std::runtime_error(
          "Invalid number of standalone point-cloud coordinate dimensions in saved data");
    }
    return PointCloud<T>(std::move(coords));
  }

  // Shared writer for tensors whose elements may be indexed views over a
  // source coordinate buffer (PointCloud): each distinct source is
  // stored once (deduplicated by buffer address — elements sharing a source,
  // e.g. indexed subsamples, are written once),
  // then every element as its source id plus, for indexed views, its index
  // array. @p sourceKey maps an element to its source buffer address;
  // @p writeSource writes one element's source.
  template <typename ElemT, typename SourceKeyF, typename WriteSourceF>
  void write_shared_source_elements(
      std::ostream& os, const Tensor<ElemT>& tensor, SourceKeyF sourceKey, WriteSourceF writeSource)
  {
    using KeyT = decltype(sourceKey(std::declval<const ElemT&>()));

    auto sz = tensor.size();
    const auto* data = tensor.data();

    // Assign each distinct source an id in first-appearance order...
    std::map<KeyT, uint64_t, std::owner_less<KeyT>> idOf;
    std::vector<const ElemT*> sources;
    for (auto k = 0_uz; k < sz; ++k)
    {
      if (!idOf.contains(sourceKey(data[k])))
      {
        idOf.emplace(sourceKey(data[k]), static_cast<uint64_t>(sources.size()));
        sources.push_back(&data[k]);
      }
    }

    // ...write the source block...
    write_bytes<uint64_t>(os, static_cast<uint64_t>(sources.size()));
    for (const ElemT* src : sources)
    {
      writeSource(os, *src);
    }

    // ...then every element as a reference to its source.
    for (auto k = 0_uz; k < sz; ++k)
    {
      write_bytes<uint64_t>(os, idOf.at(sourceKey(data[k])));
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
        [](const PointCloud<ScalarT>& elem) { return elem.coords().storage_owner(); },
        [](std::ostream& o, const PointCloud<ScalarT>& src) { write_tensor(o, src.coords()); });
  }

  template <typename T>
  size_t serialized_tensor_size(const Tensor<T>& tensor)
  {
    if constexpr (is_nested_tensor_v<T>)
    {
      // A scalar nested tensor has one element even though Tensor::size()
      // reports zero for a rank-0 tensor.
      if (tensor.shape().empty())
      {
        return 1;
      }
    }
    return tensor.size();
  }


  template <IsTensor TensorT>
    void write_contiguous_tensor(std::ostream& os, const TensorT& tensor)
  {
    write_tensor_format(os, getTensorFormat(tensor));

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
    else
    {
      auto sz = serialized_tensor_size(tensor);
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
  void write_tensor(std::ostream& os, const NestedTensor<T>& tensor)
  {
    write_tensor_format<NestedTensor<T>>(os);
    write_element(os, tensor);
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

    auto sz = serialized_tensor_size(ret);
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
      else if constexpr (is_nested_tensor_v<T>)
        *elem = read_nested_tensor_element<typename is_nested_tensor<T>::leaf_type>(is);
      else
      {
        *elem = read_element<T>(is);
      }
    }

    return ret;
  }

  template <typename T>
  NestedTensor<T> read_nested_tensor_element(std::istream& is)
  {
    const size_t depth = static_cast<size_t>(read_bytes<uint64_t>(is));
    const bool isLeaf = read_bytes<bool>(is);
    if (isLeaf)
    {
      if (depth != 1)
      {
        throw std::runtime_error("Invalid leaf depth in nested tensor");
      }
      return NestedTensor<T>(read_element<Tensor<T>>(is));
    }
    if (depth < 2)
    {
      throw std::runtime_error("Invalid nested tensor depth");
    }
    auto children = read_element<Tensor<NestedTensor<T>>>(is);
    NestedTensor<T> result(std::move(children), depth - 1);
    if (result.depth() != depth)
    {
      throw std::runtime_error("Nested tensor depth does not match its children");
    }
    return result;
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
      SourceT source = readSource(is);
      if constexpr (is_point_cloud_v<ElemT>)
      {
        if (source.rank() != 0 && source.rank() != 2)
        {
          throw std::runtime_error(
            "Invalid number of point-cloud coordinate dimensions in saved data");
        }
      }
      sources.push_back(std::move(source));
    }

    auto sz = ret.size();
    for (auto* elem = ret.data(); elem != ret.data() + sz; ++elem)
    {
      auto id = read_bytes<std::uint64_t>(is);
      if (id >= sources.size())
      {
        throw std::runtime_error("Invalid shared-source reference in saved data");
      }
      const bool indexed = read_bytes<bool>(is);
      if (indexed)
      {
        if constexpr (is_point_cloud_v<ElemT>)
        {
          if (sources[id].rank() != 2)
          {
            throw std::runtime_error(
              "Indexed point-cloud source must have 2 dimensions");
          }
        }
        Tensor<uint64_t> indices = read_element<Tensor<uint64_t>>(is);
        if (indices.rank() != 1)
        {
          throw std::runtime_error(
            "Invalid number of point-cloud index dimensions in saved data");
        }
        for (size_t i = 0; i < indices.size(); ++i)
        {
          if (indices(i) >= sources[id].shape(0))
          {
            throw std::runtime_error("Point-cloud index out of bounds in saved data");
          }
        }
        *elem = ElemT(sources[id], std::move(indices));
      }
      else
      {
        // Sharing, not copying: PointCloud wraps the coordinate tensor.
        if constexpr (is_point_cloud_v<ElemT>)
        {
          *elem = sources[id].rank() == 0 ? ElemT() : ElemT(sources[id]);
        }
        else
        {
          *elem = ElemT(sources[id]);
        }
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


  /// The format an earlier version used for point-cloud tensors, where each
  /// point cloud's coordinate tensor was stored inline (1000). The current
  /// format uses shared coordinate sources (1001). Equal to tensorFormat<T>()
  /// for every other element type.
  template <typename T>
  TensorFormat legacyTensorFormat()
  {
    if constexpr (is_point_cloud_v<T>)
    {
      return TensorFormat{ .baseFormat = 1000, .subFormat = tensorFormat<T>().subFormat };
    }
    else
    {
      return tensorFormat<T>();
    }
  }

  /// Read a tensor body for element type T, routing on the format already read
  /// from the stream: the shared-source layout for the current point-cloud format, element-wise for legacy and unchanged formats.
  /// Both read entry points go through here so they cannot drift apart.
  template <typename T>
  Tensor<T> read_tensor_for_format(std::istream& is, TensorFormat format)
  {
    if constexpr (is_point_cloud_v<T>)
    {
      if (format == tensorFormat<T>())
      {
        return read_indexed_point_cloud_tensor<typename is_point_cloud<T>::scalar_type>(is);
      }
    }
    return read_tensor<T>(is);
  }
}

#endif // STABLEBEAR_TENSOR_IO_H
