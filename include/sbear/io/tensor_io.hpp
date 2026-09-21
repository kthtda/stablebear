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
  struct TensorFormatFlag
  {
    static constexpr std::uint32_t None = 0;
    static constexpr std::uint32_t Nested = 1 << 0;
  };

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
      Tensor<PointCloud<float32_t>, TensorProperty::Indexed>,
      Tensor<PointCloud<float64_t>, TensorProperty::Indexed>,

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
    std::uint32_t tensorProperties = TensorProperty::None;
    std::uint32_t formatFlags = TensorFormatFlag::None;

    std::string toString() const
    {
      return "(" + std::to_string(baseFormat) + ", "
        + std::to_string(subFormat) + ", "
        + std::to_string(tensorProperties) + ", "
        + std::to_string(formatFlags) + ")";
    }

    bool operator==(const TensorFormat&) const = default;
    bool operator!=(const TensorFormat&) const = default;

    [[nodiscard]] bool same_base_and_subformat(
        const TensorFormat& other) const
    {
      return baseFormat == other.baseFormat
        && subFormat == other.subFormat;
    }

    [[nodiscard]] bool matches(
        const TensorFormat& expected, int fileFormatVersion) const
    {
      return fileFormatVersion >= 3
        ? *this == expected
        : same_base_and_subformat(expected);
    }

    template <typename T>
    [[nodiscard]] bool matches_type(int fileFormatVersion) const;
  };

  template <typename U>
  TensorFormat tensorFormatV3()
  {
    using namespace std::string_literals;
    using T = std::decay_t<U>;

    if constexpr (is_nested_tensor_v<T>)
    {
      const auto leafFormat = tensorFormatV3<typename is_nested_tensor<T>::leaf_type>();
      return TensorFormat{
        .baseFormat = leafFormat.baseFormat,
        .subFormat = leafFormat.subFormat,
        .tensorProperties = leafFormat.tensorProperties,
        .formatFlags = leafFormat.formatFlags | TensorFormatFlag::Nested
      };
    }
    else if constexpr (IsTensor<T>)
    {
      using value_type = typename T::value_type;
      auto format = tensorFormatV3<value_type>();
      format.tensorProperties = static_cast<std::uint32_t>(T::PropertyFlags);
      return format;
    }
    else if constexpr (std::is_same_v<T, float32_t>) { return TensorFormat{ .baseFormat = 1, .subFormat = 32 }; }
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

    // Earlier point-cloud tensors use base format 1000 and store every cloud
    // as a full coordinate tensor. V3 uses base format 1001 and stores each
    // distinct coordinate source once plus per-element source metadata.
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

  template <typename U>
  TensorFormat tensorValueFormatV1_V2()
  {
    using T = std::decay_t<U>;
    using namespace std::string_literals;

    if constexpr (std::is_same_v<T, float32_t>) { return { 1, 32, 0, 0 }; }
    else if constexpr (std::is_same_v<T, float64_t>) { return { 1, 64, 0, 0 }; }
    else if constexpr (std::is_same_v<T, int32_t>) { return { 2, 32, 0, 0 }; }
    else if constexpr (std::is_same_v<T, int64_t>) { return { 2, 64, 0, 0 }; }
    else if constexpr (std::is_same_v<T, uint32_t>) { return { 3, 32, 0, 0 }; }
    else if constexpr (std::is_same_v<T, uint64_t>) { return { 3, 64, 0, 0 }; }
    else if constexpr (std::is_same_v<T, bool>) { return { 4, 8, 0, 0 }; }
    else if constexpr (std::is_same_v<T, Pcf<float32_t, float32_t>>) { return { 100, 32, 0, 0 }; }
    else if constexpr (std::is_same_v<T, Pcf<float64_t, float64_t>>) { return { 100, 64, 0, 0 }; }
    else if constexpr (std::is_same_v<T, Pcf<int32_t, int32_t>>) { return { 101, 32, 0, 0 }; }
    else if constexpr (std::is_same_v<T, Pcf<int64_t, int64_t>>) { return { 101, 64, 0, 0 }; }
    else if constexpr (std::is_same_v<T, PointCloud<float32_t>>) { return { 1000, 32, 0, 0 }; }
    else if constexpr (std::is_same_v<T, PointCloud<float64_t>>) { return { 1000, 64, 0, 0 }; }
    else if constexpr (std::is_same_v<T, SymmetricMatrix<float32_t>>) { return { 1100, 32, 0, 0 }; }
    else if constexpr (std::is_same_v<T, SymmetricMatrix<float64_t>>) { return { 1100, 64, 0, 0 }; }
    else if constexpr (std::is_same_v<T, DistanceMatrix<float32_t>>) { return { 1120, 32, 0, 0 }; }
    else if constexpr (std::is_same_v<T, DistanceMatrix<float64_t>>) { return { 1120, 64, 0, 0 }; }
    else if constexpr (std::is_same_v<T, ph::Barcode<float32_t>>) { return { 10000, 32, 0, 0 }; }
    else if constexpr (std::is_same_v<T, ph::Barcode<float64_t>>) { return { 10000, 64, 0, 0 }; }

    throw std::runtime_error(
      "Tensor type "s + sb::detail::unmangled_typename<T>()
      + " not supported by format versions 1 and 2.");
  }

  template <typename U>
    requires (IsTensor<std::decay_t<U>>
              && !is_nested_tensor_v<std::decay_t<U>>)
  TensorFormat tensorFormatV1_V2()
  {
    using TensorT = std::decay_t<U>;
    using value_type = typename TensorT::value_type;
    return tensorValueFormatV1_V2<value_type>();
  }

  template <typename T>
  bool TensorFormat::matches_type(int fileFormatVersion) const
  {
    if constexpr (IsTensor<std::decay_t<T>>)
    {
      if constexpr (is_nested_tensor_v<std::decay_t<T>>)
      {
        return fileFormatVersion >= 3
          && matches(tensorFormatV3<std::decay_t<T>>(), fileFormatVersion);
      }
      else
      {
        const auto expected = (fileFormatVersion >= 3)
          ? tensorFormatV3<std::decay_t<T>>()
          : tensorFormatV1_V2<std::decay_t<T>>();
        return matches(expected, fileFormatVersion);
      }
    }
    else
    {
      return matches(tensorFormatV3<std::decay_t<T>>(), fileFormatVersion);
    }
  }

  template <IsTensor TensorT>
  void write_tensor(std::ostream& os, const TensorT& tensor);

  // A standalone cloud is a logical value, not a view into its owner. Store
  // only its selected coordinates so owner-backed clouds cannot pull their
  // parent tensor (or unrelated cells) into a pickle.
  template <typename T>
  void write_value(std::ostream& os, const PointCloud<T>& cloud)
  {
    write_tensor(os, cloud.materialize());
  }

  template <typename T>
  void write_value(std::ostream& os, const sb::Tensor<T>& t)
  {
    io::detail::write_tensor(os, t);
  }

  template <typename T>
  Tensor<T> read_tensor(BinaryReader& reader);

  template <typename T>
  NestedTensor<T> read_nested_tensor_element(BinaryReader& reader);

  template <typename T>
  void write_value(std::ostream& os, const NestedTensor<T>& value)
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

  template <typename T>
  void write_value(std::ostream& os, const T& value)
  {
    write_element(os, value);
  }

  inline void write_type_format(std::ostream& os, TensorFormat format)
  {
    write_bytes<std::int32_t>(os, format.baseFormat);
    write_bytes<std::int32_t>(os, format.subFormat);
    write_bytes<std::uint32_t>(os, format.tensorProperties);
    write_bytes<std::uint32_t>(os, format.formatFlags);
  }

  template <typename T>
  void write_type_format(std::ostream& os)
  {
    write_type_format(os, tensorFormatV3<T>());
  }

  inline TensorFormat read_type_format(BinaryReader& reader)
  {
    auto& is = reader.stream();
    TensorFormat format{
      .baseFormat = read_bytes<std::int32_t>(is),
      .subFormat = read_bytes<std::int32_t>(is),
      .tensorProperties = TensorProperty::None,
      .formatFlags = TensorFormatFlag::None
    };
    if (reader.format_version() >= 3)
    {
      format.tensorProperties = read_bytes<std::uint32_t>(is);
      format.formatFlags = read_bytes<std::uint32_t>(is);
    }
    return format;
  }

  template <typename NestedT>
    requires is_nested_tensor_v<std::decay_t<NestedT>>
  bool matches_nested_tensor_format(
      const BinaryReader& reader, TensorFormat actual)
  {
    return actual.template matches_type<std::decay_t<NestedT>>(
      reader.format_version());
  }

  template <IsTensor TensorT>
  TensorT read_element(BinaryReader& reader)
  {
    auto format = read_type_format(reader);
    if (!format.template matches_type<TensorT>(reader.format_version()))
    {
      throw std::runtime_error(
        "Unexpected tensor of type " + format.toString() + " where "
        + tensorFormatV3<TensorT>().toString() + " was expected.");
    }
    return io::detail::read_tensor<typename TensorT::value_type>(reader);
  }

  template <typename T>
  PointCloud<T> read_point_cloud(BinaryReader& reader)
  {
    auto coords = read_element<Tensor<T>>(reader);
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
      std::ostream& os, const Tensor<ElemT>& tensor,
      SourceKeyF sourceKey, WriteSourceF writeSource)
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
  void write_point_cloud_elements(
      std::ostream& os, const Tensor<PointCloud<ScalarT>>& tensor)
  {
    write_shared_source_elements(
        os, tensor,
        [](const PointCloud<ScalarT>& elem) { return elem.coords().storage_owner(); },
        [](std::ostream& output, const PointCloud<ScalarT>& src) {
          write_tensor(output, src.coords());
        });
  }

  // Write a tensor-level indexed point-cloud tensor without materializing its
  // logical coordinates.  Iterating through the logical view composes any
  // source-cloud indexing with the tensor-level selections, so sliced,
  // transposed, and repeated views become self-contained while coordinate
  // buffers remain deduplicated.
  template <typename ScalarT>
  void write_indexed_point_cloud_tensor(
      std::ostream& os,
      const Tensor<PointCloud<ScalarT>, TensorProperty::Indexed>& tensor)
  {
    write_type_format(os, tensorFormatV3<decltype(tensor)>());

    write_bytes<std::uint64_t>(os, tensor.shape().size());
    std::vector<uint64_t> contiguousStrides(tensor.shape().size());
    if (!contiguousStrides.empty())
    {
      contiguousStrides.back() = 1;
      for (ptrdiff_t i = static_cast<ptrdiff_t>(contiguousStrides.size()) - 2;
           i >= 0; --i)
      {
        contiguousStrides[i] = contiguousStrides[i + 1]
          * static_cast<uint64_t>(tensor.shape()[i + 1]);
      }
    }
    for (auto i = 0_uz; i < tensor.shape().size(); ++i)
    {
      write_bytes<std::uint64_t>(os, tensor.shape()[i]);
      write_bytes<std::uint64_t>(os, contiguousStrides[i]);
    }

    const size_t count = tensor.shape().empty() ? size_t{1} : tensor.size();
    using KeyT = std::shared_ptr<const void>;
    std::map<KeyT, uint64_t, std::owner_less<KeyT>> idOf;
    std::vector<Tensor<ScalarT>> sources;
    for (size_t i = 0; i < count; ++i)
    {
      const PointCloud<ScalarT> cloud = tensor.flat(i);
      const KeyT key = cloud.coords().storage_owner();
      if (!idOf.contains(key))
      {
        idOf.emplace(key, static_cast<uint64_t>(sources.size()));
        sources.push_back(cloud.coords());
      }
    }

    write_bytes<uint64_t>(os, static_cast<uint64_t>(sources.size()));
    for (const auto& source : sources)
    {
      write_tensor(os, source);
    }

    const bool hasSelections = tensor.has_indices();
    write_bytes<bool>(os, hasSelections);
    for (size_t i = 0; i < count; ++i)
    {
      const PointCloud<ScalarT> cloud = tensor.flat(i);
      write_bytes<uint64_t>(os, idOf.at(cloud.coords().storage_owner()));
      if (hasSelections)
      {
        write_tensor(os, cloud.indices());
      }
    }
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
    write_type_format(os, tensorFormatV3<TensorT>());

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
      if constexpr (std::is_same_v<value_type, bool>)
      {
        for (size_t offset = 0; offset < sz; offset += 8)
        {
          std::uint8_t packed = 0;
          const size_t end = std::min(offset + 8, sz);
          for (size_t i = offset; i < end; ++i)
          {
            packed |= static_cast<std::uint8_t>(tensor.data()[i] ? 1 : 0)
              << (i - offset);
          }
          write_bytes<std::uint8_t>(os, packed);
        }
      }
      else
      {
        for (auto const * elem = tensor.data(); elem != tensor.data() + sz; ++elem)
        {
          write_value(os, *elem);
        }
      }
    }
  }

  template <IsTensor TensorT>
  void write_tensor(std::ostream& os, const TensorT& tensor)
  {
    using value_type = typename TensorT::value_type;
    if constexpr (TensorT::IsIndexed && is_point_cloud_v<value_type>)
    {
      write_indexed_point_cloud_tensor<
        typename is_point_cloud<value_type>::scalar_type>(os, tensor);
    }
    else if (!tensor.is_contiguous())
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
    else
    {
      write_contiguous_tensor(os, tensor);
    }
  }

  template <typename T>
  void write_tensor(std::ostream& os, const NestedTensor<T>& tensor)
  {
    write_type_format<NestedTensor<T>>(os);
    write_value(os, tensor);
  }



  template <typename T>
  Tensor<T> read_tensor(BinaryReader& reader)
  {
    auto& is = reader.stream();
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
    if constexpr (std::is_same_v<T, bool>)
    {
      if (reader.format_version() >= 3)
      {
        for (size_t offset = 0; offset < sz; offset += 8)
        {
          const std::uint8_t packed = read_bytes<std::uint8_t>(is);
          const size_t end = std::min(offset + 8, sz);
          for (size_t i = offset; i < end; ++i)
          {
            ret.data()[i] = (packed & (std::uint8_t{1} << (i - offset))) != 0;
          }
        }
        return ret;
      }
    }

    for (auto * elem = ret.data(); elem != ret.data() + sz; ++elem)
    {
      if constexpr (is_barcode_v<T>)
        *elem = read_barcode<typename is_barcode<T>::scalar_type>(is);
      else if constexpr (is_compressed_matrix_v<T>)
        *elem = read_compressed_matrix<T>(is);
      else if constexpr (is_point_cloud_v<T>)
        // Earlier point-cloud tensors (baseFormat 1000) store a complete
        // coordinate tensor for every element.
        *elem = T(read_element<Tensor<typename is_point_cloud<T>::scalar_type>>(reader));
      else if constexpr (is_nested_tensor_v<T>)
        *elem = read_nested_tensor_element<typename is_nested_tensor<T>::leaf_type>(reader);
      else
      {
        *elem = read_element<T>(is);
      }
    }

    return ret;
  }

  template <typename T>
  NestedTensor<T> read_nested_tensor_element(BinaryReader& reader)
  {
    auto& is = reader.stream();
    const size_t depth = static_cast<size_t>(read_bytes<uint64_t>(is));
    const bool isLeaf = read_bytes<bool>(is);
    if (isLeaf)
    {
      if (depth != 1)
      {
        throw std::runtime_error("Invalid leaf depth in nested tensor");
      }
      return NestedTensor<T>(read_element<Tensor<T>>(reader));
    }
    if (depth < 2)
    {
      throw std::runtime_error("Invalid nested tensor depth");
    }
    auto children = read_element<Tensor<NestedTensor<T>>>(reader);
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
  Tensor<ElemT> read_shared_source_tensor(
      BinaryReader& reader, ReadSourceF readSource)
  {
    auto& is = reader.stream();
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
      SourceT source = readSource(reader);
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
        Tensor<uint64_t> indices = read_element<Tensor<uint64_t>>(reader);
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

  // Read the V3 shared-source point-cloud layout (baseFormat 1001).
  template <typename ScalarT>
  Tensor<PointCloud<ScalarT>> read_indexed_point_cloud_tensor(
      BinaryReader& reader)
  {
    return read_shared_source_tensor<PointCloud<ScalarT>, Tensor<ScalarT>>(
        reader, [](BinaryReader& input) {
          return read_element<Tensor<ScalarT>>(input);
        });
  }

  template <typename ScalarT>
  void validate_indexed_point_cloud_source(const Tensor<ScalarT>& coordinates)
  {
    if (coordinates.rank() != 2)
    {
      throw std::runtime_error(
        "Indexed point-cloud coordinate source must have 2 dimensions");
    }
  }

  inline void validate_indexed_point_cloud_source_reference(
      std::uint64_t sourceId, size_t sourceCount)
  {
    if (sourceId >= sourceCount)
    {
      throw std::runtime_error(
        "Invalid indexed point-cloud source reference in saved data");
    }
  }

  inline void validate_indexed_point_cloud_selection(
      const Tensor<uint64_t>& selection, size_t sourcePointCount)
  {
    if (selection.rank() != 1)
    {
      throw std::runtime_error(
        "Invalid number of indexed point-cloud selection dimensions in saved data");
    }
    for (size_t i = 0; i < selection.size(); ++i)
    {
      if (selection(i) >= sourcePointCount)
      {
        throw std::runtime_error(
          "Indexed point-cloud selection out of bounds in saved data");
      }
    }
  }

  // Read the V3 tensor-level indexed point-cloud layout (baseFormat 1001 with
  // the Indexed tensor-property bit).
  // The payload contains a deduplicated coordinate-source table followed by
  // an aligned source reference and owned selection for each logical cell.
  template <typename ScalarT>
  Tensor<PointCloud<ScalarT>, TensorProperty::Indexed>
  read_tensor_level_indexed_point_cloud_tensor(BinaryReader& reader)
  {
    auto& is = reader.stream();
    auto shapeSz = read_bytes<std::uint64_t>(is);
    std::vector<size_t> shape(shapeSz);
    std::vector<ptrdiff_t> strides(shapeSz);
    for (auto i = 0_uz; i < shapeSz; ++i)
    {
      shape[i] = read_bytes<std::uint64_t>(is);
      strides[i] = static_cast<ptrdiff_t>(read_bytes<std::uint64_t>(is));
    }

    Tensor<PointCloud<ScalarT>> source(shape);
    if (source.strides() != strides)
    {
      throw std::runtime_error(
        "Incorrect strides in saved indexed tensor (expected "
        + index_to_string(source.strides()) + " but got "
        + index_to_string(strides) + ")");
    }

    const auto numSources = read_bytes<std::uint64_t>(is);
    std::vector<Tensor<ScalarT>> sources;
    sources.reserve(numSources);
    for (auto i = 0_uz; i < numSources; ++i)
    {
      Tensor<ScalarT> coords = read_element<Tensor<ScalarT>>(reader);
      validate_indexed_point_cloud_source(coords);
      sources.push_back(std::move(coords));
    }

    const bool hasSelections = read_bytes<bool>(is);
    Tensor<NestedTensor<uint64_t>> selections(shape);
    const size_t count = shape.empty() ? size_t{1} : source.size();
    for (size_t i = 0; i < count; ++i)
    {
      const auto sourceId = read_bytes<std::uint64_t>(is);
      validate_indexed_point_cloud_source_reference(sourceId, sources.size());

      source.flat(i) = PointCloud<ScalarT>(sources[sourceId]);
      if (hasSelections)
      {
        Tensor<uint64_t> indices = read_element<Tensor<uint64_t>>(reader);
        validate_indexed_point_cloud_selection(
          indices, sources[sourceId].shape(0));
        selections.flat(i) = NestedTensor<uint64_t>(std::move(indices));
      }
    }

    using IndexedTensor =
      Tensor<PointCloud<ScalarT>, TensorProperty::Indexed>;
    if (!hasSelections)
    {
      return IndexedTensor(std::move(source), std::nullopt);
    }
    NestedTensor<uint64_t> ownedSelections(std::move(selections), 1);
    return IndexedTensor(
      std::move(source), std::move(ownedSelections));
  }


  template <IsTensor TensorT>
  bool is_tensor_format(const BinaryReader& reader, TensorFormat format)
  {
    if constexpr (TensorT::IsIndexed)
    {
      if (reader.format_version() < 3)
      {
        return false;
      }
    }
    return format.template matches_type<TensorT>(reader.format_version());
  }

  /// Read a tensor body for TensorT, routing on its full type and the format
  /// already read from the stream: lazy indexed point-cloud storage, the
  /// ordinary V3 shared-source layout, or the element-wise V1/V2 layout.
  /// Both read entry points go through here so they cannot drift apart.
  template <IsTensor TensorT>
  TensorT read_tensor_for_format(BinaryReader& reader, TensorFormat format)
  {
    const auto expectedFormat = reader.format_version() >= 3
      ? tensorFormatV3<TensorT>()
      : tensorFormatV1_V2<TensorT>();
    if (!is_tensor_format<TensorT>(reader, format))
    {
      throw std::runtime_error(
        "Unexpected tensor format " + format.toString() + " where "
        + expectedFormat.toString() + " was expected.");
    }

    using value_type = typename TensorT::value_type;
    if constexpr (TensorT::IsIndexed && is_point_cloud_v<value_type>)
    {
      return read_tensor_level_indexed_point_cloud_tensor<
        typename is_point_cloud<value_type>::scalar_type>(reader);
    }
    else if constexpr (is_point_cloud_v<value_type>)
    {
      if (reader.format_version() >= 3)
      {
        return read_indexed_point_cloud_tensor<
          typename is_point_cloud<value_type>::scalar_type>(reader);
      }
      return read_tensor<value_type>(reader);
    }
    else
    {
      return read_tensor<value_type>(reader);
    }
  }

  inline TensorFormat read_type_format(std::istream& is)
  {
    BinaryReader reader(is);
    return read_type_format(reader);
  }

  template <typename T>
  Tensor<T> read_tensor(std::istream& is)
  {
    BinaryReader reader(is);
    return read_tensor<T>(reader);
  }

  template <typename ScalarT>
  Tensor<PointCloud<ScalarT>, TensorProperty::Indexed>
  read_tensor_level_indexed_point_cloud_tensor(std::istream& is)
  {
    BinaryReader reader(is);
    return read_tensor_level_indexed_point_cloud_tensor<ScalarT>(reader);
  }
}

#endif // STABLEBEAR_TENSOR_IO_H
