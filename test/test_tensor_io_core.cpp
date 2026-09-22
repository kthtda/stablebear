#include <gtest/gtest.h>

#include <sbear/tensor.hpp>
#include <sbear/walk.hpp>
#include <sbear/io/tensor_io.hpp>
#include <sbear/functional/pcf.hpp>
#include <sbear/persistence/barcode.hpp>

#include <sstream>
#include <cstring>

namespace
{
  using sb::io::detail::TensorFormat;
  using sb::io::detail::tensorFormatV3;

  // ============================================================================
  // TensorFormat mapping for supported core types
  // ============================================================================

  TEST(TensorIoCore, TensorFormatScalarAndCompositeTypes)
  {
    using sb::float32_t;
    using sb::float64_t;

    {
      auto fmt = tensorFormatV3<float32_t>();
      EXPECT_EQ(1, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = tensorFormatV3<float64_t>();
      EXPECT_EQ(1, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
    {
      auto fmt = tensorFormatV3<sb::Pcf<float32_t, float32_t>>();
      EXPECT_EQ(100, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = tensorFormatV3<sb::Pcf<float64_t, float64_t>>();
      EXPECT_EQ(100, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
    {
      auto fmt = tensorFormatV3<sb::PointCloud<float32_t>>();
      EXPECT_EQ(1000, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = tensorFormatV3<sb::PointCloud<float64_t>>();
      EXPECT_EQ(1000, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
    {
      auto fmt = tensorFormatV3<sb::Tensor<sb::PointCloud<float32_t>>>();
      EXPECT_EQ(1000, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = tensorFormatV3<sb::Tensor<sb::PointCloud<float64_t>>>();
      EXPECT_EQ(1000, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
    {
      auto fmt = sb::io::detail::tensorFormatV1_V2<
        sb::Tensor<sb::PointCloud<float32_t>>>();
      EXPECT_EQ(1000, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = sb::io::detail::tensorFormatV3<
        sb::Tensor<sb::PointCloud<float32_t>>>();
      EXPECT_EQ(1000, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = sb::io::detail::tensorFormatV3<
        sb::Tensor<sb::PointCloud<float32_t>, sb::TensorProperty::Indexed>>();
      EXPECT_EQ(1000, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
      EXPECT_EQ(sb::TensorProperty::Indexed, fmt.tensorProperties);
      EXPECT_EQ(sb::io::detail::TensorFormatFlag::None, fmt.formatFlags);
    }
    {
      auto fmt = sb::io::detail::tensorFormatV3<
        sb::Tensor<sb::PointCloud<float64_t>, sb::TensorProperty::Indexed>>();
      EXPECT_EQ(1000, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
      EXPECT_EQ(sb::TensorProperty::Indexed, fmt.tensorProperties);
      EXPECT_EQ(sb::io::detail::TensorFormatFlag::None, fmt.formatFlags);
    }
    {
      auto fmt = tensorFormatV3<sb::ph::Barcode<float32_t>>();
      EXPECT_EQ(10000, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = tensorFormatV3<sb::ph::Barcode<float64_t>>();
      EXPECT_EQ(10000, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
  }

  TEST(TensorIoCore, NestedTensorFormatsPreserveLeafFormat)
  {
    {
      auto fmt = tensorFormatV3<sb::NestedTensor<sb::float32_t>>();
      EXPECT_EQ(fmt.baseFormat, 1);
      EXPECT_EQ(fmt.subFormat, 32);
      EXPECT_EQ(fmt.formatFlags, sb::io::detail::TensorFormatFlag::Nested);
    }
    {
      auto fmt = tensorFormatV3<sb::NestedTensor<sb::float64_t>>();
      EXPECT_EQ(fmt.baseFormat, 1);
      EXPECT_EQ(fmt.subFormat, 64);
      EXPECT_EQ(fmt.formatFlags, sb::io::detail::TensorFormatFlag::Nested);
    }
    {
      auto fmt = tensorFormatV3<sb::NestedTensor<sb::int32_t>>();
      EXPECT_EQ(fmt.baseFormat, 2);
      EXPECT_EQ(fmt.subFormat, 32);
      EXPECT_EQ(fmt.formatFlags, sb::io::detail::TensorFormatFlag::Nested);
    }
    {
      auto fmt = tensorFormatV3<sb::NestedTensor<sb::int64_t>>();
      EXPECT_EQ(fmt.baseFormat, 2);
      EXPECT_EQ(fmt.subFormat, 64);
      EXPECT_EQ(fmt.formatFlags, sb::io::detail::TensorFormatFlag::Nested);
    }
    {
      auto fmt = tensorFormatV3<sb::NestedTensor<sb::uint32_t>>();
      EXPECT_EQ(fmt.baseFormat, 3);
      EXPECT_EQ(fmt.subFormat, 32);
      EXPECT_EQ(fmt.formatFlags, sb::io::detail::TensorFormatFlag::Nested);
    }
    {
      auto fmt = tensorFormatV3<sb::NestedTensor<sb::uint64_t>>();
      EXPECT_EQ(fmt.baseFormat, 3);
      EXPECT_EQ(fmt.subFormat, 64);
      EXPECT_EQ(fmt.formatFlags, sb::io::detail::TensorFormatFlag::Nested);
    }
  }

  TEST(TensorIoCore, TensorFormatReaderUsesFileVersion)
  {
    const TensorFormat v3Format{
      .baseFormat = 1000,
      .subFormat = 64,
      .tensorProperties = sb::TensorProperty::Indexed,
      .formatFlags = sb::io::detail::TensorFormatFlag::Nested
    };

    std::stringstream v3;
    sb::io::detail::write_type_format(v3, v3Format);
    EXPECT_EQ(v3.str().size(), 4 * sizeof(std::uint32_t));
    sb::io::detail::BinaryReader v3Reader(v3, 3);
    EXPECT_EQ(sb::io::detail::read_type_format(v3Reader), v3Format);

    std::stringstream v2;
    sb::io::detail::write_bytes<std::int32_t>(v2, v3Format.baseFormat);
    sb::io::detail::write_bytes<std::int32_t>(v2, v3Format.subFormat);
    sb::io::detail::write_bytes<std::uint32_t>(v2, 0x12345678u);
    EXPECT_EQ(v2.str().size(), 3 * sizeof(std::uint32_t));

    sb::io::detail::BinaryReader v2Reader(v2, 2);
    const TensorFormat restored = sb::io::detail::read_type_format(v2Reader);
    EXPECT_EQ(restored.baseFormat, v3Format.baseFormat);
    EXPECT_EQ(restored.subFormat, v3Format.subFormat);
    EXPECT_EQ(restored.tensorProperties, sb::TensorProperty::None);
    EXPECT_EQ(restored.formatFlags, sb::io::detail::TensorFormatFlag::None);
    EXPECT_EQ(sb::io::detail::read_bytes<std::uint32_t>(v2), 0x12345678u);
  }

  TEST(TensorIoCore, NestedTensorFormatStartsAtVersion3)
  {
    std::istringstream input;
    const auto format = tensorFormatV3<sb::NestedTensor<sb::float32_t>>();

    sb::io::detail::BinaryReader v1Reader(input, 1);
    EXPECT_FALSE(
      sb::io::detail::matches_nested_tensor_format<
        sb::NestedTensor<sb::float32_t>>(v1Reader, format));

    sb::io::detail::BinaryReader v2Reader(input, 2);
    EXPECT_FALSE(
      sb::io::detail::matches_nested_tensor_format<
        sb::NestedTensor<sb::float32_t>>(v2Reader, format));

    sb::io::detail::BinaryReader v3Reader(input, 3);
    EXPECT_TRUE(
      sb::io::detail::matches_nested_tensor_format<
        sb::NestedTensor<sb::float32_t>>(v3Reader, format));
  }

  TEST(TensorIoCore, Version3PacksBoolTensorElements)
  {
    sb::Tensor<bool> tensor({ 9 });
    for (size_t i = 0; i < tensor.size(); ++i)
    {
      tensor(i) = i % 3 == 0;
    }

    std::stringstream stream;
    sb::io::detail::write_tensor(stream, tensor);
    constexpr size_t tensorFormatSize = 4 * sizeof(std::uint32_t);
    constexpr size_t shapeSize = 3 * sizeof(std::uint64_t);
    constexpr size_t packedPayloadSize = 2;
    EXPECT_EQ(
      stream.str().size(),
      tensorFormatSize + shapeSize + packedPayloadSize);

    EXPECT_EQ(
      sb::io::detail::read_type_format(stream),
      tensorFormatV3<sb::Tensor<bool>>());
    EXPECT_EQ(sb::io::detail::read_tensor<bool>(stream), tensor);
  }

  TEST(TensorIoCore, TensorFormatThrowsOnUnsupportedType)
  {
    struct Unsupported {};
    EXPECT_THROW((void)tensorFormatV3<Unsupported>(), std::runtime_error);
  }

  // ============================================================================
  // write_contiguous_tensor / read_tensor roundtrip for scalar tensor
  // ============================================================================

  TEST(TensorIoCore, ContiguousScalarTensorRoundtrip)
  {
    using T = sb::float32_t;
    sb::Tensor<T> t({ 2, 3 });
    T v = static_cast<T>(0);
    t.apply([&v](T& x) { x = v++; });

    std::stringstream ss;
    sb::io::detail::write_contiguous_tensor(ss, t);

    std::string all = ss.str();

    std::istringstream iss(all);

    sb::io::detail::read_type_format(iss);

    auto roundtrip = sb::io::detail::read_tensor<T>(iss);
    EXPECT_EQ(roundtrip, t);
  }

  // ============================================================================
  // write_tensor handles non-contiguous views by copying internally
  // ============================================================================

  TEST(TensorIoCore, NonContiguousTensorRoundtripViaWriteTensor)
  {
    using T = double;

    sb::Tensor<T> base({ 4, 4 });
    T v = 0;
    base.apply([&v](T& x) { x = v++; });

    // Take a strided view so that is_contiguous() is false
    auto view = base[std::vector<sb::Slice>{ sb::range(0, 4, 2), sb::all() }];
    ASSERT_FALSE(view.is_contiguous());

    std::stringstream ss;
    sb::io::detail::write_tensor(ss, view);

    std::string all = ss.str();

    std::istringstream iss(all);

    sb::io::detail::read_type_format(iss);
    auto rt = sb::io::detail::read_tensor<T>(iss);

    EXPECT_EQ(rt.shape(), view.shape());

    sb::walk(view, [&](const std::vector<size_t>& idx)
    {
      EXPECT_EQ(rt(idx), view(idx));
    });
  }

  // ============================================================================
  // read_tensor detects stride mismatches
  // ============================================================================

  TEST(TensorIoCore, ReadTensorThrowsOnStrideMismatch)
  {
    using T = double;

    sb::Tensor<T> t({ 2, 3 });
    T v = 0;
    t.apply([&v](T& x) { x = v++; });

    std::stringstream ss;
    sb::io::detail::write_contiguous_tensor(ss, t);

    std::string all = ss.str();
    ASSERT_GE(all.size(), 16u);

    // Work on the payload that starts after the TensorFormat header
    std::string payload(all.begin() + 16, all.end());

    // Layout in payload:
    // [shapeSz:uint64][shape0:uint64][stride0:uint64][shape1:uint64][stride1:uint64]...
    std::uint64_t* raw = reinterpret_cast<std::uint64_t*>(payload.data());
    std::uint64_t shapeSz = raw[0];
    ASSERT_EQ(shapeSz, 2u);

    // Corrupt first stride (located after shapeSz + shape0 + shape1)
    std::size_t strideIdx = 1 + static_cast<std::size_t>(shapeSz) + 0;
    raw[strideIdx] = 999u;

    std::istringstream iss(payload);

    EXPECT_THROW((void)sb::io::detail::read_tensor<T>(iss), std::runtime_error);
  }

  TEST(TensorIoCore, IndexedPointCloudSourcesMustBeCoordinateMatrices)
  {
    EXPECT_NO_THROW(
      sb::io::detail::validate_indexed_point_cloud_source(
        sb::Tensor<float>({ 2, 3 })));
    EXPECT_THROW(
      sb::io::detail::validate_indexed_point_cloud_source(
        sb::Tensor<float>({ 2 })),
      std::runtime_error);
  }

  TEST(TensorIoCore, IndexedPointCloudSourceReferencesMustExist)
  {
    EXPECT_NO_THROW(
      sb::io::detail::validate_indexed_point_cloud_source_reference(0, 1));
    EXPECT_THROW(
      sb::io::detail::validate_indexed_point_cloud_source_reference(1, 1),
      std::runtime_error);
    EXPECT_THROW(
      sb::io::detail::validate_indexed_point_cloud_source_reference(0, 0),
      std::runtime_error);
  }

  TEST(TensorIoCore, IndexedPointCloudSelectionsMustNameCoordinateRows)
  {
    sb::Tensor<uint64_t> valid({ 2 });
    valid(0) = 1;
    valid(1) = 0;
    EXPECT_NO_THROW(
      sb::io::detail::validate_indexed_point_cloud_selection(valid, 2));

    sb::Tensor<uint64_t> notAVector({ 1, 1 });
    EXPECT_THROW(
      sb::io::detail::validate_indexed_point_cloud_selection(notAVector, 2),
      std::runtime_error);

    sb::Tensor<uint64_t> outOfBounds({ 1 });
    outOfBounds(0) = 2;
    EXPECT_THROW(
      sb::io::detail::validate_indexed_point_cloud_selection(outOfBounds, 2),
      std::runtime_error);
  }

} // namespace
