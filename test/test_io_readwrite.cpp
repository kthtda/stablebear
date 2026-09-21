#include <gtest/gtest.h>

#include <sbear/io.hpp>
#include <sbear/tensor.hpp>
#include <sbear/walk.hpp>

#include <sstream>
#include <stdexcept>

namespace
{
  template<typename T>
  class IoReadWriteTest : public ::testing::Test
  {
  };

  using FloatTypes = ::testing::Types<sb::float32_t, sb::float64_t>;
  TYPED_TEST_SUITE(IoReadWriteTest, FloatTypes);

  // ============================================================================
  // Full write/read roundtrip for float tensors
  // ============================================================================

  TYPED_TEST(IoReadWriteTest, FloatTensorRoundtrip)
  {
    using TensorT = sb::Tensor<TypeParam>;

    TensorT tensor({ 3, 4 });
    sb::walk(tensor, [&tensor](const std::vector<size_t>& idx)
    {
      tensor(idx) = static_cast<TypeParam>(idx[0] * 10 + idx[1]);
    });

    std::stringstream ss;
    sb::write(tensor, ss);

    std::istringstream iss(ss.str());
    auto retTensor = sb::read<TensorT>(iss);

    EXPECT_EQ(tensor, retTensor);
  }

  TYPED_TEST(IoReadWriteTest, FloatTensorRoundtrip1d)
  {
    using TensorT = sb::Tensor<TypeParam>;

    TensorT tensor({ 5 });
    for (size_t i = 0; i < 5; ++i)
      tensor(i) = static_cast<TypeParam>(i * 1.5);

    std::stringstream ss;
    sb::write(tensor, ss);

    std::istringstream iss(ss.str());
    auto retTensor = sb::read<TensorT>(iss);

    EXPECT_EQ(tensor, retTensor);
  }

  TYPED_TEST(IoReadWriteTest, FloatTensorRoundtrip3d)
  {
    using TensorT = sb::Tensor<TypeParam>;

    TensorT tensor({ 2, 3, 4 });
    sb::walk(tensor, [&tensor](const std::vector<size_t>& idx)
    {
      tensor(idx) = static_cast<TypeParam>(100 * idx[0] + 10 * idx[1] + idx[2]);
    });

    std::stringstream ss;
    sb::write(tensor, ss);

    std::istringstream iss(ss.str());
    auto retTensor = sb::read<TensorT>(iss);

    EXPECT_EQ(tensor, retTensor);
  }

// ============================================================================
// Full write/read roundtrip for Pcf tensors
// ============================================================================

  TYPED_TEST(IoReadWriteTest, PcfTensorRoundtrip)
  {
    using PcfT = sb::Pcf<TypeParam, TypeParam>;
    using TensorT = sb::Tensor<PcfT>;

    TensorT tensor({ 2, 2 });
    sb::walk(tensor, [&tensor](const std::vector<size_t>& idx)
    {
      std::vector<typename PcfT::point_type> pts;
      pts.emplace_back(TypeParam(0), static_cast<TypeParam>(idx[0] * 10 + idx[1]));
      pts.emplace_back(TypeParam(1), static_cast<TypeParam>(idx[0] + idx[1]));
      tensor(idx) = PcfT(std::move(pts));
    });

    std::stringstream ss;
    sb::write(tensor, ss);

    std::istringstream iss(ss.str());
    auto retTensor = sb::read<TensorT>(iss);

    EXPECT_EQ(tensor, retTensor);
  }

  TEST(IoReadWriteTest, NestedTensorRoundtrip)
  {
    using Leaf = sb::Tensor<uint64_t>;
    using Nested = sb::NestedTensor<uint64_t>;

    auto leaf = [](std::initializer_list<uint64_t> values)
    {
      Leaf result({ values.size() });
      size_t i = 0;
      for (uint64_t value : values)
      {
        result(i++) = value;
      }
      return result;
    };
    auto roundtrip = [](const Nested& tensor)
    {
      std::stringstream output;
      sb::write(tensor, output);
      std::istringstream input(output.str());
      return sb::read<Nested>(input);
    };

    sb::Tensor<Leaf> tensorSource({ 2, 2 });
    tensorSource({ 0, 0 }) = leaf({ 3, 3 });
    tensorSource({ 0, 1 }) = leaf({});
    tensorSource({ 1, 0 }) = leaf({ 4 });
    tensorSource({ 1, 1 }) = leaf({ 9, 6, 7 });
    const Nested tensor = sb::to_nested_tensor(tensorSource);

    EXPECT_EQ(roundtrip(tensor), tensor);

    sb::Tensor<Leaf> scalarSource(std::vector<size_t>{});
    const std::vector<size_t> scalarIndex;
    scalarSource(scalarIndex) = leaf({ 3, 3, 7 });
    const Nested scalar = sb::to_nested_tensor(scalarSource);

    EXPECT_EQ(roundtrip(scalar), scalar);

    sb::Tensor<Leaf> branch({ 2 });
    branch(0) = leaf({ 1, 2 });
    branch(1) = leaf({ 3 });

    sb::Tensor<Leaf> emptyBranch({ 0 });

    sb::Tensor<sb::Tensor<Leaf>> recursiveChildren({ 2 });
    recursiveChildren(0) = std::move(branch);
    recursiveChildren(1) = std::move(emptyBranch);
    const Nested recursive = sb::to_nested_tensor(recursiveChildren);
    const Nested recursiveRoundtrip = roundtrip(recursive);

    EXPECT_EQ(recursiveRoundtrip, recursive);
    EXPECT_EQ(recursiveRoundtrip.depth(), 3);
    EXPECT_EQ(recursiveRoundtrip.nested()(0).depth(), 2);
    EXPECT_EQ(recursiveRoundtrip.nested()(0).nested()(0).depth(), 1);
    EXPECT_EQ(recursiveRoundtrip.nested()(1).nested().shape(), std::vector<size_t>({ 0 }));
  }

  TYPED_TEST(IoReadWriteTest, TensorLevelIndexedPointCloudRoundtrip)
  {
    using Cloud = sb::PointCloud<TypeParam>;
    using DenseTensor = sb::Tensor<Cloud>;
    using IndexedTensor =
      sb::Tensor<Cloud, sb::TensorProperty::Indexed>;

    sb::Tensor<TypeParam> coordinates({ 5, 2 });
    for (size_t i = 0; i < coordinates.shape(0); ++i)
    {
      coordinates({ i, 0 }) = static_cast<TypeParam>(10 * i);
      coordinates({ i, 1 }) = static_cast<TypeParam>(10 * i + 1);
    }
    DenseTensor source({ 1 });
    source(0) = Cloud(coordinates);

    sb::Tensor<sb::NestedTensor<uint64_t>> selectionChildren({ 1, 3 });
    auto selection = [](std::initializer_list<uint64_t> values)
    {
      sb::Tensor<uint64_t> result({ values.size() });
      size_t i = 0;
      for (uint64_t value : values)
      {
        result(i++) = value;
      }
      return sb::NestedTensor<uint64_t>(std::move(result));
    };
    selectionChildren({ 0, 0 }) = selection({ 3, 1, 3 });
    selectionChildren({ 0, 1 }) = selection({});
    selectionChildren({ 0, 2 }) = selection({ 4, 0 });
    sb::NestedTensor<uint64_t> selections(
      std::move(selectionChildren), 1);

    const IndexedTensor indexed = sb::make_indexed_tensor(
      source, std::move(selections));
    const IndexedTensor view = indexed.transpose();

    std::stringstream output;
    sb::write(view, output);

    std::istringstream typedInput(output.str());
    IndexedTensor restored = sb::read<IndexedTensor>(typedInput);
    EXPECT_EQ(restored.shape(), (std::vector<size_t>{ 3, 1 }));
    EXPECT_EQ(restored, view);
    EXPECT_TRUE(restored.has_indices());
    EXPECT_TRUE(restored.flat(0).is_indexed());
    EXPECT_EQ(restored.flat(0).indices()(0), 3);
    EXPECT_EQ(restored.flat(0).indices()(1), 1);
    EXPECT_EQ(restored.flat(0).indices()(2), 3);
    EXPECT_EQ(restored.flat(1).n_points(), 0);
    EXPECT_EQ(
      restored.flat(0).coords().storage_owner(),
      restored.flat(2).coords().storage_owner());

    // Loading retains the shared indexed state and its one-time
    // materialization behavior on mutation.
    restored.writable_at({ 0, 0 }).mutable_coords()({ 0, 0 }) =
      static_cast<TypeParam>(-1);
    EXPECT_FALSE(restored.has_indices());
    EXPECT_EQ(restored({ 0, 0 })(0, 0), static_cast<TypeParam>(-1));
    EXPECT_EQ(restored({ 2, 0 })(0, 0), static_cast<TypeParam>(40));

    std::stringstream materializedOutput;
    sb::write(restored, materializedOutput);
    std::istringstream materializedInput(materializedOutput.str());
    IndexedTensor materializedRoundtrip =
      sb::read<IndexedTensor>(materializedInput);
    EXPECT_EQ(materializedRoundtrip, restored);
    EXPECT_FALSE(materializedRoundtrip.has_indices());

    std::istringstream untypedInput(output.str());
    auto any = sb::read_any_tensor(untypedInput);
    EXPECT_TRUE(std::holds_alternative<IndexedTensor>(any));
    EXPECT_EQ(std::get<IndexedTensor>(any), view);
  }

// ============================================================================
// Empty (scalar/0-d) tensor roundtrip
// ============================================================================

  TYPED_TEST(IoReadWriteTest, EmptyTensorRoundtrip)
  {
    using TensorT = sb::Tensor<TypeParam>;

    TensorT tensor;

    std::stringstream ss;
    sb::write(tensor, ss);

    std::istringstream iss(ss.str());
    auto retTensor = sb::read<TensorT>(iss);

    EXPECT_EQ(tensor, retTensor);
  }

// ============================================================================
// Error: unrecognized file format (bad magic bytes)
// ============================================================================

  TYPED_TEST(IoReadWriteTest, ThrowsOnBadMagicBytes)
  {
    using TensorT = sb::Tensor<TypeParam>;

    std::istringstream iss("this is not a valid sb file");
    EXPECT_THROW(sb::read<TensorT>(iss), std::runtime_error);
  }

// ============================================================================
// Error: wrong format version
// ============================================================================

  TYPED_TEST(IoReadWriteTest, ThrowsOnWrongFormatVersion)
  {
    using TensorT = sb::Tensor<TypeParam>;

    // Write a valid tensor
    TensorT tensor({ 2 });
    tensor(0) = TypeParam(1);
    tensor(1) = TypeParam(2);

    std::stringstream ss;
    sb::write(tensor, ss);

    // Patch the format version in the stream. The header is:
    // "\1MPCF" (legacy magic, 5 bytes) + endianness (1 byte) + format version (sizeof(int) bytes)
    std::string data = ss.str();
    constexpr size_t versionOffset = 6; // after "\1MPCF" (legacy magic) + "e"/"E"
    sb::int32_t badVersion = 9999;
    std::memcpy(data.data() + versionOffset, &badVersion, sizeof(sb::int32_t));

    std::istringstream iss(data);
    EXPECT_THROW(sb::read<TensorT>(iss), std::runtime_error);
  }

// ============================================================================
// Error: truncated stream
// ============================================================================

  TYPED_TEST(IoReadWriteTest, ThrowsOnTruncatedStream)
  {
    using TensorT = sb::Tensor<TypeParam>;

    TensorT tensor({ 4 });
    sb::walk(tensor, [&tensor](const std::vector<size_t>& idx)
    {
      tensor(idx) = static_cast<TypeParam>(idx[0]);
    });

    std::stringstream ss;
    sb::write(tensor, ss);

    // Truncate to half the data
    auto data = ss.str();
    data = data.substr(0, data.size() / 2);

    std::istringstream iss(data);
    EXPECT_THROW(sb::read<TensorT>(iss), std::runtime_error);
  }

// ============================================================================
// Error: wrong tensor type on read
// ============================================================================

  TYPED_TEST(IoReadWriteTest, ThrowsOnTensorTypeMismatch)
  {
    // Write float32 tensor, try to read as float64 (and vice versa)
    using WriteT = sb::Tensor<sb::float32_t>;
    using ReadT = sb::Tensor<sb::float64_t>;

    WriteT tensor({ 2 });
    tensor(0) = 1.0f;
    tensor(1) = 2.0f;

    std::stringstream ss;
    sb::write(tensor, ss);

    std::istringstream iss(ss.str());
    EXPECT_THROW(sb::read<ReadT>(iss), std::runtime_error);
  }

// ============================================================================
// write produces non-empty output with valid magic bytes
// ============================================================================

  TYPED_TEST(IoReadWriteTest, WrittenDataStartsWithMagicBytes)
  {
    using TensorT = sb::Tensor<TypeParam>;

    TensorT tensor({ 2 });
    tensor(0) = TypeParam(0);
    tensor(1) = TypeParam(1);

    std::stringstream ss;
    sb::write(tensor, ss);

    auto data = ss.str();
    ASSERT_GE(data.size(), 5u);
    EXPECT_EQ(data[0], '\1');
    EXPECT_EQ(data[1], 'M');
    EXPECT_EQ(data[2], 'P');
    EXPECT_EQ(data[3], 'C');
    EXPECT_EQ(data[4], 'F');
  }

// ============================================================================
// Multiple write/read cycles (data survives two roundtrips)
// ============================================================================

  TYPED_TEST(IoReadWriteTest, TwoRoundtrips)
  {
    using TensorT = sb::Tensor<TypeParam>;

    TensorT tensor({ 3 });
    tensor(0) = TypeParam(1);
    tensor(1) = TypeParam(2);
    tensor(2) = TypeParam(3);

    std::stringstream ss1;
    sb::write(tensor, ss1);

    std::istringstream iss1(ss1.str());
    auto tensor2 = sb::read<TensorT>(iss1);

    std::stringstream ss2;
    sb::write(tensor2, ss2);

    std::istringstream iss2(ss2.str());
    auto tensor3 = sb::read<TensorT>(iss2);

    EXPECT_EQ(tensor, tensor3);
  }

  TYPED_TEST(IoReadWriteTest, ThrowsOnFutureFormatVersion)
  {
    using TensorT = sb::Tensor<TypeParam>;

    TensorT tensor({ 2 });
    tensor(0) = TypeParam(1);
    tensor(1) = TypeParam(2);

    std::stringstream ss;
    sb::write(tensor, ss);

    // Patch format version to something far in the future
    std::string data = ss.str();
    constexpr size_t versionOffset = 6;
    sb::int32_t futureVersion = 9999;
    std::memcpy(data.data() + versionOffset, &futureVersion, sizeof(sb::int32_t));

    std::istringstream iss(data);
    EXPECT_THROW(sb::read<TensorT>(iss), std::runtime_error);
  }

} // namespace
