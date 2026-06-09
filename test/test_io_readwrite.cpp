// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

// ============================================================================
// Format version backward compatibility
// ============================================================================

  TYPED_TEST(IoReadWriteTest, ReadsFormatVersion1WithoutPlatformField)
  {
    using TensorT = sb::Tensor<TypeParam>;

    // Write a v2 tensor, then patch it back to v1 by removing the platform field
    TensorT tensor({ 3 });
    tensor(0) = TypeParam(1);
    tensor(1) = TypeParam(2);
    tensor(2) = TypeParam(3);

    std::stringstream ss;
    sb::write(tensor, ss);

    // The v2 header is: "\1MPCF" (legacy magic, 5) + endianness (1) + version (4) + version_string + date_string + platform_string + ...
    // We need to patch version to 1 and remove the platform string.
    std::string data = ss.str();
    constexpr size_t versionOffset = 6; // after "\1MPCF" (legacy magic) + "e"/"E"

    // Read the v2 header to find where the platform string starts and ends
    std::istringstream probe(data);
    sb::io::detail::read_binary_string(probe, 5); // header id
    sb::io::detail::read_binary_string(probe, 1); // endianness
    sb::io::detail::read_bytes<int>(probe);        // format version
    sb::io::detail::read_string(probe);            // version string
    sb::io::detail::read_string(probe);            // date string
    auto beforePlatform = probe.tellg();
    sb::io::detail::read_string(probe);            // platform string
    auto afterPlatform = probe.tellg();

    // Build a v1 stream: everything before platform + everything after platform, with version patched to 1
    std::string v1data = data.substr(0, beforePlatform) + data.substr(afterPlatform);
    sb::int32_t v1 = 1;
    std::memcpy(v1data.data() + versionOffset, &v1, sizeof(sb::int32_t));

    std::istringstream iss(v1data);
    auto retTensor = sb::read<TensorT>(iss);
    EXPECT_EQ(tensor, retTensor);
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
