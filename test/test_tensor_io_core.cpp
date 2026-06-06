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
  using sb::io::detail::tensorFormat;
  using sb::io::detail::getTensorFormat;

  // ============================================================================
  // TensorFormat mapping for supported core types
  // ============================================================================

  TEST(TensorIoCore, TensorFormatScalarAndCompositeTypes)
  {
    using sb::float32_t;
    using sb::float64_t;

    {
      auto fmt = tensorFormat<float32_t>();
      EXPECT_EQ(1, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<float64_t>();
      EXPECT_EQ(1, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<sb::Pcf<float32_t, float32_t>>();
      EXPECT_EQ(100, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<sb::Pcf<float64_t, float64_t>>();
      EXPECT_EQ(100, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<sb::PointCloud<float32_t>>();
      EXPECT_EQ(1001, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<sb::PointCloud<float64_t>>();
      EXPECT_EQ(1001, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<sb::ph::Barcode<float32_t>>();
      EXPECT_EQ(10000, fmt.baseFormat);
      EXPECT_EQ(32, fmt.subFormat);
    }
    {
      auto fmt = tensorFormat<sb::ph::Barcode<float64_t>>();
      EXPECT_EQ(10000, fmt.baseFormat);
      EXPECT_EQ(64, fmt.subFormat);
    }
  }

  TEST(TensorIoCore, TensorFormatThrowsOnUnsupportedType)
  {
    struct Unsupported {};
    EXPECT_THROW((void)tensorFormat<Unsupported>(), std::runtime_error);
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

    sb::io::detail::read_tensor_format(iss);

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

    sb::io::detail::read_tensor_format(iss);
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
    ASSERT_GE(all.size(), 8u);

    // Work on the payload that starts after the TensorFormat header
    std::string payload(all.begin() + 8, all.end());

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

} // namespace

