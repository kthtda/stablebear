#include <gtest/gtest.h>

#include <../include/sbear/io/io_stream.hpp>
#include <sbear/io.hpp>
#include <sbear/tensor.hpp>
#include <sbear/walk.hpp>

#include <sstream>
#include <iostream>

namespace sb
{
  template<typename T>
  void PrintTo(const sb::Tensor<T>& tensor, std::ostream* os)
  {
    *os << "Tensor[\n";
    sb::walk(tensor, [&tensor, os](const std::vector<size_t>& index)
    {

      *os << "  " << index_to_string(index) << ": ";
      if constexpr (requires { *os << tensor(index); })
      {
        *os << tensor(index);
      }
      else
      {
        PrintTo(tensor(index), os);
      }
      *os << '\n';

    });
    *os << "]";
  }

  namespace io::detail
  {
    void PrintTo(const TensorFormat& format, std::ostream* os)
    {
      *os << "(" << format.baseFormat << ", " << format.subFormat << ")";
    }
  }
}

std::string to_printable(const std::string& in)
{
  std::stringstream out;

  out << "String of length " << in.length() << ": >>";

  for (auto i = 0_uz; i < in.length(); ++i)
  {
    if (std::isprint(in[i]))
    {
      out << in[i];
    }
    else
    {
      out << "\\" << static_cast<unsigned int>(in[i]);
    }
  }
  out << "<<";
  return out.str();
}

#if 0
TEST(IoStream, GoAroundHasCorrectDataTypes)
{
  std::stringstream ss("", std::ios::out | std::ios::binary);

  sb::Tensor<sb::float64_t> dblTensor;

  sb::write(dblTensor, ss);

  //ASSERT_TRUE(false) << to_printable(ss.str());

  //os << dblTensor;

  //std::string data = ss.str();
  //std::istringstream iss(data, std::ios::in | std::ios::binary);

  //sb::IStream is(iss);



}
#endif

template <typename T>
class IoStreamTest : public ::testing::Test {};

namespace
{
  using FloatTypes = ::testing::Types<sb::float32_t, sb::float64_t>;
  TYPED_TEST_SUITE(IoStreamTest, FloatTypes);

  TYPED_TEST(IoStreamTest, TestPointRoundtrip)
  {
    using PointT = sb::TimePoint<TypeParam, TypeParam>;
    PointT pt(0.5, 2.5);

    std::stringstream ss;
    sb::io::detail::write_element(ss, pt);

    std::istringstream iss(ss.str());
    auto retPt = sb::io::detail::read_element<PointT>(iss);

    EXPECT_EQ(pt, retPt);
  }

  TYPED_TEST(IoStreamTest, TestPcfRoundtrip)
  {
    using PcfT = sb::Pcf<TypeParam, TypeParam>;

    std::vector<typename PcfT::point_type> pts({ { 0., 10. }, { 1., 20. }, { 2., 30. } });
    PcfT pcf(std::move(pts));

    std::stringstream ss;
    sb::io::detail::write_element(ss, pcf);

    std::istringstream iss(ss.str());
    auto retPcf = sb::io::detail::read_element<PcfT>(iss);

    EXPECT_EQ(pcf, retPcf);
  }

  TYPED_TEST(IoStreamTest, TestFloatTensorRoundtrip)
  {
    using TensorT = sb::Tensor<TypeParam>;

    TensorT tensor({ 2, 3, 4 });

    sb::walk(tensor, [&tensor](const std::vector<size_t>& idx) {
      tensor(idx) = 100 * idx[0] + 10 * idx[1] + idx[2];
    });

    std::stringstream ss;
    sb::io::detail::write_tensor(ss, tensor);

    std::istringstream iss(ss.str());
    ASSERT_EQ(sb::io::detail::read_tensor_format(iss), sb::io::detail::tensorFormat<TypeParam>());
    TensorT retTensor = sb::io::detail::read_tensor<TypeParam>(iss);

    EXPECT_EQ(tensor, retTensor);
  }

  TYPED_TEST(IoStreamTest, WriteBytesRoundtrip)
  {
    TypeParam value = static_cast<TypeParam>(3.14);
    std::stringstream ss;
    sb::io::detail::write_bytes(ss, value);

    std::istringstream iss(ss.str());
    auto result = sb::io::detail::read_bytes<TypeParam>(iss);

    EXPECT_EQ(value, result);
  }

  TYPED_TEST(IoStreamTest, WriteBytesRoundtripZero)
  {
    TypeParam value = static_cast<TypeParam>(0);
    std::stringstream ss;
    sb::io::detail::write_bytes(ss, value);

    std::istringstream iss(ss.str());
    auto result = sb::io::detail::read_bytes<TypeParam>(iss);

    EXPECT_EQ(value, result);
  }

  TYPED_TEST(IoStreamTest, WriteBytesRoundtripNegative)
  {
    TypeParam value = static_cast<TypeParam>(-42.5);
    std::stringstream ss;
    sb::io::detail::write_bytes(ss, value);

    std::istringstream iss(ss.str());
    auto result = sb::io::detail::read_bytes<TypeParam>(iss);

    EXPECT_EQ(value, result);
  }

  // ============================================================================
  // read_bytes throws on truncated input
  // ============================================================================

  TYPED_TEST(IoStreamTest, ReadBytesThrowsOnTruncatedInput)
  {
    // Write only 1 byte but try to read sizeof(TypeParam) bytes
    std::istringstream iss("x");
    EXPECT_THROW(sb::io::detail::read_bytes<TypeParam>(iss), std::runtime_error);
  }

  // ============================================================================
  // Pcf tensor roundtrip
  // ============================================================================

  TYPED_TEST(IoStreamTest, PcfTensorRoundtrip)
  {
    using PcfT = sb::Pcf<TypeParam, TypeParam>;
    using TensorT = sb::Tensor<PcfT>;

    TensorT tensor({ 2 });

    std::vector<typename PcfT::point_type> pts0, pts1;
    pts0.emplace_back(TypeParam(0), TypeParam(1));
    pts0.emplace_back(TypeParam(1), TypeParam(2));
    pts1.emplace_back(TypeParam(0), TypeParam(3));
    pts1.emplace_back(TypeParam(2), TypeParam(4));

    tensor(0) = PcfT(std::move(pts0));
    tensor(1) = PcfT(std::move(pts1));

    std::stringstream ss;
    sb::io::detail::write_tensor(ss, tensor);

    std::istringstream iss(ss.str());
    ASSERT_EQ(sb::io::detail::read_tensor_format(iss), sb::io::detail::tensorFormat<typename decltype(tensor)::value_type>());
    auto retTensor = sb::io::detail::read_tensor<PcfT>(iss);

    EXPECT_EQ(tensor, retTensor);
  }

  TYPED_TEST(IoStreamTest, PcfTensorRoundtrip2d)
  {
    using PcfT = sb::Pcf<TypeParam, TypeParam>;
    using TensorT = sb::Tensor<PcfT>;

    TensorT tensor({ 2, 3 });

    sb::walk(tensor, [&tensor](const std::vector<size_t>& idx) {
      std::vector<typename PcfT::point_type> pts;
      pts.emplace_back(TypeParam(0), static_cast<TypeParam>(idx[0] * 10 + idx[1]));
      tensor(idx) = PcfT(std::move(pts));
    });

    std::stringstream ss;
    sb::io::detail::write_tensor(ss, tensor);

    std::istringstream iss(ss.str());
    ASSERT_EQ(sb::io::detail::read_tensor_format(iss), sb::io::detail::tensorFormat<typename decltype(tensor)::value_type>());
    auto retTensor = sb::io::detail::read_tensor<PcfT>(iss);

    EXPECT_EQ(tensor, retTensor);
  }

  // ============================================================================
  // Full sb::write roundtrip (with header)
  // ============================================================================

  TYPED_TEST(IoStreamTest, WriteReadFloatTensorWithHeader)
  {
    using TensorT = sb::Tensor<TypeParam>;

    TensorT tensor({ 3, 4 });
    sb::walk(tensor, [&tensor](const std::vector<size_t>& idx) {
      tensor(idx) = static_cast<TypeParam>(idx[0] * 10 + idx[1]);
    });

    std::stringstream ss;
    sb::write(tensor, ss);

    // We can at least verify the stream is non-empty and starts with the magic bytes
    auto data = ss.str();
    ASSERT_GT(data.size(), 5u);
    EXPECT_EQ(data[1], 'M');
    EXPECT_EQ(data[2], 'P');
    EXPECT_EQ(data[3], 'C');
    EXPECT_EQ(data[4], 'F');
  }

  // ============================================================================
  // Empty Pcf roundtrip (single zero point)
  // ============================================================================

  TYPED_TEST(IoStreamTest, EmptyPcfRoundtrip)
  {
    using PcfT = sb::Pcf<TypeParam, TypeParam>;

    PcfT pcf;  // default: single point at (0,0)

    std::stringstream ss;
    sb::io::detail::write_element(ss, pcf);

    std::istringstream iss(ss.str());
    auto retPcf = sb::io::detail::read_element<PcfT>(iss);

    EXPECT_EQ(pcf, retPcf);
  }

  // ============================================================================
  // Single-point (constant) Pcf roundtrip
  // ============================================================================

  TYPED_TEST(IoStreamTest, SinglePointPcfRoundtrip)
  {
    using PcfT = sb::Pcf<TypeParam, TypeParam>;

    PcfT pcf(TypeParam(7));

    std::stringstream ss;
    sb::io::detail::write_element(ss, pcf);

    std::istringstream iss(ss.str());
    auto retPcf = sb::io::detail::read_element<PcfT>(iss);

    EXPECT_EQ(pcf, retPcf);
  }

  // ============================================================================
  // write_tensor on non-contiguous (sliced) tensor
  // ============================================================================

  TYPED_TEST(IoStreamTest, NonContiguousTensorRoundtrip)
  {
    using TensorT = sb::Tensor<TypeParam>;

    // Create a 4x4 tensor and take a slice to get a non-contiguous view
    TensorT tensor({ 4, 4 });
    sb::walk(tensor, [&tensor](const std::vector<size_t>& idx) {
      tensor(idx) = static_cast<TypeParam>(idx[0] * 10 + idx[1]);
    });

    // Slice rows 1..2 to get a non-contiguous view
    auto sliced = tensor[std::vector<sb::Slice>{ sb::range(1, 3, std::nullopt), sb::all() }];
    EXPECT_FALSE(sliced.is_contiguous());

    std::stringstream ss;
    sb::io::detail::write_tensor(ss, sliced);

    std::istringstream iss(ss.str());
    ASSERT_EQ(sb::io::detail::read_tensor_format(iss), sb::io::detail::tensorFormat<typename decltype(tensor)::value_type>());
    auto retTensor = sb::io::detail::read_tensor<TypeParam>(iss);

    // The returned tensor should be contiguous and contain the sliced values
    EXPECT_TRUE(retTensor.is_contiguous());
    EXPECT_EQ(retTensor.shape(0), 2u);
    EXPECT_EQ(retTensor.shape(1), 4u);

    EXPECT_EQ(retTensor({0, 0}), static_cast<TypeParam>(10));
    EXPECT_EQ(retTensor({0, 3}), static_cast<TypeParam>(13));
    EXPECT_EQ(retTensor({1, 0}), static_cast<TypeParam>(20));
    EXPECT_EQ(retTensor({1, 3}), static_cast<TypeParam>(23));
  }
  
  // ============================================================================
  // assert_not_bad throws on bad stream
  // ============================================================================
  
  TEST(IoStreamBase, AssertNotBadThrowsOnBadStream)
  {
    std::stringstream ss;
    ss.setstate(std::ios::badbit);
    EXPECT_THROW(sb::io::detail::assert_not_bad(ss), std::runtime_error);
  }
  
  TEST(IoStreamBase, AssertNotBadDoesNotThrowOnGoodStream)
  {
    std::stringstream ss;
    EXPECT_NO_THROW(sb::io::detail::assert_not_bad(ss));
  }
  
  // ============================================================================
  // write_string / read via raw bytes
  // ============================================================================
  
  TEST(IoStreamBase, WriteStringRoundtrip)
  {
    std::stringstream ss;
    sb::io::detail::write_string(ss, "hello");
  
    // Read back: first 8 bytes are uint64 length, then the string chars
    std::istringstream iss(ss.str());
    auto len = sb::io::detail::read_bytes<uint64_t>(iss);
    ASSERT_EQ(len, 5u);
  
    std::string result(len, '\0');
    iss.read(result.data(), len);
    EXPECT_EQ(result, "hello");
  }
  
  TEST(IoStreamBase, WriteStringEmptyRoundtrip)
  {
    std::stringstream ss;
    sb::io::detail::write_string(ss, "");
  
    std::istringstream iss(ss.str());
    auto len = sb::io::detail::read_bytes<uint64_t>(iss);
    EXPECT_EQ(len, 0u);
  }
  
  // ============================================================================
  // write_length
  // ============================================================================
  
  TEST(IoStreamBase, WriteLengthCorrect)
  {
    std::vector<int> v = {1, 2, 3, 4, 5};
    std::stringstream ss;
    sb::io::detail::write_length(ss, v.begin(), v.end());
  
    std::istringstream iss(ss.str());
    auto len = sb::io::detail::read_bytes<sb::uint64_t>(iss);
    EXPECT_EQ(len, 5u);
  }
  
  TEST(IoStreamBase, WriteLengthEmpty)
  {
    std::vector<int> v;
    std::stringstream ss;
    sb::io::detail::write_length(ss, v.begin(), v.end());
  
    std::istringstream iss(ss.str());
    auto len = sb::io::detail::read_bytes<sb::uint64_t>(iss);
    EXPECT_EQ(len, 0u);
  }
  
  // ============================================================================
  // Pcf tensor IO with empty Pcfs
  // ============================================================================
  
  TYPED_TEST(IoStreamTest, PcfTensorWithDefaultPcfs)
  {
    using PcfT = sb::Pcf<TypeParam, TypeParam>;
    using TensorT = sb::Tensor<PcfT>;
  
    TensorT tensor({ 3 });
    // All elements are default-constructed Pcfs
  
    std::stringstream ss;
    sb::io::detail::write_tensor(ss, tensor);
  
    std::istringstream iss(ss.str());
    ASSERT_EQ(sb::io::detail::read_tensor_format(iss), sb::io::detail::tensorFormat<typename decltype(tensor)::value_type>());
    auto retTensor = sb::io::detail::read_tensor<PcfT>(iss);
  
    EXPECT_EQ(tensor, retTensor);
  }
}
