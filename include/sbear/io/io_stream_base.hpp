#ifndef STABLEBEAR_IO_STREAM_BASE_H
#define STABLEBEAR_IO_STREAM_BASE_H


#include "../config.hpp"

#include <cstddef>
#include <vector>
#include <iostream>
#include <string_view>

//#define SB_IO_DEBUG

namespace sb::io::detail
{
  template <typename CharT, typename Traits>
  void assert_not_bad(std::basic_ios<CharT, Traits>& stream)
  {
    if (stream.bad())
    {
      throw std::runtime_error("Bad stream.");
    }
  }

  inline void write_binary_string(std::ostream& os, std::string_view str)
  {
    os.write(str.data(), str.size());
    assert_not_bad(os);
  }

  inline std::string read_binary_string(std::istream& is, std::streamsize nBytes)
  {
    std::string ret;
    ret.resize(nBytes);
    is.read(ret.data(), nBytes);
    if (is.gcount() != nBytes)
    {
      throw std::runtime_error("Unexpected EOF");
    }
    return ret;
  }

  template <typename T>
  void write_bytes(std::ostream& os, const T& v)
  {
#ifdef SB_IO_DEBUG
    std::cout << "WRITE " << sizeof(T) << " B: " << v <<  '\n';
    auto * c = reinterpret_cast<const char*>(&v);
    auto sz = sizeof(v);
    std::vector<unsigned char> chars;
    for (auto* cc = c; cc != c + sz; ++cc)
    {
      chars.push_back(*cc);
    }
#endif
    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
    assert_not_bad(os);
  }

  template <typename T>
  T read_bytes(std::istream& is)
  {
#ifdef SB_IO_DEBUG
    std::cout << "READ " << sizeof(T) << " B: ";
#endif
    T v;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    assert_not_bad(is);
    if (is.gcount() != sizeof(v))
    {
      throw std::runtime_error("Incorrect number of bytes returned (file may be corrupted)");
    }
#ifdef SB_IO_DEBUG
    std::cout << v << '\n';
#endif
    return v;
  }

  template <std::forward_iterator FwdIt>
  void write_length(std::ostream& os, FwdIt begin, FwdIt end)
  {
    auto len = std::distance(begin, end);
    if (len < 0)
    {
      throw std::runtime_error("Negative length container (this is a bug, please report it!).");
    }
    write_bytes<sb::uint64_t>(os, len);
  }

  inline sb::uint64_t read_length(std::istream& is)
  {
    return read_bytes<sb::uint64_t>(is);
  }

  template <std::forward_iterator FwdIt>
  void write_bytes(std::ostream& os, FwdIt begin, FwdIt end)
  {
    write_length(os, begin, end);
    for (auto it = begin; it != end; ++it)
    {
      write_bytes<typename FwdIt::value_type>(os, *it);
    }
  }

  inline void write_string(std::ostream& os, const std::string& str)
  {
    write_bytes<uint64_t>(os, str.length());
    os << str;
  }

  inline std::string read_string(std::istream& is)
  {
    // Purely artificial limit to avoid corrupted reads resulting in massive strings
    static constexpr const uint64_t maxLen = 32767;

    auto len = read_bytes<uint64_t>(is);
    if (len > maxLen)
    {
      throw std::runtime_error("Read string of length " + std::to_string(len) + " which exceeds the " + std::to_string(maxLen) + " limit (this is likely a corrupted file).");
    }

    auto lenss = static_cast<std::streamsize>(len);
    std::string ret(len, '\0');
    is.read(ret.data(), lenss);
    if (is.gcount() != lenss)
    {
      throw std::runtime_error("Unexpected EOF");
    }

    return ret;
  }

  template <std::forward_iterator FwdIt>
  void write_elements(std::ostream& os, FwdIt begin, FwdIt end);

  template <ArithmeticType T>
  T read_element(std::istream& is);

  template <ArithmeticType T>
  void write_element(std::ostream& os, T val);

  template <typename T, typename AT = std::allocator<T>>
  std::vector<T, AT> read_vector(std::istream& is);
}

#endif // STABLEBEAR_IO_STREAM_BASE_H
