#ifndef STABLEBEAR_PY_IO_H
#define STABLEBEAR_PY_IO_H

#include "nb.hpp"

#include <sbear/io.hpp>

#include <iostream>
#include <streambuf>
#include <string>
#include <cstdio>

namespace nb = nanobind;

namespace sb_py
{
  class PythonIStreamBuf : public std::streambuf {
  public:
    explicit PythonIStreamBuf(nb::object file) : file_(file) {}
  protected:
    int_type underflow() override {
      auto bytes = nb::cast<nb::bytes>(file_.attr("read")(4096));
      buffer_.assign(bytes.c_str(), bytes.size());
      if (buffer_.empty()) return traits_type::eof();
      setg(buffer_.data(), buffer_.data(), buffer_.data() + buffer_.size());
      return traits_type::to_int_type(buffer_[0]);
    }
  private:
    nb::object file_;
    std::string buffer_;
  };

  class PythonOStreamBuf : public std::streambuf {
  public:
    explicit PythonOStreamBuf(nb::object file) : file_(file) {}
  protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
      file_.attr("write")(nb::bytes(s, n));
      return n;
    }
    int_type overflow(int_type c) override {
      if (c != EOF) {
        char ch = static_cast<char>(c);
        file_.attr("write")(nb::bytes(&ch, 1));
      }
      return c;
    }
  private:
    nb::object file_;
  };

  /// Write @p tensor to the Python binary file @p file, as sb.save does.
  template <typename TensorT>
  void save_tensor(const TensorT& tensor, nb::object file)
  {
    PythonOStreamBuf buf(file);
    std::ostream os(&buf);
    sb::write(tensor, os);
  }

  /// Write the standalone @p object to the Python binary file @p file.
  template <typename T>
  void save_object(const T& object, nb::object file)
  {
    PythonOStreamBuf buf(file);
    std::ostream os(&buf);
    sb::write_object(object, os);
  }

  void register_io(nb::module_& m);
}

#endif //STABLEBEAR_PY_IO_H
