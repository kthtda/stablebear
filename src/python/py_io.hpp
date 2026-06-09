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


#ifndef STABLEBEAR_PY_IO_H
#define STABLEBEAR_PY_IO_H

#include "pybind.hpp"
#include <iostream>
#include <streambuf>
#include <string>
#include <cstdio>

namespace py = pybind11;

namespace sb_py
{
  class PythonIStreamBuf : public std::streambuf {
  public:
    explicit PythonIStreamBuf(py::object file) : file_(file) {}
  protected:
    int_type underflow() override {
      auto bytes = file_.attr("read")(4096);
      auto view = py::bytes(bytes);
      buffer_ = std::string(view);
      if (buffer_.empty()) return traits_type::eof();
      setg(buffer_.data(), buffer_.data(), buffer_.data() + buffer_.size());
      return traits_type::to_int_type(buffer_[0]);
    }
  private:
    py::object file_;
    std::string buffer_;
  };

  class PythonOStreamBuf : public std::streambuf {
  public:
    explicit PythonOStreamBuf(py::object file) : file_(file) {}
  protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
      file_.attr("write")(py::bytes(s, n));
      return n;
    }
    int_type overflow(int_type c) override {
      if (c != EOF) {
        char ch = static_cast<char>(c);
        file_.attr("write")(py::bytes(&ch, 1));
      }
      return c;
    }
  private:
    py::object file_;
  };

  void register_io(pybind11::module_& m);
}

#endif //STABLEBEAR_PY_IO_H
