/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef STABLEBEAR_PY_NP_SUPPORT_H
#define STABLEBEAR_PY_NP_SUPPORT_H

#include <pybind11/numpy.h>

#include <sbear/concepts.hpp>

#include <string>
#include <sstream>
#include <algorithm>

template <typename T>
std::string shape_to_string(pybind11::array_t<T> arr)
{
  std::stringstream ss;

  ss << "(";
  for (auto i = 0; i < arr.ndim(); ++i)
  {
    if (i != 0)
    {
      ss << ", ";
    }
    ss << arr.shape(i);
  }
  ss << ")";
  return ss.str();
}

template <typename T>
T& get_element(pybind11::array_t<T>& arr, const std::vector<pybind11::ssize_t>& idx)
{
  auto offset = std::inner_product(idx.begin(), idx.end(), arr.strides(), 0_uz);
  offset /= arr.itemsize();
  return *(static_cast<T*>(arr.request().ptr) + offset);
}

template <typename T>
class NumpyTensor
{
public:
  using value_type = T;

  explicit NumpyTensor(pybind11::array_t<T> arr)
    : m_arr(arr)
  { }

  [[nodiscard]] std::vector<size_t> shape() const
  {
    return std::vector<size_t>(m_arr.shape(), m_arr.shape() + m_arr.ndim());
  }

  [[nodiscard]] size_t shape(size_t i) const
  {
    return m_arr.shape(i);
  }

  [[nodiscard]] std::vector<size_t> strides() const
  {
    std::vector<size_t> s;
    s.resize(m_arr.ndim());
    std::transform(m_arr.strides(), m_arr.strides() + m_arr.ndim(), s.begin(), [this](pybind11::ssize_t n) {
      return static_cast<size_t>(n / m_arr.itemsize());
    });
    return s;
  }

  [[nodiscard]] size_t rank() const
  {
    return m_arr.ndim();
  }

  [[nodiscard]] size_t size() const
  {
    return static_cast<size_t>(m_arr.size());
  }

  [[nodiscard]] T& operator()(const std::vector<std::size_t>& idx)
  {
    auto offset = std::inner_product(idx.begin(), idx.end(), m_arr.strides(), pybind11::ssize_t{0});
    offset /= m_arr.itemsize();
    return *(m_arr.mutable_data() + offset);
  }

  [[nodiscard]] const T& operator()(const std::vector<std::size_t>& idx) const
  {
    auto offset = std::inner_product(idx.begin(), idx.end(), m_arr.strides(), pybind11::ssize_t{0});
    offset /= m_arr.itemsize();
    return *(m_arr.data() + offset);
  }

  template <typename... Ix>
  [[nodiscard]] T& operator()(Ix... index)
  {
    return m_arr.mutable_at(std::forward<Ix>(index)...);
  }

  template <typename... Ix>
  [[nodiscard]] const T& operator()(Ix... index) const
  {
    return m_arr.at(std::forward<Ix>(index)...);
  }

private:
  pybind11::array_t<T> m_arr;
};

static_assert(sb::IsTensor<NumpyTensor<int>>);

#endif //STABLEBEAR_PY_NP_SUPPORT_H