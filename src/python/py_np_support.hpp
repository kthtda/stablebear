#ifndef STABLEBEAR_PY_NP_SUPPORT_H
#define STABLEBEAR_PY_NP_SUPPORT_H

#include "nb.hpp"

#include <nanobind/ndarray.h>

#include <sbear/concepts.hpp>
#include <sbear/config.hpp>

#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <sstream>
#include <vector>

namespace sb_py
{
  /// A NumPy array argument: any CPU array, converted to T if its dtype
  /// differs. Read-only arrays are accepted.
  template <typename T>
  using NpIn = nb::ndarray<const T, nb::device::cpu>;

  /// A NumPy array returned to Python.
  template <typename T>
  using NpArray = nb::ndarray<nb::numpy, T>;

  /// A new zero-initialized C-ordered NumPy array that owns its memory.
  template <typename T>
  NpArray<T> new_array(const std::vector<size_t>& shape)
  {
    const size_t n = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
    T* data = new T[n]();
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return NpArray<T>(data, shape.size(), shape.data(), owner);
  }

  /// A NumPy array viewing memory owned by `owner`, with strides in elements.
  template <typename T>
  NpArray<T> array_view(T* data, const std::vector<size_t>& shape, const std::vector<int64_t>& strides,
                        nb::handle owner)
  {
    return NpArray<T>(data, shape.size(), shape.data(), owner, strides.data());
  }

  /// Bind __array__ so that np.asarray(obj) gives a NumPy array viewing
  /// obj's memory. `view` takes the object and its Python handle, which the
  /// array keeps alive.
  template <typename Cls, typename View>
  void bind_array_view(Cls& cls, View view)
  {
    using T = typename Cls::Type;
    cls.def("__array__", [view](nb::handle self, nb::handle dtype, nb::handle copy) -> nb::object {
      nb::object array = nb::cast(view(nb::cast<const T&>(self), self));
      if (!dtype.is_none())
      {
        nb::object converted = array.attr("astype")(dtype, nb::arg("copy") = false);
        // copy=False forbids copying, as NumPy requires of __array__
        if (!converted.is(array) && nb::isinstance<nb::bool_>(copy) && !nb::cast<bool>(copy))
        {
          throw nb::value_error("Unable to avoid copy while creating an array as requested.");
        }
        array = converted;
      }
      if (nb::isinstance<nb::bool_>(copy) && nb::cast<bool>(copy))
      {
        array = array.attr("copy")();
      }
      return array;
    }, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none());
  }
}

template <typename Array>
std::string shape_to_string(const Array& arr)
{
  std::stringstream ss;

  ss << "(";
  for (size_t i = 0; i < arr.ndim(); ++i)
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

/// A strided view of an nb::ndarray as an sb tensor. T is const for a
/// read-only view. The array must outlive the view.
template <typename T>
class NumpyTensor
{
public:
  using value_type = std::remove_const_t<T>;

  template <typename... Args>
  explicit NumpyTensor(const nb::ndarray<Args...>& arr)
    : m_data(static_cast<T*>(arr.data()))
    , m_shape(arr.ndim())
    , m_strides(arr.ndim())
  {
    for (size_t i = 0; i < arr.ndim(); ++i)
    {
      m_shape[i] = arr.shape(i);
      m_strides[i] = arr.stride(i);
    }
  }

  [[nodiscard]] std::vector<size_t> shape() const
  {
    return m_shape;
  }

  [[nodiscard]] size_t shape(size_t i) const
  {
    return m_shape[i];
  }

  [[nodiscard]] std::vector<size_t> strides() const
  {
    return std::vector<size_t>(m_strides.begin(), m_strides.end());
  }

  [[nodiscard]] size_t rank() const
  {
    return m_shape.size();
  }

  [[nodiscard]] size_t size() const
  {
    return std::accumulate(m_shape.begin(), m_shape.end(), size_t{1}, std::multiplies<>());
  }

  [[nodiscard]] T& operator()(const std::vector<std::size_t>& idx) const
  {
    int64_t offset = 0;
    for (size_t i = 0; i < idx.size(); ++i)
    {
      offset += static_cast<int64_t>(idx[i]) * m_strides[i];
    }
    return m_data[offset];
  }

  template <typename... Ix>
  [[nodiscard]] T& operator()(Ix... index) const
  {
    int64_t offset = 0;
    size_t i = 0;
    ((offset += static_cast<int64_t>(index) * m_strides[i++]), ...);
    return m_data[offset];
  }

private:
  T* m_data;
  std::vector<size_t> m_shape;
  std::vector<int64_t> m_strides;
};

static_assert(sb::IsTensor<NumpyTensor<int>>);

#endif //STABLEBEAR_PY_NP_SUPPORT_H
