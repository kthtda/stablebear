#include "py_plf.hpp"

#include <nanobind/ndarray.h>

#include <sbear/functional/plf.hpp>
#include "../pypcf_support.hpp"
#include "../py_np_support.hpp"

#include <stdexcept>
#include <vector>

namespace nb = nanobind;

namespace
{

  /// A Plf from an (n, 2) array of (time, value) rows. Throws
  /// std::invalid_argument, a ValueError in Python, for any other shape or
  /// for invalid breakpoints.
  template <typename Tt, typename Tv>
  sb::Plf<Tt, Tv> construct_plf(sb_py::NpIn<Tt> arr)
  {
    if (arr.ndim() != 2 || arr.shape(1) != 2)
    {
      throw std::invalid_argument("PLF breakpoints should be an (n, 2) array (supplied shape is "
                                  + ::shape_to_string(arr) + ").");
    }

    NumpyTensor<const Tt> data(arr);
    std::vector<sb::TimePoint<Tt, Tv>> points(arr.shape(0));
    for (size_t i = 0; i < points.size(); ++i)
    {
      points[i].t = data(i, 0);
      points[i].v = data(i, 1);
    }
    return sb::Plf<Tt, Tv>(std::move(points));
  }

  template <typename Tt, typename Tv>
  void register_bindings(nb::module_& m, const std::string& suffix)
  {
    using TPlf = sb::Plf<Tt, Tv>;

    nb::class_<TPlf> cls(m, ("Plf" + suffix).c_str());
    sb_py::bind_array_view(cls, [](const TPlf& self, nb::handle owner) {
      return sb::detail::to_numpy(self, owner);
    });
    cls
      .def(nb::init<>())
      .def("__init__", [](TPlf* self, sb_py::NpIn<Tt> arr) {
        new (self) TPlf(construct_plf<Tt, Tv>(arr));
      })
      .def("size", [](const TPlf& self) { return self.size(); })
      ;
  }

}

void sb_py::register_plf(nb::module_& m)
{
  register_bindings<sb::float32_t, sb::float32_t>(m, "_f32_f32");
  register_bindings<sb::float64_t, sb::float64_t>(m, "_f64_f64");
}
