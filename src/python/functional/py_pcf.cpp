#include "py_pcf.hpp"
#include "../py_future.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <sbear/functional/pcf.hpp>
#include "../pypcf_support.hpp"
#include <sbear/algorithm.hpp>
#include <sbear/algorithms/functional/iterate_rectangles.hpp>
#include <sbear/executor.hpp>
#include <sbear/task.hpp>

#include "../py_np_support.hpp"

#include <iostream>
#include <sstream>

namespace py = pybind11;

namespace
{

#define STRINGIFY(x) #x

  template <typename Tt, typename Tv>
  class ReductionWrapper
  {
  public:
    using reduction_function = Tt(*)(Tt, Tt, Tv, Tv);
    explicit ReductionWrapper(unsigned long long addr) : m_fn(reinterpret_cast<reduction_function>(addr)) { }
    Tt operator()(Tt left, Tt right, Tv top, Tv bottom)
    {
      return m_fn(left, right, top, bottom);
    }
  private:
    reduction_function m_fn;
  };

  template <typename Tt, typename Tv>
  class Backend
  {
  public:
    static sb::Pcf<Tt, Tv> add(const sb::Pcf<Tt, Tv>& f, const sb::Pcf<Tt, Tv>& g)
    {
      return f + g;
    }

    static sb::Pcf<Tt, Tv> combine(const sb::Pcf<Tt, Tv>& f, const sb::Pcf<Tt, Tv>& g, unsigned long long cb)
    {
      ReductionWrapper<Tt, Tv> reduction(cb);
      return sb::combine(f, g,
        [&reduction](const sb::Rectangle<Tt, Tv>& rect) -> Tt {
          return reduction(rect.left, rect.right, rect.f_value, rect.g_value);
        });
    }

    static sb::Pcf<Tt, Tv> average(const std::vector<sb::Pcf<Tt, Tv>>& fs)
    {
      return sb::average(fs);
    }

    static sb::Pcf<Tt, Tv> parallel_reduce(const std::vector<sb::Pcf<Tt, Tv>>& fs, unsigned long long cb){ \
      ReductionWrapper<Tt, Tv> reduction(cb);
      return sb::parallel_reduce(fs.begin(), fs.end(),
        [&reduction](const sb::Rectangle<Tt, Tv>& rect) -> Tt
        {
          return reduction(rect.left, rect.right, rect.f_value, rect.g_value);
        });
    }

    static Tv single_l1_norm(const sb::Pcf<Tt, Tv>& f)
    {
      return sb::l1_norm(f);
    }

    static Tv single_l2_norm(const sb::Pcf<Tt, Tv>& f)
    {
      return sb::l2_norm(f);
    }

    static Tv single_lp_norm(const sb::Pcf<Tt, Tv>& f, /* let's stick with float64_t here to make life a bit easier */ sb::float64_t p)
    {
      return sb::lp_norm(f, Tv(p));
    }

    static Tv single_linfinity_norm(const sb::Pcf<Tt, Tv>& f)
    {
      return sb::linfinity_norm(f);
    }

    static std::vector<sb::Rectangle<Tt, Tv>> iterate_rectangles(const sb::Pcf<Tt, Tv>& f, const sb::Pcf<Tt, Tv>& g, Tt a, Tt b)
    {
      std::vector<sb::Rectangle<Tt, Tv>> result;
      sb::iterate_rectangles(f.points(), g.points(),
        [&result](const sb::Rectangle<Tt, Tv>& rect) {
          result.push_back(rect);
        }, a, b);
      return result;
    }

  };

  template <typename Tt, typename Tv>
  class PyBindings
  {
  public:
    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      using TPcf = sb::Pcf<Tt, Tv>;
      using point_type = typename TPcf::point_type;

      py::class_<sb::Pcf<Tt, Tv>>(m, ("Pcf" + suffix).c_str(), py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<>([](py::array_t<Tt> arr){ return sb::detail::construct_pcf<Tt, Tv>(arr); }))
        .def("get_time_type", [](TPcf& /* self */) -> std::string { return STRINGIFY(Tt); })
        .def("get_value_type", [](TPcf& /* self */) -> std::string { return STRINGIFY(Tv); })
        .def("debug_print", &TPcf::debug_print) \
        .def_buffer([](TPcf& self) { return sb::detail::to_numpy<sb::Pcf<Tt, Tv>>(self); })
        .def("size", [](const TPcf& self){ return self.points().size(); })
        .def("copy", [](const TPcf& self){ return TPcf(self); })
        .def("__add__", [](const TPcf& self, const TPcf& rhs) -> TPcf { return self + rhs; })
        .def("__add__", [](const TPcf& self, Tv c) -> TPcf { return self + c; })
        .def("__radd__", [](const TPcf& self, Tv c) -> TPcf { return c + self; })
        .def("__sub__", [](const TPcf& self, const TPcf& rhs) -> TPcf { return self - rhs; })
        .def("__sub__", [](const TPcf& self, Tv c) -> TPcf { return self - c; })
        .def("__rsub__", [](const TPcf& self, Tv c) -> TPcf { return c - self; })
        .def("__mul__", [](const TPcf& self, const TPcf& rhs) -> TPcf { return self * rhs; })
        .def("__mul__", [](const TPcf& self, Tv c) -> TPcf { return self * c; })
        .def("__rmul__", [](const TPcf& self, Tv c) -> TPcf { return c * self; })
        .def("__truediv__", [](const TPcf& self, const TPcf& rhs) -> TPcf { return self / rhs; })
        .def("__truediv__", [](const TPcf& self, Tv c) -> TPcf { return self / c; })
        .def("__rtruediv__", [](const TPcf& self, Tv c) -> TPcf { return c / self; })
        .def("__neg__", [](const TPcf& self) -> TPcf { return -self; })
        .def("__pow__", [](const TPcf& self, Tv c) -> TPcf {
          auto result = sb::pow(self, c);
          bool warn = false;
          for (const auto& pt : result.points())
          {
            if (std::isnan(pt.v) || std::isinf(pt.v))
            {
              warn = true;
              break;
            }
          }
          // PyErr_WarnEx returns -1 when the warning is escalated to an
          // exception (e.g. under `-W error` / simplefilter('error')); in that
          // case a Python exception is pending, so propagate it instead of
          // returning normally and letting pybind11 raise SystemError.
          if (warn && PyErr_WarnEx(PyExc_RuntimeWarning,
                "invalid or infinite value encountered in pcf pow", 1) != 0)
            throw py::error_already_set();
          return result;
        })
        .def("__call__", [](const TPcf& self, Tt t) -> Tv { return self.evaluate(t); })
        .def("__call__", [](const TPcf& self, py::array_t<Tt> times) -> py::array_t<Tv> {
          // Flatten to 1D for processing, remember original shape
          auto original_shape = std::vector<py::ssize_t>(times.shape(), times.shape() + times.ndim());
          auto flat_times = times.reshape({times.size()});
          NumpyTensor<Tt> in(flat_times);
          auto n = static_cast<size_t>(flat_times.size());

          // Argsort
          std::vector<size_t> order(n);
          std::iota(order.begin(), order.end(), 0);
          std::sort(order.begin(), order.end(), [&in](size_t a, size_t b) {
            return in(a) < in(b);
          });

          // Build sorted times and evaluate
          py::array_t<Tt> sorted_times({static_cast<py::ssize_t>(n)});
          NumpyTensor<Tt> sorted_in(sorted_times);
          for (size_t i = 0; i < n; ++i)
          {
            sorted_in(i) = in(order[i]);
          }

          py::array_t<Tv> sorted_result({static_cast<py::ssize_t>(n)});
          NumpyTensor<Tv> sorted_out(sorted_result);
          self.evaluate(sorted_in, sorted_out, n);

          // Unsort results back to original order
          py::array_t<Tv> flat_result({static_cast<py::ssize_t>(n)});
          NumpyTensor<Tv> out(flat_result);
          for (size_t i = 0; i < n; ++i)
          {
            out(order[i]) = sorted_out(i);
          }

          return flat_result.reshape(original_shape);
        })

        ;

      using RectT = sb::Rectangle<Tt, Tv>;
      py::class_<RectT>(m, ("Rectangle" + suffix).c_str())
        .def(py::init<>())
        .def(py::init<Tt, Tt, Tv, Tv>(), py::arg("left"), py::arg("right"), py::arg("f_value"), py::arg("g_value"))
        .def_readwrite("left", &RectT::left)
        .def_readwrite("right", &RectT::right)
        .def_readwrite("f_value", &RectT::f_value)
        .def_readwrite("g_value", &RectT::g_value)
        .def("__repr__", [](const RectT& r) {
          std::ostringstream os;
          os << "Rectangle(left=" << r.left << ", right=" << r.right
             << ", fv=" << r.f_value << ", gv=" << r.g_value << ")";
          return os.str();
        })
        .def("__eq__", &RectT::operator==)
        ;

      py::class_<Backend<Tt, Tv>> backend(m, ("Backend" + suffix).c_str());
      backend
        .def(py::init<>())
        .def_static("add", &Backend<Tt, Tv>::add)
        .def_static("combine", &Backend<Tt, Tv>::combine)
        .def_static("average", &Backend<Tt, Tv>::average)
        .def_static("parallel_reduce", &Backend<Tt, Tv>::parallel_reduce)

        .def_static("single_l1_norm", &Backend<Tt, Tv>::single_l1_norm)
        .def_static("single_l2_norm", &Backend<Tt, Tv>::single_l2_norm)
        .def_static("single_lp_norm", &Backend<Tt, Tv>::single_lp_norm)
        .def_static("single_linfinity_norm", &Backend<Tt, Tv>::single_linfinity_norm)

        .def_static("iterate_rectangles", &Backend<Tt, Tv>::iterate_rectangles,
          py::arg("f"), py::arg("g"),
          py::arg("a") = point_type::zero_time(),
          py::arg("b") = point_type::infinite_time())
        ;

      sb_py::register_bindings_future<TPcf>(m, suffix);
    }
  };

}

void sb_py::register_pcf(pybind11::module_& m)
{
  PyBindings<sb::float32_t, sb::float32_t>::register_bindings(m, "_f32_f32");
  PyBindings<sb::float64_t, sb::float64_t>::register_bindings(m, "_f64_f64");
  PyBindings<sb::int32_t, sb::int32_t>::register_bindings(m, "_i32_i32");
  PyBindings<sb::int64_t, sb::int64_t>::register_bindings(m, "_i64_i64");
}
