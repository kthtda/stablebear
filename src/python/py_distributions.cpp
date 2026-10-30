#include "py_distributions.hpp"

#include "py_np_support.hpp"

#include <sbear/random/distribution.hpp>
#include <sbear/walk.hpp>

#include <cmath>

namespace nb = nanobind;

namespace
{
  /// Evaluate weights elementwise, preserving the input shape. Values are
  /// read as long double, so that 64-bit integers keep their precision.
  template <typename DistributionT, typename ValueT>
  sb_py::NpArray<double> evaluate(const DistributionT& distribution, sb_py::NpIn<ValueT> values)
  {
    const NumpyTensor<const ValueT> input(values);
    auto result = sb_py::new_array<double>(input.shape());
    NumpyTensor<double> output(result);
    sb::walk(input, [&](const std::vector<size_t>& index) {
      const auto value = static_cast<long double>(input(index));
      if (!std::isfinite(value))
        throw std::invalid_argument("values must be finite");
      output(index) = static_cast<double>(distribution.weight(value));
    });
    return result;
  }

  /// Bind evaluate for float64, int64, and uint64 values; other arrays are
  /// converted to float64. float64 comes first so that a conversion never
  /// truncates floats to integers.
  template <typename DistributionT, typename Cls>
  void bind_evaluate(Cls& cls)
  {
    cls.def("evaluate", &evaluate<DistributionT, double>, nb::arg("values"))
      .def("evaluate", &evaluate<DistributionT, int64_t>, nb::arg("values"))
      .def("evaluate", &evaluate<DistributionT, uint64_t>, nb::arg("values"));
  }
}

void sb_py::register_distributions(nb::module_& m)
{
  using namespace sb::random;
  nb::class_<Gaussian> gaussian(m, "_Gaussian");
  bind_evaluate<Gaussian>(gaussian);
  gaussian
    .def(nb::init<double, double>(), nb::arg("mean"), nb::arg("std"))
    .def_prop_ro("mean", &Gaussian::mean)
    .def_prop_ro("std", &Gaussian::standard_deviation);
  nb::class_<Uniform> uniform(m, "_Uniform");
  bind_evaluate<Uniform>(uniform);
  uniform
    .def(nb::init<double, double>(), nb::arg("start"), nb::arg("end"))
    .def_prop_ro("start", &Uniform::start)
    .def_prop_ro("end", &Uniform::end);
  nb::class_<Mixture> mixture(m, "_Mixture");
  bind_evaluate<Mixture>(mixture);
  mixture
    .def(nb::init<const std::vector<std::pair<double, Distribution>>&>(), nb::arg("terms"))
    .def_prop_ro("terms", &Mixture::terms);
}
