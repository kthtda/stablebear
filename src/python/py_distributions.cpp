#include "py_distributions.hpp"

#include "py_np_support.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sbear/random/distribution.hpp>
#include <sbear/walk.hpp>

#include <cmath>

namespace py = pybind11;

namespace
{
  using Values = py::array_t<long double, py::array::forcecast>;

  /// Evaluate weights elementwise, preserving the input shape.
  template <typename DistributionT>
  py::array_t<double> evaluate(const DistributionT& distribution, const Values& values)
  {
    const NumpyTensor<long double> input(values);
    py::array_t<double> result(std::vector<py::ssize_t>(values.shape(), values.shape() + values.ndim()));
    NumpyTensor<double> output(result);
    sb::walk(input, [&](const std::vector<size_t>& index) {
      if (!std::isfinite(input(index)))
        throw std::invalid_argument("values must be finite");
      output(index) = static_cast<double>(distribution.weight(input(index)));
    });
    return result;
  }
}

void sb_py::register_distributions(py::module_& m)
{
  using namespace sb::random;
  py::class_<Gaussian>(m, "_Gaussian")
    .def(py::init<double, double>(), py::arg("mean"), py::arg("std"))
    .def_property_readonly("mean", &Gaussian::mean)
    .def_property_readonly("std", &Gaussian::standard_deviation)
    .def("evaluate", &evaluate<Gaussian>, py::arg("values"));
  py::class_<Uniform>(m, "_Uniform")
    .def(py::init<double, double>(), py::arg("start"), py::arg("end"))
    .def_property_readonly("start", &Uniform::start)
    .def_property_readonly("end", &Uniform::end)
    .def("evaluate", &evaluate<Uniform>, py::arg("values"));
  py::class_<Mixture>(m, "_Mixture")
    .def(py::init<const std::vector<std::pair<double, Distribution>>&>(), py::arg("terms"))
    .def_property_readonly("terms", &Mixture::terms)
    .def("evaluate", &evaluate<Mixture>, py::arg("values"));
}
