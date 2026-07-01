#include "py_subsample.hpp"

#include <sbear/sampling/subsample.hpp>

#include <string>
#include <utility>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PySubsampleBindings
  {
  public:
    using TensorT = sb::Tensor<T>;
    using Gen = sb::DefaultRandomGenerator;

    static py::tuple sample_subsets_gaussian(const TensorT& reference, const TensorT& query, T mean,
                                             T sigma, size_t sampleSize, size_t nInstances,
                                             bool replace, const Gen* gen)
    {
      return sample_subsets_impl(reference, query, sb::sampling::Gaussian<T>{mean, sigma}, sampleSize,
                                 nInstances, replace, gen);
    }

    static py::tuple sample_subsets_uniform(const TensorT& reference, const TensorT& query, T low,
                                            T high, size_t sampleSize, size_t nInstances,
                                            bool replace, const Gen* gen)
    {
      return sample_subsets_impl(reference, query, sb::sampling::Uniform<T>{low, high}, sampleSize,
                                 nInstances, replace, gen);
    }

    // ----- distance-matrix input: query is a tensor of reference row indices -----

    static py::tuple sample_subsets_distmat_gaussian(const sb::DistanceMatrix<T>& source,
                                                     const sb::Tensor<uint64_t>& query, T mean, T sigma,
                                                     size_t sampleSize, size_t nInstances, bool replace,
                                                     const Gen* gen)
    {
      return sample_subsets_distmat_impl(source, query, sb::sampling::Gaussian<T>{mean, sigma},
                                         sampleSize, nInstances, replace, gen);
    }

    static py::tuple sample_subsets_distmat_uniform(const sb::DistanceMatrix<T>& source,
                                                    const sb::Tensor<uint64_t>& query, T low, T high,
                                                    size_t sampleSize, size_t nInstances, bool replace,
                                                    const Gen* gen)
    {
      return sample_subsets_distmat_impl(source, query, sb::sampling::Uniform<T>{low, high}, sampleSize,
                                         nInstances, replace, gen);
    }

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<PySubsampleBindings>(m, ("Subsample" + suffix).c_str())
          .def_static("sample_subsets_gaussian", &sample_subsets_gaussian, py::arg("reference"),
                      py::arg("query"), py::arg("mean"), py::arg("sigma"), py::arg("sample_size"),
                      py::arg("n_instances"), py::arg("replace"),
                      py::arg("generator").none(true) = py::none())
          .def_static("sample_subsets_uniform", &sample_subsets_uniform, py::arg("reference"),
                      py::arg("query"), py::arg("low"), py::arg("high"), py::arg("sample_size"),
                      py::arg("n_instances"), py::arg("replace"),
                      py::arg("generator").none(true) = py::none())
          .def_static("sample_subsets_distmat_gaussian", &sample_subsets_distmat_gaussian,
                      py::arg("source"), py::arg("query"), py::arg("mean"), py::arg("sigma"),
                      py::arg("sample_size"), py::arg("n_instances"), py::arg("replace"),
                      py::arg("generator").none(true) = py::none())
          .def_static("sample_subsets_distmat_uniform", &sample_subsets_distmat_uniform,
                      py::arg("source"), py::arg("query"), py::arg("low"), py::arg("high"),
                      py::arg("sample_size"), py::arg("n_instances"), py::arg("replace"),
                      py::arg("generator").none(true) = py::none());
    }

  private:
    static const Gen& pick(const Gen* gen) { return gen ? *gen : sb::default_generator(); }

    template <typename ElemT>
    static py::tuple to_tuple(sb::sampling::SubsampleHandle<ElemT> handle)
    {
      return py::make_tuple(std::move(handle.task), std::move(handle.samples));
    }

    // Templated over the distribution functor (the "templated distribution"), like
    // pdist_impl<TOperation> in py_distance.cpp: pair the Euclidean filter with
    // @p distribution and launch the core stoppable sampler.
    template <typename DistF>
    static py::tuple sample_subsets_impl(const TensorT& reference, const TensorT& query,
                                         DistF distribution, size_t sampleSize, size_t nInstances,
                                         bool replace, const Gen* gen)
    {
      return to_tuple(sb::sampling::sample_subsets(
          sb::PointCloud<T>(reference), sb::PointCloud<T>(query), sb::sampling::EuclideanDistance<T>{},
          distribution, sampleSize, nInstances, replace, pick(gen), sb::default_executor()));
    }

    template <typename DistF>
    static py::tuple sample_subsets_distmat_impl(const sb::DistanceMatrix<T>& source,
                                                 const sb::Tensor<uint64_t>& query, DistF distribution,
                                                 size_t sampleSize, size_t nInstances, bool replace,
                                                 const Gen* gen)
    {
      return to_tuple(sb::sampling::sample_subsets_distmat(source, query, distribution, sampleSize,
                                                           nInstances, replace, pick(gen),
                                                           sb::default_executor()));
    }
  };

}

namespace sb_py
{
  void register_sampling_subsample(py::module_& m)
  {
    PySubsampleBindings<sb::float32_t>::register_bindings(m, "32");
    PySubsampleBindings<sb::float64_t>::register_bindings(m, "64");
  }
}
