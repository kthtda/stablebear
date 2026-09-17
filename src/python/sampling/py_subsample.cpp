#include "py_subsample.hpp"

#include <sbear/sampling/subsample.hpp>

#include <pybind11/stl.h>  // std::variant caster

#include <string>
#include <utility>
#include <variant>

namespace py = pybind11;

namespace
{

  using sb::sampling::EuclideanDistance;
  using sb::sampling::Gaussian;
  using sb::sampling::Uniform;

  // ===========================================================================
  // Built-in functors
  //
  // The entry points accept exactly these functors, passed from Python as
  // descriptor objects. pybind11's variant caster converts the descriptor to
  // the matching alternative and std::visit dispatches to the core call
  // instantiated for that combination. Adding a built-in = add the core
  // functor, list it here, and register its descriptor class in
  // register_descriptors.
  // ===========================================================================

  template <typename T>
  using FilterVariant = std::variant<EuclideanDistance<T>>;

  template <typename T>
  using DistVariant = std::variant<Gaussian<T>, Uniform<T>>;

  /// Bindings for one floating-point precision, registered as a Python class
  /// (Subsample32/Subsample64) that namespaces the descriptor classes and the
  /// static entry points.
  template <typename T>
  class PySubsampleBindings
  {
  public:
    using TensorT = sb::Tensor<T>;
    using Gen = sb::DefaultRandomGenerator;
    using PyClass = py::class_<PySubsampleBindings>;

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      PyClass cls(m, ("Subsample" + suffix).c_str());
      register_descriptors(cls);
      register_entry_points(cls);
    }

  private:
    // -------------------------------------------------------------------------
    // Descriptor classes, nested under the precision namespace
    // (Subsample32.Gaussian etc.). The descriptors are the core functors
    // themselves, constructed member-by-member; parameter validation lives in
    // the Python spec classes.
    // -------------------------------------------------------------------------
    static void register_descriptors(PyClass& cls)
    {
      py::class_<EuclideanDistance<T>>(cls, "Euclidean")
          .def(py::init<>());
      py::class_<Gaussian<T>>(cls, "Gaussian")
          .def(py::init<T, T>(), py::arg("mean"), py::arg("sigma"));
      py::class_<Uniform<T>>(cls, "Uniform")
          .def(py::init<T, T>(), py::arg("low"), py::arg("high"));
    }

    // -------------------------------------------------------------------------
    // Entry points: named static functions taking the functor variants. Each
    // visitor holds the non-variant arguments; std::visit instantiates its
    // call operator once per functor combination.
    // -------------------------------------------------------------------------

    struct SampleSubsetsCall
    {
      const TensorT& reference;
      const TensorT& query;
      size_t sampleSize;
      size_t nInstances;
      bool replace;
      Gen* gen;

      template <typename FilterF, typename DistF>
      py::tuple operator()(const FilterF& filter, const DistF& distribution) const
      {
        // The weighting/preparation phases block this thread in pure C++ for
        // O(n_query * n_reference); release the GIL so other Python threads
        // keep running. Reacquired before to_tuple builds Python objects.
        auto handle = [&] {
          py::gil_scoped_release release;
          return sb::sampling::sample_subsets(
              sb::PointCloud<T>(reference), sb::PointCloud<T>(query), filter, distribution,
              sampleSize, nInstances, replace, resolve_generator(gen), sb::default_executor());
        }();
        return to_tuple(std::move(handle));
      }
    };

    /// Point-cloud input: subsample @p reference relative to each point of
    /// @p query, weighting by @p distribution of @p filter of each point pair.
    static py::tuple sample_subsets(const TensorT& reference, const TensorT& query,
                                    const FilterVariant<T>& filter,
                                    const DistVariant<T>& distribution, size_t sampleSize,
                                    size_t nInstances, bool replace, Gen* gen)
    {
      return std::visit(
          SampleSubsetsCall{reference, query, sampleSize, nInstances, replace, gen},
          filter, distribution);
    }

    struct SampleSubsetsDistmatCall
    {
      const sb::DistanceMatrix<T>& source;
      const sb::Tensor<uint64_t>& query;
      size_t sampleSize;
      size_t nInstances;
      bool replace;
      Gen* gen;

      template <typename DistF>
      py::tuple operator()(const DistF& distribution) const
      {
        // Same GIL story as SampleSubsetsCall above.
        auto handle = [&] {
          py::gil_scoped_release release;
          return sb::sampling::sample_subsets_distmat(
              source, query, distribution, sampleSize, nInstances, replace,
              resolve_generator(gen), sb::default_executor());
        }();
        return to_tuple(std::move(handle));
      }
    };

    /// Distance-matrix input: @p query is a tensor of reference row indices.
    /// The "filter" is inherently the stored distance, so there is no filter
    /// argument.
    static py::tuple sample_subsets_distmat(const sb::DistanceMatrix<T>& source,
                                            const sb::Tensor<uint64_t>& query,
                                            const DistVariant<T>& distribution,
                                            size_t sampleSize, size_t nInstances, bool replace,
                                            Gen* gen)
    {
      return std::visit(
          SampleSubsetsDistmatCall{source, query, sampleSize, nInstances, replace, gen},
          distribution);
    }

    static void register_entry_points(PyClass& cls)
    {
      cls.def_static("sample_subsets", &PySubsampleBindings::sample_subsets,
          py::arg("reference"), py::arg("query"), py::arg("filter"), py::arg("distribution"),
          py::arg("sample_size"), py::arg("n_instances"), py::arg("replace"),
          py::arg("generator").none(true) = py::none());

      cls.def_static("sample_subsets_distmat", &PySubsampleBindings::sample_subsets_distmat,
          py::arg("source"), py::arg("query"), py::arg("distribution"),
          py::arg("sample_size"), py::arg("n_instances"), py::arg("replace"),
          py::arg("generator").none(true) = py::none());
    }

    // -------------------------------------------------------------------------
    // Argument/result adapters
    // -------------------------------------------------------------------------

    /// Resolve the nullable generator argument (Python None -> nullptr) to a
    /// concrete generator, falling back to the global default. Mutable: the
    /// draw reserves its seed block from the resolved generator, advancing it.
    static Gen& resolve_generator(Gen* gen)
    {
      return gen ? *gen : sb::default_generator();
    }

    /// Unpack a SubsampleHandle into the (task, samples) pair returned to Python.
    template <typename ElemT>
    static py::tuple to_tuple(sb::sampling::SubsampleHandle<ElemT> handle)
    {
      return py::make_tuple(std::move(handle.task), std::move(handle.samples));
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
