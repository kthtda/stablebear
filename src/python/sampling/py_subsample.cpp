#include "py_subsample.hpp"
#include "../py_async_support.hpp"

#include <sbear/sampling/subsample.hpp>

#include <pybind11/stl.h> // std::variant caster

#include <memory>
#include <string>
#include <variant>

namespace py = pybind11;

namespace
{

  using sb::sampling::EuclideanDistance;
  using sb::sampling::Gaussian;
  using sb::sampling::NoFilter;
  using sb::sampling::Uniform;

  // ===========================================================================
  // Built-in functors
  //
  // The entry points accept exactly these functors, passed from Python as
  // descriptor objects. pybind11's variant caster converts the descriptor to
  // the matching alternative and std::visit spawns the task type instantiated
  // for that combination, so the built-in weighting inlines into the draw
  // path. Adding a built-in = add the core functor, list it here, and register
  // its descriptor class in register_filters / register_distributions.
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
    using TaskPtr = std::unique_ptr<sb::StoppableTask<void>>;

    static void register_bindings(py::module_ &m, const std::string &suffix)
    {
      PyClass cls(m, ("Subsample" + suffix).c_str());
      register_filters(cls);
      register_distributions(cls);
      register_entry_points(cls);
    }

  private:
    // -------------------------------------------------------------------------
    // Descriptor classes, nested under the precision namespace
    // (Subsample32.Euclidean, Subsample32.Gaussian etc.). The descriptors are
    // the core functors themselves, constructed member-by-member; parameter
    // validation lives in the Python spec classes.
    // -------------------------------------------------------------------------

    static void register_filters(PyClass &cls)
    {
      py::class_<EuclideanDistance<T>>(cls, "Euclidean").def(py::init<>());
    }

    static void register_distributions(PyClass &cls)
    {
      py::class_<Gaussian<T>>(cls, "Gaussian").def(py::init<T, T>(), py::arg("mean"), py::arg("sigma"));
      py::class_<Uniform<T>>(cls, "Uniform").def(py::init<T, T>(), py::arg("low"), py::arg("high"));
    }

    // -------------------------------------------------------------------------
    // Entry points: spawn functions taking the functor variants, mirroring the
    // Ripser spawn_*_task bindings. Each visitor holds the non-variant
    // arguments; std::visit instantiates its call operator once per functor
    // combination. Spawning is cheap — only the task's synchronous prologue
    // (validation, output allocation, seed-block reservation) runs on this
    // thread; the draws run on executor threads, which never touch Python, so
    // no GIL release is needed here.
    // -------------------------------------------------------------------------

    struct SpawnPcloudTaskCall
    {
      const TensorT &reference;
      const TensorT &query;
      sb::Tensor<sb::PointCloud<T>> &out;
      size_t sampleSize = 0;
      size_t nInstances = 0;
      bool replace = false;
      Gen *gen = nullptr;

      template <typename FilterF, typename DistF>
      TaskPtr operator()(const FilterF &filter, const DistF &distribution) const
      {
        using Task = sb::sampling::SubsampleTask<T, FilterF, DistF>;
        return sb_py::execute_stoppable_task<Task>(
            sb::PointCloud<T>(reference), sb::PointCloud<T>(query), filter, distribution, out, sampleSize, nInstances,
            replace, resolve_generator(gen));
      }
    };

    /// Point-cloud input: spawn the task subsampling @p reference relative to
    /// each point of @p query, weighting by @p distribution of @p filter of
    /// each point pair. The task reallocates @p out to (n_query, n_instances)
    /// in its prologue and fills it in place; the caller keeps @p out and the
    /// returned task alive until the task completes.
    static TaskPtr spawn_subsample_pcloud_task(
        const TensorT &reference, const TensorT &query, sb::Tensor<sb::PointCloud<T>> &out,
        const FilterVariant<T> &filter, const DistVariant<T> &distribution, size_t sampleSize, size_t nInstances,
        bool replace, Gen *gen)
    {
      return std::visit(
          SpawnPcloudTaskCall{reference, query, out, sampleSize, nInstances, replace, gen}, filter, distribution);
    }

    struct SpawnPcloudIndexQueryTaskCall
    {
      const TensorT &reference;
      const sb::Tensor<uint64_t> &query;
      sb::Tensor<sb::PointCloud<T>> &out;
      size_t sampleSize = 0;
      size_t nInstances = 0;
      bool replace = false;
      Gen *gen = nullptr;

      template <typename FilterF, typename DistF>
      TaskPtr operator()(const FilterF &filter, const DistF &distribution) const
      {
        using Task = sb::sampling::SubsampleIndexQueryTask<T, FilterF, DistF>;
        return sb_py::execute_stoppable_task<Task>(
            sb::PointCloud<T>(reference), query, filter, distribution, out, sampleSize, nInstances, replace,
            resolve_generator(gen));
      }
    };

    /// Point-cloud input with the query given as reference row indices (query
    /// point q is reference row query(q)) — the coordinates never leave C++.
    /// Overloads the coordinate-query spawn above.
    static TaskPtr spawn_subsample_pcloud_task(
        const TensorT &reference, const sb::Tensor<uint64_t> &query, sb::Tensor<sb::PointCloud<T>> &out,
        const FilterVariant<T> &filter, const DistVariant<T> &distribution, size_t sampleSize, size_t nInstances,
        bool replace, Gen *gen)
    {
      return std::visit(
          SpawnPcloudIndexQueryTaskCall{reference, query, out, sampleSize, nInstances, replace, gen}, filter,
          distribution);
    }

    struct SpawnDistmatTaskCall
    {
      const sb::DistanceMatrix<T> &reference;
      const sb::Tensor<uint64_t> &query;
      sb::Tensor<sb::DistanceMatrix<T>> &out;
      size_t sampleSize = 0;
      size_t nInstances = 0;
      bool replace = false;
      Gen *gen = nullptr;

      template <typename DistF>
      TaskPtr operator()(const DistF &distribution) const
      {
        using Task = sb::sampling::SubsampleDistMatTask<T, DistF>;
        return sb_py::execute_stoppable_task<Task>(
            reference, query, NoFilter{}, distribution, out, sampleSize, nInstances, replace, resolve_generator(gen));
      }
    };

    /// Distance-matrix input: @p query is a tensor of reference row indices.
    /// The "filter" is inherently the stored distance, so there is no filter
    /// argument. Same output contract as the point-cloud spawn.
    static TaskPtr spawn_subsample_distmat_task(
        const sb::DistanceMatrix<T> &reference, const sb::Tensor<uint64_t> &query, sb::Tensor<sb::DistanceMatrix<T>> &out,
        const DistVariant<T> &distribution, size_t sampleSize, size_t nInstances, bool replace, Gen *gen)
    {
      return std::visit(SpawnDistmatTaskCall{reference, query, out, sampleSize, nInstances, replace, gen}, distribution);
    }

    static void register_entry_points(PyClass &cls)
    {
      cls.def_static(
          "spawn_subsample_pcloud_task",
          py::overload_cast<
              const TensorT &, const TensorT &, sb::Tensor<sb::PointCloud<T>> &, const FilterVariant<T> &,
              const DistVariant<T> &, size_t, size_t, bool, Gen *>(&PySubsampleBindings::spawn_subsample_pcloud_task),
          py::arg("reference"), py::arg("query"), py::arg("out"), py::arg("filter"), py::arg("distribution"),
          py::arg("sample_size"), py::arg("n_instances"), py::arg("replace"),
          py::arg("generator").none(true) = py::none());

      cls.def_static(
          "spawn_subsample_pcloud_task",
          py::overload_cast<
              const TensorT &, const sb::Tensor<uint64_t> &, sb::Tensor<sb::PointCloud<T>> &, const FilterVariant<T> &,
              const DistVariant<T> &, size_t, size_t, bool, Gen *>(&PySubsampleBindings::spawn_subsample_pcloud_task),
          py::arg("reference"), py::arg("query"), py::arg("out"), py::arg("filter"), py::arg("distribution"),
          py::arg("sample_size"), py::arg("n_instances"), py::arg("replace"),
          py::arg("generator").none(true) = py::none());

      cls.def_static(
          "spawn_subsample_distmat_task", &PySubsampleBindings::spawn_subsample_distmat_task, py::arg("reference"),
          py::arg("query"), py::arg("out"), py::arg("distribution"), py::arg("sample_size"), py::arg("n_instances"),
          py::arg("replace"), py::arg("generator").none(true) = py::none());
    }

    // -------------------------------------------------------------------------
    // Argument adapters
    // -------------------------------------------------------------------------

    /// Resolve the nullable generator argument (Python None -> nullptr) to a
    /// concrete generator, falling back to the global default. Mutable: the
    /// spawn reserves the draw's seed block from the resolved generator,
    /// advancing it.
    static Gen &resolve_generator(Gen *gen)
    {
      return (gen != nullptr) ? *gen : sb::default_generator();
    }
  };

} // namespace

namespace sb_py
{
  void register_sampling_subsample(py::module_ &m)
  {
    PySubsampleBindings<sb::float32_t>::register_bindings(m, "32");
    PySubsampleBindings<sb::float64_t>::register_bindings(m, "64");
  }
} // namespace sb_py
