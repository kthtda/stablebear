#ifndef STABLEBEAR_COMPUTE_PERSISTENCE_H
#define STABLEBEAR_COMPUTE_PERSISTENCE_H

#include "../tensor.hpp"
#include "../point_cloud.hpp"
#include "../distance_matrix.hpp"
#include "../ufunc.hpp"

#include "barcode.hpp"
#include "persistence_pair.hpp"

#include "ripser/ripser.hpp"

#include <span>
#include <string>
#include <type_traits>
#include <utility>

namespace sb::ph
{
  namespace detail
  {
    /// Run Ripser on any distance-matrix-like object (must have .size() and operator()(i,j)).
    /// Computes the enclosing radius threshold, builds the compressed lower-triangular
    /// matrix, runs Ripser, and writes the barcode of dimension i into out[i].
    template <typename DistMatT, typename T>
    void run_ripser(const DistMatT& distanceMatrix, size_t n, std::span<Barcode<T>> out, size_t maxDim, bool reducedHomology)
    {
      // A single point has no pairwise distances, so ripser's compressed
      // distance matrix would be empty and init_rows() would dereference
      // invalid memory.  Handle this trivially: one essential H0 bar
      // (unreduced) or nothing (reduced), and empty bars in higher dims.
      if (n <= 1)
      {
        for (auto i = 0_uz; i < maxDim + 1; ++i)
        {
          std::vector<PersistencePair<T>> bars;
          if (i == 0 && !reducedHomology && n == 1)
          {
            bars.emplace_back(T{0}, std::numeric_limits<T>::infinity());
          }
          out[i] = std::move(bars);
        }
        return;
      }

      rips::value_t threshold = std::numeric_limits<rips::value_t>::infinity();
      for (auto i = 0_uz; i < n; ++i)
      {
        auto r = -std::numeric_limits<rips::value_t>::infinity();
        for (auto j = 0_uz; j < n; ++j)
        {
          r = std::max(r, static_cast<rips::value_t>(distanceMatrix(i, j)));
        }
        threshold = std::min(threshold, r);
      }

      rips::value_t ratio = static_cast<rips::value_t>(1);
      rips::coefficient_t modulus = 2;

      rips::compressed_lower_distance_matrix dist(distanceMatrix);
      rips::ripser<rips::compressed_lower_distance_matrix> ripser(std::move(dist), maxDim, threshold, ratio, modulus);
      ripser.compute_barcodes();

      for (auto i = 0_uz; i < maxDim + 1; ++i)
      {
        auto const & intervals = ripser.get_intervals(i);

        std::vector<PersistencePair<T>> bars;
        bars.reserve(intervals.size() + 1);

        // Ripser computes reduced homology. When unreduced homology is
        // requested, insert the essential H0 class (born at 0, never dies).
        if (i == 0 && !reducedHomology)
        {
          bars.emplace_back(T{0}, std::numeric_limits<T>::infinity());
        }

        for (auto const& rpair : intervals)
        {
          bars.emplace_back(static_cast<T>(rpair.birth), static_cast<T>(rpair.death));
        }

        out[i] = std::move(bars);
      }
    }

    /// Build a Euclidean distance matrix from @p points and run Ripser into @p out.
    /// n_points()/dim()/operator()(i, j) read through any indexing transparently, so an
    /// indexed subsample (sharing a source cloud) needs no special handling.
    template <typename T>
    void run_euclidean_ripser(const PointCloud<T>& points, std::span<Barcode<T>> out,
                              size_t maxDim, bool reducedHomology)
    {
      const size_t nPoints = points.n_points();
      const size_t dim = points.dim();

      std::vector<std::vector<rips::value_t>> rpoints;
      rpoints.reserve(nPoints);

      for (auto i = 0_uz; i < nPoints; ++i)
      {
        rpoints.emplace_back();
        auto & curRPoint = rpoints.back();
        curRPoint.resize(dim);
        for (auto j = 0_uz; j < dim; ++j)
        {
          curRPoint[j] = points(i, j);
        }
      }

      rips::euclidean_distance_matrix distanceMatrix(std::move(rpoints));
      run_ripser(distanceMatrix, nPoints, out, maxDim, reducedHomology);
    }
  }

  /// Vietoris-Rips persistence of one point cloud (Euclidean distances) or one
  /// distance matrix, as an operation with extra dimensions (maxDim + 1,): out[i] is
  /// the barcode of dimension i.
  template <typename ElemT, typename T>
  struct PersistenceOperation
  {
    using input_type = ElemT;
    using output_type = Barcode<T>;

    size_t maxDim = 1;
    bool reducedHomology = false;

    [[nodiscard]] std::vector<size_t> extra_dims() const { return {maxDim + 1}; }

    void operator()(const ElemT& input, std::span<Barcode<T>> out) const
    {
      if constexpr (std::is_same_v<ElemT, PointCloud<T>>)
      {
        // Empty clouds do not contribute persistence intervals.
        if (input.n_points() == 0 || input.dim() == 0)
        {
          return;
        }
        detail::run_euclidean_ripser(input, out, maxDim, reducedHomology);
      }
      else
      {
        if (input.size() == 0)
        {
          return;
        }
        detail::run_ripser(input, input.size(), out, maxDim, reducedHomology);
      }
    }
  };

  /// Persistence of every point cloud or distance matrix in a container: a
  /// tensor (the result has the input's shape followed by maxDim + 1) or a
  /// time series (each channel's shape is followed by maxDim + 1).
  template <typename Container>
  using PersistenceTask = UfuncTask<
      PersistenceOperation<typename Container::value_type, typename Container::value_type::value_type>,
      Container>;

  template <typename Container>
  auto make_persistence_task(Container input, size_t maxDim = 1, bool reducedHomology = false)
  {
    using ElemT = typename Container::value_type;
    return make_ufunc_task(
        std::move(input),
        PersistenceOperation<ElemT, typename ElemT::value_type>{.maxDim = maxDim, .reducedHomology = reducedHomology},
        "Computing persistence", "pointcloud");
  }

}

#endif //STABLEBEAR_COMPUTE_PERSISTENCE_H
