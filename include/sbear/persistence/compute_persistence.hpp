#ifndef STABLEBEAR_COMPUTE_PERSISTENCE_H
#define STABLEBEAR_COMPUTE_PERSISTENCE_H

#include "../tensor.hpp"
#include "../point_cloud.hpp"
#include "../distance_matrix.hpp"
#include "../ufunc.hpp"

#include "barcode.hpp"
#include "persistence_pair.hpp"

#include "ripser/ripser.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>
#include <type_traits>
#include <utility>

namespace sb::ph
{
  namespace detail
  {
    /// The points {a, b, c, d} of the critical edges of a bar: its birth is
    /// the distance between points a and b, and its death that between c
    /// and d. -1 where an endpoint has no edge: a birth at 0 in dimension
    /// 0, and an infinite death.
    using CriticalEdges = std::array<std::ptrdiff_t, 4>;

    /// Run Ripser on any distance-matrix-like object (must have .size() and operator()(i,j)).
    /// Computes the enclosing radius threshold, builds the compressed lower-triangular
    /// matrix, runs Ripser, and writes the barcode of dimension i into out[i].
    /// With @p criticalEdges, of the length of @p out, also writes the
    /// critical edges of each bar of dimension i into criticalEdges[i].
    template <typename DistMatT, typename T>
    void run_ripser(const DistMatT& distanceMatrix, size_t n, std::span<Barcode<T>> out, size_t maxDim, bool reducedHomology,
                    std::span<std::vector<CriticalEdges>> criticalEdges = {})
    {
      for (auto& edges : criticalEdges)
      {
        edges.clear();
      }
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
            if (!criticalEdges.empty())
            {
              criticalEdges[i].push_back({-1, -1, -1, -1});
            }
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
      ripser.record_critical_edges = !criticalEdges.empty();
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
          if (!criticalEdges.empty())
          {
            criticalEdges[i].push_back({-1, -1, -1, -1});
          }
        }

        for (auto const& rpair : intervals)
        {
          bars.emplace_back(static_cast<T>(rpair.birth), static_cast<T>(rpair.death));
        }

        if (!criticalEdges.empty())
        {
          for (auto const& edges : ripser.get_critical_edges(i))
          {
            criticalEdges[i].push_back({edges[0], edges[1], edges[2], edges[3]});
          }
        }

        out[i] = std::move(bars);
      }
    }

    /// Build a Euclidean distance matrix from @p nPoints points of @p dim
    /// coordinates, coordinate(i, j) being coordinate j of point i, and run
    /// Ripser into @p out. Requires dim > 0.
    template <typename T, typename Coordinate>
    void run_euclidean_ripser(size_t nPoints, size_t dim, Coordinate coordinate, std::span<Barcode<T>> out,
                              size_t maxDim, bool reducedHomology,
                              std::span<std::vector<CriticalEdges>> criticalEdges = {})
    {
      std::vector<std::vector<rips::value_t>> rpoints;
      rpoints.reserve(nPoints);

      for (auto i = 0_uz; i < nPoints; ++i)
      {
        rpoints.emplace_back();
        auto & curRPoint = rpoints.back();
        curRPoint.resize(dim);
        for (auto j = 0_uz; j < dim; ++j)
        {
          curRPoint[j] = coordinate(i, j);
        }
      }

      rips::euclidean_distance_matrix distanceMatrix(std::move(rpoints));
      run_ripser(distanceMatrix, nPoints, out, maxDim, reducedHomology, criticalEdges);
    }

    /// n_points()/dim()/operator()(i, j) read through any indexing transparently, so an
    /// indexed subsample (sharing a source cloud) needs no special handling.
    template <typename T>
    void run_euclidean_ripser(const PointCloud<T>& points, std::span<Barcode<T>> out,
                              size_t maxDim, bool reducedHomology,
                              std::span<std::vector<CriticalEdges>> criticalEdges = {})
    {
      run_euclidean_ripser<T>(points.n_points(), points.dim(), [&](size_t i, size_t j) { return points(i, j); },
                              out, maxDim, reducedHomology, criticalEdges);
    }
  }

  /// Vietoris-Rips persistence of one point cloud with the critical edges
  /// of its bars: barcodes[i] is the barcode of dimension i, and
  /// criticalEdges[i][k] gives the points whose distances are the birth and
  /// the death of its bar k.
  template <typename T>
  void persistence_with_critical_edges(const PointCloud<T>& points, size_t maxDim, bool reducedHomology,
                                       std::vector<Barcode<T>>& barcodes,
                                       std::vector<std::vector<detail::CriticalEdges>>& criticalEdges)
  {
    barcodes.assign(maxDim + 1, Barcode<T>());
    criticalEdges.assign(maxDim + 1, {});
    if (points.n_points() == 0 || points.dim() == 0)
    {
      return;
    }
    detail::run_euclidean_ripser(points, std::span<Barcode<T>>(barcodes), maxDim, reducedHomology,
                                 std::span<std::vector<detail::CriticalEdges>>(criticalEdges));
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

  /// Vietoris-Rips persistence of the Euclidean distances between the rows of
  /// an (n, d) block of coordinates, one point per row, as for a point cloud
  /// of n points in d dimensions. The core dimensions are (n, d), so a tensor
  /// of shape (..., n, d) holds a batch of clouds.
  template <typename T>
  struct CoordinatesPersistenceOperation
  {
    using input_type = T;
    using output_type = Barcode<T>;
    static constexpr size_t core_rank = 2;

    size_t maxDim = 1;
    bool reducedHomology = false;

    [[nodiscard]] std::vector<size_t> extra_dims() const { return {maxDim + 1}; }

    void operator()(CoreBlock<T> points, std::span<Barcode<T>> out) const
    {
      const auto nPoints = points.shape[0];
      const auto dim = points.shape[1];
      // Empty clouds do not contribute persistence intervals.
      if (nPoints == 0 || dim == 0)
      {
        return;
      }
      detail::run_euclidean_ripser<T>(nPoints, dim, [&](size_t i, size_t j) { return points.values[i * dim + j]; },
                                      out, maxDim, reducedHomology);
    }
  };

  /// The persistence operation for elements of type ElemT: coordinates for a
  /// floating-point element, a point cloud or distance matrix otherwise.
  template <typename ElemT>
  struct persistence_operation
  {
    using type = PersistenceOperation<ElemT, typename ElemT::value_type>;
  };

  template <typename ElemT>
    requires std::is_floating_point_v<ElemT>
  struct persistence_operation<ElemT>
  {
    using type = CoordinatesPersistenceOperation<ElemT>;
  };

  template <typename ElemT>
  using persistence_operation_t = typename persistence_operation<ElemT>::type;

  /// Persistence of every point cloud, distance matrix, or (n, d) block of
  /// coordinates in a container: a tensor (the result has the input's batch
  /// shape followed by maxDim + 1) or a time series (each channel's shape,
  /// without core dimensions, is followed by maxDim + 1).
  template <typename Container>
  using PersistenceTask = UfuncTask<persistence_operation_t<typename Container::value_type>, Container>;

  template <typename Container>
  auto make_persistence_task(Container input, size_t maxDim = 1, bool reducedHomology = false)
  {
    using Op = persistence_operation_t<typename Container::value_type>;
    return make_ufunc_task(
        std::move(input), Op{.maxDim = maxDim, .reducedHomology = reducedHomology},
        "Computing persistence", "pointcloud");
  }

}

#endif //STABLEBEAR_COMPUTE_PERSISTENCE_H
