#ifndef STABLEBEAR_POINT_PROCESS_SUBSAMPLE_H
#define STABLEBEAR_POINT_PROCESS_SUBSAMPLE_H

#include "../executor.hpp"
#include "../point_cloud.hpp"
#include "../random_generator.hpp"
#include "../walk.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace sb::pp
{
  namespace detail
  {
    template <ArithmeticType T>
    Tensor<T> copy_logical_coordinates(const PointCloud<T>& cloud)
    {
      if (!cloud.is_indexed())
      {
        if (cloud.coords().rank() == 0)
        {
          return Tensor<T>({0, 0});
        }
        return cloud.coords().copy();
      }

      Tensor<T> result({cloud.n_points(), cloud.dim()});
      for (size_t i = 0; i < cloud.n_points(); ++i)
      {
        for (size_t j = 0; j < cloud.dim(); ++j)
        {
          result({i, j}) = cloud(i, j);
        }
      }
      return result;
    }

    template <typename EngineT>
    Tensor<uint64_t> draw_uniform_indices(size_t population, size_t count, bool replace, EngineT& engine)
    {
      Tensor<uint64_t> result({count});
      if (count == 0)
      {
        return result;
      }

      if (replace)
      {
        std::uniform_int_distribution<size_t> draw(0, population - 1);
        for (size_t i = 0; i < count; ++i)
        {
          result(i) = static_cast<uint64_t>(draw(engine));
        }
        return result;
      }

      std::vector<uint64_t> candidates(population);
      std::iota(candidates.begin(), candidates.end(), uint64_t(0));
      for (size_t i = 0; i < count; ++i)
      {
        std::uniform_int_distribution<size_t> draw(i, population - 1);
        const size_t selected = draw(engine);
        std::swap(candidates[i], candidates[selected]);
        result(i) = candidates[i];
      }
      return result;
    }

    template <ArithmeticType T>
    Tensor<uint64_t> discard_duplicate_coordinates(const Tensor<T>& coordinates, const Tensor<uint64_t>& drawn)
    {
      std::vector<uint64_t> kept;
      kept.reserve(drawn.size());
      const size_t dim = coordinates.shape(1);

      for (size_t i = 0; i < drawn.size(); ++i)
      {
        const uint64_t candidate = drawn(i);
        const bool duplicate = std::any_of(kept.begin(), kept.end(), [&](uint64_t previous) {
          for (size_t j = 0; j < dim; ++j)
          {
            if (coordinates({static_cast<size_t>(candidate), j})
                != coordinates({static_cast<size_t>(previous), j}))
            {
              return false;
            }
          }
          return true;
        });
        if (!duplicate)
        {
          kept.push_back(candidate);
        }
      }

      Tensor<uint64_t> result({kept.size()});
      for (size_t i = 0; i < kept.size(); ++i)
      {
        result(i) = kept[i];
      }
      return result;
    }
  }

  template <ArithmeticType T, TensorProperties Properties>
  Tensor<PointCloud<T>, TensorProperty::Indexed> subsample(
      const Tensor<PointCloud<T>, Properties>& points, size_t nPoints, size_t nSamples, bool replace, bool allowPartial,
      bool discardDuplicates, DefaultRandomGenerator& gen, Executor& exec)
  {
    if (nPoints == 0)
    {
      throw std::invalid_argument("n_points must be positive");
    }
    if (nSamples == 0)
    {
      throw std::invalid_argument("n_samples must be positive");
    }

    walk(points, [&](const std::vector<size_t>& index) {
      const auto& cloud = points(index);
      if (cloud.coords().rank() != 0 && cloud.coords().rank() != 2)
      {
        throw std::invalid_argument("point clouds must have 2 dimensions");
      }
      const size_t available = cloud.n_points();
      if (!replace && !allowPartial && available < nPoints)
      {
        throw std::invalid_argument("n_points exceeds the number of input points");
      }
      if (replace && !allowPartial && available == 0)
      {
        throw std::invalid_argument("cannot sample with replacement from an empty point cloud");
      }
    });

    std::vector<size_t> outputShape(points.shape().begin(), points.shape().end());
    outputShape.push_back(nSamples);
    Tensor<PointCloud<T>> source(points.shape());
    Tensor<Tensor<uint64_t>> selections(outputShape);
    const size_t nOutputs = std::accumulate(
      outputShape.begin(), outputShape.end(), size_t{1}, std::multiplies<size_t>());
    const auto seedBlock = gen.reserve(nOutputs);

    parallel_walk(points, [&](const std::vector<size_t>& inputIndex) {
      const PointCloud<T>& input = points(inputIndex);
      Tensor<T> coordinates = detail::copy_logical_coordinates(input);
      source(inputIndex) = PointCloud<T>(std::move(coordinates));
      const Tensor<T>& sourceCoordinates = source(inputIndex).coords();
      const size_t available = input.n_points();
      const size_t count = replace
          ? (available == 0 ? size_t(0) : nPoints)
          : std::min(nPoints, available);

      size_t inputFlat = 0;
      for (size_t axis = 0; axis < inputIndex.size(); ++axis)
      {
        inputFlat = inputFlat * points.shape(axis) + inputIndex[axis];
      }

      std::vector<size_t> outputIndex(inputIndex);
      outputIndex.push_back(0);
      for (size_t sample = 0; sample < nSamples; ++sample)
      {
        auto engine = seedBlock.sub_generator(inputFlat * nSamples + sample);
        Tensor<uint64_t> indices = detail::draw_uniform_indices(available, count, replace, engine);
        if (discardDuplicates)
        {
          indices = detail::discard_duplicate_coordinates(sourceCoordinates, indices);
        }
        outputIndex.back() = sample;
        selections(outputIndex) = std::move(indices);
      }
    }, exec);

    return make_indexed_tensor_from_owned_indices(
      source, to_nested_tensor(selections));
  }
}

#endif
