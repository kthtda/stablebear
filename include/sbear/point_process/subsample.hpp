#ifndef STABLEBEAR_POINT_PROCESS_SUBSAMPLE_H
#define STABLEBEAR_POINT_PROCESS_SUBSAMPLE_H

#include "../executor.hpp"
#include "../distance_matrix.hpp"
#include "../point_cloud.hpp"
#include "../random_generator.hpp"
#include "../walk.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sb::pp
{
  namespace detail
  {
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

    template <ArithmeticType T, typename Layout>
      requires FixedRankLayout<Layout, 2>
    Tensor<uint64_t> discard_duplicate_coordinates(
        const FixedRankTensor<T, 2, Layout>& coordinates,
        const Tensor<uint64_t>& drawn, std::vector<uint64_t>& kept)
    {
      kept.clear();
      kept.reserve(drawn.size());
      const size_t dim = coordinates.shape(1);
      std::unordered_multimap<size_t, uint64_t> seen;
      seen.reserve(drawn.size());

      for (size_t i = 0; i < drawn.size(); ++i)
      {
        const uint64_t candidate = drawn(i);
        size_t hash = 0;
        bool hasNaN = false;
        for (size_t j = 0; j < dim; ++j)
        {
          const T value = coordinates(static_cast<size_t>(candidate), j);
          hasNaN |= value != value;
          hash ^= std::hash<T>{}(value) + size_t{0x9e3779b9} + (hash << 6) + (hash >> 2);
        }
        // NaN-containing rows never compare equal, even to themselves.
        if (hasNaN)
        {
          kept.push_back(candidate);
          continue;
        }
        const auto [first, last] = seen.equal_range(hash);
        const bool duplicate = std::any_of(first, last, [&](const auto& entry) {
          const uint64_t previous = entry.second;
          for (size_t j = 0; j < dim; ++j)
          {
            if (coordinates(static_cast<size_t>(candidate), j)
                != coordinates(static_cast<size_t>(previous), j))
            {
              return false;
            }
          }
          return true;
        });
        if (!duplicate)
        {
          kept.push_back(candidate);
          seen.emplace(hash, candidate);
        }
      }

      Tensor<uint64_t> result({kept.size()});
      for (size_t i = 0; i < kept.size(); ++i)
      {
        result(i) = kept[i];
      }
      return result;
    }

    inline Tensor<uint64_t> discard_duplicate_indices(
        const Tensor<uint64_t>& drawn, std::vector<uint64_t>& kept)
    {
      kept.clear();
      kept.reserve(drawn.size());
      std::unordered_set<uint64_t> seen;
      seen.reserve(drawn.size());
      for (size_t i = 0; i < drawn.size(); ++i)
      {
        const auto index = drawn(i);
        if (seen.insert(index).second)
          kept.push_back(index);
      }

      Tensor<uint64_t> result({kept.size()});
      for (size_t i = 0; i < kept.size(); ++i)
        result(i) = kept[i];
      return result;
    }

    template <ArithmeticType T>
    size_t sample_population(const PointCloud<T>& cloud)
    {
      return cloud.n_points();
    }

    template <ArithmeticType T>
    size_t sample_population(const DistanceMatrix<T>& matrix)
    {
      return matrix.size();
    }

    template <ArithmeticType T>
    Tensor<uint64_t> discard_sample_duplicates(
        const PointCloud<T>& source, const Tensor<uint64_t>& drawn,
        std::vector<uint64_t>& kept)
    {
      return discard_duplicate_coordinates(source.coords(), drawn, kept);
    }

    template <ArithmeticType T>
    Tensor<uint64_t> discard_sample_duplicates(
        const DistanceMatrix<T>&, const Tensor<uint64_t>& drawn,
        std::vector<uint64_t>& kept)
    {
      return discard_duplicate_indices(drawn, kept);
    }
  }

  template <IndexableTensorElement ElementT, TensorProperties Properties>
  Tensor<ElementT, TensorProperty::Indexed> subsample(
      const Tensor<ElementT, Properties>& values, size_t nPoints, size_t nSamples, bool replace, bool allowPartial,
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

    walk(values, [&](const std::vector<size_t>& index) {
      const auto value = values(index);
      const size_t available = detail::sample_population(value);
      if (!replace && !allowPartial && available < nPoints)
        throw std::invalid_argument("n_points exceeds the number of input points");
      if (replace && !allowPartial && available == 0)
        throw std::invalid_argument("cannot sample with replacement from an empty input");
    });

    std::vector<size_t> outputShape(values.shape().begin(), values.shape().end());
    outputShape.push_back(nSamples);
    Tensor<ElementT> source(values.shape());
    Tensor<Tensor<uint64_t>> selections(outputShape);
    const size_t nOutputs = std::accumulate(
      outputShape.begin(), outputShape.end(), size_t{1}, std::multiplies<size_t>());
    const auto seedBlock = gen.reserve(nOutputs);

    parallel_walk(values, [&](const std::vector<size_t>& inputIndex) {
      const ElementT input = values(inputIndex);
      source(inputIndex) = input.copy();
      const size_t available = detail::sample_population(input);
      const size_t count = replace
          ? (available == 0 ? size_t(0) : nPoints)
          : std::min(nPoints, available);

      size_t inputFlat = 0;
      for (size_t axis = 0; axis < inputIndex.size(); ++axis)
      {
        inputFlat = inputFlat * values.shape(axis) + inputIndex[axis];
      }

      std::vector<size_t> outputIndex(inputIndex);
      outputIndex.push_back(0);
      std::vector<uint64_t> kept;
      for (size_t sample = 0; sample < nSamples; ++sample)
      {
        auto engine = seedBlock.sub_generator(inputFlat * nSamples + sample);
        Tensor<uint64_t> indices = detail::draw_uniform_indices(available, count, replace, engine);
        if (discardDuplicates)
          indices = detail::discard_sample_duplicates(source(inputIndex), indices, kept);
        outputIndex.back() = sample;
        selections(outputIndex) = std::move(indices);
      }
    }, exec);

    return make_indexed_tensor_from_owned_indices(
      source, to_nested_tensor(selections));
  }
}

#endif
