#ifndef STABLEBEAR_POINT_PROCESS_SUBSAMPLE_RELATIVE_HPP
#define STABLEBEAR_POINT_PROCESS_SUBSAMPLE_RELATIVE_HPP

#include "subsample.hpp"
#include "../random/distribution.hpp"
#include "../sampling/weighted_draw.hpp"
#include "../task.hpp"

#include <optional>
#include <string>
#include <variant>

namespace sb::pp
{
  namespace detail
  {
    /// Euclidean distance between point @p i of cloud @p x and point @p j of
    /// cloud @p y. Named generically (x/y rather than query/reference) so it
    /// reads naturally anywhere a distance between two clouds is wanted. The
    /// samplers validate the shared dimension once up front, so this hot-path
    /// operator stays branch-free. Accumulates with hypot in long double so
    /// large coordinates do not overflow where long double equals double.
    template <typename T>
    struct EuclideanDistance
    {
      [[nodiscard]] long double operator()(const PointCloud<T> &x, size_t i, const PointCloud<T> &y, size_t j) const
      {
        long double acc = 0;
        for (size_t k = 0; k < x.dim(); ++k)
        {
          const long double d = static_cast<long double>(x.coords()(i, k)) - y.coords()(j, k);
          acc = std::hypot(acc, d);
        }
        return acc;
      }
    };
  }

  template <typename T>
  using RelativeQuery = std::variant<std::monostate, Tensor<int32_t>, Tensor<int64_t>,
                                     Tensor<uint32_t>, Tensor<uint64_t>, PointCloud<T>>;

  /// Normalize logical indices into one independently owned, validated snapshot.
  template <typename I>
  Tensor<uint64_t> prepare_relative_indices(const Tensor<I>& input, size_t population)
  {
    if (input.rank() != 1)
      throw std::invalid_argument("query indices must be one-dimensional");
    Tensor<uint64_t> result({input.size()});
    for (size_t k = 0; k < input.size(); ++k)
    {
      const I value = input(k);
      if constexpr (std::is_signed_v<I>)
      {
        if (value < 0)
        {
          const uint64_t magnitude = static_cast<uint64_t>(-(value + 1)) + 1;
          if (magnitude > population)
            throw std::invalid_argument("query index is out of range");
          result(k) = population - magnitude;
          continue;
        }
      }
      if (std::cmp_greater_equal(value, population))
        throw std::invalid_argument("query index is out of range");
      result(k) = static_cast<uint64_t>(value);
    }
    return result;
  }

  /// One snapshot of the reference, with indices shared by every output cell.
  /// Each query/distribution row is prepared once and reused for all samples.
  template <IndexableTensorElement ElementT, typename EngineT = DefaultRandomGenerator::engine_type>
  class RelativeSubsampleTask : public StoppableTask<void>
  {
    using Generator = RandomGenerator<EngineT>;
    using T = typename ElementT::value_type;
    using Output = Tensor<ElementT, TensorProperty::Indexed>;
    static constexpr bool isCloud = std::is_same_v<ElementT, PointCloud<T>>;

  public:
    RelativeSubsampleTask(const ElementT& reference, RelativeQuery<T> query,
                         std::vector<random::Distribution> distributions,
                         size_t nPoints, size_t nSamples, bool distributionAxis,
                         bool replace, sampling::PartialPolicy partialPolicy, bool discardDuplicates,
                         Generator& generator)
      : m_source(std::vector<size_t>{}),
        m_distributions(std::move(distributions)), m_nPoints(nPoints), m_nSamples(nSamples),
        m_distributionAxis(distributionAxis), m_replace(replace),
        m_partialPolicy(partialPolicy),
        m_discardDuplicates(discardDuplicates), m_generator(generator)
    {
      if (nPoints == 0 || nSamples == 0)
        throw std::invalid_argument("n_points and n_samples must be positive");
      if (m_distributions.empty() || (!distributionAxis && m_distributions.size() != 1))
        throw std::invalid_argument("expected one distribution or a nonempty distribution list");
      m_source.flat(0) = reference.copy();
      validate_reference();
      const size_t population = detail::sample_population(m_source.flat(0));
      std::visit([&](const auto& input) {
        using Query = std::decay_t<decltype(input)>;
        if constexpr (std::is_same_v<Query, std::monostate>)
        {
          // The implicit full query needs no allocated index sequence.
          m_query = std::monostate{};
          m_nQuery = population;
        }
        else if constexpr (std::is_same_v<Query, PointCloud<T>>)
        {
          if constexpr (isCloud)
          {
            if (input.dim() != reference.dim())
              throw std::invalid_argument("reference and query must have the same dimension");
            auto snapshot = input.copy();
            validate_coordinates(snapshot);
            m_nQuery = snapshot.n_points();
            m_query = std::move(snapshot);
          }
          else
          {
            throw std::invalid_argument("distance-matrix queries must be reference indices");
          }
        }
        else
        {
          m_query = prepare_relative_indices(input, population);
          m_nQuery = input.size();
        }
      }, query);

      std::vector<size_t> shape{m_nQuery};
      if (distributionAxis)
        shape.push_back(m_distributions.size());
      shape.push_back(nSamples);
      m_selections = Tensor<Tensor<uint64_t>>(shape);
      m_empty = Tensor<uint8_t>({m_nQuery, m_distributions.size()});
    }

    Output result()
    {
      wait();
      if (work_completed() != work_total())
        throw std::runtime_error("relative subsampling was cancelled");
      if (!m_result)
        m_result = make_indexed_tensor_from_owned_indices(m_source, to_nested_tensor(m_selections));
      return *m_result;
    }

    std::vector<std::pair<size_t, size_t>> empty_regions() const
    {
      std::vector<std::pair<size_t, size_t>> result;
      for (size_t q = 0; q < m_nQuery; ++q)
      {
        for (size_t d = 0; d < m_distributions.size(); ++d)
        {
          if (m_empty({q, d}))
            result.emplace_back(q, d);
        }
      }
      return result;
    }

  private:
    static void validate_coordinates(const PointCloud<T>& cloud)
    {
      for (size_t i = 0; i < cloud.n_points(); ++i)
      {
        for (size_t j = 0; j < cloud.dim(); ++j)
        {
          if (!std::isfinite(cloud.coords()(i, j)))
            throw std::invalid_argument("coordinates must be finite");
        }
      }
    }

    void validate_reference() const
    {
      const ElementT& reference = m_source.flat(0);
      if constexpr (isCloud)
      {
        validate_coordinates(reference);
      }
      else
      {
        for (size_t i = 0; i < reference.size(); ++i)
        {
          for (size_t j = 0; j < i; ++j)
          {
            if (std::isnan(reference(i, j)) || reference(i, j) < 0)
              throw std::invalid_argument("distances must be nonnegative and not NaN");
          }
        }
      }
    }

    long double distance(size_t q, size_t r) const
    {
      const ElementT& reference = m_source.flat(0);
      const auto* indices = std::get_if<Tensor<uint64_t>>(&m_query);
      const size_t sourceQuery = indices ? static_cast<size_t>((*indices)(q)) : q;
      if constexpr (isCloud)
      {
        const auto* queryCloud = std::get_if<PointCloud<T>>(&m_query);
        const auto result = queryCloud
          ? detail::EuclideanDistance<T>{}(*queryCloud, q, reference, r)
          : detail::EuclideanDistance<T>{}(reference, sourceQuery, reference, r);
        if (!std::isfinite(result))
          throw std::overflow_error("Euclidean distance exceeds numerical range");
        return result;
      }
      else
      {
        return reference(sourceQuery, r);
      }
    }

    void check_cancelled() const
    {
      if (stop_requested())
        throw std::runtime_error("relative subsampling was cancelled");
    }

    // Taskflow copies initially empty scratch into each sample partition;
    // each partition reuses its buffers while all prepared data stays read-only.
    void draw_samples(size_t q, size_t d, const std::vector<long double>& row,
                      const std::vector<uint64_t>& sourceIndices, size_t count,
                      Generator& queryGenerator, tf::Executor& executor)
    {
      // One engine per sample, from a block reserved per distribution.
      const auto seeds = queryGenerator.reserve(m_nSamples);
      std::vector<size_t> outputIndex{q};
      if (m_distributionAxis)
        outputIndex.push_back(d);
      outputIndex.push_back(0);
      tf::Taskflow samples;
      samples.for_each_index(size_t{0}, m_nSamples, size_t{1},
        [&, outputIndex, kept = std::vector<uint64_t>{},
         keyed = std::vector<std::pair<long double, uint64_t>>{}](size_t s) mutable {
        check_cancelled();
        auto engine = seeds.sub_generator(s);
        Tensor<uint64_t> indices({0});
        if (count != 0)
        {
          indices = m_replace
            ? sampling::detail::draw_with_replacement<long double>(row, count, engine)
            : sampling::detail::draw_without_replacement<long double>(row, sourceIndices, count, engine, keyed);
        }
        if (m_discardDuplicates)
          indices = detail::discard_sample_duplicates(m_source.flat(0), indices, kept);
        outputIndex.back() = s;
        m_selections(outputIndex) = std::move(indices);
        add_progress(1);
      });
      // Cooperatively wait for all nested work, also on failure, before row reuse.
      executor.corun(samples);
    }

    void sample_distribution(size_t q, size_t d, const std::vector<long double>& distances,
                             std::vector<long double>& row, std::vector<uint64_t>& sourceIndices,
                             Generator& queryGenerator, tf::Executor& executor)
    {
      const size_t population = distances.size();
      row.resize(population);
      random::log_weights(m_distributions[d], distances, row);
      const auto eligible = static_cast<size_t>(std::ranges::count_if(row, [](long double w) { return std::isfinite(w); }));
      add_progress(population);
      if (sampling::is_insufficient(m_partialPolicy, m_replace, m_nPoints, eligible))
      {
        throw std::invalid_argument("insufficient positive support for query " + std::to_string(q)
                                    + ", distribution " + std::to_string(d));
      }
      const size_t count = sampling::sample_count(m_partialPolicy, m_replace, m_nPoints, eligible);
      m_empty({q, d}) = count == 0;
      sourceIndices.clear();
      if (m_replace)
      {
        sampling::detail::log_probabilities_to_cdf<long double>(row);
      }
      else
      {
        // Compact once, retaining source order for every sample's random draws.
        sourceIndices.reserve(eligible);
        for (size_t r = 0; r < population; ++r)
        {
          if (std::isfinite(row[r]))
          {
            row[sourceIndices.size()] = row[r];
            sourceIndices.push_back(static_cast<uint64_t>(r));
          }
        }
        row.resize(eligible);
      }

      draw_samples(q, d, row, sourceIndices, count, queryGenerator, executor);
    }

    void sample_query(size_t q, size_t population, Generator& queryGenerator, tf::Executor& executor)
    {
      std::vector<long double> distances(population);
      for (size_t r = 0; r < population; ++r)
        distances[r] = distance(q, r);
      add_progress(population);

      std::vector<long double> row;
      std::vector<uint64_t> sourceIndices;
      for (size_t d = 0; d < m_distributions.size(); ++d)
      {
        check_cancelled();
        sample_distribution(q, d, distances, row, sourceIndices, queryGenerator, executor);
      }
    }

    tf::Future<void> run_async(Executor& exec) override
    {
      const size_t population = detail::sample_population(m_source.flat(0));
      next_step(m_nQuery * population + m_empty.size() * (population + m_nSamples),
                "Relative subsampling", "item");
      auto* executor = exec.cpu();
      // Each query owns a generator, seeded from its engine, and reserves one
      // block per distribution in order, so seeds follow the loops rather than
      // the output layout.
      // DynamicPartitioner creates at most one query partition per worker.
      // Each finishes its nested samples before claiming another query, bounding
      // live distance/weight rows even while cooperative waits execute other work.
      return parallel_for_each_index_async(m_nQuery, m_generator,
        [this, executor, population](size_t q, EngineT& engine) {
          check_cancelled();
          Generator queryGenerator(engine());
          sample_query(q, population, queryGenerator, *executor);
        }, exec, tf::DynamicPartitioner<>(1));
    }

    Tensor<ElementT> m_source;
    std::variant<std::monostate, Tensor<uint64_t>, PointCloud<T>> m_query;
    std::vector<random::Distribution> m_distributions;
    size_t m_nPoints, m_nSamples, m_nQuery = 0;
    bool m_distributionAxis, m_replace;
    sampling::PartialPolicy m_partialPolicy;
    bool m_discardDuplicates;
    Tensor<Tensor<uint64_t>> m_selections;
    Tensor<uint8_t> m_empty;
    Generator& m_generator;
    std::optional<Output> m_result;
  };
}
#endif
