#ifndef STABLEBEAR_UFUNC_H
#define STABLEBEAR_UFUNC_H

#include "executor.hpp"
#include "task.hpp"
#include "tensor.hpp"
#include "timeseries.hpp"

#include <taskflow/algorithm/for_each.hpp>

#include <algorithm>
#include <functional>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sb
{
  /// A contiguous block of input for an operation with core dimensions: its
  /// values in row-major order and its shape (the core dimensions).
  template <typename T>
  struct CoreBlock
  {
    std::span<const T> values;
    std::span<const size_t> shape;
  };

  /// The number of trailing input dimensions an operation consumes per call,
  /// like the core dimensions of a NumPy gufunc signature; 0 unless the
  /// operation declares a core_rank.
  template <typename Op>
  inline constexpr size_t core_rank_v = [] {
    if constexpr (requires { Op::core_rank; })
      return size_t{Op::core_rank};
    else
      return size_t{0};
  }();

  /// What an operation receives per call: one element, or a CoreBlock of its
  /// core dimensions.
  template <typename Op>
  using operation_input_t = std::conditional_t<
      core_rank_v<Op> == 0, const typename Op::input_type&, CoreBlock<typename Op::input_type>>;

  /// An operation in the style of a NumPy (g)ufunc. It maps one input element,
  /// or one block of core_rank trailing input dimensions, to a block of output
  /// elements of shape extra_dims(), the same for every call. Parameters are
  /// bound in the operation.
  template <typename Op>
  concept Operation = requires(const Op& op, operation_input_t<Op> in,
                               std::span<typename Op::output_type> out)
  {
    { op.extra_dims() } -> std::convertible_to<std::vector<size_t>>;
    op(in, out);
  };

  namespace detail
  {
    [[nodiscard]] inline size_t product(std::span<const size_t> dims)
    {
      return std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    }

    /// Split @p shape into its leading batch dimensions and its trailing
    /// @p coreRank core dimensions.
    [[nodiscard]] inline std::pair<std::vector<size_t>, std::vector<size_t>>
    split_core(std::span<const size_t> shape, size_t coreRank)
    {
      if (shape.size() < coreRank)
      {
        throw std::invalid_argument(
            "Expected input of rank at least " + std::to_string(coreRank) + ", got rank "
            + std::to_string(shape.size()));
      }
      const auto split = shape.begin() + static_cast<std::ptrdiff_t>(shape.size() - coreRank);
      return {std::vector<size_t>(shape.begin(), split), std::vector<size_t>(split, shape.end())};
    }
  }

  /// Says where a container's inputs are and how to build the output of an
  /// operation with core rank CoreRank and output element type Out. An
  /// adapter numbers the calls 0, ..., size() - 1; input(i) is call i's input,
  /// an element or a CoreBlock, and block(i) points at its block of output
  /// elements, which finish() turns into the result.
  template <typename Container, size_t CoreRank, typename Out>
  class UfuncAdapter;

  /// Tensor: the result has the input's batch shape (its shape without the
  /// core dimensions) followed by extra_dims.
  template <typename In, TensorProperties Properties, size_t CoreRank, typename Out>
  class UfuncAdapter<Tensor<In, Properties>, CoreRank, Out>
  {
  public:
    using result_type = Tensor<Out>;

    UfuncAdapter(const Tensor<In, Properties>& input, const std::vector<size_t>& extraDims)
      : m_input(input), m_blockSize(detail::product(extraDims))
    {
      auto [batch, core] = detail::split_core(input.shape(), CoreRank);
      m_size = detail::product(batch);  // a 0-d batch is one call
      m_core = std::move(core);
      m_coreSize = detail::product(m_core);
      if constexpr (CoreRank > 0)
      {
        // Blocks are read as contiguous spans.
        if (!m_input.is_contiguous())
        {
          m_input = m_input.copy();
        }
      }
      batch.insert(batch.end(), extraDims.begin(), extraDims.end());
      m_output = Tensor<Out>(batch);
    }

    [[nodiscard]] size_t size() const { return m_size; }

    [[nodiscard]] decltype(auto) input(size_t i) const
    {
      if constexpr (CoreRank == 0)
      {
        return m_input.flat(i);
      }
      else
      {
        return CoreBlock<In>{std::span<const In>(m_input.data() + i * m_coreSize, m_coreSize), m_core};
      }
    }

    /// The output is freshly allocated, so it is contiguous and call i's
    /// block starts at i * blockSize.
    [[nodiscard]] Out* block(size_t i) { return m_output.data() + i * m_blockSize; }

    [[nodiscard]] result_type finish() { return std::move(m_output); }

  private:
    Tensor<In, Properties> m_input;
    size_t m_blockSize;
    size_t m_size = 0;
    std::vector<size_t> m_core;
    size_t m_coreSize = 1;
    Tensor<Out> m_output;
  };

  /// Time series: the calls of all channels' observations, channel by channel.
  /// Each output channel keeps its input's times and snap settings, and its
  /// shape is the input shape without the core dimensions, followed by
  /// extra_dims; the result keeps the query unit.
  template <typename In, size_t CoreRank, typename Out>
  class UfuncAdapter<TimeSeries<In>, CoreRank, Out>
  {
  public:
    using result_type = TimeSeries<Out>;

    UfuncAdapter(const TimeSeries<In>& input, const std::vector<size_t>& extraDims)
      : m_input(input), m_extraDims(extraDims), m_blockSize(detail::product(extraDims))
    {
      size_t offset = 0;
      for (size_t c = 0; c < input.n_channels(); ++c)
      {
        const auto& channel = input.channel(c);
        auto [batch, core] = detail::split_core(channel.shape(), CoreRank);
        const auto coreSize = detail::product(core);
        const auto count = channel.size() * detail::product(batch);
        m_offsets.push_back(offset);
        m_batches.push_back(std::move(batch));
        m_cores.push_back(std::move(core));
        m_coreSizes.push_back(coreSize);
        m_outputs.emplace_back(count * m_blockSize);
        offset += count;
      }
      m_size = offset;
    }

    [[nodiscard]] size_t size() const { return m_size; }

    [[nodiscard]] decltype(auto) input(size_t i) const
    {
      const auto c = channel_of(i);
      const auto& channel = m_input.channel(c);
      if constexpr (CoreRank == 0)
      {
        return channel.flat_value(i - m_offsets[c]);
      }
      else
      {
        // The block is copied into scratch for this worker thread; the
        // operation uses it before the thread makes its next call.
        thread_local std::vector<In> scratch;
        const auto coreSize = m_coreSizes[c];
        const auto first = (i - m_offsets[c]) * coreSize;
        scratch.resize(coreSize);
        for (size_t e = 0; e < coreSize; ++e)
        {
          scratch[e] = channel.flat_value(first + e);
        }
        return CoreBlock<In>{std::span<const In>(scratch.data(), coreSize), m_cores[c]};
      }
    }

    [[nodiscard]] Out* block(size_t i)
    {
      const auto c = channel_of(i);
      return m_outputs[c].data() + (i - m_offsets[c]) * m_blockSize;
    }

    [[nodiscard]] result_type finish()
    {
      std::vector<Channel<Out>> channels;
      for (size_t c = 0; c < m_input.n_channels(); ++c)
      {
        const auto& source = m_input.channel(c);
        auto shape = m_batches[c];
        shape.insert(shape.end(), m_extraDims.begin(), m_extraDims.end());
        channels.emplace_back(source.times(), std::move(shape), std::move(m_outputs[c]),
                              source.snap_tolerance(), source.snap_eps());
      }
      return TimeSeries<Out>(std::move(channels), m_input.unit());
    }

  private:
    /// The channel holding call i: the last one starting at or before it.
    [[nodiscard]] size_t channel_of(size_t i) const
    {
      return static_cast<size_t>(std::upper_bound(m_offsets.begin(), m_offsets.end(), i) - m_offsets.begin()) - 1;
    }

    const TimeSeries<In>& m_input;
    std::vector<size_t> m_extraDims;
    size_t m_blockSize;
    std::vector<size_t> m_offsets;
    std::vector<std::vector<size_t>> m_batches;
    std::vector<std::vector<size_t>> m_cores;
    std::vector<size_t> m_coreSizes;
    std::vector<std::vector<Out>> m_outputs;
    size_t m_size = 0;
  };

  /// Applies an operation to every element of a container in parallel; the
  /// container's UfuncAdapter decides the result's shape.
  template <Operation Op, typename Container = Tensor<typename Op::input_type>>
  class UfuncTask : public StoppableTask<void>
  {
  public:
    using input_type = typename Op::input_type;
    using output_type = typename Op::output_type;
    using Adapter = UfuncAdapter<Container, core_rank_v<Op>, output_type>;

    UfuncTask(Container input, Op op, std::string description, std::string unit)
      : m_input(std::move(input)), m_op(std::move(op)),
        m_description(std::move(description)), m_unit(std::move(unit))
    { }

    /// The output, available once the task has finished.
    [[nodiscard]] const typename Adapter::result_type& result() const
    {
      if (!m_result)
      {
        throw std::logic_error("The task has not finished");
      }
      return *m_result;
    }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      const auto extraDims = std::vector<size_t>(m_op.extra_dims());
      const size_t blockSize = std::accumulate(
          extraDims.begin(), extraDims.end(), size_t{1}, std::multiplies<>());
      m_adapter.emplace(m_input, extraDims);

      const size_t n = m_adapter->size();
      next_step(n, m_description, m_unit);

      tf::Taskflow flow;
      auto apply = flow.for_each_index(size_t{0}, n, size_t{1}, [this, blockSize](size_t i) {
        if (stop_requested())
        {
          return;
        }
        m_op(m_adapter->input(i), std::span<output_type>(m_adapter->block(i), blockSize));
        add_progress(1);
      });
      auto finish = flow.emplace([this] { m_result = m_adapter->finish(); });
      apply.precede(finish);
      return exec.cpu()->run(std::move(flow));
    }

    Container m_input;
    Op m_op;
    std::optional<Adapter> m_adapter;
    std::optional<typename Adapter::result_type> m_result;
    std::string m_description;
    std::string m_unit;
  };

  /// A task applying @p op to every element of @p input, a tensor or a time
  /// series; the container type is deduced.
  template <typename Container, Operation Op>
  auto make_ufunc_task(Container input, Op op, std::string description, std::string unit)
  {
    return std::make_unique<UfuncTask<Op, Container>>(
        std::move(input), std::move(op), std::move(description), std::move(unit));
  }
}

#endif // STABLEBEAR_UFUNC_H
