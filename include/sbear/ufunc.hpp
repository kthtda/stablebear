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
#include <utility>
#include <vector>

namespace sb
{
  /// An element-wise operation in the style of a NumPy (g)ufunc. It maps one
  /// input element to a block of output elements of shape extra_dims(), the
  /// same for every element. Parameters are bound in the operation.
  template <typename Op>
  concept Operation = requires(const Op& op, const typename Op::input_type& in,
                               std::span<typename Op::output_type> out)
  {
    { op.extra_dims() } -> std::convertible_to<std::vector<size_t>>;
    op(in, out);
  };

  /// Says where a container's input elements are and how to build the output
  /// of an operation with output element type Out. An adapter numbers the
  /// input elements 0, ..., size() - 1; element(i) is input element i, and
  /// block(i) points at its block of output elements, which finish() turns
  /// into the result.
  template <typename Container, typename Out>
  class UfuncAdapter;

  /// Tensor: the result has the input's shape followed by extra_dims.
  template <typename In, TensorProperties Properties, typename Out>
  class UfuncAdapter<Tensor<In, Properties>, Out>
  {
  public:
    using result_type = Tensor<Out>;

    UfuncAdapter(const Tensor<In, Properties>& input, const std::vector<size_t>& extraDims)
      : m_input(input),
        m_blockSize(std::accumulate(extraDims.begin(), extraDims.end(), size_t{1}, std::multiplies<>()))
    {
      auto shape = std::vector<size_t>(input.shape().begin(), input.shape().end());
      shape.insert(shape.end(), extraDims.begin(), extraDims.end());
      m_output = Tensor<Out>(shape);
    }

    /// A 0-d input is one element, although Tensor::size() reports 0 for it.
    [[nodiscard]] size_t size() const
    {
      return std::accumulate(m_input.shape().begin(), m_input.shape().end(), size_t{1}, std::multiplies<>());
    }

    [[nodiscard]] decltype(auto) element(size_t i) const { return m_input.flat(i); }

    /// The output is freshly allocated, so it is contiguous and element i's
    /// block starts at i * blockSize.
    [[nodiscard]] Out* block(size_t i) { return m_output.data() + i * m_blockSize; }

    [[nodiscard]] result_type finish() { return std::move(m_output); }

  private:
    const Tensor<In, Properties>& m_input;
    size_t m_blockSize;
    Tensor<Out> m_output;
  };

  /// Time series: the elements of all channels' observations, channel by
  /// channel. Each output channel keeps its input's times and snap settings,
  /// and its shape is the input shape followed by extra_dims; the result keeps
  /// the query unit.
  template <typename In, typename Out>
  class UfuncAdapter<TimeSeries<In>, Out>
  {
  public:
    using result_type = TimeSeries<Out>;

    UfuncAdapter(const TimeSeries<In>& input, const std::vector<size_t>& extraDims)
      : m_input(input), m_extraDims(extraDims),
        m_blockSize(std::accumulate(extraDims.begin(), extraDims.end(), size_t{1}, std::multiplies<>()))
    {
      size_t offset = 0;
      for (size_t c = 0; c < input.n_channels(); ++c)
      {
        m_offsets.push_back(offset);
        const auto count = input.channel(c).flat_values().size();
        m_outputs.emplace_back(count * m_blockSize);
        offset += count;
      }
      m_size = offset;
    }

    [[nodiscard]] size_t size() const { return m_size; }

    [[nodiscard]] const In& element(size_t i) const
    {
      const auto c = channel_of(i);
      return m_input.channel(c).flat_values()[i - m_offsets[c]];
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
        auto shape = source.shape();
        shape.insert(shape.end(), m_extraDims.begin(), m_extraDims.end());
        channels.emplace_back(source.times(), std::move(shape), std::move(m_outputs[c]),
                              source.snap_tolerance(), source.snap_eps());
      }
      return TimeSeries<Out>(std::move(channels), m_input.unit());
    }

  private:
    /// The channel holding element i: the last one starting at or before it.
    [[nodiscard]] size_t channel_of(size_t i) const
    {
      return static_cast<size_t>(std::upper_bound(m_offsets.begin(), m_offsets.end(), i) - m_offsets.begin()) - 1;
    }

    const TimeSeries<In>& m_input;
    std::vector<size_t> m_extraDims;
    size_t m_blockSize;
    std::vector<size_t> m_offsets;
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
    using Adapter = UfuncAdapter<Container, output_type>;

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
        m_op(m_adapter->element(i), std::span<output_type>(m_adapter->block(i), blockSize));
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
