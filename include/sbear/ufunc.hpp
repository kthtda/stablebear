#ifndef STABLEBEAR_UFUNC_H
#define STABLEBEAR_UFUNC_H

#include "executor.hpp"
#include "task.hpp"
#include "tensor.hpp"

#include <taskflow/algorithm/for_each.hpp>

#include <functional>
#include <numeric>
#include <span>
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

  /// Applies an operation to every element of a tensor in parallel. The result
  /// has the input's shape followed by the operation's extra_dims().
  template <Operation Op, TensorProperties Properties = TensorProperty::None>
  class UfuncTask : public StoppableTask<void>
  {
  public:
    using input_type = typename Op::input_type;
    using output_type = typename Op::output_type;

    UfuncTask(Tensor<input_type, Properties> input, Op op,
              std::string description, std::string unit)
      : m_input(std::move(input)), m_op(std::move(op)),
        m_description(std::move(description)), m_unit(std::move(unit))
    { }

    /// The output; complete once the task has finished.
    [[nodiscard]] const Tensor<output_type>& result() const { return m_output; }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      const auto extraDims = std::vector<size_t>(m_op.extra_dims());
      const size_t blockSize = std::accumulate(
          extraDims.begin(), extraDims.end(), size_t{1}, std::multiplies<>());

      auto shape = std::vector<size_t>(m_input.shape().begin(), m_input.shape().end());
      shape.insert(shape.end(), extraDims.begin(), extraDims.end());
      m_output = Tensor<output_type>(shape);

      // A 0-d input is one element, although Tensor::size() reports 0 for it.
      const size_t n = std::accumulate(m_input.shape().begin(), m_input.shape().end(),
                                       size_t{1}, std::multiplies<>());
      next_step(n, m_description, m_unit);

      // The output is freshly allocated, so it is contiguous and element i's
      // block starts at i * blockSize.
      tf::Taskflow flow;
      flow.for_each_index(size_t{0}, n, size_t{1}, [this, blockSize](size_t i) {
        if (stop_requested())
        {
          return;
        }
        m_op(m_input.flat(i),
             std::span<output_type>(m_output.data() + i * blockSize, blockSize));
        add_progress(1);
      });
      return exec.cpu()->run(std::move(flow));
    }

    Tensor<input_type, Properties> m_input;
    Op m_op;
    Tensor<output_type> m_output;
    std::string m_description;
    std::string m_unit;
  };
}

#endif // STABLEBEAR_UFUNC_H
