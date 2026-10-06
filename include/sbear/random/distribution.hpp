#ifndef STABLEBEAR_RANDOM_DISTRIBUTION_H
#define STABLEBEAR_RANDOM_DISTRIBUTION_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace sb::random
{

  namespace detail
  {
    inline constexpr long double negativeInfinity = -std::numeric_limits<long double>::infinity();

    inline long double log_add(long double a, long double b)
    {
      if (a == negativeInfinity)
        return b;
      if (b == negativeInfinity)
        return a;
      const auto maximum = std::max(a, b);
      return maximum + std::log1p(std::exp(std::min(a, b) - maximum));
    }

    /// Shift log weights so their probabilities sum to one. A row that is
    /// all -inf (empty support) is left unchanged.
    inline void normalize_log_row(std::span<long double> row)
    {
      long double maximum = negativeInfinity;
      for (const auto value : row)
        maximum = std::max(maximum, value);
      if (!std::isfinite(maximum))
        return;
      long double sum = 0;
      for (auto& value : row)
      {
        // Shift first: adding log(sum) to an enormous negative maximum can
        // lose it entirely, even when several far-tail points have equal mass.
        value -= maximum;
        sum += std::exp(value);
      }
      const auto logSum = std::log(sum);
      for (auto& value : row)
        value -= logSum;
    }
  }

  /// Normalized Gaussian density. log_weight stays finite many standard
  /// deviations from the mean, where the density itself underflows to zero
  /// (past ~38 sigma in float64).
  class Gaussian
  {
  public:
    /// Standard normal; also makes Distribution default-constructible.
    Gaussian() = default;

    Gaussian(double mean, double standardDeviation) : m_mean(mean), m_sigma(standardDeviation)
    {
      if (!std::isfinite(mean) || !std::isfinite(standardDeviation) || standardDeviation <= 0)
        throw std::invalid_argument("Gaussian requires a finite mean and finite positive std");
    }

    [[nodiscard]] long double mean() const noexcept { return m_mean; }
    [[nodiscard]] long double standard_deviation() const noexcept { return m_sigma; }

    /// Density at @p value; zero where it underflows or the exponent overflows.
    [[nodiscard]] long double weight(long double value) const noexcept
    {
      return std::exp(log_density(value));
    }

    /// Log density at a finite @p value. Throws if the exponent overflows.
    [[nodiscard]] long double log_weight(long double value) const
    {
      const auto result = log_density(value);
      if (!std::isfinite(result))
        throw std::overflow_error("Gaussian log weight exceeds numerical range");
      return result;
    }

  private:
    [[nodiscard]] long double log_density(long double value) const noexcept
    {
      const auto d = (value - m_mean) / m_sigma;
      return -0.5L * d * d - std::log(m_sigma) - 0.5L * std::log(2 * std::numbers::pi_v<long double>);
    }

    long double m_mean = 0.L;
    long double m_sigma = 1.L;
  };

  /// Density 1 / (end - start) on a finite interval, or weight 1 on an
  /// unbounded interval. Both cases are zero outside [start, end).
  class Uniform
  {
  public:
    Uniform(double start, double end) : m_start(start), m_end(end)
    {
      if (std::isnan(start) || std::isnan(end) || !(start < end))
        throw std::invalid_argument("Uniform requires start < end (start may be -inf and end may be +inf)");
    }

    [[nodiscard]] long double start() const noexcept { return m_start; }
    [[nodiscard]] long double end() const noexcept { return m_end; }

    [[nodiscard]] long double weight(long double value) const noexcept
    {
      if (!(value >= m_start && value < m_end))
        return 0;
      if (!std::isfinite(m_start) || !std::isfinite(m_end))
        return 1;
      const auto width = m_end - m_start;
      if (std::isfinite(width))
        return 1 / width;
      // Finite endpoints can have a width exceeding the floating-point range.
      return 0.5L / (m_end / 2 - m_start / 2);
    }

    [[nodiscard]] long double log_weight(long double value) const noexcept
    {
      return std::log(weight(value));
    }

  private:
    long double m_start;
    long double m_end;
  };

  class Mixture;

  /// Any built-in distribution. Interfaces take this variant and dispatch
  /// once per row rather than once per value.
  using Distribution = std::variant<Gaussian, Uniform, Mixture>;

  /// Weighted sum of component distributions. Coefficients are normalized to
  /// sum to one at construction, independently of any evaluation points.
  /// Zero-coefficient components are kept but never evaluated.
  class Mixture
  {
  public:
    explicit Mixture(const std::vector<std::pair<double, Distribution>>& terms);

    [[nodiscard]] const std::vector<std::pair<long double, Distribution>>& terms() const noexcept { return m_terms; }

    [[nodiscard]] long double weight(long double value) const;

    [[nodiscard]] long double log_weight(long double value) const;

  private:
    std::vector<std::pair<long double, Distribution>> m_terms;
  };

  inline Mixture::Mixture(const std::vector<std::pair<double, Distribution>>& terms)
  {
    if (terms.empty())
      throw std::invalid_argument("Mixture needs at least one component");
    long double maximum = 0;
    for (const auto& [coefficient, component] : terms)
    {
      if (!std::isfinite(coefficient) || coefficient < 0)
        throw std::invalid_argument("Mixture coefficients must be finite and nonnegative");
      maximum = std::max<long double>(maximum, coefficient);
    }
    if (maximum == 0)
      throw std::invalid_argument("Mixture needs at least one positive coefficient");
    // Scale by the maximum first so the sum cannot overflow.
    long double total = 0;
    for (const auto& [coefficient, component] : terms)
      total += m_terms.emplace_back(coefficient / maximum, component).first;
    for (auto& term : m_terms)
      term.first /= total;
  }

  inline long double Mixture::weight(long double value) const
  {
    long double sum = 0;
    for (const auto& [coefficient, component] : m_terms)
    {
      if (coefficient > 0)
      {
        sum += coefficient * std::visit([&](const auto& d) { return d.weight(value); }, component);
      }
    }
    return sum;
  }

  inline long double Mixture::log_weight(long double value) const
  {
    long double result = detail::negativeInfinity;
    for (const auto& [coefficient, component] : m_terms)
    {
      if (coefficient > 0)
      {
        result = detail::log_add(result, std::log(coefficient)
          + std::visit([&](const auto& d) { return d.log_weight(value); }, component));
      }
    }
    return result;
  }

  /// Fill @p row with the log probabilities of @p values under @p distribution,
  /// normalized over the values to sum to one. Tiny positive weights stay
  /// finite. Infinite values and values outside the support get -inf; if no
  /// value has support, the row remains all -inf.
  inline void log_weights(const Distribution& distribution, std::span<const long double> values,
                          std::span<long double> row)
  {
    if (row.size() != values.size())
      throw std::invalid_argument("value and weight rows must have equal length");
    if (std::ranges::any_of(values, [](long double v) { return std::isnan(v); }))
      throw std::invalid_argument("values must not be NaN");
    std::visit([&](const auto& d) {
      for (size_t i = 0; i < values.size(); ++i)
        row[i] = std::isinf(values[i]) ? detail::negativeInfinity : d.log_weight(values[i]);
    }, distribution);
    detail::normalize_log_row(row);
  }

} // namespace sb::random

#endif
