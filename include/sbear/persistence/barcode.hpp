// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef STABLEBEAR_BARCODE_H
#define STABLEBEAR_BARCODE_H

#include "persistence_pair.hpp"

#include <iosfwd>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace sb::ph
{
  template <typename T>
  class Barcode
  {
  public:
    [[nodiscard]] static bool is_infinite(T val)
    {
      return val == std::numeric_limits<T>::infinity() || val == std::numeric_limits<T>::max();
    }

    /**
     * Constructs an empty barcode (i.e., a barcode with no bars)
     */
    Barcode() = default;

    explicit Barcode(const std::vector<sb::ph::PersistencePair<T>>& bars)
      : m_bars(bars)
    { }

    explicit Barcode(std::vector<sb::ph::PersistencePair<T>>&& bars)
        : m_bars(std::move(bars))
    { }

    /// Converting constructor: convert from a Barcode with different precision.
    template <typename U>
    explicit Barcode(const Barcode<U>& other)
    {
      m_bars.reserve(other.bars().size());
      for (const auto& bar : other.bars())
        m_bars.emplace_back(static_cast<T>(bar.birth), static_cast<T>(bar.death));
    }

    Barcode& operator=(const std::vector<PersistencePair<T>>& bars)
    {
      m_bars = bars;
      return *this;
    }

    Barcode& operator=(std::vector<PersistencePair<T>>&& bars)
    {
      m_bars = std::move(bars);
      return *this;
    }

    /**
     * Does a 1-1 comparison between two `Barcode` objects. For efficiency reasons, equality is only `true` if the bars
     * occur *in the same order*, even though, mathematically, this doesn't matter. For a more proper (but slower)
     * comparison, see `is_isomorphic_to`.
     * @param rhs The `Barcode` to compare against.
     * @return `true` if all bars are the same and occur in the same order, otherwise `false`.
     */
    [[nodiscard]] bool operator==(const Barcode& rhs) const
    {
      return m_bars == rhs.m_bars;
    }

    /**
     * Tests whether two barcodes are isomorphic, i.e. whether their bars agree
     * as multisets (order independent). Endpoints are compared with a numerical
     * tolerance so that barcodes computed via different (but mathematically
     * equivalent) routes — e.g. from a point cloud versus a precomputed distance
     * matrix — still compare equal despite floating-point rounding. Two endpoints
     * `a` and `b` are considered equal when `|a - b| <= atol + rtol * |b|`, with
     * infinite endpoints matched exactly. Pass `atol = rtol = 0` for a bitwise
     * comparison.
     * @param rhs The `Barcode` to compare against.
     * @param atol Absolute tolerance for endpoint comparison.
     * @param rtol Relative tolerance for endpoint comparison.
     * @return `true` if the barcodes are isomorphic within tolerance.
     */
    [[nodiscard]] bool is_isomorphic_to(const Barcode& rhs, double atol = 1e-8, double rtol = 1e-5) const
    {
      if (m_bars.size() != rhs.m_bars.size())
      {
        return false;
      }

      std::vector<sb::ph::PersistencePair<T>> thisBars(m_bars);
      std::vector<sb::ph::PersistencePair<T>> rhsBars(rhs.m_bars);

      std::sort(thisBars.begin(), thisBars.end());
      std::sort(rhsBars.begin(), rhsBars.end());

      // Exact comparison: a single contiguous compare of the sorted bars,
      // avoiding the per-endpoint tolerance branches below.
      if (atol == 0.0 && rtol == 0.0)
      {
        return thisBars == rhsBars;
      }

      auto endpointsClose = [atol, rtol](T a, T b) {
        bool aInf = is_infinite(a);
        bool bInf = is_infinite(b);
        if (aInf || bInf)
          return aInf && bInf; // infinities must match exactly
        auto da = static_cast<double>(a);
        auto db = static_cast<double>(b);
        return std::abs(da - db) <= atol + rtol * std::abs(db);
      };

      for (std::size_t i = 0; i < thisBars.size(); ++i)
      {
        if (!endpointsClose(thisBars[i].birth, rhsBars[i].birth) ||
            !endpointsClose(thisBars[i].death, rhsBars[i].death))
        {
          return false;
        }
      }

      return true;
    }

    [[nodiscard]] const std::vector<sb::ph::PersistencePair<T>>& bars() const { return m_bars; }

  private:
    std::vector<sb::ph::PersistencePair<T>> m_bars;
  };

  template <typename T>
  std::ostream& operator<<(std::ostream& os, const Barcode<T>& barcode)
  {
    os << "{";

    constexpr const int nBarsBeforeNewline = 10;
    int nBars = nBarsBeforeNewline;

    for (auto it = barcode.bars().begin(); it != barcode.bars().end(); ++it)
    {
      auto const & bar = *it;

      os << "[";

      if (bar.birth == -std::numeric_limits<T>::infinity() || bar.birth == -std::numeric_limits<T>::max())
      {
        os << "-oo";
      }
      else
      {
        os << bar.birth;
      }

      os << ", ";

      if (bar.death == std::numeric_limits<T>::infinity() || bar.death == std::numeric_limits<T>::max())
      {
        os << "oo";
      }
      else
      {
        os << bar.death;
      }

      os << ")";

      if (std::next(it) != barcode.bars().end())
      {
        if (--nBars == 0)
        {
          os << ",\n  ";
          nBars = nBarsBeforeNewline;
        }
        else
        {
          os << ", ";
        }
      }

    }
    os << "}";

    return os;
  }
}

#endif //STABLEBEAR_BARCODE_H