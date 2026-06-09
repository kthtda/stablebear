/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef PERSISTENCE_PAIR_H
#define PERSISTENCE_PAIR_H

#include <limits>
#include <vector>

namespace sb::ph
{
  template <typename T>
  struct PersistencePair
  {
    using value_type = T;

    value_type birth = static_cast<value_type>(0.);
    value_type death = std::numeric_limits<value_type>::infinity();

    PersistencePair() = default;
    explicit PersistencePair(value_type b, value_type d = std::numeric_limits<value_type>::infinity()) : birth(b), death(d) { }

    [[nodiscard]] bool isDeathFinite() const noexcept { return death != std::numeric_limits<value_type>::infinity(); }

    [[nodiscard]] bool operator==(const PersistencePair& rhs) const
    {
      return birth == rhs.birth && death == rhs.death;
    }

    /**
     * Lexicographical order on (birth, death)
     * @param rhs `PersistencePair` to compare against
     * @return `true` if `*this` is lexicographically smaller than `rhs`
     */
    [[nodiscard]] bool operator<(const PersistencePair& rhs) const noexcept
    {
      return birth < rhs.birth || (birth == rhs.birth && death < rhs.death);
    }
  };

}

#endif //PERSISTENCE_PAIR_H
