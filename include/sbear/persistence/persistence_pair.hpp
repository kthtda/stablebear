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
