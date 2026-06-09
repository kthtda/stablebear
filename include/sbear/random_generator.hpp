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

#ifndef SB_RANDOM_GENERATOR_H
#define SB_RANDOM_GENERATOR_H

#include "xoroshiro128pp.hpp"

#include <cstddef>
#include <cstdint>
#include <random>

namespace sb
{

  namespace detail
  {
    inline uint64_t splitmix64(uint64_t x)
    {
      /*
        splitmix64 is based on Sebastiano Vigna's reference implementation
        (available at https://prng.di.unimi.it/splitmix64.c), which is licensed
        as follows:

          Written in 2015 by Sebastiano Vigna (vigna@acm.org)

          To the extent possible under law, the author has dedicated all copyright
          and related and neighboring rights to this software to the public domain
          worldwide.

          Permission to use, copy, modify, and/or distribute this software for any
          purpose with or without fee is hereby granted.

          THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
          WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
          MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
          ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
          WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
          ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
          IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
      */

      x += 0x9e3779b97f4a7c15ULL;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
      return x ^ (x >> 31);
    }

    /// Default: hash seed via splitmix64 and forward to single-argument constructor.
    template <typename EngineT>
    EngineT make_engine(uint64_t seed)
    {
      return EngineT(splitmix64(seed));
    }

    /// Xoroshiro128pp: chain splitmix64 twice to fill both state words.
    template <>
    inline Xoroshiro128pp make_engine<Xoroshiro128pp>(uint64_t seed)
    {
      uint64_t s0 = splitmix64(seed);
      uint64_t s1 = splitmix64(s0);
      return Xoroshiro128pp(s0, s1);
    }
  }

  template <typename EngineT = Xoroshiro128pp>
  class RandomGenerator
  {
  public:
    using engine_type = EngineT;

    RandomGenerator()
      : m_seed(std::random_device{}())
    {
    }

    explicit RandomGenerator(uint64_t seed)
      : m_seed(seed)
    {
    }

    void seed(uint64_t seed) noexcept { m_seed = seed; }

    [[nodiscard]] EngineT sub_generator(size_t flatIndex) const
    {
      return detail::make_engine<EngineT>(m_seed + flatIndex);
    }

  private:
    uint64_t m_seed;
  };

  using DefaultRandomGenerator = RandomGenerator<Xoroshiro128pp>;

  inline DefaultRandomGenerator& default_generator()
  {
    static DefaultRandomGenerator gen;
    return gen;
  }

  inline void seed(uint64_t s)
  {
    default_generator().seed(s);
  }

}

#endif
