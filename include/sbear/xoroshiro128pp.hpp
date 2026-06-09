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

#ifndef SB_XOROSHIRO128PP_H
#define SB_XOROSHIRO128PP_H

#include "detail/xoroshiro128pp_impl.hpp"

#include <cstdint>
#include <limits>

namespace sb
{

  /// C++ UniformRandomBitGenerator wrapping xoroshiro128++ (Blackman & Vigna, 2019).
  class Xoroshiro128pp
  {
  public:
    using result_type = uint64_t;

    Xoroshiro128pp(uint64_t s0, uint64_t s1)
      : m_s0(s0)
      , m_s1(s1)
    {
    }

    uint64_t operator()()
    {
      return detail::xoroshiro128pp::next(m_s0, m_s1);
    }

    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return std::numeric_limits<uint64_t>::max(); }

  private:
    uint64_t m_s0;
    uint64_t m_s1;
  };

}

#endif
