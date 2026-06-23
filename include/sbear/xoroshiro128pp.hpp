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
