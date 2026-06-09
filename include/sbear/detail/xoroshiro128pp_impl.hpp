/*
  xoroshiro128++ 1.0 reference implementation by David Blackman and
  Sebastiano Vigna (available at
  https://prng.di.unimi.it/xoroshiro128plusplus.c). We keep only the parts
  we use (no jump methods) and namespace it.

  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

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

#ifndef SB_DETAIL_XOROSHIRO128PP_IMPL_H
#define SB_DETAIL_XOROSHIRO128PP_IMPL_H

#include <cstdint>

namespace sb::detail::xoroshiro128pp
{

  inline uint64_t rotl(uint64_t x, int k)
  {
    return (x << k) | (x >> (64 - k));
  }

  inline uint64_t next(uint64_t& s0, uint64_t& s1)
  {
    const uint64_t result = rotl(s0 + s1, 17) + s0;

    s1 ^= s0;
    s0 = rotl(s0, 49) ^ s1 ^ (s1 << 21); // a, b
    s1 = rotl(s1, 28); // c

    return result;
  }

}

#endif
