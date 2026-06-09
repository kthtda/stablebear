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

#ifndef SB_RECTANGLE_H
#define SB_RECTANGLE_H

#include <ostream>

namespace sb
{
  template <typename Tt, typename Tv>
  struct Rectangle
  {
    Tt left = 0;
    Tt right = 0;
    Tv f_value = 0;
    Tv g_value = 0;

    Rectangle() = default;
    Rectangle(Tt l, Tt r, Tv fv, Tv gv)
      : left(l), right(r), f_value(fv), g_value(gv)
    { }

    // Builder methods for readable construction in tests:
    //   Rectangle<T,T>().l(0).r(1).fv(2).gv(3)
    Rectangle& l(Tt t) { left = t; return *this; }
    Rectangle& r(Tt t) { right = t; return *this; }
    Rectangle& fv(Tv v) { f_value = v; return *this; }
    Rectangle& gv(Tv v) { g_value = v; return *this; }

    bool operator==(const Rectangle& rhs) const
    {
      return left == rhs.left && right == rhs.right && f_value == rhs.f_value && g_value == rhs.g_value;
    }

    bool operator!=(const Rectangle& rhs) const
    {
      return left != rhs.left || right != rhs.right || f_value != rhs.f_value || g_value != rhs.g_value;
    }
    
    template <typename Ut, typename Uv>
    friend std::ostream& operator<<(std::ostream&, const Rectangle<Ut, Uv>&);
  };

  template <typename Tt, typename Tv>
  struct Segment
  {
    Tt left = 0;
    Tt right = 0;
    Tv value = 0;

    Segment() = default;
    Segment(Tt l, Tt r, Tv v)
      : left(l), right(r), value(v)
    { }
  };
  
  template <typename Tt, typename Tv>
  std::ostream& operator<<(std::ostream& os, const sb::Rectangle<Tt, Tv>& rect)
  {
    os << "Rectangle(.l = " << rect.left << ", .r = " << rect.right << ", .fv = " << rect.f_value << ", .gv = " << rect.g_value << ")";
    return os;
  }
}

#endif
