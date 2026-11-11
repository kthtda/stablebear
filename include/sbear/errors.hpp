#ifndef STABLEBEAR_ERRORS_H
#define STABLEBEAR_ERRORS_H

#include <stdexcept>

namespace sb
{
  /// An argument of the wrong type or kind, such as durations for a series
  /// without a unit. Python raises it as a TypeError.
  class TypeError : public std::invalid_argument
  {
  public:
    using std::invalid_argument::invalid_argument;
  };
}

#endif // STABLEBEAR_ERRORS_H
