#ifndef STABLEBEAR_VERSION_H
#define STABLEBEAR_VERSION_H

#include <string>

namespace sb
{
  // These get automatically generated from version.cpp.in in the project root directory via CMake. The generated file
  // containing the definitions is in the binary output dir (version.cpp)

  extern const std::string PROJECT_NAME;
  extern const std::string PROJECT_TITLE;
  extern const std::string PROJECT_VERSION;
  extern const std::string PROJECT_VERSION_FULL;
  extern const std::string PROJECT_BUILD_DATE;
}

#endif //STABLEBEAR_VERSION_H
