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
