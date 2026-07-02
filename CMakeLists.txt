cmake_minimum_required(VERSION 3.19)

include(FetchContent)

# Force MSVC release CRT (/MD) for wheel builds. Must be set before project() so it applies to all targets.
# CMP0091 NEW (default since cmake_minimum_required 3.15) makes this honor CMAKE_MSVC_RUNTIME_LIBRARY.
# Without this, conda compiler-activation env vars (vs2017_win-64 pulled in by cuda-compiler) can leak
# debug CRT into the link, producing wheels that bundle non-redistributable
# msvcp140d/vcruntime140d/ucrtbased. Local non-SKBUILD builds keep CMake default behavior.
if (WIN32 AND SKBUILD)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING "MSVC runtime library")
endif ()

if (SKBUILD)
  message(STATUS "!!! Not Minimal module build (SKBUILD ON) !!!")
else ()
  message(STATUS "!!! Minimal module build (SKBUILD OFF) !!!")
endif ()

message(STATUS "!!! CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "!!! CMAKE_MSVC_RUNTIME_LIBRARY: ${CMAKE_MSVC_RUNTIME_LIBRARY}")

# Probe for a physical NVIDIA GPU via nvidia-smi (shipped with the driver).
# This is only used to pick the auto-default below; an explicit
# BUILD_WITH_CUDA=1 still forces a CUDA build even with no GPU present
# (e.g. CI hosts that cross-compile CUDA without a card).
function (detect_nvidia_gpu OUTVAR)
  find_program(_NVIDIA_SMI nvidia-smi)
  if (_NVIDIA_SMI)
    execute_process(
      COMMAND "${_NVIDIA_SMI}" -L
      OUTPUT_VARIABLE _SMI_OUT
      RESULT_VARIABLE _SMI_RES
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (_SMI_RES EQUAL 0 AND _SMI_OUT MATCHES "GPU")
      set(${OUTVAR} ON PARENT_SCOPE)
      return ()
    endif ()
  endif ()
  set(${OUTVAR} OFF PARENT_SCOPE)
endfunction ()

if (DEFINED ENV{BUILD_WITH_CUDA})
  if ($ENV{BUILD_WITH_CUDA} MATCHES "0")
    set(BUILD_WITH_CUDA OFF)
  elseif (APPLE)
    set(BUILD_WITH_CUDA OFF)
  else ()
    set(BUILD_WITH_CUDA ON)
  endif ()
elseif (APPLE)
  set(BUILD_WITH_CUDA OFF)
else ()
  # No explicit request: only build CUDA if an NVIDIA GPU is detected.
  detect_nvidia_gpu(_HAS_NVIDIA_GPU)
  if (_HAS_NVIDIA_GPU)
    set(BUILD_WITH_CUDA ON)
    message(STATUS "!!! NVIDIA GPU detected; defaulting BUILD_WITH_CUDA=ON")
  else ()
    set(BUILD_WITH_CUDA OFF)
    message(STATUS "!!! No NVIDIA GPU detected; defaulting BUILD_WITH_CUDA=OFF (set BUILD_WITH_CUDA=1 to force)")
  endif ()
endif ()

if ((DEFINED ENV{SKIP_STUBGEN}) AND (NOT $ENV{SKIP_STUBGEN} MATCHES "0"))
  set(SKIP_STUBGEN ON)
else ()
  set(SKIP_STUBGEN OFF)
endif ()

if ((DEFINED ENV{SKIP_BACK_COPY}) AND (NOT $ENV{SKIP_BACK_COPY} MATCHES "0"))
  set(SKIP_BACK_COPY ON)
else ()
  set(SKIP_BACK_COPY OFF)
endif ()

if (SKBUILD)
  set(BUILD_TESTER OFF)
  set(SKIP_STUBGEN ON) # Too many problems with stubgen and skbuild at the moment, we don't really need the stubs anyway in the release since the C++ bits are all internal...
else ()
  set(BUILD_TESTER ON)
endif ()

set(BUILD_BENCHMARKS OFF)

set(Python_FIND_VIRTUALENV STANDARD)
find_package(Python 3.10 REQUIRED COMPONENTS Development.Module Interpreter)


# --- Windows/NuGet hardening: ensure Python import library is findable by the linker ---
# pybind11 headers emit #pragma comment(lib, "pythonXX.lib") on MSVC, which requires the
# Python libs/ directory to be in the linker search path.  Pre-installed Python versions
# (3.10-3.12) live under C:\hostedtoolcache\windows\Python\ which MSVC already searches,
# but newer versions installed only via the cibuildwheel NuGet cache are not in that path.
# Always add sys.base_prefix/libs unconditionally on Windows.
if (WIN32)
  execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.base_prefix)"
      OUTPUT_VARIABLE _PY_BASE_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if (_PY_BASE_PREFIX AND EXISTS "${_PY_BASE_PREFIX}/libs")
    link_directories("${_PY_BASE_PREFIX}/libs")
    message(STATUS "!!! Python libs dir added to linker path: ${_PY_BASE_PREFIX}/libs")

    # For free-threaded Python (e.g. cp314t), the import library is named
    # python314t.lib, but Python's own pyconfig.hpp emits:
    #   #pragma comment(lib, "python314.lib")
    # which causes LNK1104.  Create a copy with the expected name in the
    # build directory and add that directory to the linker search path.
    set(_v_base "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
    set(_ft_lib "${_PY_BASE_PREFIX}/libs/${_v_base}t.lib")
    set(_std_lib "${_PY_BASE_PREFIX}/libs/${_v_base}.lib")
    if (EXISTS "${_ft_lib}" AND NOT EXISTS "${_std_lib}")
      set(_py_alias_dir "${CMAKE_CURRENT_BINARY_DIR}/_py_lib_alias")
      file(MAKE_DIRECTORY "${_py_alias_dir}")
      execute_process(
          COMMAND "${CMAKE_COMMAND}" -E copy "${_ft_lib}" "${_py_alias_dir}/${_v_base}.lib"
      )
      link_directories("${_py_alias_dir}")
      message(STATUS "!!! Created ${_v_base}.lib alias for free-threaded Python in ${_py_alias_dir}")
    endif()
  endif()

  # Always link the import library explicitly via Python::Module.  Up to Python 3.13,
  # pyconfig.h auto-linked pythonXY.lib through #pragma comment(lib, ...), so having the
  # libs/ directory on the linker path was enough.  CPython 3.14 removed that pragma,
  # which left extension modules with unresolved Py* externals unless the .lib is named
  # on the link line.  (Python_LIBRARY being set, e.g. by scikit-build-core hints, does
  # not guarantee FindPython propagates it to Python::Module for Development.Module.)
  execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(int(bool(sysconfig.get_config_var('Py_GIL_DISABLED'))))"
      OUTPUT_VARIABLE _PY_GIL_DISABLED
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(_v "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
  if (_PY_GIL_DISABLED STREQUAL "1")
    # Free-threaded build: prefer the t-suffixed import library.
    set(_PY_LIB_SEARCH_ORDER "${_v}t.lib" "${_v}t_d.lib" "${_v}.lib" "${_v}_d.lib")
  else()
    set(_PY_LIB_SEARCH_ORDER "${_v}.lib" "${_v}_d.lib" "${_v}t.lib" "${_v}t_d.lib")
  endif()
  set(_PY_LIB_CANDIDATE "")
  foreach(_name IN LISTS _PY_LIB_SEARCH_ORDER)
    if (EXISTS "${_PY_BASE_PREFIX}/libs/${_name}")
      set(_PY_LIB_CANDIDATE "${_PY_BASE_PREFIX}/libs/${_name}")
      break()
    endif()
  endforeach()
  if (NOT _PY_LIB_CANDIDATE)
    message(FATAL_ERROR "Could not locate Python import library (.lib) in ${_PY_BASE_PREFIX}/libs")
  endif()
  set(Python_LIBRARY "${_PY_LIB_CANDIDATE}" CACHE FILEPATH "Python import library" FORCE)
  message(STATUS "!!! Python import library resolved to: ${_PY_LIB_CANDIDATE}")
  if (TARGET Python::Module)
    set_property(TARGET Python::Module APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${_PY_LIB_CANDIDATE}")
  endif()
endif()


if (BUILD_WITH_CUDA)
  message(STATUS "!!! Building with CUDA !!!")

  # If CUDAToolkit_ROOT is not set, try to find a pip-installed CUDA toolkit
  if (NOT DEFINED CUDAToolkit_ROOT AND NOT DEFINED ENV{CUDAToolkit_ROOT})
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "if True:
        import importlib.util, pathlib
        for pkg in ('nvidia.cu13', 'nvidia.cu12', 'nvidia.cuda_nvcc', 'nvidia.cuda_runtime'):
            spec = importlib.util.find_spec(pkg)
            if spec and spec.submodule_search_locations:
                print(pathlib.Path(spec.submodule_search_locations[0]).resolve())
                break"
        OUTPUT_VARIABLE _PIP_CUDA_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if (_PIP_CUDA_ROOT)
      message(STATUS "!!! Found pip-installed CUDA toolkit: ${_PIP_CUDA_ROOT}")
      set(CUDAToolkit_ROOT "${_PIP_CUDA_ROOT}")
      set(CMAKE_CUDA_COMPILER "${_PIP_CUDA_ROOT}/bin/nvcc")

      # Pip-installed CUDA uses lib/ instead of lib64/; help nvcc and FindCUDAToolkit find the libraries
      if (EXISTS "${_PIP_CUDA_ROOT}/lib" AND NOT EXISTS "${_PIP_CUDA_ROOT}/lib64")
        # Pip CUDA ships only versioned .so files (e.g. libcudart.so.13).
        # Create unversioned symlinks in the build dir so the linker can
        # resolve -lcudart etc., without modifying the pip installation.
        set(_PIP_CUDA_LINK_DIR "${CMAKE_CURRENT_BINARY_DIR}/_pip_cuda_links")
        file(MAKE_DIRECTORY "${_PIP_CUDA_LINK_DIR}")
        file(GLOB _PIP_CUDA_VERSIONED_LIBS "${_PIP_CUDA_ROOT}/lib/lib*.so.*")
        foreach(_VLIB IN LISTS _PIP_CUDA_VERSIONED_LIBS)
          get_filename_component(_VLIB_NAME "${_VLIB}" NAME)
          string(REGEX REPLACE "\\.so\\..*" ".so" _UNVERSIONED_NAME "${_VLIB_NAME}")
          if (NOT EXISTS "${_PIP_CUDA_LINK_DIR}/${_UNVERSIONED_NAME}")
            file(CREATE_LINK "${_VLIB}" "${_PIP_CUDA_LINK_DIR}/${_UNVERSIONED_NAME}" SYMBOLIC)
          endif ()
        endforeach()

        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -L${_PIP_CUDA_ROOT}/lib -L${_PIP_CUDA_LINK_DIR}")
        link_directories("${_PIP_CUDA_LINK_DIR}" "${_PIP_CUDA_ROOT}/lib")
      endif ()
    endif ()
  endif ()

  SET(LANG_CUDA CUDA)

  if (NOT SKBUILD)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    # Query the CUDA version before project() enables the CUDA language,
    # because CMAKE_CUDA_ARCHITECTURES must be set before the try-compile
    # that happens during enable_language(CUDA). CUDAToolkit_VERSION is
    # not available until find_package(CUDAToolkit), which is too late.
    if (CMAKE_CUDA_COMPILER)
      set(_NVCC_PATH "${CMAKE_CUDA_COMPILER}")
    else()
      find_program(_NVCC_PATH nvcc HINTS ENV CUDA_PATH PATH_SUFFIXES bin)
    endif()
    if (_NVCC_PATH)
      execute_process(
        COMMAND "${_NVCC_PATH}" --version
        OUTPUT_VARIABLE _NVCC_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      string(REGEX MATCH "release ([0-9]+)\\.([0-9]+)" _NVCC_MATCH "${_NVCC_VERSION_OUTPUT}")
      set(_CUDA_MAJOR "${CMAKE_MATCH_1}")
      set(_CUDA_MINOR "${CMAKE_MATCH_2}")
      set(_CUDA_VER "${_CUDA_MAJOR}.${_CUDA_MINOR}")
      message(STATUS "!!! Detected CUDA ${_CUDA_VER} from nvcc for architecture selection")
    endif()

    # Build SASS for each listed architecture, plus PTX for the highest
    # so that newer GPUs can JIT-compile at first launch.
    #
    # Maxwell (5.0, 5.2) supported through CUDA 12.6, dropped in 12.7+.
    # Blackwell (100, 120) requires CUDA 12.8+.
    # Mirrors PyTorch's architecture list. Older GPUs (Pascal, Volta) and
    # in-between archs (Ada Lovelace sm_89) are covered by binary compatibility
    # or PTX JIT from sm_90.
    set(CMAKE_CUDA_ARCHITECTURES
      75-real
      80-real
      86-real
    )
    if (NOT _CUDA_VER OR _CUDA_VER VERSION_LESS "12.7")
      # Maxwell support: CUDA <= 12.6
      list(PREPEND CMAKE_CUDA_ARCHITECTURES 50-real)
    endif()
    if (_CUDA_VER VERSION_GREATER_EQUAL "12.8")
      # Blackwell native support
      list(APPEND CMAKE_CUDA_ARCHITECTURES 100-real 120-real)
    endif()
    # Emit PTX for the highest real arch for forward compatibility.
    # sm_90 is the common baseline for both CUDA 12 and 13 builds.
    list(APPEND CMAKE_CUDA_ARCHITECTURES 90)
  endif()
else ()
  message(STATUS "!!! Building without CUDA !!!")

  SET(LANG_CUDA)
endif ()

set(PROJ_NAME stablebear)

# Read version info out of pyproject.toml
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" TOML_CONTENT)

# 1. Find the [project] section
string(FIND "${TOML_CONTENT}" "[project]" PROJECT_SECTION_START)

# 2. Slice the string from [project] onwards to avoid [build-system] versions
string(SUBSTRING "${TOML_CONTENT}" ${PROJECT_SECTION_START} -1 PROJECT_BODY)

# 3. Extract the version string
if (PROJECT_BODY MATCHES [[version[ \t]*=[ \t]*"([^"]+)"]])
  set(PROJ_VERSION_FULL "${CMAKE_MATCH_1}")

  # Extract only the numeric part for CMake's project() command
  string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" PROJ_VERSION_NUMERIC "${PROJ_VERSION_FULL}")

  message(STATUS "!!! Detected Full Version: ${PROJ_VERSION_FULL}")
  message(STATUS "!!! Numeric Version for CMake: ${PROJ_VERSION_NUMERIC}")

  set(PROJ_VERSION "${PROJ_VERSION_NUMERIC}")
else ()
  message(FATAL "Could not detect version number from pyproject.toml!")
endif ()

project(${PROJ_NAME} VERSION ${PROJ_VERSION_NUMERIC} LANGUAGES CXX ${LANG_CUDA})

if (BUILD_WITH_CUDA)
  # After project() the CUDA compiler path is known; derive the toolkit root from it
  # if CUDAToolkit_ROOT is still unset (handles the case where nvcc is on PATH but the
  # toolkit root was not automatically detected).
  if (NOT CUDAToolkit_ROOT AND CMAKE_CUDA_COMPILER)
    get_filename_component(_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(_CUDA_DERIVED_ROOT "${_CUDA_BIN_DIR}" DIRECTORY)

    message(STATUS "Search ${_CUDA_DERIVED_ROOT}")
    if (EXISTS "${_CUDA_DERIVED_ROOT}/include/cuda_runtime.h")
      set(CUDAToolkit_ROOT "${_CUDA_DERIVED_ROOT}" CACHE PATH "CUDA toolkit root (derived from nvcc path)" FORCE)
      message(STATUS "!!! Derived CUDAToolkit_ROOT from CMAKE_CUDA_COMPILER: ${CUDAToolkit_ROOT}")
    endif ()
  endif ()

  if (_PIP_CUDA_ROOT AND EXISTS "${_PIP_CUDA_ROOT}/lib" AND NOT EXISTS "${_PIP_CUDA_ROOT}/lib64")
    # Pip-installed CUDA uses lib/ instead of lib64/ and ships only versioned
    # .so files (e.g. libcudart.so.13 instead of libcudart.so).
    # FindCUDAToolkit can't find these, so we locate them ourselves.
    file(GLOB _PIP_CUDART "${_PIP_CUDA_ROOT}/lib/libcudart.so*")
    if (_PIP_CUDART)
      list(GET _PIP_CUDART 0 _PIP_CUDART_LIB)
      set(CUDA_CUDART "${_PIP_CUDART_LIB}" CACHE FILEPATH "CUDA runtime library" FORCE)
    endif ()
  endif ()
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

  set(SB_CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDAToolkit_INCLUDE_DIRS})
  if (CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/include/cuda_runtime.h")
    list(APPEND SB_CUDA_INCLUDE_DIRS "${CUDAToolkit_ROOT}/include")
  endif ()
  list(REMOVE_DUPLICATES SB_CUDA_INCLUDE_DIRS)
  include_directories(${SB_CUDA_INCLUDE_DIRS})
endif ()

if ((DEFINED ENV{ENABLE_COVERAGE}) AND (NOT $ENV{ENABLE_COVERAGE} MATCHES "0"))
  set(ENABLE_COVERAGE ON)
else ()
  set(ENABLE_COVERAGE OFF)
endif ()

# GPUCov: CUDA device code coverage (reads ENABLE_CUDA_COVERAGE env var automatically)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m gpucov --cmake-dir
    OUTPUT_VARIABLE _GPUCOV_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _GPUCOV_CMAKE_ERR
    RESULT_VARIABLE _GPUCOV_CMAKE_RESULT
)
if (_GPUCOV_CMAKE_RESULT EQUAL 0)
  include("${_GPUCOV_CMAKE_DIR}/GPUCovConfig.cmake")
endif ()

message(STATUS "!!! PROJECT_NAME: ${PROJECT_NAME}, PROJECT_VERSION: ${PROJECT_VERSION} !!!")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CPP_MODULE_NAME_CPU "_sb_cpu")
set(CPP_MODULE_NAMES ${CPP_MODULE_NAME_CPU})

if (BUILD_WITH_CUDA)
  set(_CUDA_MOD_SUFFIX "")
  if (DEFINED CUDAToolkit_VERSION_MAJOR)
    set(_CUDA_MOD_SUFFIX "${CUDAToolkit_VERSION_MAJOR}")
  elseif (DEFINED CUDAToolkit_VERSION)
    string(REGEX MATCH "^[0-9]+" _CUDA_MOD_SUFFIX "${CUDAToolkit_VERSION}")
  endif ()

  if (_CUDA_MOD_SUFFIX)
    set(CPP_MODULE_NAME_CUDA "_sb_cuda${_CUDA_MOD_SUFFIX}")
  else ()
    set(CPP_MODULE_NAME_CUDA "_sb_cuda")
  endif ()
  list(APPEND CPP_MODULE_NAMES ${CPP_MODULE_NAME_CUDA})
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CXX)

# ============================================================
# x86-64 microarchitecture baseline
# ============================================================
# Default depends on the build context:
#   * A local source build (SKBUILD on, CIBUILDWHEEL not set) — i.e. the
#     user doing "pip install ." or "pip install --no-binary stablebear" —
#     defaults to "native" so the extension is tuned for this exact CPU.
#   * macOS x86_64 (any non-native build) defaults to "v2": SSE3/SSSE3/
#     SSE4.1/SSE4.2/POPCNT, no AVX. The x86_64 distribution wheel is built
#     and tested on an Apple Silicon CI runner under Rosetta 2, which does
#     not implement AVX/AVX2/FMA — so an AVX wheel could neither be tested
#     there nor (once the last Intel CI runner is retired in 2027) built on
#     real Intel hardware. v2 is supported by Rosetta and by every Intel
#     Mac, at the cost of AVX2/FMA autovectorization on x86_64 Macs.
#   * All other builds (cibuildwheel-produced distribution wheels, and
#     the minimal plain-cmake developer build) default to "v3":
#     AVX, AVX2, FMA, BMI1/2, F16C, LZCNT, MOVBE. This covers essentially
#     every x86-64 laptop and workstation from 2013+ (Haswell / Excavator
#     / Zen 1+). Notable exceptions are Atom-derived chips (Celeron N /
#     Pentium Silver / Jasper Lake, pre-2022).
#
# A Python runtime check in stablebear/_sb_cpp.py raises a clear
# ImportError on CPUs that lack the required extensions. The check is
# skipped for "native" / v1 / v2 builds (recorded in _build_info.py).
#
# Override with -DSB_X86_64_LEVEL=v2 (or v1) at configure time to
# build for older CPUs. v4 (AVX-512) is accepted but rarely useful.
if (SKBUILD AND NOT "$ENV{CIBUILDWHEEL}" STREQUAL "1")
  set(_SB_X86_64_DEFAULT "native")
elseif (APPLE)
  set(_SB_X86_64_DEFAULT "v2")
else ()
  set(_SB_X86_64_DEFAULT "v3")
endif ()
set(SB_X86_64_LEVEL "${_SB_X86_64_DEFAULT}" CACHE STRING
    "x86-64 microarchitecture baseline for CPU code (native, v1, v2, v3, v4)")
set_property(CACHE SB_X86_64_LEVEL PROPERTY STRINGS native v1 v2 v3 v4)

# MSVC has no -march=native. For SB_X86_64_LEVEL=native we probe the
# host CPU at configure time via CPUID and pick the best /arch: flag —
# this matters because a user doing `pip install .` precisely because
# the shipped AVX2 wheel doesn't work on their CPU must NOT end up with
# /arch:AVX2. Result is cached so the probe runs only once per build
# tree. Produces "/arch:AVX512", "/arch:AVX2", "/arch:AVX", or "" (no
# flag — MSVC default, SSE2-baseline, works on any x86-64 CPU).
function(sb_detect_msvc_native_arch out_var)
  if (DEFINED _SB_MSVC_NATIVE_ARCH_CACHE)
    set(${out_var} "${_SB_MSVC_NATIVE_ARCH_CACHE}" PARENT_SCOPE)
    return()
  endif ()

  set(_probe "${CMAKE_BINARY_DIR}/_sb_cpuid_probe.cpp")
  file(WRITE "${_probe}" [=[
#include <intrin.h>
#include <stdio.h>
int main() {
    int info[4];
    __cpuid(info, 0);
    int max_leaf = info[0];
    int ecx1 = 0;
    if (max_leaf >= 1) { __cpuid(info, 1); ecx1 = info[2]; }
    int ebx7 = 0;
    if (max_leaf >= 7) { __cpuidex(info, 7, 0); ebx7 = info[1]; }
    if (ebx7 & (1 << 16)) { printf("AVX512"); return 0; }
    if (ebx7 & (1 <<  5)) { printf("AVX2");   return 0; }
    if (ecx1 & (1 << 28)) { printf("AVX");    return 0; }
    printf("NONE");
    return 0;
}
]=])
  try_run(_run_result _compile_result
      "${CMAKE_BINARY_DIR}/_sb_cpuid_probe_build"
      "${_probe}"
      RUN_OUTPUT_VARIABLE _probe_output)
  set(_flag "")
  if (_compile_result AND _run_result EQUAL 0)
    if (_probe_output STREQUAL "AVX512")
      set(_flag "/arch:AVX512")
    elseif (_probe_output STREQUAL "AVX2")
      set(_flag "/arch:AVX2")
    elseif (_probe_output STREQUAL "AVX")
      set(_flag "/arch:AVX")
    endif ()
  endif ()
  set(_SB_MSVC_NATIVE_ARCH_CACHE "${_flag}" CACHE INTERNAL "")
  message(STATUS "!!! MSVC native CPU probe: '${_probe_output}' -> '${_flag}'")
  set(${out_var} "${_flag}" PARENT_SCOPE)
endfunction()

function(sb_apply_cpu_arch_flags tgt)
  if (MSVC)
    if (SB_X86_64_LEVEL STREQUAL "native")
      sb_detect_msvc_native_arch(_native_flag)
      if (_native_flag)
        target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${_native_flag}>)
      endif ()
    elseif (SB_X86_64_LEVEL STREQUAL "v3")
      target_compile_options(${tgt} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    elseif (SB_X86_64_LEVEL STREQUAL "v4")
      target_compile_options(${tgt} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
    endif ()
    # v1/v2 have no direct MSVC /arch: flag (default is SSE2; SSE4.2
    # intrinsics are header-gated and work at runtime on capable CPUs).
    return()
  endif ()

  if (SB_X86_64_LEVEL STREQUAL "v1")
    return()
  endif ()

  if (SB_X86_64_LEVEL STREQUAL "native")
    set(_arch_flag "-march=native")
  else ()
    set(_arch_flag "-march=x86-64-${SB_X86_64_LEVEL}")
  endif ()

  if (APPLE)
    # On macOS (including universal2 builds), -Xarch_x86_64 applies the
    # following flag only when producing the x86_64 slice, so arm64 slices
    # are unaffected.
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Xarch_x86_64>
        $<$<COMPILE_LANGUAGE:CXX>:${_arch_flag}>)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|x64)$")
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${_arch_flag}>)
  endif ()
endfunction()

message(STATUS "!!! x86-64 baseline: ${SB_X86_64_LEVEL}")

# Record the build-time CPU arch level so _sb_cpp.py knows whether a
# runtime check for AVX2/FMA is needed.
set(_SB_BUILD_INFO_PY "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py")
file(WRITE "${_SB_BUILD_INFO_PY}"
    "# Auto-generated by CMake; do not edit.\n"
    "CPU_ARCH_LEVEL = \"${SB_X86_64_LEVEL}\"\n")

if (WIN32)
  if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
  else ()
    add_compile_options(-Wa,-mbig-obj)
  endif ()
endif ()

include_directories("3rd/taskflow")

set(PYBIND11_FINDPYTHON ON)
add_subdirectory(3rd/pybind11)

# Adapted from https://community.gigperformer.com/t/getting-cmake-variables-from-c/17711
string(TIMESTAMP TODAY "%d/%m/%Y")
set(PROJECT_BUILD_DATE ${TODAY})
configure_file(version.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/version.cpp @ONLY)

SET(SB_LIB_SOURCES_CUDA
    include/sbear/cuda/cuda_block_executor.cuh
    include/sbear/cuda/cuda_device_array.cuh
    include/sbear/cuda/cuda_functional_support.cuh
    include/sbear/cuda/cuda_matrix_integrate.cuh
    include/sbear/cuda/cuda_matrix_integrate_structs.cuh
    include/sbear/cuda/cuda_offset_data_manager.cuh
    include/sbear/cuda/cuda_pcf_data_manager.cuh
    include/sbear/cuda/cuda_pcf_kernel.cuh
    include/sbear/cuda/cuda_util.cuh
    include/sbear/cuda/pcf_block_op.cuh
    include/sbear/cuda/triangle_skip_mode.hpp
)

SET(SB_HEADER_FILES
    config.hpp
    concepts.hpp
    version.hpp
    functional/pcf.hpp
    functional/piecewise_function.hpp
    functional/time_point.hpp
    functional/rectangle.hpp
    functional/operations.cuh
    platform.hpp
    random.hpp
    task.hpp
    tensor.hpp tensor.tpp
    tensor_to_string.hpp
    special_tensors.hpp
    executor.hpp
    point_cloud.hpp
    io.hpp
    io/io_stream.hpp
    io/io_stream_base.hpp
    io/barcode_io.hpp
    io/pcf_io.hpp
    io/point_io.hpp
    io/point_cloud_io.hpp
    io/tensor_io.hpp
    algorithm.hpp
    algorithms/functional/iterate_rectangles.hpp
    algorithms/functional/reduce.hpp
    algorithms/functional/matrix_integrate.hpp
    algorithms/functional/matrix_reduce.hpp
    algorithms/subdivide.hpp
    algorithms/functional/apply_functional.hpp
    block_matrix_support.cuh
    persistence/barcode.hpp
    persistence/compute_persistence.hpp
    persistence/persistence_pair.hpp
    persistence/stable_rank.hpp
    persistence/ripser/ripser.hpp
    settings.hpp)

SET(SB_SRC_FILES
    executor.cpp
    io_stream.cpp)

list(TRANSFORM SB_HEADER_FILES PREPEND "include/sbear/")
list(TRANSFORM SB_SRC_FILES PREPEND "src/")

SET(SB_LIB_SOURCES
    ${SB_HEADER_FILES}
    ${SB_SRC_FILES}
)

SET(SB_PY_PERSISTENCE_SOURCES
    py_barcode.hpp py_barcode.cpp
    py_barcode_summary.hpp py_barcode_summary.cpp
    py_persistence_pair.hpp py_persistence_pair.cpp
    py_ripser.hpp py_ripser.cpp
    pymodule_persistence.hpp pymodule_persistence.cpp
)

list(TRANSFORM SB_PY_PERSISTENCE_SOURCES PREPEND "persistence/")

SET(SB_PY_POINT_PROCESS_SOURCES
    py_poisson.hpp py_poisson.cpp
    pymodule_point_process.hpp pymodule_point_process.cpp
)

list(TRANSFORM SB_PY_POINT_PROCESS_SOURCES PREPEND "point_process/")

SET(SB_PY_SOURCES
    pybind.hpp
    pymodule.cpp
    functional/py_random.hpp functional/py_random.cpp
    pypcf_support.hpp
    py_np_support.hpp
    py_future.hpp
    functional/py_pcf.hpp functional/py_pcf.cpp
    py_np_tensor_convert.hpp py_np_tensor_convert.cpp
    functional/py_make_from_serial_content.hpp functional/py_make_from_serial_content.cpp
    functional/py_distance.hpp functional/py_distance.cpp
    functional/py_inner_product.hpp functional/py_inner_product.cpp
    functional/py_norms.hpp functional/py_norms.cpp
    functional/py_reductions.hpp functional/py_reductions.cpp
    py_tensor.hpp py_tensor.cpp
    py_io.hpp py_io.cpp
    py_symmetric_matrix.hpp py_symmetric_matrix.cpp
    py_distance_matrix.hpp py_distance_matrix.cpp
    ${SB_PY_PERSISTENCE_SOURCES}
    ${SB_PY_POINT_PROCESS_SOURCES}
)

list(TRANSFORM SB_PY_SOURCES PREPEND "src/python/")

SET(SB_VERSION_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)

# ============================================================
# IDE source groups — group by functional area, not language
# ============================================================
source_group("Functional"               REGULAR_EXPRESSION "include/sbear/functional/.*")
source_group("Functional/Bindings"      REGULAR_EXPRESSION "src/python/functional/.*")
source_group("Algorithms"               REGULAR_EXPRESSION "include/sbear/algorithms/.*")
source_group("IO"                       REGULAR_EXPRESSION "(include/sbear/io[./].*|src/io_stream\\.cpp)")
source_group("IO/Bindings"             REGULAR_EXPRESSION "src/python/py_io\\.(h|cpp)")
source_group("Persistence"             REGULAR_EXPRESSION "include/sbear/persistence/.*")
source_group("Persistence/Bindings"    REGULAR_EXPRESSION "src/python/persistence/.*")
source_group("PointProcess"            REGULAR_EXPRESSION "include/sbear/point_process/.*")
source_group("PointProcess/Bindings"   REGULAR_EXPRESSION "src/python/point_process/.*")
source_group("CUDA"                    REGULAR_EXPRESSION "(include/sbear/cuda/.*|src/cuda/.*)")
source_group("Tensor"                  REGULAR_EXPRESSION "include/sbear/tensor[^/]*")
source_group("Tensor/Bindings"         REGULAR_EXPRESSION "src/python/py_(np_tensor_convert|tensor)\\.(h|cpp)")
source_group("Core"                    REGULAR_EXPRESSION "include/sbear/(config|concepts|version|platform|executor|task|point_cloud|special_tensors|random|block_matrix_support|algorithm|tensor_to_string)\\.")
source_group("Core/Bindings"          REGULAR_EXPRESSION "src/python/(pybind|pymodule|py_np_support|pypcf_support|py_future)\\.(h|cpp)")
source_group("Core Sources"           REGULAR_EXPRESSION "src/[^/]+\\.cpp")
source_group("IO" FILES src/io_stream.cpp)

# Standalone GPU detection module (no CUDA dependency)
pybind11_add_module(_gpu_detect MODULE src/gpu_detect/gpu_detect.cpp)
target_link_libraries(_gpu_detect PRIVATE pybind11::headers)
if (APPLE)
  target_link_libraries(_gpu_detect PRIVATE "-framework IOKit" "-framework CoreFoundation")
elseif (WIN32)
  target_link_libraries(_gpu_detect PRIVATE dxgi)
endif ()

set(SB_PRIVATE_INCLUDES "include/sbear")
set(SB_PUBLIC_INCLUDES "include/")
if (BUILD_WITH_CUDA)
  list(APPEND SB_PRIVATE_INCLUDES ${SB_CUDA_INCLUDE_DIRS})
endif ()

if (BUILD_WITH_CUDA)
  # Build only the .cu files with NVCC, in a separate static library.
  # Everything else stays compiled by the host C++ compiler.
  add_library(sb_cuda STATIC
      src/cuda/cuda_matrix_integrate.cu
  )
  target_include_directories(sb_cuda PRIVATE ${SB_PRIVATE_INCLUDES})
  target_include_directories(sb_cuda PUBLIC ${SB_PUBLIC_INCLUDES})
  target_include_directories(sb_cuda PRIVATE "3rd/taskflow")
  target_include_directories(sb_cuda PRIVATE ${SB_CUDA_INCLUDE_DIRS})
  target_compile_definitions(sb_cuda PRIVATE BUILD_WITH_CUDA=1)

  set_target_properties(sb_cuda PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      POSITION_INDEPENDENT_CODE ON
  )

  target_compile_options(sb_cuda PRIVATE
      --expt-relaxed-constexpr
      -Xcudafe=--diag_suppress=partial_override
      -Xcudafe=--diag_suppress=virtual_function_decl_hidden
      -Xcudafe=--diag_suppress=initialization_not_reachable
  )

  if (GPUCOV_FOUND)
    # Collect all .cuh files from the CUDA and header source lists
    set(_GPUCOV_FILES)
    foreach(_src IN LISTS SB_LIB_SOURCES_CUDA)
      if (_src MATCHES "\\.cuh$")
        list(APPEND _GPUCOV_FILES ${_src})
      endif ()
    endforeach()
    foreach(_src IN LISTS SB_HEADER_FILES)
      if (_src MATCHES "\\.cuh$")
        list(APPEND _GPUCOV_FILES ${_src})
      endif ()
    endforeach()
    # Add the .cu sources from the sb_cuda target
    get_target_property(_CUDA_SRCS sb_cuda SOURCES)
    foreach(_src IN LISTS _CUDA_SRCS)
      if (_src MATCHES "\\.cu$")
        list(APPEND _GPUCOV_FILES ${_src})
      endif ()
    endforeach()
    list(REMOVE_DUPLICATES _GPUCOV_FILES)

    gpucov_instrument_target(sb_cuda
        FILES
            ${_GPUCOV_FILES}
        INCLUDE_PATHS
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/3rd/taskflow"
        CUDA_INCLUDE
            "${CUDAToolkit_INCLUDE_DIRS}"
    )
  endif ()
endif ()

# Create both CPU and (optionally) CUDA Python modules
pybind11_add_module(${CPP_MODULE_NAME_CPU} MODULE ${SB_LIB_SOURCES} ${SB_PY_SOURCES} ${SB_VERSION_SOURCE})
# Do NOT define BUILD_WITH_CUDA for the CPU module: code uses #ifdef BUILD_WITH_CUDA,
# so a value of 0 would still enable CUDA code paths.

if (BUILD_WITH_CUDA)
  pybind11_add_module(${CPP_MODULE_NAME_CUDA} MODULE ${SB_LIB_SOURCES} ${SB_LIB_SOURCES_CUDA} ${SB_PY_SOURCES} ${SB_VERSION_SOURCE})
  target_compile_definitions(${CPP_MODULE_NAME_CUDA} PRIVATE BUILD_WITH_CUDA=1)
  target_include_directories(${CPP_MODULE_NAME_CUDA} PRIVATE ${SB_CUDA_INCLUDE_DIRS})
  if (CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/include/cuda_runtime.h")
    target_compile_options(${CPP_MODULE_NAME_CUDA} PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/I${CUDAToolkit_ROOT}/include>
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-I${CUDAToolkit_ROOT}/include>
    )
  endif ()
  target_link_libraries(${CPP_MODULE_NAME_CUDA} PRIVATE sb_cuda ${CUDAToolkit_CUDART_LIBRARY})
endif ()

foreach(MOD_NAME IN LISTS CPP_MODULE_NAMES)
  target_compile_definitions(${MOD_NAME} PRIVATE SB_MODULE_NAME=${MOD_NAME})
  target_include_directories(${MOD_NAME} PRIVATE ${SB_PRIVATE_INCLUDES})
  target_include_directories(${MOD_NAME} PUBLIC ${SB_PUBLIC_INCLUDES})
  target_link_libraries(${MOD_NAME} PRIVATE pybind11::headers)
  target_compile_definitions(${MOD_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION_FULL})
  sb_apply_cpu_arch_flags(${MOD_NAME})
endforeach()

# NOTE: _gpu_detect is intentionally built without x86-64-vN flags so it
# remains loadable on CPUs that lack AVX2. The Python runtime check in
# _sb_cpp.py uses a stdlib-only path (independent of _gpu_detect) and
# reports a clear error before attempting to import the main backend.

# Convenience target so CLion (and plain cmake --build) can build all Python modules at once
add_custom_target(_sb DEPENDS ${CPP_MODULE_NAMES})

# IDE target folders
set_target_properties(_gpu_detect PROPERTIES FOLDER "Python Modules")
set_target_properties(_sb PROPERTIES FOLDER "Python Modules")
foreach(MOD_NAME IN LISTS CPP_MODULE_NAMES)
  set_target_properties(${MOD_NAME} PROPERTIES FOLDER "Python Modules")
endforeach()
if (BUILD_WITH_CUDA)
  set_target_properties(sb_cuda PROPERTIES FOLDER "CUDA")
endif ()

if (MSVC)
  # MSVC CUDA has some problems with picking up include directories in IntelliSense. We can solve this
  # by using VC++ Include Directories instead of on the compiler-level
  get_filename_component(taskflow_DIR "3rd/taskflow" REALPATH)
  set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath);${taskflow_DIR};${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif ()

# --- Detect free-threaded Python (PEP 703) ---
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('Py_GIL_DISABLED'))"
  OUTPUT_VARIABLE PY_GIL_DISABLED_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (PY_GIL_DISABLED_RAW STREQUAL "1")
  message(STATUS "!!! Building against FREE-THREADED Python: enabling Py_GIL_DISABLED !!!")
  foreach(MOD_NAME IN LISTS CPP_MODULE_NAMES)
    target_compile_definitions(${MOD_NAME} PRIVATE Py_GIL_DISABLED=1)
  endforeach()
endif()

# ============================================================
# Coverage configuration (GCC or Clang/LLVM)
# ============================================================
if (ENABLE_COVERAGE)
  # Create the interface library only once even if this block is re-entered.
  if (NOT TARGET coverage_flags)
    add_library(coverage_flags INTERFACE)
  endif ()

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "!!! Coverage ENABLED (LLVM/Clang) !!!")
    # Disable optimization for accurate coverage; use LLVM source-based coverage.
    target_compile_options(coverage_flags INTERFACE
      -O0 -g
      -fprofile-instr-generate
      -fcoverage-mapping
    )
    # With Clang/LLVM, pass profiling to the linker too (do NOT link gcov).
    target_link_options(coverage_flags INTERFACE -fprofile-instr-generate)

  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "!!! Coverage ENABLED (GCC/gcov) !!!")
    # Existing GCC/gcov-style coverage (unchanged).
    target_compile_options(coverage_flags INTERFACE
      -O0 -g
      --coverage # Shorthand for arcs/test-coverage
      -fprofile-abs-path # Essential for finding headers in GHA
      -fno-inline
      -fno-inline-small-functions
      -fno-default-inline
    )
    target_link_options(coverage_flags INTERFACE --coverage)
    target_link_libraries(coverage_flags INTERFACE gcov)

  else ()
    message(FATAL_ERROR "Coverage only supported with GCC or Clang/LLVM")
  endif ()
else ()
  message(STATUS "!!! Coverage DISABLED !!!")
endif ()

if (BUILD_TESTER)
  # Build the tester with CUDA when a GPU is available. Standard GitHub Actions
  # runners have no GPU, so default to CPU-only there. Self-hosted GPU runners
  # set ENABLE_CUDA_COVERAGE=1 (or can set BUILD_CUDA_TESTER=ON explicitly).
  if (BUILD_WITH_CUDA AND (NOT DEFINED ENV{GITHUB_ACTIONS} OR GPUCOV_ENABLE))
    set(BUILD_CUDA_TESTER ON)
  else ()
    set(BUILD_CUDA_TESTER OFF)
    message(STATUS "!!! Tester built without CUDA (no GPU on this runner) !!!")
  endif ()

  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  set(gtest_force_shared_crt ON)

  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory("3rd/googletest")

  enable_testing()

  set(SB_TEST_SOURCES_COMMON
      test/test_iterate_rectangles.cpp
      test/test_iterate_segments.cpp
      test/test_parallel_reduce.cpp
      test/test_tensor.cpp
      test/test_tensor_broadcast.cpp
      test/test_tensor_iteration.cpp
      test/test_special_tensors.cpp
      test/test_io_stream.cpp
      test/test_io_readwrite.cpp
      test/test_pcf_algebra_and_norms.cpp
      test/test_tensor_io_core.cpp
      test/test_barcode_and_stable_rank.cpp
      test/test_tensor_masked.cpp
      test/test_block_scheduler.cpp
      test/test_result_writer.cpp
      test/test_offset_data_manager.cpp
      test/test_walk_random.cpp
      test/test_subdivide.cpp
  )

  if (BUILD_CUDA_TESTER)
    add_executable(sb_test
        test/test_block_matrix_support.cu
        test/test_l1_dist.cu
        test/test_block_integrate.cu
        test/test_block_integrate_bounds.cu
        test/test_norms.cu
        ${SB_TEST_SOURCES_COMMON}
    )
  else ()
    add_executable(sb_test ${SB_TEST_SOURCES_COMMON})
  endif ()

  target_include_directories(sb_test PRIVATE "include/")
  if (BUILD_WITH_CUDA)
    target_include_directories(sb_test PRIVATE ${SB_CUDA_INCLUDE_DIRS})
    if (CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/include/cuda_runtime.h")
      target_compile_options(sb_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/I${CUDAToolkit_ROOT}/include>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-I${CUDAToolkit_ROOT}/include>
      )
    endif ()
  endif ()

  if (BUILD_CUDA_TESTER)
    add_library(sb_nopy STATIC ${SB_LIB_SOURCES} ${SB_LIB_SOURCES_CUDA} ${SB_VERSION_SOURCE})
    target_compile_definitions(sb_nopy PUBLIC BUILD_WITH_CUDA=1)
    target_link_libraries(sb_nopy PRIVATE sb_cuda)
  else ()
    add_library(sb_nopy STATIC ${SB_LIB_SOURCES} ${SB_VERSION_SOURCE})
  endif ()
  sb_apply_cpu_arch_flags(sb_nopy)
  sb_apply_cpu_arch_flags(sb_test)
  target_include_directories(sb_nopy PRIVATE ${SB_PRIVATE_INCLUDES})
  target_include_directories(sb_nopy PUBLIC ${SB_PUBLIC_INCLUDES})
  if (BUILD_WITH_CUDA)
    target_include_directories(sb_nopy PRIVATE ${SB_CUDA_INCLUDE_DIRS})
    if (CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/include/cuda_runtime.h")
      target_compile_options(sb_nopy PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/I${CUDAToolkit_ROOT}/include>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-I${CUDAToolkit_ROOT}/include>
      )
    endif ()
  endif ()

  if (BUILD_CUDA_TESTER)
    set_target_properties(sb_nopy PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared  # Must match the Python module's runtime
    )
    set_target_properties(sb_test PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
    )
  endif ()

  set_target_properties(sb_test PROPERTIES FOLDER "Testing")
  set_target_properties(sb_nopy PROPERTIES FOLDER "Testing")

  target_link_libraries(sb_test PUBLIC GTest::gtest_main sb_nopy)

  if (BUILD_CUDA_TESTER)
    target_link_libraries(sb_test PUBLIC sb_cuda CUDA::cudart_static)
  endif ()

  if (ENABLE_COVERAGE)
    target_link_libraries(sb_nopy PRIVATE coverage_flags)
    target_link_libraries(sb_test PRIVATE coverage_flags)
    foreach(MOD_NAME IN LISTS CPP_MODULE_NAMES)
      target_link_libraries(${MOD_NAME} PRIVATE coverage_flags)
    endforeach()
  endif ()

  if (WIN32)
    get_filename_component(python_dir "${Python_EXECUTABLE}" DIRECTORY)
    set_target_properties(sb_test PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=${python_dir}")
  endif ()


endif ()

if (BUILD_BENCHMARKS)
  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_MULTITHREADED ON)

  find_package(Boost COMPONENTS program_options system REQUIRED)
  find_package(Threads REQUIRED)

  add_executable(sb_bench
      benchmarking/bench.cpp
      benchmarking/benchmark.hpp
      benchmarking/bench_reduction.hpp benchmarking/bench_reduction.cpp
  )

  set_target_properties(sb_bench PROPERTIES
      CUDA_RUNTIME_LIBRARY Static
  )

  target_link_libraries(sb_bench PUBLIC ${CPP_MODULE_NAME_CPU} ${Boost_LIBRARIES} Threads::Threads)

  if (BUILD_WITH_CUDA)
    target_link_libraries(sb_bench PUBLIC ${CPP_MODULE_NAME_CUDA} CUDA::cudart_static)
  endif ()
endif ()

foreach(MOD_NAME IN LISTS CPP_MODULE_NAMES)
  if (MSVC)
    target_compile_options(${MOD_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>")
  endif ()
endforeach()

if (NOT MSVC)
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror=return-type")
endif ()

if (POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW) # Normalize paths
endif ()

if (NOT SKBUILD)
  message(STATUS "Python Executable: ${Python_EXECUTABLE}")

  execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "if True:
      import sysconfig
      print(sysconfig.get_path('purelib'))"
      OUTPUT_VARIABLE PYTHON_SITE_PRE
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  file(TO_CMAKE_PATH "${PYTHON_SITE_PRE}" PYTHON_SITE)

  SET(SITE_PKG_STABLEBEAR ${PYTHON_SITE}/stablebear)
  set(INSTALL_TARGET ${SITE_PKG_STABLEBEAR})

  message(STATUS "Minimal module will be installed into ${SITE_PKG_STABLEBEAR}")
else ()
  set(INSTALL_TARGET stablebear)
endif ()

install(TARGETS ${CPP_MODULE_NAMES} _gpu_detect DESTINATION ${INSTALL_TARGET})
install(FILES "${_SB_BUILD_INFO_PY}" DESTINATION ${INSTALL_TARGET})

if ((LINUX OR (NOT SKBUILD)) AND (NOT SKIP_STUBGEN)) # Stubgen does not seem happy on CI
  foreach(MOD_NAME IN LISTS CPP_MODULE_NAMES)
    add_custom_command(
        TARGET ${MOD_NAME}
        POST_BUILD
        COMMAND ${Python_EXECUTABLE} -m pybind11_stubgen ${MOD_NAME}
        --output-dir "${CMAKE_CURRENT_BINARY_DIR}/stubs"
        --ignore-invalid-expressions comment
        COMMENT "Generating Python stubs for ${MOD_NAME}..."
    )

    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/stubs/${MOD_NAME}/"
        DESTINATION "${INSTALL_TARGET}/${MOD_NAME}"
        FILES_MATCHING
        PATTERN "*.pyi"
    )
  endforeach()

  add_custom_command(
      TARGET ${CPP_MODULE_NAME_CPU}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E touch
      ${CMAKE_CURRENT_BINARY_DIR}/py.typed
      COMMENT "Creating py.typed"
  )

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/py.typed" "")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/py.typed" DESTINATION ${INSTALL_TARGET})
endif ()

if (SKBUILD)
  # Install Python source files alongside compiled module
  install(
      DIRECTORY stablebear/
      DESTINATION ${INSTALL_TARGET}
      FILES_MATCHING PATTERN "*.py"
      PATTERN "__pycache__" EXCLUDE
  )

elseif (NOT SKBUILD)


  file(GLOB PY_PKG_FILES "stablebear/*.py")
  #message(STATUS ${PY_PKG_FILES})
  install(
      DIRECTORY stablebear
      DESTINATION ${PYTHON_SITE}

      # Allow symlinking in CPP module for completions in IDE during development (__pycache__ should never be installed)
      # Note: _sb_cpp.py is the backend dispatcher and must be installed; .so files are excluded via \.so$ already
      # Exclude stub symlink dirs (_sb_cpu/_sb_cuda*) to avoid clashing with installed stubs/extensions.
      REGEX "(__pycache__|\\.so$|\\.pyi$|py\\.typed|/_sb_[^./]+$)" EXCLUDE
  )


  if (NOT SKIP_BACK_COPY)

    # Copy on Windows, symlink otherwise
    if (WIN32)
      SET(LINK_TYPE copy_if_different)
    else ()
      SET(LINK_TYPE create_symlink)
    endif ()

    # Remove stale files from the old single-module layout (_sb without cpu/cuda suffix)
    install(CODE "
      # Helper: remove all _sb.cpython-*.so from a directory (whether symlink or real file)
      macro(remove_old_sb_so _dir)
        file(GLOB _OLD_SO_FILES \"\${_dir}/_sb.cpython-*.so\")
        foreach(_F IN LISTS _OLD_SO_FILES)
          message(STATUS \"Removing stale old-layout module: \${_F}\")
          file(REMOVE \"\${_F}\")
        endforeach()
      endmacro()

      set(SRC_DIR \"${CMAKE_CURRENT_SOURCE_DIR}/stablebear\")

      # Remove old _sb stub-dir symlink from source tree
      set(_OLD_STUB_LINK \"\${SRC_DIR}/_sb\")
      if(IS_SYMLINK \"\${_OLD_STUB_LINK}\")
        message(STATUS \"Removing stale symlink \${_OLD_STUB_LINK}\")
        file(REMOVE \"\${_OLD_STUB_LINK}\")
      endif()

      # Remove old _sb.cpython-*.so from source tree and from site-packages
      # (the .so shadows the _sb.py dispatcher because Python prefers extension modules)
      remove_old_sb_so(\"\${SRC_DIR}\")
      remove_old_sb_so(\"${SITE_PKG_STABLEBEAR}\")
    ")

    foreach(MOD_NAME IN LISTS CPP_MODULE_NAMES)
      install(CODE "
      message(STATUS \"Syncing built extension ${MOD_NAME} back to source directory\")

      set(SRC_DIR \"${CMAKE_CURRENT_SOURCE_DIR}/stablebear\")

      # Extension file
      set(EXT_FILE \"$<TARGET_FILE:${MOD_NAME}>\")
      get_filename_component(EXT_NAME \"\${EXT_FILE}\" NAME)
      set(EXT_DEST \"\${SRC_DIR}/\${EXT_NAME}\")

      if(UNIX)
        # Remove stale symlink before (re-)creating — handles build-dir changes
        if(IS_SYMLINK \"\${EXT_DEST}\")
          file(REMOVE \"\${EXT_DEST}\")
        endif()
        file(CREATE_LINK \"\${EXT_FILE}\" \"\${EXT_DEST}\" SYMBOLIC)
      else()
        file(COPY \"\${EXT_FILE}\" DESTINATION \"\${SRC_DIR}\")
      endif()

      if (NOT SKIP_STUBGEN)

        # Stub file
        set(STUB_SRC_DIR \"${CMAKE_CURRENT_BINARY_DIR}/stubs/${MOD_NAME}\")
        set(STUB_DEST_DIR \"\${SRC_DIR}/${MOD_NAME}\")

        if(UNIX)
          # Check if the destination already exists (as a folder or a symlink)
          if(NOT EXISTS \"\${STUB_DEST_DIR}\")
            message(STATUS \"Creating initial symlink for stubs: \${STUB_DEST_DIR}\")
            file(CREATE_LINK \"\${STUB_SRC_DIR}\" \"\${STUB_DEST_DIR}\" SYMBOLIC)
          else()
            # Optional: Check if it's pointing to the WRONG place (e.g., after a build dir change)
            file(READ_SYMLINK \"\${STUB_DEST_DIR}\" CURRENT_LINK_TARGET)
            if(NOT \"\${CURRENT_LINK_TARGET}\" STREQUAL \"\${STUB_SRC_DIR}\")
              message(STATUS \"Updating stale symlink for stubs...\")
              file(REMOVE_RECURSE \"\${STUB_DEST_DIR}\")
              file(CREATE_LINK \"\${STUB_SRC_DIR}\" \"\${STUB_DEST_DIR}\" SYMBOLIC)
            else()
              message(STATUS \"Symlink for stubs already exists and is correct.\")
            endif()
          endif()
        endif()

        # py.typed (small file — copy everywhere)
        file(COPY
          \"${CMAKE_CURRENT_BINARY_DIR}/py.typed\"
          DESTINATION \"\${SRC_DIR}\")
      endif()
      ")
    endforeach()

    # Optional: Clean the symlink when running 'make clean'
    #set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${SYMLINK_NAME}")
  endif ()

endif ()

# ============================================================
# clean_stablebear — remove symlinks from source tree and all
#                 stablebear content from site-packages
# ============================================================
if (NOT SKBUILD)
  add_custom_target(clean_stablebear
    COMMAND ${CMAKE_COMMAND}
      -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stablebear
      -DSITE_DIR=${SITE_PKG_STABLEBEAR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/clean_stablebear.cmake
    COMMENT "Cleaning stablebear symlinks and site-packages content"
  )
endif ()

# ============================================================
# Valgrind / debug info configuration (Linux only)
# ============================================================

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND BUILD_TESTER)
  add_library(valgrind_flags INTERFACE)

  target_compile_options(valgrind_flags INTERFACE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-fno-pie>)
  target_link_options(valgrind_flags INTERFACE $<$<CONFIG:Debug>:-no-pie>)

  find_program(LSB_RELEASE lsb_release)
  if(LSB_RELEASE)
    execute_process(
      COMMAND ${LSB_RELEASE} -rs
      OUTPUT_VARIABLE DISTRO_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(DISTRO_VERSION VERSION_LESS "24.04")
      message(STATUS "Ubuntu < 24.04 detected, forcing -gdwarf-4 for valgrind compatibility")
      target_compile_options(valgrind_flags INTERFACE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-gdwarf-4>)
    endif()
  endif()

  target_link_libraries(sb_nopy PRIVATE valgrind_flags)
  target_link_libraries(sb_test PRIVATE valgrind_flags)
endif()

message(STATUS "!!! Stub file:          ${STUB_FILE}")
message(STATUS "!!! Install target:     ${INSTALL_TARGET}")

message(STATUS "!!! C compiler:         ${CMAKE_C_COMPILER}")
message(STATUS "!!! C++ compiler:       ${CMAKE_CXX_COMPILER}")
message(STATUS "!!! Compiler ID:        ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "!!! Python_ROOT_DIR:    ${Python_ROOT_DIR}")
message(STATUS "!!! Python_EXECUTABLE:  ${Python_EXECUTABLE}")
message(STATUS "!!! CUDAToolkit_ROOT:             ${CUDAToolkit_ROOT}")
message(STATUS "!!! CUDAToolkit_INCLUDE_DIRS:     ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "!!! CUDAToolkit_CUDART_LIBRARY:   ${CUDAToolkit_CUDART_LIBRARY}")
message(STATUS "!!! CMAKE_LIBRARY_PATH:           ${CMAKE_LIBRARY_PATH}")
