cmake_minimum_required(VERSION 3.27)
project(spectre_native LANGUAGES CXX)

# Issue #5, D2: the newest MLX release is the floor, so the code base starts on
# current fixes and does not need a version-upgrade pass later.
set(SPECTRE_MINIMUM_MLX_VERSION 0.32.5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# QUIET plus our own message: CMake's default "could not find package" output
# does not tell a newcomer which version is needed or how to install it.
find_package(MLX ${SPECTRE_MINIMUM_MLX_VERSION} CONFIG QUIET)
if(NOT MLX_FOUND)
  message(FATAL_ERROR
    "MLX >= ${SPECTRE_MINIMUM_MLX_VERSION} was not found.\n"
    "  * Homebrew:     brew install mlx   (only if it already ships ${SPECTRE_MINIMUM_MLX_VERSION}+; check 'brew info mlx')\n"
    "  * From source:  build tag v${SPECTRE_MINIMUM_MLX_VERSION} of https://github.com/ml-explore/mlx and pass\n"
    "                  -DCMAKE_PREFIX_PATH=<install prefix> (see native/README.md).")
endif()
message(STATUS "Using MLX ${MLX_VERSION}")

include(FetchContent)

# Pinned release tags (D5). Each dependency's own tests and tools are off so
# the configure and build stay fast.
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
  GIT_SHALLOW TRUE)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  GIT_SHALLOW TRUE)

# yaml-cpp 0.8.0 (the newest release) declares cmake_minimum_required below 3.5,
# which CMake 4 refuses; 3.10 is also the point below which CMake prints a
# deprecation warning. The floor is set only while yaml-cpp configures so our
# own policies and the other dependencies are untouched.
set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
FetchContent_MakeAvailable(yaml-cpp)
unset(CMAKE_POLICY_VERSION_MINIMUM)
FetchContent_MakeAvailable(googletest)

# Warnings are errors for our code only; third-party targets keep their own flags.
# An option rather than a hard-wired flag so a contributor on a newer compiler,
# whose new warnings are not ours to fix first, can still build with
# -DSPECTRE_WARNINGS_AS_ERRORS=OFF. CI keeps the default.
option(SPECTRE_WARNINGS_AS_ERRORS "Treat warnings in spectre code as errors" ON)
add_library(spectre_warnings INTERFACE)
target_compile_options(spectre_warnings INTERFACE -Wall -Wextra -Wpedantic)
if(SPECTRE_WARNINGS_AS_ERRORS)
  target_compile_options(spectre_warnings INTERFACE -Werror)
endif()

add_library(spectre_core STATIC
  src/config/config_types.cpp
  src/config/yaml_config_loader.cpp
  src/mixing/chunk_layout.cpp
  src/mixing/causal_query_pool.cpp)
target_include_directories(spectre_core PUBLIC include)
# MLX is linked now, before any tensor code exists, so a broken toolchain
# surfaces in T0.1 rather than in the first numerical ticket.
target_link_libraries(spectre_core PUBLIC mlx PRIVATE yaml-cpp::yaml-cpp spectre_warnings)

enable_testing()
include(GoogleTest)

# Oracles are plain C++ with NO MLX dependency (Issue #5, trap 12): they must
# not share code paths with what they check.
add_library(spectre_test_oracles STATIC tests/oracles/naive_reference.cpp)
target_include_directories(spectre_test_oracles PUBLIC tests/oracles)
target_link_libraries(spectre_test_oracles PRIVATE spectre_warnings)

# One executable per test category keeps CTest labels (unit, component,
# integration, ...) accurate: gtest_discover_tests labels a whole executable,
# and the presets include or exclude tests by label.
function(spectre_add_test_suite suite_name test_label)
  add_executable(${suite_name} ${ARGN})
  target_include_directories(${suite_name} PRIVATE tests/support)
  target_link_libraries(${suite_name} PRIVATE
    spectre_core spectre_test_oracles GTest::gtest_main spectre_warnings)
  gtest_discover_tests(${suite_name} PROPERTIES LABELS ${test_label} DISCOVERY_MODE PRE_TEST)
endfunction()

spectre_add_test_suite(spectre_unit_tests unit
  tests/unit/test_config_types.cpp
  tests/unit/test_yaml_config_loader.cpp
  tests/unit/test_mlx_toolchain.cpp
  tests/unit/test_chunk_layout.cpp
  tests/unit/test_causal_query_pool.cpp)
