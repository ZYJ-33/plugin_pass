cmake_minimum_required(VERSION 3.25)
project(benchmark_generator_pass)

set(CMAKE_CXX_STANDARD 17)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION}")

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

set(_SOURCE_FILES
        BenchMarkGenerator.cpp)


add_llvm_pass_plugin(BenchmarkGenerator
        ${_SOURCE_FILES}
        LINK_COMPONENTS
        Support Analysis Core)

