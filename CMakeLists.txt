cmake_minimum_required(VERSION 3.16)
cmake_policy(VERSION 3.16)
project(Ragnar LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# dependencies
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies.cmake)

add_subdirectory(src)
