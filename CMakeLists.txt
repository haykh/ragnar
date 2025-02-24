cmake_minimum_required(VERSION 3.16)
project(Ragnar LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG 4.5.01)
FetchContent_MakeAvailable(Kokkos)

FetchContent_Declare(
  toml11
  GIT_REPOSITORY https://github.com/ToruNiina/toml11.git
  GIT_TAG v4.4.0)
FetchContent_MakeAvailable(toml11)

find_package(HighFive REQUIRED)
add_subdirectory(src)
