add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/styling.cmake)

include(FetchContent)
find_package(HighFive QUIET)
if(NOT HighFive_FOUND)
  set(HIGHFIVE_USE_BOOST OFF)
  find_package(hdf5 REQUIRED)
  message(
    STATUS "${Yellow}HighFive not found, fetching from source${ColorReset}")
  FetchContent_Declare(
    HighFive
    GIT_REPOSITORY https://github.com/highfive-devs/highfive.git
    GIT_TAG v2.10.1)
  FetchContent_MakeAvailable(HighFive)
else()
  message(STATUS "${Green}HighFive found${ColorReset}")
endif()

find_package(Kokkos QUIET)
if(NOT Kokkos_FOUND)
  message(STATUS "${Yellow}Kokkos not found, fetching from source${ColorReset}")
  FetchContent_Declare(
    Kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG 4.5.01)
  FetchContent_MakeAvailable(Kokkos)
else()
  message(STATUS "${Green}Kokkos found${ColorReset}")
endif()

find_package(toml11 QUIET)
if(NOT toml11_FOUND)
  message(STATUS "${Yellow}toml11 not found, fetching from source${ColorReset}")
  set(TOML11_BUILD_TESTS OFF)
  set(TOML11_BUILD_EXAMPLES OFF)
  FetchContent_Declare(
    toml11
    GIT_REPOSITORY https://github.com/ToruNiina/toml11.git
    GIT_TAG v4.4.0)
  FetchContent_MakeAvailable(toml11)
else()
  message(STATUS "${Green}toml11 found${ColorReset}")
endif()
