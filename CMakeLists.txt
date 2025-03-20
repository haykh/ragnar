cmake_minimum_required(VERSION 3.16)
cmake_policy(VERSION 3.16)
project(Ragnar LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# dependencies
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies.cmake)

# main code
set(LIBRARY ragnar)
set(SUBDIRS utils containers plugins physics)
set(DEPS rgnr_containers rgnr_plugins rgnr_physics)
link_libraries(Kokkos::kokkos HighFive pybind11::pybind11)

foreach(subdir ${SUBDIRS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/${subdir})
endforeach()

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

pybind11_add_module(${LIBRARY} ${SOURCES})

add_dependencies(${LIBRARY} ${DEPS})
target_link_libraries(${LIBRARY} PUBLIC ${DEPS})

include(CTest)
enable_testing()

file(GLOB_RECURSE TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.py)

foreach(test_source ${TEST_SOURCES})
  get_filename_component(test_name ${test_source} NAME)
  string(REGEX REPLACE ".py" "" test_name ${test_name})
  add_test(NAME ${test_name} COMMAND ${PYTHON_EXECUTABLE} -m pytest
                                     ${test_source})
endforeach()
