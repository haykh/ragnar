cmake_minimum_required(VERSION 3.16)
cmake_policy(VERSION 3.16)

set(LIBRARY ragnar)
project(${LIBRARY} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RAGNAR_USE_HDF5 CACHE BOOL ON)
if(RAGNAR_USE_HDF5)
  message(STATUS "Using HDF5")
  add_compile_definitions(USE_HDF5)
else()
  message(STATUS "Not using HDF5")
endif()

# dependencies
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies.cmake)

# main code
set(SUBDIRS utils containers physics)
set(DEPS rgnr_utils rgnr_containers rgnr_physics)

if(RAGNAR_USE_HDF5)
  list(APPEND SUBDIRS io plugins)
  list(APPEND DEPS rgnr_io rgnr_plugins)
endif()

link_libraries(Kokkos::kokkos pybind11::pybind11)

if(RAGNAR_USE_HDF5)
  link_libraries(HighFive)
endif()

set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/pyinterface.cpp)
foreach(subdir ${SUBDIRS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/${subdir})
  file(GLOB_RECURSE LOC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/${subdir}/*.cpp)
  list(APPEND SOURCES ${LOC_SOURCES})
endforeach()

pybind11_add_module(${LIBRARY} ${SOURCES})

add_dependencies(${LIBRARY} ${DEPS})
target_link_libraries(${LIBRARY} PUBLIC ${DEPS})

install(TARGETS ${LIBRARY} LIBRARY DESTINATION .)

include(CTest)
enable_testing()

file(GLOB_RECURSE TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.py)

foreach(test_source ${TEST_SOURCES})
  get_filename_component(test_name ${test_source} NAME)
  string(REGEX REPLACE ".py" "" test_name ${test_name})
  add_test(NAME ${test_name} COMMAND ${PYTHON_EXECUTABLE} -m pytest
                                     ${test_source})
endforeach()
