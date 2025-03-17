set(LIBRARY ragnar)
set(SUBDIRS utils containers plugins physics)
set(DEPS rgnr_containers rgnr_plugins rgnr_physics)
link_libraries(Kokkos::kokkos HighFive toml11::toml11 pybind11::pybind11)

foreach(subdir ${SUBDIRS})
  add_subdirectory(${subdir})
endforeach()

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

pybind11_add_module(${LIBRARY} ${SOURCES})

add_dependencies(${LIBRARY} ${DEPS})
target_link_libraries(${LIBRARY} PUBLIC ${DEPS})
