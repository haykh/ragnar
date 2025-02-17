set(EXECUTABLE main.xc)
set(SUBDIRS utils containers plugins physics)
set(DEPS rgnr_plugins rgnr_containers rgnr_physics)
link_libraries(Kokkos::kokkos HighFive)

foreach(subdir ${SUBDIRS})
  add_subdirectory(${subdir})
endforeach()

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(${EXECUTABLE} ${SOURCES})

add_dependencies(${EXECUTABLE} ${DEPS})
target_link_libraries(${EXECUTABLE} PUBLIC ${DEPS})
