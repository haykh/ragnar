set(LIB rgnr_physics)
set(DEPS rgnr_utils rgnr_containers)

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(${LIB} ${SOURCES})
target_link_libraries(${LIB} PUBLIC ${DEPS})

target_include_directories(
  ${LIB}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)
