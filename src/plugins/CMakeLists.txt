set(LIB rgnr_plugins)
set(DEPS rgnr_utils rgnr_containers rgnr_io)

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(${LIB} ${SOURCES})
target_link_libraries(${LIB} PUBLIC ${DEPS})

target_include_directories(
  ${LIB}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)
