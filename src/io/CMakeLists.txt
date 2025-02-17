set(LIB rgnr_io)
set(DEPS rgnr_utils)

add_library(${LIB})
target_link_libraries(${LIB} PUBLIC ${DEPS})

target_include_directories(
  ${LIB}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)
