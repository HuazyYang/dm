
project(dm_algorithm)

set(DM_ISPC_SOURCES
  dm_miscs.ispc
  dm_transpose.ispc
  dm_matrix.ispc
)

set(DM_CXX_SOURCES
  miscs.cpp
  dm_matrix.cpp
)

add_library(dm_double
  STATIC
  ${DM_ISPC_SOURCES}
)
target_compile_definitions(dm_double PRIVATE "DM_REAL_AS_DOUBLE=1")
add_library(dm_float
  STATIC
  ${DM_ISPC_SOURCES}
)
target_compile_definitions(dm_float PRIVATE "DM_REAL_AS_FLOAT=1")

add_library(${PROJECT_NAME}
  ${DM_CXX_SOURCES}
)
target_link_libraries(${PROJECT_NAME} PUBLIC dm_float dm_double)

set_target_properties(
  dm_double PROPERTIES
  ISPC_INSTRUCTION_SETS
  "sse2-i32x8;sse4-i32x8;avx1-i32x8;avx2-i32x8"
  ISPC_HEADER_SUFFIX
  "_double_ispc.h"
  # COMPILE_OPTIONS
  # "$<$<COMPILE_LANGUAGE:ISPC>:--opt=disable-fma>"
)

set_target_properties(
  dm_float PROPERTIES
  ISPC_INSTRUCTION_SETS
  "sse2-i32x8;sse4-i32x8;avx1-i32x8;avx2-i32x8"
  ISPC_HEADER_SUFFIX
  "_float_ispc.h"
  # COMPILE_OPTIONS
  # "$<$<COMPILE_LANGUAGE:ISPC>:--opt=disable-fma>"
)

target_include_directories(${PROJECT_NAME} PUBLIC ../)