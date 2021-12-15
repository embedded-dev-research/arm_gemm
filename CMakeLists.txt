cmake_minimum_required(VERSION 3.12)
project(arm_gemm)

file(GLOB_RECURSE ALL_SOURCE_FILES src/*.cpp app/*.cpp include/*.hpp)
include_directories(include)

option(USE_STYLE_CHECKER OFF)
if( USE_STYLE_CHECKER )
    message(STATUS "Enable style check: see google style")
    include(cmake/cpplint.cmake)
    cpplint_add_subdirectory(app)
    cpplint_add_subdirectory(include)
    cpplint_add_subdirectory(src)
endif( USE_STYLE_CHECKER )


set(ProjectId ${PROJECT_NAME})
set(PACK_LIB "${ProjectId}_lib")
add_library(${PACK_LIB} STATIC ${ALL_SOURCE_FILES} )

add_executable( ${ProjectId} ${ALL_SOURCE_FILES} )
target_link_libraries(${ProjectId} ${PACK_LIB})