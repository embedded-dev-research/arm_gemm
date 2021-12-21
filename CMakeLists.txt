cmake_minimum_required(VERSION 3.21)

#include(ExternalProject)
#ExternalProject_Add(project_opencl_headers
#        SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/opencl_headers
#        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/opencl_headers
#        CONFIGURE_COMMAND ""
#        INSTALL_COMMAND ""
#        BUILD_COMMAND make
#        BUILD_IN_SOURCE 1
#        )
#add_library(opencl_headers STATIC IMPORTED)
#add_dependencies(project_opencl_headers opencl_headers)

project(arm_gemm)

find_package( OpenMP )
if( OpenMP_FOUND )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif( OpenMP_FOUND )

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

#find_package(OpenCLHeaders REQUIRED)

set(ProjectId ${PROJECT_NAME})
set(PACK_LIB "${ProjectId}_lib")
add_library(${PACK_LIB} STATIC ${ALL_SOURCE_FILES} )

add_executable( ${ProjectId} ${ALL_SOURCE_FILES} )
#target_link_libraries(${ProjectId} ${PACK_LIB} OpenCL::Headers)
