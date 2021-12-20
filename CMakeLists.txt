cmake_minimum_required(VERSION 2.11)
project(mpi-mxv)

set(CMAKE_CXX_STANDARD 17)
add_executable(mpi_mxv src/main.cpp)

find_package(MPI REQUIRED)
target_include_directories(mpi_mxv PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(mpi_mxv PRIVATE ${MPI_CXX_LIBRARIES})

if (${CMAKE_BUILD_TYPE} STREQUAL Release)
    message(STATUS "Configure flags for Release build")
    target_compile_options(mpi_mxv PUBLIC /O2)
    target_compile_options(mpi_mxv PUBLIC /fp:precise)
endif ()