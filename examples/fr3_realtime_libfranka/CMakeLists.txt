cmake_minimum_required(VERSION 3.16)

project(fr3_realtime_libfranka LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

find_package(Franka QUIET)
if(NOT Franka_FOUND)
  set(FR3_LIBFRANKA_SOURCE
      "${CMAKE_CURRENT_SOURCE_DIR}/libfranka-0.21.1"
      CACHE PATH "Path to a libfranka source checkout")
  if(EXISTS "${FR3_LIBFRANKA_SOURCE}/CMakeLists.txt")
    message(STATUS "FrankaConfig.cmake not found; building bundled libfranka from ${FR3_LIBFRANKA_SOURCE}")
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GENERATE_PYLIBFRANKA OFF CACHE BOOL "" FORCE)
    if(DEFINED ENV{CONDA_PREFIX} AND NOT DEFINED EIGEN3_INCLUDE_DIRS)
      set(EIGEN3_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include/eigen3" CACHE PATH "" FORCE)
    endif()
    add_subdirectory("${FR3_LIBFRANKA_SOURCE}" "${CMAKE_BINARY_DIR}/libfranka" EXCLUDE_FROM_ALL)
  else()
    message(FATAL_ERROR
      "Could not find FrankaConfig.cmake and no libfranka source checkout was found at:\n"
      "  ${FR3_LIBFRANKA_SOURCE}\n"
      "Either install libfranka into $CONDA_PREFIX, or clone it here with:\n"
      "  git clone --branch 0.21.1 --depth 1 --recurse-submodules "
      "https://github.com/frankarobotics/libfranka.git libfranka-0.21.1")
  endif()
endif()
find_package(cppzmq REQUIRED)
find_path(MSGPACK_INCLUDE_DIR msgpack.hpp REQUIRED)

add_library(fr3_protocol INTERFACE)
target_include_directories(fr3_protocol INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${MSGPACK_INCLUDE_DIR}
)

add_executable(robot_server src/robot_server.cpp)
target_link_libraries(robot_server PRIVATE fr3_protocol Franka::Franka cppzmq Threads::Threads)

add_executable(readonly_state_publisher src/readonly_state_publisher.cpp)
target_link_libraries(readonly_state_publisher PRIVATE fr3_protocol Franka::Franka cppzmq Threads::Threads)

add_executable(go_home src/go_home.cpp)
target_link_libraries(go_home PRIVATE fr3_protocol Franka::Franka Threads::Threads)
