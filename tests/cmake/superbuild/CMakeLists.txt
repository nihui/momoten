# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22.1)
project(momoten_superbuild_check LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(MOMOTEN_BUILD_DRIVER OFF CACHE BOOL "" FORCE)
set(MOMOTEN_BUILD_TRANSLATOR ON CACHE BOOL "" FORCE)
set(MOMOTEN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MOMOTEN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MOMOTEN_BUILD_AMALGAMATION OFF CACHE BOOL "" FORCE)
set(MOMOTEN_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../.." momoten)

if(NOT CMAKE_CXX_STANDARD EQUAL 17)
    message(FATAL_ERROR "momoten changed its parent C++ standard")
endif()
if(NOT TARGET momoten::translator OR NOT TARGET momoten::headers)
    message(FATAL_ERROR "momoten superbuild targets are missing")
endif()
