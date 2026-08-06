# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22.1)
project(ncnn_momoten_integration LANGUAGES C CXX)

enable_testing()
option(MOMOTEN_NCNN_BUILD_ALL_TESTS
    "Build and register the upstream ncnn test suite through momoten" OFF)
set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS ${MOMOTEN_NCNN_BUILD_ALL_TESTS} CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MOMOTEN_BUILD_DRIVER ON CACHE BOOL "" FORCE)
set(MOMOTEN_BUILD_TRANSLATOR ON CACHE BOOL "" FORCE)
set(MOMOTEN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MOMOTEN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MOMOTEN_BUILD_AMALGAMATION OFF CACHE BOOL "" FORCE)
set(MOMOTEN_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../../../../ncnn" ncnn)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../../.." momoten)
include("${CMAKE_CURRENT_LIST_DIR}/../../ncnn_momoten.cmake")
ncnn_momoten_register_ncnn_tests()
