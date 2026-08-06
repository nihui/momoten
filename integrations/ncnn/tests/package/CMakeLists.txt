# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(ncnn_momoten_package_integration LANGUAGES CXX)

enable_testing()
find_package(ncnn CONFIG REQUIRED)
find_package(momoten CONFIG REQUIRED)
set(NCNN_VULKAN ON)
include("${CMAKE_CURRENT_LIST_DIR}/../../ncnn_momoten.cmake")
ncnn_momoten_register_ncnn_tests()
