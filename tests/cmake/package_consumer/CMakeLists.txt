# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(momoten_package_consumer LANGUAGES CXX)

find_package(momoten CONFIG REQUIRED)

if(NOT TARGET momoten::headers OR NOT TARGET momoten::driver)
    message(FATAL_ERROR "momoten package does not provide its documented targets")
endif()

get_target_property(momoten_include_directories
    momoten::headers INTERFACE_INCLUDE_DIRECTORIES)
set(momoten_version_header_found OFF)
foreach(momoten_include_directory IN LISTS momoten_include_directories)
    if(EXISTS "${momoten_include_directory}/momoten/spv_to_clc.h")
        message(FATAL_ERROR
            "the build-tree translator header was installed as a public package API")
    endif()
    if(EXISTS "${momoten_include_directory}/momoten/version.h")
        set(momoten_version_header_found ON)
    endif()
endforeach()
if(NOT momoten_version_header_found)
    message(FATAL_ERROR "momoten package does not install its version header")
endif()
