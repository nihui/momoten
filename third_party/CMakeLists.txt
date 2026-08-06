# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

function(momoten_require_submodule relative_path sentinel)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${relative_path}/${sentinel}")
        message(FATAL_ERROR
            "momoten dependency '${relative_path}' is missing or not initialized.\n"
            "Run: git submodule update --init --recursive")
    endif()
endfunction()

if(NOT TARGET spirv-cross-core AND NOT TARGET spirv-cross-glsl)
    momoten_require_submodule(spirv-cross CMakeLists.txt)

    set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_GLSL ON CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_MSL OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_SKIP_INSTALL ON CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_WERROR OFF CACHE BOOL "" FORCE)

    add_subdirectory(spirv-cross EXCLUDE_FROM_ALL)
elseif(NOT TARGET spirv-cross-core OR NOT TARGET spirv-cross-glsl)
    message(FATAL_ERROR
        "The parent build must provide both spirv-cross-core and spirv-cross-glsl, or neither")
endif()

if(NOT TARGET SPIRV-Tools-opt)
    momoten_require_submodule(spirv-headers include/spirv/unified1/spirv.hpp)
    momoten_require_submodule(spirv-tools CMakeLists.txt)

    set(momoten_had_cxx_standard OFF)
    if(DEFINED CMAKE_CXX_STANDARD)
        set(momoten_had_cxx_standard ON)
        set(momoten_saved_cxx_standard "${CMAKE_CXX_STANDARD}")
    endif()
    set(CMAKE_CXX_STANDARD 17)

    set(SPIRV-Headers_SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/spirv-headers")
    set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
    set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
    set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
    set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
    set(SPIRV_TOOLS_BUILD_STATIC ON CACHE BOOL "" FORCE)

    add_subdirectory(spirv-tools EXCLUDE_FROM_ALL)

    if(momoten_had_cxx_standard)
        set(CMAKE_CXX_STANDARD "${momoten_saved_cxx_standard}")
    else()
        unset(CMAKE_CXX_STANDARD)
    endif()
endif()

if(MOMOTEN_BUILD_DRIVER OR MOMOTEN_BUILD_TESTS)
    momoten_require_submodule(opencl-headers CL/cl.h)

    if(NOT TARGET momoten_opencl_headers)
        add_library(momoten_opencl_headers INTERFACE)
        target_include_directories(momoten_opencl_headers INTERFACE
            "${CMAKE_CURRENT_SOURCE_DIR}/opencl-headers")
    endif()

    if(MOMOTEN_TEST_OPENCL_LIBRARY AND NOT TARGET momoten_test_opencl)
        add_library(momoten_test_opencl UNKNOWN IMPORTED GLOBAL)
        set_target_properties(momoten_test_opencl PROPERTIES
            IMPORTED_LOCATION "${MOMOTEN_TEST_OPENCL_LIBRARY}"
            INTERFACE_LINK_LIBRARIES momoten_opencl_headers)
    endif()
endif()

if(MOMOTEN_BUILD_DRIVER)
    momoten_require_submodule(vulkan-headers include/vulkan/vulkan.h)

    if(NOT TARGET momoten_vulkan_headers)
        add_library(momoten_vulkan_headers INTERFACE)
        target_include_directories(momoten_vulkan_headers INTERFACE
            "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-headers/include")
    endif()
endif()
