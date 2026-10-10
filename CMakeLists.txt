# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22.1)

set(MOMOTEN_VERSION 0.1.0)
set(MOMOTEN_VERSION_MAJOR 0)
set(MOMOTEN_VERSION_MINOR 1)
set(MOMOTEN_VERSION_PATCH 0)
set(MOMOTEN_IS_TOP_LEVEL OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(MOMOTEN_IS_TOP_LEVEL ON)
    project(momoten
        VERSION ${MOMOTEN_VERSION}
        DESCRIPTION "Experimental restricted Vulkan API subset over OpenCL"
        HOMEPAGE_URL "https://github.com/nihui/momoten"
        LANGUAGES C CXX)
endif()

include(GNUInstallDirs)

option(MOMOTEN_BUILD_DRIVER "Build the shared restricted-API implementation" ON)
option(MOMOTEN_BUILD_TRANSLATOR "Build the SPIR-V to OpenCL C translator library" ON)
option(MOMOTEN_BUILD_TOOLS "Build momoten command-line tools" ON)
option(MOMOTEN_BUILD_AMALGAMATION
    "Generate and compile the single-file momoten distribution"
    OFF)

if(DEFINED NCNN_MOMOTEN_BUILD_TESTS AND NOT DEFINED MOMOTEN_BUILD_TESTS)
    set(MOMOTEN_BUILD_TESTS "${NCNN_MOMOTEN_BUILD_TESTS}" CACHE BOOL
        "Build momoten generic translator and runtime tests")
else()
    option(MOMOTEN_BUILD_TESTS "Build momoten generic translator and runtime tests"
        ${MOMOTEN_IS_TOP_LEVEL})
endif()

option(MOMOTEN_INSTALL "Install momoten implementation, headers and package files"
    ${MOMOTEN_IS_TOP_LEVEL})

if(DEFINED NCNN_MOMOTEN_TEST_OPENCL_LIBRARY AND
   NOT DEFINED MOMOTEN_TEST_OPENCL_LIBRARY)
    set(MOMOTEN_TEST_OPENCL_LIBRARY "${NCNN_MOMOTEN_TEST_OPENCL_LIBRARY}" CACHE FILEPATH
        "Optional OpenCL loader used only by the direct translator test")
else()
    set(MOMOTEN_TEST_OPENCL_LIBRARY "${MOMOTEN_TEST_OPENCL_LIBRARY}" CACHE FILEPATH
        "Optional OpenCL loader used only by the direct translator test")
endif()

if(MOMOTEN_BUILD_DRIVER OR MOMOTEN_BUILD_TRANSLATOR OR MOMOTEN_BUILD_TOOLS
   OR MOMOTEN_BUILD_TESTS OR MOMOTEN_BUILD_AMALGAMATION)
    add_subdirectory(third_party)
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/momoten/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/momoten/version.h"
    @ONLY)

add_library(momoten_headers INTERFACE)
add_library(momoten::headers ALIAS momoten_headers)
target_include_directories(momoten_headers INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
set_target_properties(momoten_headers PROPERTIES EXPORT_NAME headers)

set(momoten_source_manifest
    "${CMAKE_CURRENT_SOURCE_DIR}/src/source_manifest.txt")
file(STRINGS "${momoten_source_manifest}" momoten_source_manifest_lines)
set(momoten_translator_parts)
set(momoten_driver_parts)
set(momoten_manifest_parts)
set(momoten_manifest_driver_seen OFF)
foreach(momoten_manifest_line IN LISTS momoten_source_manifest_lines)
    string(STRIP "${momoten_manifest_line}" momoten_manifest_line)
    if(momoten_manifest_line STREQUAL "" OR
       momoten_manifest_line MATCHES "^#")
        continue()
    endif()
    if(NOT momoten_manifest_line MATCHES "^(translator|driver)[ \t]+(.+)$")
        message(FATAL_ERROR
            "Invalid momoten source manifest entry: ${momoten_manifest_line}")
    endif()
    set(momoten_manifest_group "${CMAKE_MATCH_1}")
    set(momoten_manifest_path "${CMAKE_MATCH_2}")
    if(momoten_manifest_group STREQUAL "translator" AND
       momoten_manifest_driver_seen)
        message(FATAL_ERROR
            "Translator manifest entry appears after driver entry: ${momoten_manifest_path}")
    endif()
    if(momoten_manifest_group STREQUAL "driver")
        set(momoten_manifest_driver_seen ON)
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${momoten_manifest_path}")
        message(FATAL_ERROR
            "momoten source manifest path does not exist: ${momoten_manifest_path}")
    endif()
    list(APPEND momoten_${momoten_manifest_group}_parts
        "${momoten_manifest_path}")
    list(APPEND momoten_manifest_parts "${momoten_manifest_path}")
endforeach()

set(momoten_translator_sources ${momoten_translator_parts})
list(FILTER momoten_translator_sources INCLUDE REGEX "\\.cpp$")
set(momoten_driver_sources ${momoten_driver_parts})
list(FILTER momoten_driver_sources INCLUDE REGEX "\\.cpp$")

if(MOMOTEN_BUILD_DRIVER OR MOMOTEN_BUILD_TRANSLATOR OR MOMOTEN_BUILD_TOOLS OR MOMOTEN_BUILD_TESTS)
    add_library(momoten_translator STATIC ${momoten_translator_sources})
    add_library(momoten::translator ALIAS momoten_translator)
    target_compile_features(momoten_translator PUBLIC cxx_std_11)
    target_link_libraries(momoten_translator
        PUBLIC momoten_headers
        PRIVATE spirv-cross-core spirv-cross-glsl SPIRV-Tools-opt)
    set_target_properties(momoten_translator PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME translator)
endif()

foreach(momoten_pic_target spirv-cross-core spirv-cross-glsl SPIRV-Tools-opt SPIRV-Tools-static)
    if(TARGET ${momoten_pic_target})
        set_target_properties(${momoten_pic_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
endforeach()

if(MOMOTEN_BUILD_DRIVER)
    add_library(momoten_driver SHARED ${momoten_driver_sources})
    add_library(momoten::driver ALIAS momoten_driver)
    target_compile_features(momoten_driver PRIVATE cxx_std_11)
    target_compile_definitions(momoten_driver PRIVATE CL_TARGET_OPENCL_VERSION=100)
    target_link_libraries(momoten_driver PRIVATE
        momoten_translator
        momoten_opencl_headers
        momoten_vulkan_headers
        ${CMAKE_DL_LIBS})
    set_target_properties(momoten_driver PROPERTIES
        OUTPUT_NAME momoten
        EXPORT_NAME driver
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

    # Android's dynamic loader expects the unversioned libmomoten.so name.
    # Avoid a versioned SONAME and symlinks in APK/native-library packaging.
    if(NOT ANDROID)
        set_target_properties(momoten_driver PROPERTIES
            VERSION ${MOMOTEN_VERSION}
            SOVERSION 0)
    endif()

    if(UNIX AND NOT APPLE)
        target_link_options(momoten_driver PRIVATE
            "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/momoten.exports.map")
        set_property(TARGET momoten_driver APPEND PROPERTY
            LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/momoten.exports.map")
    endif()
endif()

if(MOMOTEN_BUILD_AMALGAMATION)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    set(momoten_amalgamated_source
        "${CMAKE_CURRENT_BINARY_DIR}/dist/momoten.cpp")
    set(momoten_amalgamation_inputs ${momoten_manifest_parts})
    list(TRANSFORM momoten_amalgamation_inputs
        PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

    add_custom_command(
        OUTPUT "${momoten_amalgamated_source}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/dist"
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/amalgamate.py"
            --output "${momoten_amalgamated_source}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/amalgamate.py"
            "${momoten_source_manifest}"
            ${momoten_amalgamation_inputs}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Generating ${momoten_amalgamated_source}"
        VERBATIM)
    add_library(momoten_amalgamated_translator STATIC
        "${momoten_amalgamated_source}")
    target_compile_features(momoten_amalgamated_translator PUBLIC cxx_std_11)
    target_compile_definitions(momoten_amalgamated_translator PRIVATE
        MOMOTEN_TRANSLATOR_ONLY=1)
    target_link_libraries(momoten_amalgamated_translator PRIVATE
        spirv-cross-core spirv-cross-glsl SPIRV-Tools-opt)
    set_target_properties(momoten_amalgamated_translator PROPERTIES
        POSITION_INDEPENDENT_CODE ON)

    if(MOMOTEN_BUILD_DRIVER)
        add_library(momoten_amalgamated_driver SHARED
            "${momoten_amalgamated_source}")
        target_compile_features(momoten_amalgamated_driver PRIVATE cxx_std_11)
        target_compile_definitions(momoten_amalgamated_driver PRIVATE
            CL_TARGET_OPENCL_VERSION=100)
        target_link_libraries(momoten_amalgamated_driver PRIVATE
            spirv-cross-core
            spirv-cross-glsl
            SPIRV-Tools-opt
            momoten_opencl_headers
            momoten_vulkan_headers
            ${CMAKE_DL_LIBS})
        set_target_properties(momoten_amalgamated_driver PROPERTIES
            OUTPUT_NAME momoten-amalgamated
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON)
        if(UNIX AND NOT APPLE)
            target_link_options(momoten_amalgamated_driver PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/momoten.exports.map")
            set_property(TARGET momoten_amalgamated_driver APPEND PROPERTY
                LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/momoten.exports.map")
        endif()
    endif()

    add_custom_target(momoten_amalgamation)
    add_dependencies(momoten_amalgamation
        momoten_amalgamated_translator)
    if(TARGET momoten_amalgamated_driver)
        add_dependencies(momoten_amalgamation
            momoten_amalgamated_driver)
    endif()
endif()

if(MOMOTEN_BUILD_TOOLS)
    if(NOT TARGET momoten_translator)
        message(FATAL_ERROR "MOMOTEN_BUILD_TOOLS requires the translator")
    endif()
    add_executable(momoten_spv_to_clc tools/momoten_spv_to_clc.cpp)
    target_compile_features(momoten_spv_to_clc PRIVATE cxx_std_11)
    target_link_libraries(momoten_spv_to_clc PRIVATE momoten::translator)
    set_target_properties(momoten_spv_to_clc PROPERTIES
        OUTPUT_NAME momoten-spv-to-clc)
endif()

if(MOMOTEN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MOMOTEN_INSTALL)
    include(CMakePackageConfigHelpers)

    set(momoten_install_targets momoten_headers)
    if(TARGET momoten_driver)
        list(APPEND momoten_install_targets momoten_driver)
    endif()
    install(TARGETS ${momoten_install_targets}
        EXPORT momotenTargets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    if(TARGET momoten_spv_to_clc)
        install(TARGETS momoten_spv_to_clc RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
    endif()

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/include/momoten/version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/momoten")
    if(MOMOTEN_BUILD_AMALGAMATION)
        install(FILES "${momoten_amalgamated_source}"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/momoten")
    endif()
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/momotenConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/momotenConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/momoten")
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/momotenConfigVersion.cmake"
        VERSION ${MOMOTEN_VERSION}
        COMPATIBILITY SameMajorVersion)
    install(EXPORT momotenTargets
        FILE momotenTargets.cmake
        NAMESPACE momoten::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/momoten")
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/momotenConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/momotenConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/momoten")
endif()
