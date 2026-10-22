# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

add_test(NAME momoten_version_configuration
    COMMAND "${CMAKE_COMMAND}" -DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/..
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_test.cmake")
set_tests_properties(momoten_version_configuration PROPERTIES
    LABELS "momoten;cmake;version")

add_executable(momoten_driver_version_test
    api/driver_version_test.cpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/translator/integer_dot_product.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/instance.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/driver_policy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/device_profile.cpp")
target_compile_features(momoten_driver_version_test PRIVATE cxx_std_11)
target_compile_definitions(momoten_driver_version_test PRIVATE
    CL_TARGET_OPENCL_VERSION=100)
target_include_directories(momoten_driver_version_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src")
target_link_libraries(momoten_driver_version_test PRIVATE
    momoten::headers momoten_opencl_headers momoten_vulkan_headers)
add_test(NAME momoten_driver_version COMMAND momoten_driver_version_test)
set_tests_properties(momoten_driver_version PROPERTIES LABELS "momoten;api;version")

if(MOMOTEN_BUILD_AMALGAMATION)
    add_test(NAME momoten_amalgamation_current
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../tools/amalgamate.py"
            --version-header "${CMAKE_CURRENT_BINARY_DIR}/../include/momoten/version.h"
            --check "${momoten_amalgamated_source}")
    set_tests_properties(momoten_amalgamation_current PROPERTIES
        LABELS "momoten;amalgamation;version")
endif()

add_executable(momoten_optimizer_flags_test translator/optimizer_flags_test.cpp)
target_compile_features(momoten_optimizer_flags_test PRIVATE cxx_std_11)
target_link_libraries(momoten_optimizer_flags_test PRIVATE SPIRV-Tools-opt)
add_test(NAME momoten_optimizer_flags COMMAND momoten_optimizer_flags_test)
set_tests_properties(momoten_optimizer_flags PROPERTIES
    LABELS "momoten;translator;compatibility")

add_executable(momoten_opencl_source_test translator/opencl_source_test.cpp)
target_compile_features(momoten_opencl_source_test PRIVATE cxx_std_11)
target_include_directories(momoten_opencl_source_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src")
target_link_libraries(momoten_opencl_source_test PRIVATE momoten::translator)
add_test(NAME momoten_opencl_source COMMAND momoten_opencl_source_test)
set_tests_properties(momoten_opencl_source PROPERTIES
    LABELS "momoten;translator;compatibility")

add_executable(momoten_opencl_policy_test translator/opencl_policy_test.cpp)
target_compile_features(momoten_opencl_policy_test PRIVATE cxx_std_11)
target_include_directories(momoten_opencl_policy_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src")
target_link_libraries(momoten_opencl_policy_test PRIVATE
    momoten::translator spirv-cross-glsl)
add_test(NAME momoten_opencl_policy COMMAND momoten_opencl_policy_test)
set_tests_properties(momoten_opencl_policy PROPERTIES
    LABELS "momoten;translator;compatibility")

add_executable(momoten_memory_state_test
    runtime/memory_state_test.cpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/memory_state.cpp")
target_compile_features(momoten_memory_state_test PRIVATE cxx_std_11)
target_link_libraries(momoten_memory_state_test PRIVATE
    momoten_vulkan_headers)
add_test(NAME momoten_memory_state COMMAND momoten_memory_state_test)
set_tests_properties(momoten_memory_state PROPERTIES
    LABELS "momoten;runtime;memory")

add_executable(momoten_subgroup_validation_test runtime/subgroup_validation_test.cpp)
target_compile_features(momoten_subgroup_validation_test PRIVATE cxx_std_11)
add_test(NAME momoten_subgroup_validation COMMAND momoten_subgroup_validation_test)
set_tests_properties(momoten_subgroup_validation PROPERTIES LABELS "momoten;runtime;subgroup")

add_executable(momoten_parameter_size_test
    api/parameter_size_test.cpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/driver_policy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/device_profile.cpp")
target_compile_features(momoten_parameter_size_test PRIVATE cxx_std_11)
target_compile_definitions(momoten_parameter_size_test PRIVATE
    CL_TARGET_OPENCL_VERSION=100)
target_include_directories(momoten_parameter_size_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src")
target_link_libraries(momoten_parameter_size_test PRIVATE
    momoten::translator momoten_opencl_headers momoten_vulkan_headers)
add_test(NAME momoten_parameter_size COMMAND momoten_parameter_size_test)
set_tests_properties(momoten_parameter_size PROPERTIES
    LABELS "momoten;api;abi")

add_executable(momoten_driver_policy_test
    api/driver_policy_test.cpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/driver_policy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/device_profile.cpp")
target_compile_features(momoten_driver_policy_test PRIVATE cxx_std_11)
target_compile_definitions(momoten_driver_policy_test PRIVATE
    CL_TARGET_OPENCL_VERSION=100)
target_include_directories(momoten_driver_policy_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src")
target_link_libraries(momoten_driver_policy_test PRIVATE
    momoten::translator momoten_opencl_headers momoten_vulkan_headers)
add_test(NAME momoten_driver_policy COMMAND momoten_driver_policy_test)
set_tests_properties(momoten_driver_policy PROPERTIES
    LABELS "momoten;api;profile;integer-dot")

add_executable(momoten_device_profile_test
    api/device_profile_test.cpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/driver_policy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/device_profile.cpp")
target_compile_features(momoten_device_profile_test PRIVATE cxx_std_11)
target_compile_definitions(momoten_device_profile_test PRIVATE
    CL_TARGET_OPENCL_VERSION=100)
target_include_directories(momoten_device_profile_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src")
target_link_libraries(momoten_device_profile_test PRIVATE
    momoten::translator momoten_opencl_headers momoten_vulkan_headers)
add_test(NAME momoten_device_profile COMMAND momoten_device_profile_test)
set_tests_properties(momoten_device_profile PROPERTIES
    LABELS "momoten;api;profile;subgroup;compiler")

function(momoten_add_integer_dot_test target_name translator_target test_name)
    add_executable(${target_name} translator/integer_dot_product_test.cpp)
    target_compile_features(${target_name} PRIVATE cxx_std_11)
    target_link_libraries(${target_name} PRIVATE ${translator_target} momoten::headers SPIRV-Tools-static)
    add_test(NAME ${test_name} COMMAND ${target_name})
    set_tests_properties(${test_name} PROPERTIES LABELS "momoten;translator;integer-dot")
endfunction()

momoten_add_integer_dot_test(momoten_integer_dot_product_test
    momoten::translator momoten_integer_dot_product)
if(TARGET momoten_amalgamated_translator)
    momoten_add_integer_dot_test(momoten_amalgamated_integer_dot_product_test
        momoten_amalgamated_translator momoten_amalgamated_integer_dot_product)
endif()

function(momoten_add_float_controls_test target_name translator_target test_name)
    add_executable(${target_name}
        translator/float_controls_test.cpp
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/driver_policy.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/device_profile.cpp")
    target_compile_features(${target_name} PRIVATE cxx_std_11)
    target_compile_definitions(${target_name} PRIVATE
        CL_TARGET_OPENCL_VERSION=100)
    target_include_directories(${target_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../src")
    target_link_libraries(${target_name} PRIVATE
        ${translator_target} momoten::headers momoten_opencl_headers
        momoten_vulkan_headers SPIRV-Tools-static)
    add_test(NAME ${test_name} COMMAND ${target_name})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "momoten;translator;float-controls")
endfunction()

momoten_add_float_controls_test(momoten_float_controls_test
    momoten::translator momoten_float_controls)
if(TARGET momoten_amalgamated_translator)
    momoten_add_float_controls_test(momoten_amalgamated_float_controls_test
        momoten_amalgamated_translator momoten_amalgamated_float_controls)
    set_tests_properties(momoten_amalgamated_float_controls PROPERTIES
        LABELS "momoten;amalgamation;translator;float-controls")
endif()

function(momoten_add_workgroup_pipeline_test target_name translator_target test_name)
    add_executable(${target_name}
        api/workgroup_pipeline_test.cpp
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/pipeline.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/memory.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/pipeline_build.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/program.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/cache.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/driver_policy.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/device_profile.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/buffer_copy.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/replay.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/memory_sync.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/memory_state.cpp")
    target_compile_features(${target_name} PRIVATE cxx_std_11)
    target_compile_definitions(${target_name} PRIVATE
        CL_TARGET_OPENCL_VERSION=100)
    target_include_directories(${target_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../src")
    target_link_libraries(${target_name} PRIVATE
        ${translator_target} momoten::headers momoten_opencl_headers
        momoten_vulkan_headers SPIRV-Tools-static)
    add_test(NAME ${test_name} COMMAND ${target_name})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "momoten;api;workgroup")
endfunction()

function(momoten_add_spirv_analysis_test name translator_target prefix)
    set(target_name ${prefix}_${name}_test)
    set(test_name ${prefix}_${name})
    string(REPLACE "_" "-" label ${name})
    add_executable(${target_name} translator/${name}_test.cpp)
    target_compile_features(${target_name} PRIVATE cxx_std_11)
    target_include_directories(${target_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../src")
    target_link_libraries(${target_name} PRIVATE
        ${translator_target} momoten::headers spirv-cross-glsl SPIRV-Tools-static)
    add_test(NAME ${test_name} COMMAND ${target_name})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "momoten;translator;${label}")
endfunction()

foreach(name integer_ranges buffer_loads)
    momoten_add_spirv_analysis_test(${name} momoten::translator momoten)
    if(TARGET momoten_amalgamated_translator)
        momoten_add_spirv_analysis_test(${name}
            momoten_amalgamated_translator momoten_amalgamated)
    endif()
endforeach()

momoten_add_workgroup_pipeline_test(momoten_workgroup_pipeline_test
    momoten::translator momoten_workgroup_pipeline)
if(TARGET momoten_amalgamated_translator)
    momoten_add_workgroup_pipeline_test(momoten_amalgamated_workgroup_pipeline_test
        momoten_amalgamated_translator momoten_amalgamated_workgroup_pipeline)
    set_tests_properties(momoten_amalgamated_workgroup_pipeline PROPERTIES
        LABELS "momoten;amalgamation;api;workgroup")
endif()

# Validate that the installed runtime does not expose a development SDK.
if(TARGET momoten_driver AND MOMOTEN_INSTALL AND NOT CMAKE_CROSSCOMPILING)
    add_test(NAME momoten_runtime_install
        COMMAND "${CMAKE_COMMAND}"
            -DBUILD_DIR=${CMAKE_BINARY_DIR}
            -DSTAGE_DIR=${CMAKE_CURRENT_BINARY_DIR}/runtime-install
            -DBIN_DIR=${CMAKE_INSTALL_BINDIR}
            -DLIB_DIR=${CMAKE_INSTALL_LIBDIR}
            -DRUNTIME_FILE=$<TARGET_FILE_NAME:momoten_driver>
            -DTEST_CONFIG=$<CONFIG>
            -DTOOLS_ENABLED=${MOMOTEN_BUILD_TOOLS}
            -DBUILD_ICD_MANIFEST=${MOMOTEN_BUILD_ICD_MANIFEST}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/runtime/install_layout_test.cmake")
    set_tests_properties(momoten_runtime_install PROPERTIES
        LABELS "momoten;runtime;install")
endif()

if(TARGET momoten_driver AND NOT CMAKE_CROSSCOMPILING)
    add_test(NAME momoten_no_system_opencl_dependency
        COMMAND "${CMAKE_COMMAND}"
            -DDRIVER=$<TARGET_FILE:momoten_driver>
            -P "${CMAKE_CURRENT_SOURCE_DIR}/runtime/no_opencl_dependency_test.cmake")
    set_tests_properties(momoten_no_system_opencl_dependency PROPERTIES
        LABELS "momoten;runtime")
endif()

if(TARGET momoten_driver)
    set(momoten_32bit_driver_sources)
    foreach(momoten_driver_source IN LISTS momoten_driver_sources)
        list(APPEND momoten_32bit_driver_sources
            "${CMAKE_CURRENT_SOURCE_DIR}/../${momoten_driver_source}")
    endforeach()
    add_library(momoten_32bit_handle_compile OBJECT
        ${momoten_32bit_driver_sources})
    target_compile_features(momoten_32bit_handle_compile PRIVATE cxx_std_11)
    target_compile_definitions(momoten_32bit_handle_compile PRIVATE
        CL_TARGET_OPENCL_VERSION=100
        VK_USE_64_BIT_PTR_DEFINES=0)
    target_link_libraries(momoten_32bit_handle_compile PRIVATE
        momoten::headers
        momoten_opencl_headers
        momoten_vulkan_headers)
endif()

if(TARGET momoten_driver)
    add_executable(momoten_non_conformant_warning_test
        api/non_conformant_warning_test.cpp)
    target_compile_features(momoten_non_conformant_warning_test PRIVATE cxx_std_11)
    target_link_libraries(momoten_non_conformant_warning_test PRIVATE
        momoten::driver momoten_vulkan_headers)
    add_test(NAME momoten_non_conformant_warning
        COMMAND "${CMAKE_COMMAND}"
            -DTEST_EXECUTABLE=$<TARGET_FILE:momoten_non_conformant_warning_test>
            -DPROJECT_NAME=momoten
            -P "${CMAKE_CURRENT_SOURCE_DIR}/api/non_conformant_warning_test.cmake")
    set_tests_properties(momoten_non_conformant_warning PROPERTIES
        LABELS "momoten;api;conformance")
endif()

if(TARGET momoten_amalgamated_driver)
    if(NOT CMAKE_CROSSCOMPILING)
        add_test(NAME momoten_amalgamated_no_system_opencl_dependency
            COMMAND "${CMAKE_COMMAND}"
                -DDRIVER=$<TARGET_FILE:momoten_amalgamated_driver>
                -P "${CMAKE_CURRENT_SOURCE_DIR}/runtime/no_opencl_dependency_test.cmake")
        set_tests_properties(momoten_amalgamated_no_system_opencl_dependency PROPERTIES
            LABELS "momoten;amalgamation;runtime")
    endif()
endif()

# Mock the Android loader on POSIX hosts without Android, OpenCL, or a GPU.
if(UNIX AND NOT ANDROID)
    find_package(Threads REQUIRED)
    add_executable(momoten_android_opencl_loader_test
        runtime/android_opencl_loader_test.cpp)
    target_compile_features(momoten_android_opencl_loader_test PRIVATE cxx_std_11)
    target_compile_definitions(momoten_android_opencl_loader_test PRIVATE
        CL_TARGET_OPENCL_VERSION=100)
    target_include_directories(momoten_android_opencl_loader_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime/android_stubs")
    target_link_libraries(momoten_android_opencl_loader_test PRIVATE
        momoten_opencl_headers Threads::Threads)
    add_test(NAME momoten_android_opencl_loader
        COMMAND momoten_android_opencl_loader_test)
    set_tests_properties(momoten_android_opencl_loader PROPERTIES
        LABELS "momoten;runtime;android;loader")

    # Exercise the actual Android 7 ELF and linker helper with host Android stubs.
    # /proc/self/maps and native ELF headers make this test Linux-specific.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(momoten_android7_opencl_loader_test
            runtime/android7_opencl_loader_test.cpp
            ../src/backend/android7_loader.cpp)
        target_compile_features(momoten_android7_opencl_loader_test PRIVATE cxx_std_11)
        target_compile_definitions(momoten_android7_opencl_loader_test PRIVATE __ANDROID__)
        target_include_directories(momoten_android7_opencl_loader_test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/runtime/android_stubs")
        target_link_libraries(momoten_android7_opencl_loader_test PRIVATE
            Threads::Threads ${CMAKE_DL_LIBS})
        add_test(NAME momoten_android7_opencl_loader
            COMMAND momoten_android7_opencl_loader_test)
        set_tests_properties(momoten_android7_opencl_loader PROPERTIES
            LABELS "momoten;runtime;android;loader")
    endif()
endif()

if(DEFINED NCNN_MOMOTEN_GLSLANG_VALIDATOR AND
   NOT DEFINED MOMOTEN_GLSLANG_VALIDATOR)
    set(MOMOTEN_GLSLANG_VALIDATOR "${NCNN_MOMOTEN_GLSLANG_VALIDATOR}" CACHE FILEPATH
        "Path to glslangValidator used by momoten tests")
endif()
find_program(MOMOTEN_GLSLANG_VALIDATOR glslangValidator)

if(NOT MOMOTEN_GLSLANG_VALIDATOR)
    message(WARNING "glslangValidator not found; momoten shader tests are disabled")
    return()
endif()

set(momoten_test_spv_files)
function(momoten_compile_test_shader output_variable source_name target_env)
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${source_name}.spv")
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${source_name}")
    add_custom_command(
        OUTPUT "${output}"
        COMMAND "${MOMOTEN_GLSLANG_VALIDATOR}" -V --target-env "${target_env}"
                -o "${output}" "${source}"
        DEPENDS "${source}"
        VERBATIM)
    set(${output_variable} "${output}" PARENT_SCOPE)
    set(momoten_test_spv_files
        ${momoten_test_spv_files} "${output}" PARENT_SCOPE)
endfunction()

momoten_compile_test_shader(momoten_test_spv simple_buffer.comp vulkan1.0)
momoten_compile_test_shader(momoten_unsupported_test_spv unsupported_image.comp vulkan1.0)
momoten_compile_test_shader(momoten_atomic_test_spv atomic_compare_exchange.comp vulkan1.0)
momoten_compile_test_shader(momoten_atomic_packed_test_spv atomic_packed.comp vulkan1.0)
momoten_compile_test_shader(momoten_fp16_test_spv fp16_buffer.comp vulkan1.1)
momoten_compile_test_shader(momoten_fp16_basic_test_spv fp16_basic_buffer.comp vulkan1.1)
momoten_compile_test_shader(momoten_fp16_program_constant_test_spv fp16_program_constant.comp vulkan1.1)
momoten_compile_test_shader(momoten_fp64_test_spv fp64_buffer.comp vulkan1.1)
momoten_compile_test_shader(momoten_int16_test_spv int16_buffer.comp vulkan1.1)
momoten_compile_test_shader(momoten_scalar_16bit_test_spv scalar_16bit_buffer.comp vulkan1.1)
momoten_compile_test_shader(momoten_subgroup_basic_test_spv subgroup_basic.comp vulkan1.1)
momoten_compile_test_shader(momoten_subgroup_values_test_spv subgroup_values.comp vulkan1.1)
momoten_compile_test_shader(momoten_opencl_names_test_spv opencl_names.comp vulkan1.1)
momoten_compile_test_shader(momoten_integer_dot_product_test_spv integer_dot_product.comp vulkan1.1)
momoten_compile_test_shader(momoten_workgroup_split_test_spv workgroup_split.comp vulkan1.1)
momoten_compile_test_shader(momoten_workgroup_3d_test_spv workgroup_3d.comp vulkan1.1)
momoten_compile_test_shader(momoten_vector_extract_test_spv vector_extract_dynamic.comp vulkan1.1)
momoten_compile_test_shader(momoten_vector_extract_fp64_test_spv vector_extract_fp64.comp vulkan1.1)
momoten_compile_test_shader(momoten_vector_extract_int16_test_spv vector_extract_int16.comp vulkan1.1)
momoten_compile_test_shader(momoten_buffer_vector_load_test_spv buffer_vector_load.comp vulkan1.0)
momoten_compile_test_shader(momoten_buffer_vector_store_test_spv buffer_vector_store.comp vulkan1.0)
momoten_compile_test_shader(momoten_fp16_vector_load_test_spv fp16_vector_load.comp vulkan1.1)
momoten_compile_test_shader(momoten_coalesced_buffer_load_test_spv coalesced_buffer_load.comp vulkan1.1)
momoten_compile_test_shader(momoten_coalesced_fp16_load_test_spv coalesced_fp16_load.comp vulkan1.1)
momoten_compile_test_shader(momoten_buffer_alias_test_spv buffer_alias.comp vulkan1.1)
momoten_compile_test_shader(momoten_memory_lowering_test_spv memory_lowering.comp vulkan1.1)
momoten_compile_test_shader(momoten_buffer_read_write_test_spv buffer_read_write.comp vulkan1.1)
momoten_compile_test_shader(momoten_fp16_buffer_read_write_test_spv fp16_buffer_read_write.comp vulkan1.1)
momoten_compile_test_shader(momoten_pow_constant_base_test_spv pow_constant_base.comp vulkan1.0)
momoten_compile_test_shader(momoten_integer_ranges_test_spv integer_ranges.comp vulkan1.0)

add_custom_target(momoten-test-spv DEPENDS ${momoten_test_spv_files})

function(momoten_add_memory_lowering_test target translator)
    add_executable(${target} translator/memory_lowering_test.cpp)
    target_compile_features(${target} PRIVATE cxx_std_11)
    target_link_libraries(${target} PRIVATE ${translator} momoten::headers spirv-cross-core SPIRV-Tools-static)
    add_dependencies(${target} momoten-test-spv)
    add_test(NAME ${target} COMMAND ${target}
        "${momoten_memory_lowering_test_spv}" "${momoten_fp16_buffer_read_write_test_spv}" "${momoten_fp64_test_spv}")
    set_tests_properties(${target} PROPERTIES LABELS "momoten;translator;memory")
endfunction()

momoten_add_memory_lowering_test(momoten_memory_lowering_test momoten::translator)
if(TARGET momoten_amalgamated_translator)
    momoten_add_memory_lowering_test(momoten_amalgamated_memory_lowering_test momoten_amalgamated_translator)
endif()

add_executable(momoten_translator_test translator/translator_test.cpp)
target_compile_features(momoten_translator_test PRIVATE cxx_std_11)
target_link_libraries(momoten_translator_test PRIVATE momoten::translator)
add_dependencies(momoten_translator_test momoten-test-spv)
add_test(NAME momoten_translator_simple
    COMMAND momoten_translator_test
        "${momoten_test_spv}"
        "${momoten_unsupported_test_spv}"
        "${momoten_atomic_test_spv}"
        "${momoten_fp16_test_spv}"
        "${momoten_int16_test_spv}"
        "${momoten_subgroup_basic_test_spv}"
        "${momoten_integer_dot_product_test_spv}"
        "${momoten_workgroup_split_test_spv}"
        "${momoten_fp64_test_spv}"
        "${momoten_fp16_program_constant_test_spv}"
        "${momoten_workgroup_3d_test_spv}"
        "${momoten_vector_extract_test_spv}"
        "${momoten_vector_extract_fp64_test_spv}")
set_tests_properties(momoten_translator_simple PROPERTIES
    LABELS "momoten;translator")

if(TARGET momoten_amalgamated_translator)
    add_executable(momoten_amalgamated_translator_test
        translator/translator_test.cpp)
    target_compile_features(momoten_amalgamated_translator_test PRIVATE cxx_std_11)
    target_link_libraries(momoten_amalgamated_translator_test PRIVATE
        momoten_amalgamated_translator momoten::headers)
    add_dependencies(momoten_amalgamated_translator_test momoten-test-spv)
    add_test(NAME momoten_amalgamated_translator_simple
        COMMAND momoten_amalgamated_translator_test
            "${momoten_test_spv}"
            "${momoten_unsupported_test_spv}"
            "${momoten_atomic_test_spv}"
            "${momoten_fp16_test_spv}"
            "${momoten_int16_test_spv}"
            "${momoten_subgroup_basic_test_spv}"
            "${momoten_integer_dot_product_test_spv}"
            "${momoten_workgroup_split_test_spv}"
            "${momoten_fp64_test_spv}"
            "${momoten_fp16_program_constant_test_spv}"
            "${momoten_workgroup_3d_test_spv}"
            "${momoten_vector_extract_test_spv}"
            "${momoten_vector_extract_fp64_test_spv}")
    set_tests_properties(momoten_amalgamated_translator_simple PROPERTIES
        LABELS "momoten;amalgamation;translator")
endif()

function(momoten_add_opencl_symbols_test target_name translator_target test_name)
    add_executable(${target_name} translator/opencl_symbols_test.cpp)
    target_compile_features(${target_name} PRIVATE cxx_std_11)
    target_link_libraries(${target_name} PRIVATE ${translator_target} momoten::headers SPIRV-Tools-static)
    add_dependencies(${target_name} momoten-test-spv)
    add_test(NAME ${test_name} COMMAND ${target_name}
        "${momoten_test_spv}"
        "${momoten_opencl_names_test_spv}"
        "${momoten_vector_extract_test_spv}"
        "${momoten_pow_constant_base_test_spv}")
    set_tests_properties(${test_name} PROPERTIES LABELS "momoten;translator;compiler")
endfunction()

momoten_add_opencl_symbols_test(momoten_opencl_symbols_test momoten::translator momoten_opencl_symbols)
if(TARGET momoten_amalgamated_translator)
    momoten_add_opencl_symbols_test(momoten_amalgamated_opencl_symbols_test
        momoten_amalgamated_translator momoten_amalgamated_opencl_symbols)
    set_tests_properties(momoten_amalgamated_opencl_symbols PROPERTIES
        LABELS "momoten;amalgamation;translator;compiler")
endif()

if(DEFINED NCNN_MOMOTEN_CLANG AND NOT DEFINED MOMOTEN_CLANG)
    set(MOMOTEN_CLANG "${NCNN_MOMOTEN_CLANG}" CACHE FILEPATH
        "Path to clang used by momoten syntax tests")
endif()
find_program(MOMOTEN_CLANG clang)
if(MOMOTEN_CLANG)
    add_test(NAME momoten_memory_lowering_syntax
        COMMAND "${CMAKE_COMMAND}"
            -DTEST=$<TARGET_FILE:momoten_memory_lowering_test>
            -DCLANG=${MOMOTEN_CLANG}
            -DSPV=${momoten_memory_lowering_test_spv}
            -DFP16_SPV=${momoten_fp16_buffer_read_write_test_spv}
            -DFP64_SPV=${momoten_fp64_test_spv}
            -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/memory-lowering
            -P "${CMAKE_CURRENT_SOURCE_DIR}/translator/memory_lowering_test.cmake")
    set_tests_properties(momoten_memory_lowering_syntax PROPERTIES
        LABELS "momoten;translator;opencl-1.0")
    # Keep syntax regressions active in runtime-only CI without installing a CLI.
    if(TARGET momoten_spv_to_clc)
        set(momoten_syntax_translator momoten_spv_to_clc)
    else()
        add_executable(momoten_test_spv_to_clc
            "${CMAKE_CURRENT_SOURCE_DIR}/../tools/momoten_spv_to_clc.cpp")
        target_compile_features(momoten_test_spv_to_clc PRIVATE cxx_std_11)
        target_link_libraries(momoten_test_spv_to_clc PRIVATE momoten::translator)
        set(momoten_syntax_translator momoten_test_spv_to_clc)
    endif()
    add_test(NAME momoten_opencl_c_1_0_syntax
        COMMAND "${CMAKE_COMMAND}"
            -DTRANSLATOR=$<TARGET_FILE:${momoten_syntax_translator}>
            -DCLANG=${MOMOTEN_CLANG}
            -DSPV_FILE=${momoten_test_spv}
            -DATOMIC_SPV_FILE=${momoten_atomic_test_spv}
            -DFP16_SPV_FILE=${momoten_fp16_test_spv}
            -DFP64_SPV_FILE=${momoten_fp64_test_spv}
            -DFP16_CONSTANT_SPV_FILE=${momoten_fp16_program_constant_test_spv}
            -DVECTOR_EXTRACT_SPV_FILE=${momoten_vector_extract_test_spv}
            -DVECTOR_EXTRACT_FP64_SPV_FILE=${momoten_vector_extract_fp64_test_spv}
            -DVECTOR_EXTRACT_INT16_SPV_FILE=${momoten_vector_extract_int16_test_spv}
            -DPOW_CONSTANT_BASE_SPV_FILE=${momoten_pow_constant_base_test_spv}
            -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/strict-cl10
            -P "${CMAKE_CURRENT_SOURCE_DIR}/translator/strict_cl10_test.cmake")
    set_tests_properties(momoten_opencl_c_1_0_syntax PROPERTIES
        LABELS "momoten;translator;opencl-1.0")

    add_test(NAME momoten_opencl_names
        COMMAND "${CMAKE_COMMAND}"
            -DTRANSLATOR=$<TARGET_FILE:${momoten_syntax_translator}>
            -DCLANG=${MOMOTEN_CLANG}
            -DSPV_FILE=${momoten_opencl_names_test_spv}
            -DGLSLANG=${MOMOTEN_GLSLANG_VALIDATOR}
            -DCOLLISION_SHADER=${CMAKE_CURRENT_SOURCE_DIR}/shaders/opencl_name_collision.comp
            -DFUNCTION_SHADER=${CMAKE_CURRENT_SOURCE_DIR}/shaders/opencl_name_function.comp
            -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/opencl-names
            -P "${CMAKE_CURRENT_SOURCE_DIR}/translator/opencl_names_test.cmake")
    set_tests_properties(momoten_opencl_names PROPERTIES
        LABELS "momoten;translator;compiler")

    add_test(NAME momoten_opencl_apple_type_aliases
        COMMAND "${CMAKE_COMMAND}"
            -DTRANSLATOR=$<TARGET_FILE:${momoten_syntax_translator}>
            -DCLANG=${MOMOTEN_CLANG}
            -DSPV_FILE=${momoten_test_spv}
            -DATOMIC_SPV_FILE=${momoten_atomic_test_spv}
            -DFP16_SPV_FILE=${momoten_fp16_test_spv}
            -DFP64_SPV_FILE=${momoten_fp64_test_spv}
            -DFP16_CONSTANT_SPV_FILE=${momoten_fp16_program_constant_test_spv}
            -DVECTOR_EXTRACT_SPV_FILE=${momoten_vector_extract_test_spv}
            -DVECTOR_EXTRACT_FP64_SPV_FILE=${momoten_vector_extract_fp64_test_spv}
            -DVECTOR_EXTRACT_INT16_SPV_FILE=${momoten_vector_extract_int16_test_spv}
            -DPOW_CONSTANT_BASE_SPV_FILE=${momoten_pow_constant_base_test_spv}
            -DAPPLE_TYPE_ALIASES=ON
            -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/apple-type-aliases
            -P "${CMAKE_CURRENT_SOURCE_DIR}/translator/strict_cl10_test.cmake")
    set_tests_properties(momoten_opencl_apple_type_aliases PROPERTIES
        LABELS "momoten;translator;opencl-1.0;apple")
endif()

if(TARGET momoten_test_opencl)
    add_executable(momoten_vendor_integer_dot_product_test runtime/vendor_integer_dot_product_test.cpp)
    target_compile_features(momoten_vendor_integer_dot_product_test PRIVATE cxx_std_11)
    target_compile_definitions(momoten_vendor_integer_dot_product_test PRIVATE CL_TARGET_OPENCL_VERSION=100)
    target_include_directories(momoten_vendor_integer_dot_product_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")
    target_link_libraries(momoten_vendor_integer_dot_product_test PRIVATE
        momoten::translator momoten_test_opencl SPIRV-Tools-static)
    add_test(NAME momoten_vendor_integer_dot_product_emulated COMMAND momoten_vendor_integer_dot_product_test)
    add_test(NAME momoten_vendor_integer_dot_product_native COMMAND momoten_vendor_integer_dot_product_test native)
    set_tests_properties(momoten_vendor_integer_dot_product_emulated momoten_vendor_integer_dot_product_native
        PROPERTIES SKIP_RETURN_CODE 77 LABELS "momoten;opencl;integer-dot")

    add_executable(momoten_opencl_test runtime/opencl_test.cpp)
    target_compile_features(momoten_opencl_test PRIVATE cxx_std_11)
    target_compile_definitions(momoten_opencl_test PRIVATE CL_TARGET_OPENCL_VERSION=100)
    target_link_libraries(momoten_opencl_test PRIVATE
        momoten::translator momoten_test_opencl spirv-cross-core)
    add_dependencies(momoten_opencl_test momoten-test-spv)
    add_test(NAME momoten_opencl_simple
        COMMAND momoten_opencl_test "${momoten_test_spv}")
    add_test(NAME momoten_opencl_fp16
        COMMAND momoten_opencl_test "${momoten_fp16_test_spv}" fp16)
    add_test(NAME momoten_opencl_fp16_basic
        COMMAND momoten_opencl_test "${momoten_fp16_basic_test_spv}" fp16)
    add_test(NAME momoten_opencl_fp64
        COMMAND momoten_opencl_test "${momoten_fp64_test_spv}" fp64)
    add_test(NAME momoten_opencl_int16
        COMMAND momoten_opencl_test "${momoten_int16_test_spv}" int16)
    add_test(NAME momoten_opencl_atomic_packed
        COMMAND momoten_opencl_test "${momoten_atomic_packed_test_spv}" atomic-packed)
    add_test(NAME momoten_opencl_scalar_16bit
        COMMAND momoten_opencl_test "${momoten_scalar_16bit_test_spv}" scalar-16bit)
    add_test(NAME momoten_opencl_subgroup_basic
        COMMAND momoten_opencl_test "${momoten_subgroup_basic_test_spv}" subgroup-basic)
    add_test(NAME momoten_opencl_integer_dot_product
        COMMAND momoten_opencl_test "${momoten_integer_dot_product_test_spv}"
            integer-dot-product)
    add_test(NAME momoten_opencl_workgroup_split
        COMMAND momoten_opencl_test "${momoten_workgroup_split_test_spv}"
            workgroup-split)
    add_test(NAME momoten_opencl_workgroup_3d
        COMMAND momoten_opencl_test "${momoten_workgroup_3d_test_spv}"
            workgroup-3d)
    add_test(NAME momoten_opencl_integer_ranges
        COMMAND momoten_opencl_test "${momoten_integer_ranges_test_spv}"
            integer-ranges)
    add_test(NAME momoten_opencl_vector_access
        COMMAND momoten_opencl_test "${momoten_vector_extract_test_spv}"
            vector-access)
    add_test(NAME momoten_opencl_vector_access_fp64
        COMMAND momoten_opencl_test "${momoten_vector_extract_fp64_test_spv}"
            vector-access-fp64)
    add_test(NAME momoten_opencl_vector_extract_int16
        COMMAND momoten_opencl_test "${momoten_vector_extract_int16_test_spv}"
            vector-extract-int16)
    add_test(NAME momoten_opencl_fp16_vector_load
        COMMAND momoten_opencl_test "${momoten_fp16_vector_load_test_spv}"
            fp16-vector-load)
    add_test(NAME momoten_opencl_coalesced_buffer_load
        COMMAND momoten_opencl_test "${momoten_coalesced_buffer_load_test_spv}"
            coalesced-buffer-load)
    add_test(NAME momoten_opencl_coalesced_fp16_load
        COMMAND momoten_opencl_test "${momoten_coalesced_fp16_load_test_spv}"
            coalesced-fp16-load)
    add_test(NAME momoten_opencl_buffer_vector_store
        COMMAND momoten_opencl_test "${momoten_buffer_vector_store_test_spv}"
            buffer-vector-store)
    add_test(NAME momoten_opencl_buffer_vector_load
        COMMAND momoten_opencl_test "${momoten_buffer_vector_load_test_spv}"
            buffer-vector-load)
    foreach(prefix "" "fp16_")
        string(REPLACE "_" "-" mode "${prefix}buffer-read-write")
        add_test(NAME momoten_opencl_${prefix}buffer_read_write_scalar
            COMMAND momoten_opencl_test "${momoten_${prefix}buffer_read_write_test_spv}" ${mode} scalar-vector-memory)
        add_test(NAME momoten_opencl_${prefix}buffer_read_write
            COMMAND momoten_opencl_test "${momoten_${prefix}buffer_read_write_test_spv}" ${mode})
        add_test(NAME momoten_opencl_${prefix}buffer_read_write_native_stores
            COMMAND momoten_opencl_test "${momoten_${prefix}buffer_read_write_test_spv}" ${mode} native-vector-memory)
        set_tests_properties(momoten_opencl_${prefix}buffer_read_write
            momoten_opencl_${prefix}buffer_read_write_scalar
            momoten_opencl_${prefix}buffer_read_write_native_stores PROPERTIES
            LABELS "momoten;runtime;direct-opencl"
            SKIP_RETURN_CODE 77)
    endforeach()
    foreach(direction buffer-alias buffer-alias-reverse)
        foreach(mode scalar-vector-memory native-vector-loads native-vector-memory)
            add_test(NAME momoten_opencl_${direction}_${mode}
                COMMAND momoten_opencl_test "${momoten_buffer_alias_test_spv}" ${direction} ${mode})
            set_tests_properties(momoten_opencl_${direction}_${mode} PROPERTIES
                LABELS "momoten;runtime;direct-opencl" SKIP_RETURN_CODE 77)
        endforeach()
    endforeach()
    foreach(mode scalar-vector-memory native-vector-loads native-vector-memory)
        add_test(NAME momoten_opencl_memory_lowering_${mode}
            COMMAND momoten_opencl_test "${momoten_memory_lowering_test_spv}" memory-lowering ${mode})
        set_tests_properties(momoten_opencl_memory_lowering_${mode} PROPERTIES
            LABELS "momoten;runtime;direct-opencl" SKIP_RETURN_CODE 77)
        add_test(NAME momoten_opencl_fp16_buffer_alias_${mode}
            COMMAND momoten_opencl_test "${momoten_fp16_buffer_read_write_test_spv}" fp16-buffer-alias ${mode})
        set_tests_properties(momoten_opencl_fp16_buffer_alias_${mode} PROPERTIES
            LABELS "momoten;runtime;direct-opencl" SKIP_RETURN_CODE 77)
    endforeach()
    add_test(NAME momoten_opencl_fp64_buffer_alias
        COMMAND momoten_opencl_test "${momoten_fp64_test_spv}" fp64-buffer-alias)
    set_tests_properties(momoten_opencl_fp64_buffer_alias PROPERTIES
        LABELS "momoten;runtime;direct-opencl" SKIP_RETURN_CODE 77)
    add_test(NAME momoten_opencl_memory_lowering_copied
        COMMAND momoten_opencl_test "${momoten_memory_lowering_test_spv}" memory-lowering-copied native-vector-memory)
    set_tests_properties(momoten_opencl_memory_lowering_copied PROPERTIES
        LABELS "momoten;runtime;direct-opencl" SKIP_RETURN_CODE 77)
    foreach(width scalar vec2 vec3 vec4)
        add_test(NAME momoten_opencl_pow_${width}
            COMMAND momoten_opencl_test "${momoten_pow_constant_base_test_spv}"
                pow-${width})
        set_tests_properties(momoten_opencl_pow_${width} PROPERTIES
            LABELS "momoten;runtime;direct-opencl"
            SKIP_RETURN_CODE 77)
    endforeach()
    set_tests_properties(momoten_opencl_simple momoten_opencl_fp16
        momoten_opencl_fp16_basic
        momoten_opencl_fp64
        momoten_opencl_int16
        momoten_opencl_atomic_packed momoten_opencl_scalar_16bit
        momoten_opencl_subgroup_basic momoten_opencl_integer_dot_product
        momoten_opencl_workgroup_split momoten_opencl_workgroup_3d
        momoten_opencl_integer_ranges
        momoten_opencl_vector_access momoten_opencl_vector_access_fp64
        momoten_opencl_vector_extract_int16 momoten_opencl_buffer_vector_load
        momoten_opencl_buffer_vector_store momoten_opencl_fp16_vector_load
        momoten_opencl_coalesced_buffer_load momoten_opencl_coalesced_fp16_load PROPERTIES
        LABELS "momoten;runtime;direct-opencl"
        SKIP_RETURN_CODE 77)
    # These fixtures synchronize more than one invocation in a workgroup.
    set_property(TEST momoten_opencl_simple momoten_opencl_subgroup_basic
        momoten_opencl_vector_access APPEND PROPERTY
        LABELS "requires-opencl-multilane-barrier")
else()
    message(STATUS
        "MOMOTEN_TEST_OPENCL_LIBRARY is unset; the optional direct-OpenCL test is disabled")
endif()

if(TARGET momoten_driver)
    add_executable(momoten_abi_guard_test api/abi_guard_test.cpp)
    target_compile_features(momoten_abi_guard_test PRIVATE cxx_std_11)
    target_include_directories(momoten_abi_guard_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../src")
    target_link_libraries(momoten_abi_guard_test PRIVATE
        momoten_vulkan_headers)
    add_test(NAME momoten_abi_guard COMMAND momoten_abi_guard_test)
    set_tests_properties(momoten_abi_guard PROPERTIES
        LABELS "momoten;api;abi")

    add_executable(momoten_proc_registry_test api/proc_registry_test.cpp)
    target_compile_features(momoten_proc_registry_test PRIVATE cxx_std_11)
    target_link_libraries(momoten_proc_registry_test PRIVATE
        momoten::driver momoten_vulkan_headers ${CMAKE_DL_LIBS})
    add_test(NAME momoten_proc_registry
        COMMAND momoten_proc_registry_test $<TARGET_FILE:momoten_driver>)
    set_tests_properties(momoten_proc_registry PROPERTIES
        LABELS "momoten;api;proc"
        SKIP_RETURN_CODE 77)

    add_executable(momoten_profile_test api/profile_test.cpp)
    target_compile_features(momoten_profile_test PRIVATE cxx_std_11)
    target_link_libraries(momoten_profile_test PRIVATE
        momoten::driver momoten_vulkan_headers)
    add_test(NAME momoten_api_profile COMMAND momoten_profile_test)
    set_tests_properties(momoten_api_profile PROPERTIES
        LABELS "momoten;api;profile"
        SKIP_RETURN_CODE 77)

    add_executable(momoten_driver_test runtime/driver_test.cpp)
    target_compile_features(momoten_driver_test PRIVATE cxx_std_11)
    target_compile_definitions(momoten_driver_test PRIVATE CL_TARGET_OPENCL_VERSION=100)
    target_include_directories(momoten_driver_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")
    target_link_libraries(momoten_driver_test PRIVATE
        momoten::driver momoten::headers momoten_vulkan_headers momoten_opencl_headers)
    add_dependencies(momoten_driver_test momoten-test-spv)

    add_test(NAME momoten_driver_simple
        COMMAND momoten_driver_test "${momoten_test_spv}"
            "${momoten_subgroup_basic_test_spv}"
            "${momoten_integer_dot_product_test_spv}"
            "${momoten_subgroup_values_test_spv}")
    set_tests_properties(momoten_driver_simple PROPERTIES
        LABELS "momoten;runtime;requires-opencl-multilane-barrier"
        SKIP_RETURN_CODE 77)

    add_test(NAME momoten_program_cache
        COMMAND "${CMAKE_COMMAND}"
            -DDRIVER_TEST=$<TARGET_FILE:momoten_driver_test>
            -DSPV_FILE=${momoten_test_spv}
            -DCACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/program-cache
            -P "${CMAKE_CURRENT_SOURCE_DIR}/runtime/cache_test.cmake")
    set_tests_properties(momoten_program_cache PROPERTIES
        LABELS "momoten;runtime;cache;requires-opencl-multilane-barrier"
        SKIP_REGULAR_EXPRESSION "MOMOTEN_TEST_SKIP: no compatible OpenCL device for program cache test")
endif()

add_test(NAME momoten_source_architecture
    COMMAND "${CMAKE_COMMAND}"
        -DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/..
        -P "${CMAKE_CURRENT_SOURCE_DIR}/api/source_architecture_test.cmake")
set_tests_properties(momoten_source_architecture PROPERTIES
    LABELS "momoten;architecture")

if(TARGET momoten_amalgamated_driver)
    add_executable(momoten_amalgamated_proc_registry_test
        api/proc_registry_test.cpp)
    target_compile_features(momoten_amalgamated_proc_registry_test PRIVATE cxx_std_11)
    target_link_libraries(momoten_amalgamated_proc_registry_test PRIVATE
        momoten_amalgamated_driver momoten_vulkan_headers ${CMAKE_DL_LIBS})
    add_test(NAME momoten_amalgamated_proc_registry
        COMMAND momoten_amalgamated_proc_registry_test
            $<TARGET_FILE:momoten_amalgamated_driver>)
    set_tests_properties(momoten_amalgamated_proc_registry PROPERTIES
        LABELS "momoten;amalgamation;api;proc"
        SKIP_RETURN_CODE 77)

    add_executable(momoten_amalgamated_profile_test api/profile_test.cpp)
    target_compile_features(momoten_amalgamated_profile_test PRIVATE cxx_std_11)
    target_link_libraries(momoten_amalgamated_profile_test PRIVATE
        momoten_amalgamated_driver momoten_vulkan_headers)
    add_test(NAME momoten_amalgamated_api_profile
        COMMAND momoten_amalgamated_profile_test)
    set_tests_properties(momoten_amalgamated_api_profile PROPERTIES
        LABELS "momoten;amalgamation;api;profile"
        SKIP_RETURN_CODE 77)

    add_executable(momoten_amalgamated_driver_test runtime/driver_test.cpp)
    target_compile_features(momoten_amalgamated_driver_test PRIVATE cxx_std_11)
    target_compile_definitions(momoten_amalgamated_driver_test PRIVATE CL_TARGET_OPENCL_VERSION=100)
    target_include_directories(momoten_amalgamated_driver_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")
    target_link_libraries(momoten_amalgamated_driver_test PRIVATE
        momoten_amalgamated_driver momoten::headers momoten_vulkan_headers momoten_opencl_headers)
    add_dependencies(momoten_amalgamated_driver_test momoten-test-spv)
    add_test(NAME momoten_amalgamated_driver_simple
        COMMAND momoten_amalgamated_driver_test "${momoten_test_spv}"
            "${momoten_subgroup_basic_test_spv}"
            "${momoten_integer_dot_product_test_spv}"
            "${momoten_subgroup_values_test_spv}")
    set_tests_properties(momoten_amalgamated_driver_simple PROPERTIES
        LABELS "momoten;amalgamation;runtime;requires-opencl-multilane-barrier"
        SKIP_RETURN_CODE 77)
endif()
