# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

add_executable(momoten_memory_state_test
    runtime/memory_state_test.cpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/memory_state.cpp")
target_compile_features(momoten_memory_state_test PRIVATE cxx_std_11)
target_link_libraries(momoten_memory_state_test PRIVATE
    momoten_vulkan_headers)
add_test(NAME momoten_memory_state COMMAND momoten_memory_state_test)
set_tests_properties(momoten_memory_state PROPERTIES
    LABELS "momoten;runtime;memory")

add_executable(momoten_parameter_size_test
    api/parameter_size_test.cpp
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

function(momoten_add_float_controls_test target_name translator_target test_name)
    add_executable(${target_name}
        translator/float_controls_test.cpp
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
momoten_compile_test_shader(momoten_fp16_program_constant_test_spv fp16_program_constant.comp vulkan1.1)
momoten_compile_test_shader(momoten_fp64_test_spv fp64_buffer.comp vulkan1.1)
momoten_compile_test_shader(momoten_int16_test_spv int16_buffer.comp vulkan1.1)
momoten_compile_test_shader(momoten_scalar_16bit_test_spv scalar_16bit_buffer.comp vulkan1.1)
momoten_compile_test_shader(momoten_subgroup_basic_test_spv subgroup_basic.comp vulkan1.1)
momoten_compile_test_shader(momoten_integer_dot_product_test_spv integer_dot_product.comp vulkan1.1)
momoten_compile_test_shader(momoten_workgroup_split_test_spv workgroup_split.comp vulkan1.1)

add_custom_target(momoten-test-spv DEPENDS ${momoten_test_spv_files})

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
        "${momoten_fp16_program_constant_test_spv}")
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
            "${momoten_fp16_program_constant_test_spv}")
    set_tests_properties(momoten_amalgamated_translator_simple PROPERTIES
        LABELS "momoten;amalgamation;translator")
endif()

if(DEFINED NCNN_MOMOTEN_CLANG AND NOT DEFINED MOMOTEN_CLANG)
    set(MOMOTEN_CLANG "${NCNN_MOMOTEN_CLANG}" CACHE FILEPATH
        "Path to clang used by momoten syntax tests")
endif()
find_program(MOMOTEN_CLANG clang)
if(MOMOTEN_CLANG AND TARGET momoten_spv_to_clc)
    add_test(NAME momoten_opencl_c_1_0_syntax
        COMMAND "${CMAKE_COMMAND}"
            -DTRANSLATOR=$<TARGET_FILE:momoten_spv_to_clc>
            -DCLANG=${MOMOTEN_CLANG}
            -DSPV_FILE=${momoten_test_spv}
            -DATOMIC_SPV_FILE=${momoten_atomic_test_spv}
            -DFP16_SPV_FILE=${momoten_fp16_test_spv}
            -DFP64_SPV_FILE=${momoten_fp64_test_spv}
            -DFP16_CONSTANT_SPV_FILE=${momoten_fp16_program_constant_test_spv}
            -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/strict-cl10
            -P "${CMAKE_CURRENT_SOURCE_DIR}/translator/strict_cl10_test.cmake")
    set_tests_properties(momoten_opencl_c_1_0_syntax PROPERTIES
        LABELS "momoten;translator;opencl-1.0")
endif()

if(TARGET momoten_test_opencl)
    add_executable(momoten_opencl_test runtime/opencl_test.cpp)
    target_compile_features(momoten_opencl_test PRIVATE cxx_std_11)
    target_compile_definitions(momoten_opencl_test PRIVATE CL_TARGET_OPENCL_VERSION=100)
    target_link_libraries(momoten_opencl_test PRIVATE
        momoten::translator momoten_test_opencl)
    add_dependencies(momoten_opencl_test momoten-test-spv)
    add_test(NAME momoten_opencl_simple
        COMMAND momoten_opencl_test "${momoten_test_spv}")
    add_test(NAME momoten_opencl_fp16
        COMMAND momoten_opencl_test "${momoten_fp16_test_spv}" fp16)
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
    set_tests_properties(momoten_opencl_simple momoten_opencl_fp16
        momoten_opencl_fp64
        momoten_opencl_int16
        momoten_opencl_atomic_packed momoten_opencl_scalar_16bit
        momoten_opencl_subgroup_basic momoten_opencl_integer_dot_product
        momoten_opencl_workgroup_split PROPERTIES
        LABELS "momoten;runtime;direct-opencl"
        SKIP_RETURN_CODE 77)
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
    target_link_libraries(momoten_driver_test PRIVATE
        momoten::driver momoten_vulkan_headers)
    add_dependencies(momoten_driver_test momoten-test-spv)

    add_test(NAME momoten_driver_simple
        COMMAND momoten_driver_test "${momoten_test_spv}"
            "${momoten_subgroup_basic_test_spv}"
            "${momoten_integer_dot_product_test_spv}")
    set_tests_properties(momoten_driver_simple PROPERTIES
        LABELS "momoten;runtime"
        SKIP_RETURN_CODE 77)

    add_test(NAME momoten_program_cache
        COMMAND "${CMAKE_COMMAND}"
            -DDRIVER_TEST=$<TARGET_FILE:momoten_driver_test>
            -DSPV_FILE=${momoten_test_spv}
            -DCACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/program-cache
            -P "${CMAKE_CURRENT_SOURCE_DIR}/runtime/cache_test.cmake")
    set_tests_properties(momoten_program_cache PROPERTIES
        LABELS "momoten;runtime;cache"
        SKIP_RETURN_CODE 77)
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
    target_link_libraries(momoten_amalgamated_driver_test PRIVATE
        momoten_amalgamated_driver momoten_vulkan_headers)
    add_dependencies(momoten_amalgamated_driver_test momoten-test-spv)
    add_test(NAME momoten_amalgamated_driver_simple
        COMMAND momoten_amalgamated_driver_test "${momoten_test_spv}"
            "${momoten_subgroup_basic_test_spv}"
            "${momoten_integer_dot_product_test_spv}")
    set_tests_properties(momoten_amalgamated_driver_simple PROPERTIES
        LABELS "momoten;amalgamation;runtime"
        SKIP_RETURN_CODE 77)
endif()
