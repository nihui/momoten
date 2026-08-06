# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

# Optional ncnn integration tests. The momoten core has no dependency on this
# file; ncnn includes it explicitly after creating the momoten targets.

set(MOMOTEN_NCNN_INTEGRATION_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(ncnn_momoten_register_ncnn_tests)
    if(NOT NCNN_VULKAN OR NOT TARGET momoten::driver OR NOT TARGET ncnn)
        return()
    endif()

    if(TARGET momoten_translator AND NOT TARGET momoten_ncnn_fp16_shader_test)
        add_executable(momoten_ncnn_fp16_shader_test
            "${MOMOTEN_NCNN_INTEGRATION_DIR}/tests/fp16_shader_test.cpp")
        target_compile_features(momoten_ncnn_fp16_shader_test PRIVATE cxx_std_11)
        target_link_libraries(momoten_ncnn_fp16_shader_test PRIVATE
            ncnn momoten::translator)
        add_test(NAME momoten_ncnn_fp16_shader
            COMMAND "${CMAKE_COMMAND}" -E env
                "NCNN_VULKAN_DRIVER=$<TARGET_FILE:momoten::driver>"
                $<TARGET_FILE:momoten_ncnn_fp16_shader_test>)
        set_tests_properties(momoten_ncnn_fp16_shader PROPERTIES
            LABELS "momoten;ncnn;integration"
            SKIP_RETURN_CODE 77
            TIMEOUT 1800)
    endif()

    if(NOT TARGET momoten_ncnn_model_test)
        add_executable(momoten_ncnn_model_test
            "${MOMOTEN_NCNN_INTEGRATION_DIR}/tests/model_test.cpp")
        target_compile_features(momoten_ncnn_model_test PRIVATE cxx_std_11)
        target_link_libraries(momoten_ncnn_model_test PRIVATE ncnn)
        add_test(NAME momoten_ncnn_model
            COMMAND "${CMAKE_COMMAND}" -E env
                "NCNN_VULKAN_DRIVER=$<TARGET_FILE:momoten::driver>"
                $<TARGET_FILE:momoten_ncnn_model_test>)
        set_tests_properties(momoten_ncnn_model PROPERTIES
            LABELS "momoten;ncnn;integration"
            SKIP_RETURN_CODE 77
            TIMEOUT 1800)

        add_test(NAME momoten_ncnn_model_cache
            COMMAND "${CMAKE_COMMAND}"
                -DDRIVER=$<TARGET_FILE:momoten::driver>
                -DMODEL_TEST=$<TARGET_FILE:momoten_ncnn_model_test>
                -DCACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/momoten/ncnn-model-cache
                -P "${MOMOTEN_NCNN_INTEGRATION_DIR}/tests/model_cache_test.cmake")
        set_tests_properties(momoten_ncnn_model_cache PROPERTIES
            LABELS "momoten;ncnn;integration;cache"
            SKIP_RETURN_CODE 77
            TIMEOUT 1800)
    endif()

    if(TARGET ncnntestutil AND NOT TARGET momoten_ncnn_convolution_regression_test)
        get_target_property(momoten_ncnn_tests_source_dir ncnntestutil SOURCE_DIR)
        add_executable(momoten_ncnn_convolution_regression_test
            "${MOMOTEN_NCNN_INTEGRATION_DIR}/tests/convolution_regression_test.cpp")
        target_compile_features(momoten_ncnn_convolution_regression_test PRIVATE cxx_std_11)
        target_include_directories(momoten_ncnn_convolution_regression_test PRIVATE
            "${momoten_ncnn_tests_source_dir}")
        target_link_libraries(momoten_ncnn_convolution_regression_test PRIVATE
            ncnntestutil ncnn)
        add_test(NAME momoten_ncnn_convolution_regression
            COMMAND "${CMAKE_COMMAND}" -E env
                "NCNN_VULKAN_DRIVER=$<TARGET_FILE:momoten::driver>"
                $<TARGET_FILE:momoten_ncnn_convolution_regression_test>)
        set_tests_properties(momoten_ncnn_convolution_regression PROPERTIES
            LABELS "momoten;ncnn;integration"
            SKIP_RETURN_CODE 77
            TIMEOUT 1800)

        add_executable(momoten_ncnn_innerproduct_regression_test
            "${MOMOTEN_NCNN_INTEGRATION_DIR}/tests/innerproduct_regression_test.cpp")
        target_compile_features(momoten_ncnn_innerproduct_regression_test PRIVATE cxx_std_11)
        target_include_directories(momoten_ncnn_innerproduct_regression_test PRIVATE
            "${momoten_ncnn_tests_source_dir}")
        target_link_libraries(momoten_ncnn_innerproduct_regression_test PRIVATE
            ncnntestutil ncnn)
        add_test(NAME momoten_ncnn_innerproduct_regression
            COMMAND "${CMAKE_COMMAND}" -E env
                "NCNN_VULKAN_DRIVER=$<TARGET_FILE:momoten::driver>"
                $<TARGET_FILE:momoten_ncnn_innerproduct_regression_test>)
        set_tests_properties(momoten_ncnn_innerproduct_regression PROPERTIES
            LABELS "momoten;ncnn;integration"
            SKIP_RETURN_CODE 77
            TIMEOUT 1800)
    endif()

    if(NOT TARGET ncnntestutil)
        return()
    endif()

    get_target_property(momoten_ncnn_tests_source_dir ncnntestutil SOURCE_DIR)
    get_target_property(momoten_ncnn_tests_binary_dir ncnntestutil BINARY_DIR)
    get_filename_component(momoten_ncnn_root_dir
        "${momoten_ncnn_tests_source_dir}/.." ABSOLUTE)
    get_property(momoten_ncnn_test_targets
        DIRECTORY "${momoten_ncnn_tests_source_dir}"
        PROPERTY BUILDSYSTEM_TARGETS)
    foreach(momoten_ncnn_target IN LISTS momoten_ncnn_test_targets)
        if(NOT momoten_ncnn_target MATCHES "^test_" OR
           NOT TARGET ${momoten_ncnn_target})
            continue()
        endif()

        string(REGEX REPLACE "^test_" "" momoten_ncnn_test_suffix
            "${momoten_ncnn_target}")
        set(momoten_test_name momoten_ncnn_${momoten_ncnn_test_suffix})
        add_test(NAME ${momoten_test_name}
            COMMAND "${CMAKE_COMMAND}" -E env
                "NCNN_VULKAN_DRIVER=$<TARGET_FILE:momoten::driver>"
                "MOMOTEN_CACHE=${CMAKE_CURRENT_BINARY_DIR}/momoten/ncnn-program-cache"
                "${CMAKE_COMMAND}"
                -DTEST_EXECUTABLE=$<TARGET_FILE:${momoten_ncnn_target}>
                -P "${momoten_ncnn_root_dir}/cmake/run_test.cmake")

        set(momoten_ncnn_test_working_dir "${momoten_ncnn_tests_binary_dir}")
        if(momoten_ncnn_test_suffix STREQUAL "squeezenet")
            # test_squeezenet uses ../../examples as its native model path.
            # The ncnn sub-build directory in this superbuild is one level
            # shallower than a standalone ncnn build, so run it from an
            # existing two-level source directory where that relative path
            # still resolves to ncnn/examples.
            set(momoten_ncnn_test_working_dir
                "${momoten_ncnn_root_dir}/examples/squeezencnn")
        endif()
        set_tests_properties(${momoten_test_name} PROPERTIES
            LABELS "momoten;ncnn;integration"
            SKIP_RETURN_CODE 77
            TIMEOUT 1800
            WORKING_DIRECTORY "${momoten_ncnn_test_working_dir}")
    endforeach()
endfunction()
