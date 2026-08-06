# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

if(NOT DRIVER OR NOT MODEL_TEST OR NOT CACHE_DIR)
    message(FATAL_ERROR "DRIVER, MODEL_TEST and CACHE_DIR are required")
endif()

# The target is a dedicated build-tree test directory, never user cache data.
file(REMOVE_RECURSE "${CACHE_DIR}")
file(MAKE_DIRECTORY "${CACHE_DIR}")

function(run_model result_var stdout_var stderr_var)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "NCNN_VULKAN_DRIVER=${DRIVER}"
            "MOMOTEN_CACHE=${CACHE_DIR}"
            "MOMOTEN_DEBUG=1"
            "${MODEL_TEST}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    set(${result_var} "${result}" PARENT_SCOPE)
    set(${stdout_var} "${stdout}" PARENT_SCOPE)
    set(${stderr_var} "${stderr}" PARENT_SCOPE)
endfunction()

run_model(first_result first_stdout first_stderr)
if(first_result EQUAL 77)
    message(FATAL_ERROR "OpenCL runtime unavailable")
endif()
if(NOT first_result EQUAL 0)
    message(FATAL_ERROR "model cache-miss execution failed: ${first_stdout}${first_stderr}")
endif()
if(NOT first_stderr MATCHES "OpenCL program cache miss")
    message(FATAL_ERROR "first model execution did not report cache misses: ${first_stderr}")
endif()

file(GLOB cache_files "${CACHE_DIR}/momo-*.bin")
list(LENGTH cache_files cache_file_count)
if(cache_file_count LESS 2)
    message(FATAL_ERROR "expected multiple model shader binaries, found ${cache_file_count}")
endif()

run_model(second_result second_stdout second_stderr)
if(NOT second_result EQUAL 0)
    message(FATAL_ERROR "model cache-hit execution failed: ${second_stdout}${second_stderr}")
endif()
if(NOT second_stderr MATCHES "OpenCL program cache hit")
    message(FATAL_ERROR "second model execution did not report cache hits: ${second_stderr}")
endif()
if(second_stderr MATCHES "OpenCL program cache miss")
    message(FATAL_ERROR "second model execution unexpectedly recompiled a shader: ${second_stderr}")
endif()
