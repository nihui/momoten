# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

if(NOT DRIVER_TEST OR NOT SPV_FILE OR NOT CACHE_DIR)
    message(FATAL_ERROR "DRIVER_TEST, SPV_FILE and CACHE_DIR are required")
endif()

# The target is a dedicated build-tree test directory, never user cache data.
file(REMOVE_RECURSE "${CACHE_DIR}")
file(MAKE_DIRECTORY "${CACHE_DIR}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "MOMOTEN_CACHE=${CACHE_DIR}"
        "MOMOTEN_DEBUG=1"
        "${DRIVER_TEST}" "${SPV_FILE}"
    RESULT_VARIABLE first_result
    OUTPUT_VARIABLE first_stdout
    ERROR_VARIABLE first_stderr)
if(first_result EQUAL 77)
    message(FATAL_ERROR "OpenCL runtime unavailable")
endif()
if(NOT first_result EQUAL 0)
    message(FATAL_ERROR "cache-miss execution failed: ${first_stdout}${first_stderr}")
endif()
if(NOT first_stderr MATCHES "OpenCL program cache miss")
    message(FATAL_ERROR "first execution did not report a cache miss: ${first_stderr}")
endif()
if(NOT first_stderr MATCHES "OpenCL in-memory program cache hit")
    message(FATAL_ERROR "duplicate pipeline did not reuse the in-memory program: ${first_stderr}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "MOMOTEN_CACHE=${CACHE_DIR}"
        "MOMOTEN_DEBUG=1"
        "${DRIVER_TEST}" "${SPV_FILE}"
    RESULT_VARIABLE second_result
    OUTPUT_VARIABLE second_stdout
    ERROR_VARIABLE second_stderr)
if(NOT second_result EQUAL 0)
    message(FATAL_ERROR "cache-hit execution failed: ${second_stdout}${second_stderr}")
endif()
if(NOT second_stderr MATCHES "OpenCL program cache hit")
    message(FATAL_ERROR "second execution did not report a cache hit: ${second_stderr}")
endif()

file(GLOB cache_files "${CACHE_DIR}/momo-*.bin")
list(LENGTH cache_files cache_file_count)
if(NOT cache_file_count EQUAL 1)
    message(FATAL_ERROR "expected exactly one cache binary, found ${cache_file_count}")
endif()
list(GET cache_files 0 cache_file)
file(WRITE "${cache_file}" "corrupt")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "MOMOTEN_CACHE=${CACHE_DIR}"
        "MOMOTEN_DEBUG=1"
        "${DRIVER_TEST}" "${SPV_FILE}"
    RESULT_VARIABLE corrupt_result
    OUTPUT_VARIABLE corrupt_stdout
    ERROR_VARIABLE corrupt_stderr)
if(NOT corrupt_result EQUAL 0)
    message(FATAL_ERROR "corrupt-cache fallback failed: ${corrupt_stdout}${corrupt_stderr}")
endif()
if(NOT corrupt_stderr MATCHES "OpenCL program cache miss")
    message(FATAL_ERROR "corrupt cache did not fall back to source compilation: ${corrupt_stderr}")
endif()
