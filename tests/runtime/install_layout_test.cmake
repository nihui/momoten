# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

if(NOT BUILD_DIR OR NOT STAGE_DIR OR NOT RUNTIME_FILE)
    message(FATAL_ERROR "Missing runtime installation test arguments")
endif()

file(REMOVE_RECURSE "${STAGE_DIR}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}"
        --config "${TEST_CONFIG}" --prefix "${STAGE_DIR}"
    RESULT_VARIABLE install_result
    OUTPUT_VARIABLE install_output
    ERROR_VARIABLE install_error)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "Installing momoten failed:\n${install_output}\n${install_error}")
endif()

if(WIN32)
    set(runtime_path "${STAGE_DIR}/${BIN_DIR}/${RUNTIME_FILE}")
else()
    set(runtime_path "${STAGE_DIR}/${LIB_DIR}/${RUNTIME_FILE}")
endif()
if(NOT EXISTS "${runtime_path}" OR IS_SYMLINK "${runtime_path}")
    message(FATAL_ERROR "Expected unversioned runtime library: ${runtime_path}")
endif()

# Check the entire installed payload, not only known development file names.
file(GLOB_RECURSE installed_files LIST_DIRECTORIES FALSE "${STAGE_DIR}/*")
foreach(installed_file IN LISTS installed_files)
    if(installed_file STREQUAL runtime_path)
        continue()
    endif()
    if(TOOLS_ENABLED AND installed_file MATCHES "/momoten-spv-to-clc(\\.exe)?$")
        continue()
    endif()
    message(FATAL_ERROR "Unexpected development artifact: ${installed_file}")
endforeach()
