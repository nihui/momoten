# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

if(NOT TRANSLATOR OR NOT CLANG OR NOT SPV_FILE OR NOT OUTPUT_DIR)
    message(FATAL_ERROR "TRANSLATOR, CLANG, SPV_FILE and OUTPUT_DIR are required")
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(source "${OUTPUT_DIR}/opencl_names.cl")
execute_process(
    COMMAND "${TRANSLATOR}" "${SPV_FILE}"
    RESULT_VARIABLE translate_result
    OUTPUT_FILE "${source}"
    ERROR_VARIABLE translate_log)
if(NOT translate_result EQUAL 0)
    message(FATAL_ERROR "OpenCL name translation failed: ${translate_log}")
endif()

# Check the actual generated source without depending on a runtime or device.
# A common naming policy must also remain usable with legacy compilers.
foreach(standard CL1.0 CL1.1 CL1.2 CL2.0 CL3.0)
    execute_process(
        COMMAND "${CLANG}" -target spir64 -x cl "-cl-std=${standard}"
            -fsyntax-only "${source}"
        RESULT_VARIABLE compile_result
        OUTPUT_VARIABLE compile_stdout
        ERROR_VARIABLE compile_stderr)
    if(NOT compile_result EQUAL 0)
        message(FATAL_ERROR
            "Generated names are not valid in ${standard}:\n${compile_stdout}${compile_stderr}")
    endif()
endforeach()
