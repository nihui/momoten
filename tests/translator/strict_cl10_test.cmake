# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

if(NOT TRANSLATOR OR NOT CLANG OR NOT SPV_FILE OR NOT ATOMIC_SPV_FILE OR NOT FP16_SPV_FILE OR NOT OUTPUT_DIR)
    message(FATAL_ERROR "TRANSLATOR, CLANG, SPV_FILE, ATOMIC_SPV_FILE, FP16_SPV_FILE and OUTPUT_DIR are required")
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")

function(check_opencl_c_1_0 name input)
    set(source "${OUTPUT_DIR}/${name}.cl")
    execute_process(
        COMMAND "${TRANSLATOR}" "${input}"
        RESULT_VARIABLE translate_result
        OUTPUT_FILE "${source}"
        ERROR_VARIABLE translate_log)
    if(NOT translate_result EQUAL 0)
        message(FATAL_ERROR "${name} translation failed: ${translate_log}")
    endif()

    execute_process(
        COMMAND "${CLANG}" -x cl -cl-std=CL1.0 -fsyntax-only "${source}"
        RESULT_VARIABLE compile_result
        OUTPUT_VARIABLE compile_stdout
        ERROR_VARIABLE compile_stderr)
    if(NOT compile_result EQUAL 0)
        message(FATAL_ERROR
            "${name} is not valid OpenCL C 1.0:\n${compile_stdout}${compile_stderr}\n"
            "translator log:\n${translate_log}")
    endif()
endfunction()

check_opencl_c_1_0(simple_buffer "${SPV_FILE}")
check_opencl_c_1_0(atomic_compare_exchange "${ATOMIC_SPV_FILE}")
check_opencl_c_1_0(fp16_buffer "${FP16_SPV_FILE}")
