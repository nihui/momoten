# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
execute_process(COMMAND "${TEST}" "${SPV}" "${FP16_SPV}" "${FP64_SPV}" "${OUTPUT_DIR}"
    RESULT_VARIABLE result ERROR_VARIABLE log)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Memory lowering translation failed: ${log}")
endif()

file(GLOB sources "${OUTPUT_DIR}/*.cl")
foreach(source IN LISTS sources)
    # Test the no-extension baseline independently of host/device features.
    # The half fixtures explicitly opt into byte-addressable stores.
    set(options)
    if(NOT source MATCHES "half_")
        list(APPEND options -Ucl_khr_byte_addressable_store)
    endif()
    execute_process(COMMAND "${CLANG}" -target spir64 -x cl -cl-std=CL1.0
        -fsyntax-only ${options} "${source}"
        RESULT_VARIABLE result ERROR_VARIABLE log)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${source} failed OpenCL C 1.0 compilation: ${log}")
    endif()
endforeach()
