# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

if(NOT TRANSLATOR OR NOT CLANG OR NOT SPV_FILE OR NOT ATOMIC_SPV_FILE OR NOT FP16_SPV_FILE OR NOT FP64_SPV_FILE OR NOT FP16_CONSTANT_SPV_FILE OR NOT OUTPUT_DIR)
    message(FATAL_ERROR "TRANSLATOR, CLANG, SPV_FILE, ATOMIC_SPV_FILE, FP16_SPV_FILE, FP64_SPV_FILE, FP16_CONSTANT_SPV_FILE and OUTPUT_DIR are required")
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")

set(compile_options)
if(APPLE_TYPE_ALIASES)
    # Apple's cl_kernel.h exposes vector types through object-like macros.
    # Rename Clang's actual typedefs before its builtin header is parsed, so
    # no public typedef remains to hide a translator that destroys an alias.
    foreach(base char uchar short ushort int uint long ulong float half double)
        foreach(width 2 3 4 8 16)
            list(APPEND compile_options
                "-D${base}${width}=momo_apple_${base}${width}")
        endforeach()
    endforeach()
    list(APPEND compile_options -Werror=macro-redefined)
endif()

function(check_opencl_c_1_0 name input)
    set(source "${OUTPUT_DIR}/${name}.cl")
    execute_process(
        COMMAND "${TRANSLATOR}" ${ARGN} "${input}"
        RESULT_VARIABLE translate_result
        OUTPUT_FILE "${source}"
        ERROR_VARIABLE translate_log)
    if(NOT translate_result EQUAL 0)
        message(FATAL_ERROR "${name} translation failed: ${translate_log}")
    endif()

    if(APPLE_TYPE_ALIASES)
        # Cover every emitted vector width using native OpenCL constructors,
        # including scalar splats and mixed
        # vector/scalar constructors, as well as ordinary type declarations.
        set(bases char uchar short ushort int uint long ulong float)
        if(name MATCHES "^fp16")
            list(APPEND bases half)
        elseif(name MATCHES "^fp64")
            list(APPEND bases double)
        endif()
        file(APPEND "${source}" "\nvoid momo_test_type_aliases(void)\n{\n")
        foreach(base IN LISTS bases)
            file(APPEND "${source}"
                "    ${base} scalar_${base} = (${base})(1);\n"
                "    ${base}2 pair_${base} = (${base}2)(scalar_${base});\n"
                "    ${base}3 triple_${base} = (${base}3)(pair_${base}, scalar_${base});\n"
                "    ${base}4 quad_${base} = (${base}4)(pair_${base}, pair_${base});\n"
                "    ${base}8 octet_${base} = (${base}8)(quad_${base}, quad_${base});\n"
                "    ${base}16 full_${base} = (${base}16)(octet_${base}, octet_${base});\n"
                "    ${base}16 splat_${base} = (${base}16)(scalar_${base});\n")
        endforeach()
        file(APPEND "${source}" "}\n")
    endif()

    # Syntax validation must not inherit the host CPU's OpenCL capabilities.
    # SPIR exposes optional extensions such as fp16/fp64 without requiring
    # hardware support; the separate runtime tests check actual device support.
    execute_process(
        COMMAND "${CLANG}" -target spir64 -x cl -cl-std=CL1.0 -fsyntax-only
            ${compile_options} "${source}"
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
check_opencl_c_1_0(fp64_buffer "${FP64_SPV_FILE}")
check_opencl_c_1_0(fp16_program_constant "${FP16_CONSTANT_SPV_FILE}")
if(VECTOR_EXTRACT_SPV_FILE)
    check_opencl_c_1_0(vector_extract_dynamic "${VECTOR_EXTRACT_SPV_FILE}")
endif()
if(VECTOR_EXTRACT_FP64_SPV_FILE)
    check_opencl_c_1_0(fp64_vector_extract "${VECTOR_EXTRACT_FP64_SPV_FILE}")
endif()
if(VECTOR_EXTRACT_INT16_SPV_FILE)
    check_opencl_c_1_0(vector_extract_int16 "${VECTOR_EXTRACT_INT16_SPV_FILE}")
endif()
if(POW_CONSTANT_BASE_SPV_FILE)
    check_opencl_c_1_0(pow_constant_base "${POW_CONSTANT_BASE_SPV_FILE}")
endif()

if(INT8_STORAGE_SPV_FILE)
    check_opencl_c_1_0(int8_storage "${INT8_STORAGE_SPV_FILE}" --byte-addressable-store)
    check_opencl_c_1_0(int8_vector_storage "${INT8_VECTOR_STORAGE_SPV_FILE}" --byte-addressable-store)
    check_opencl_c_1_0(int16_arithmetic "${INT16_ARITHMETIC_SPV_FILE}" --byte-addressable-store)
    check_opencl_c_1_0(fp16_storage "${STORAGE16_SPV_FILE}" --byte-addressable-store)
    check_opencl_c_1_0(storage16_emulated "${STORAGE16_SPV_FILE}" --byte-addressable-store --no-native-fp16)
    set(conversion_source "${OUTPUT_DIR}/integer-conversion.cl")
    execute_process(COMMAND "${CONVERSION_TRANSLATOR}" --conversion-output "${conversion_source}"
        RESULT_VARIABLE conversion_result)
    if(NOT conversion_result EQUAL 0)
        message(FATAL_ERROR "integer conversion fixture translation failed")
    endif()
    execute_process(COMMAND "${CLANG}" -target spir64 -x cl -cl-std=CL1.0 -fsyntax-only
        ${compile_options} "${conversion_source}" RESULT_VARIABLE conversion_result
        OUTPUT_VARIABLE conversion_stdout ERROR_VARIABLE conversion_stderr)
    if(NOT conversion_result EQUAL 0)
        message(FATAL_ERROR "integer conversion is not valid OpenCL C 1.0: ${conversion_stdout}${conversion_stderr}")
    endif()
endif()
