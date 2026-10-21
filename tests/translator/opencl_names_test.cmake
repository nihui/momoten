# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

if(NOT TRANSLATOR OR NOT CLANG OR NOT SPV_FILE OR NOT OUTPUT_DIR OR NOT GLSLANG OR NOT COLLISION_SHADER OR NOT FUNCTION_SHADER)
    message(FATAL_ERROR "TRANSLATOR, CLANG, SPV_FILE, OUTPUT_DIR, GLSLANG, COLLISION_SHADER and FUNCTION_SHADER are required")
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
function(check_opencl_names spv_file name)
    set(source "${OUTPUT_DIR}/${name}.cl")
    execute_process(
        COMMAND "${TRANSLATOR}" "${spv_file}"
        RESULT_VARIABLE translate_result
        OUTPUT_FILE "${source}"
        ERROR_VARIABLE translate_log)
    if(NOT translate_result EQUAL 0)
        message(FATAL_ERROR "OpenCL name translation failed: ${translate_log}")
    endif()

    if(ARGC GREATER 2 AND ARGV2)
        file(READ "${source}" source_text)
        string(REGEX MATCH "\nuint ([A-Za-z_][A-Za-z_0-9]*)\\(" declaration "${source_text}")
        if(NOT declaration)
            message(FATAL_ERROR "Name fixture ${name} did not retain its helper declaration")
        endif()
        set(helper_name "${CMAKE_MATCH_1}")
        string(REGEX MATCHALL "${helper_name}\\(" helper_references "${source_text}")
        list(LENGTH helper_references reference_count)
        if(reference_count LESS 2)
            message(FATAL_ERROR "Name fixture ${name} did not retain its helper call")
        endif()
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
                "Generated ${name} names are not valid in ${standard}:\n${compile_stdout}${compile_stderr}")
        endif()
    endforeach()
endfunction()

check_opencl_names("${SPV_FILE}" opencl_names)

# These builtins include compiler-only declarations absent from opencl-c.h.
# Assert that the helper survives optimization before checking all C profiles.
foreach(name to_global to_local to_private ctz enqueue_marker ordinary_helper)
    set(spv_file "${OUTPUT_DIR}/function_${name}.spv")
    execute_process(
        COMMAND "${GLSLANG}" -V --target-env vulkan1.1 "-DFUNCTION_NAME=${name}"
            -o "${spv_file}" "${FUNCTION_SHADER}"
        RESULT_VARIABLE result OUTPUT_VARIABLE log ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Function name fixture ${name} failed: ${log}${error}")
    endif()
    check_opencl_names("${spv_file}" "function_${name}" TRUE)
endforeach()

function(check_name_fixture name block buffer member)
    set(spv_file "${OUTPUT_DIR}/${name}.spv")
    execute_process(
        COMMAND "${GLSLANG}" -V --target-env vulkan1.1
            "-DBLOCK_NAME=${block}" "-DBUFFER_NAME=${buffer}" "-DMEMBER_NAME=${member}"
            -o "${spv_file}" "${COLLISION_SHADER}"
        RESULT_VARIABLE result OUTPUT_VARIABLE log ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Name fixture ${name} failed: ${log}${error}")
    endif()
    check_opencl_names("${spv_file}" "${name}")
endfunction()

# Exercise type names independently: GLSL allows only one push-constant block.
# Include collisions after renaming, CL2 types, atomic enumerators and macros.
foreach(name _global kernel_enqueue_flags_t clk_profiling_info
        memory_scope_work_item memory_scope_sub_group memory_scope_work_group
        memory_scope_device memory_scope_all_svm_devices memory_scope_all_devices
        memory_order_relaxed memory_order_acquire memory_order_release
        memory_order_acq_rel memory_order_seq_cst
        CLK_ENQUEUE_FLAGS_NO_WAIT CLK_NULL_EVENT CLK_NULL_QUEUE
        CLK_SUCCESS CLK_PROFILING_COMMAND_EXEC_TIME CL_COMPLETE
        ATOMIC_FLAG_INIT ATOMIC_VAR_INIT MAX_WORK_DIM NULL
        cl_khr_subgroups cl_khr_integer_dot_product)
    check_name_fixture("${name}" "${name}" global x)
endforeach()

# Resource macros must not capture struct members or builtin calls.
foreach(name _global CLK_SUCCESS CLK_ENQUEUE_FLAGS_WAIT_KERNEL CLK_LOCAL_MEM_FENCE
        MAX_WORK_DIM NULL cl_khr_subgroups cl_khr_integer_dot_product)
    check_name_fixture("member_${name}" Params global "${name}")
endforeach()
foreach(name get_global_id sub_group_elect MAX_WORK_DIM NULL cl_khr_subgroups cl_khr_integer_dot_product)
    check_name_fixture("resource_${name}" Params "${name}" x)
endforeach()
