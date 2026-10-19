# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

if(NOT EXISTS "${DRIVER}")
    message(FATAL_ERROR "momoten driver does not exist: ${DRIVER}")
endif()

if(APPLE)
    # GET_RUNTIME_DEPENDENCIES omits system libraries that exist only in
    # dyld's shared cache, including OpenCL.framework on recent macOS.
    # Inspect recorded load commands without resolving files on disk.
    find_program(MOMOTEN_OTOOL NAMES otool REQUIRED)
    execute_process(
        COMMAND "${MOMOTEN_OTOOL}" -L "${DRIVER}"
        RESULT_VARIABLE otool_result
        OUTPUT_VARIABLE otool_output
        ERROR_VARIABLE otool_error)
    if(NOT otool_result EQUAL 0)
        message(FATAL_ERROR "Inspecting momoten dependencies failed: ${otool_error}")
    endif()
    string(REPLACE "\n" ";" dependency_lines "${otool_output}")
    foreach(line IN LISTS dependency_lines)
        # Keep paths containing spaces intact and omit otool's version suffix.
        if(line MATCHES "^[ \t]+(.+)[ \t]+\\(compatibility version ")
            string(STRIP "${CMAKE_MATCH_1}" dependency)
            list(APPEND resolved_dependencies "${dependency}")
        endif()
    endforeach()
else()
    file(GET_RUNTIME_DEPENDENCIES
        LIBRARIES "${DRIVER}"
        RESOLVED_DEPENDENCIES_VAR resolved_dependencies
        UNRESOLVED_DEPENDENCIES_VAR unresolved_dependencies)
endif()

foreach(dependency IN LISTS resolved_dependencies unresolved_dependencies)
    get_filename_component(name "${dependency}" NAME)
    string(TOLOWER "${name}" name_lower)
    if(name_lower MATCHES "(^|lib)opencl([.-]|$)")
        message(FATAL_ERROR
            "libmomoten must self-load an OpenCL vendor and may not depend on ${dependency}")
    endif()
endforeach()
