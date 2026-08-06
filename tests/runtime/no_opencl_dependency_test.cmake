# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

if(NOT EXISTS "${DRIVER}")
    message(FATAL_ERROR "momoten driver does not exist: ${DRIVER}")
endif()

file(GET_RUNTIME_DEPENDENCIES
    LIBRARIES "${DRIVER}"
    RESOLVED_DEPENDENCIES_VAR resolved_dependencies
    UNRESOLVED_DEPENDENCIES_VAR unresolved_dependencies)

foreach(dependency IN LISTS resolved_dependencies unresolved_dependencies)
    get_filename_component(name "${dependency}" NAME)
    string(TOLOWER "${name}" name_lower)
    if(name_lower MATCHES "(^|lib)opencl([.-]|$)")
        message(FATAL_ERROR
            "libmomoten must self-load an OpenCL vendor and may not depend on ${dependency}")
    endif()
endforeach()
