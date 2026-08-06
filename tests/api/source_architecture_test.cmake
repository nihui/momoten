# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

file(READ "${SOURCE_ROOT}/src/vulkan_internal.h" vulkan_internal)
if(vulkan_internal MATCHES "backend/objects.h|opencl_loader.h|ProgramCacheKey|PendingSubmission")
    message(FATAL_ERROR "vulkan_internal.h contains backend state")
endif()

file(GLOB translator_sources
    "${SOURCE_ROOT}/src/translator/*.h"
    "${SOURCE_ROOT}/src/translator/*.cpp")
foreach(source IN LISTS translator_sources)
    file(READ "${source}" contents)
    if(contents MATCHES "vulkan_internal.h|backend/")
        message(FATAL_ERROR "translator depends on Vulkan/backend internals: ${source}")
    endif()
endforeach()

foreach(frontend instance device memory pipeline command queue sync)
    file(READ "${SOURCE_ROOT}/src/${frontend}.cpp" contents)
    if(contents MATCHES "extern[ \t\r\n]+\"C\"" OR
       contents MATCHES "MOMOTEN_CATCH_RESULT")
        message(FATAL_ERROR "${frontend}.cpp bypasses impl/ABI wrapper separation")
    endif()
endforeach()

file(READ "${SOURCE_ROOT}/src/entrypoints.cpp" entrypoints)
if(NOT entrypoints MATCHES "VulkanAbiGuard" OR
   NOT entrypoints MATCHES "vulkan_procs.inc")
    message(FATAL_ERROR "entrypoints.cpp is not generated from the guarded proc registry")
endif()

file(READ "${SOURCE_ROOT}/src/source_manifest.txt" manifest)
if(NOT manifest MATCHES "driver[ \t]+src/vulkan_procs.inc")
    message(FATAL_ERROR "vulkan_procs.inc is absent from source_manifest.txt")
endif()
