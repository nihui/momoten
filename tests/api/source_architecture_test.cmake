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
foreach(backend_part memory_sync replay runtime pipeline_build)
    if(NOT manifest MATCHES
       "driver[ \t]+src/backend/${backend_part}\\.cpp")
        message(FATAL_ERROR
            "backend/${backend_part}.cpp is absent from source_manifest.txt")
    endif()
endforeach()

if(EXISTS "${SOURCE_ROOT}/src/backend/objects.cpp")
    message(FATAL_ERROR
        "generic objects.cpp must not mix unrelated object lifetimes")
endif()

file(READ "${SOURCE_ROOT}/src/backend/runtime.cpp" runtime)
if(runtime MATCHES "replay_(copy|dispatch|barrier)|range_map|Enqueue(Read|Write)Buffer")
    message(FATAL_ERROR
        "runtime.cpp contains command replay or memory-coherence implementation")
endif()

file(READ "${SOURCE_ROOT}/src/backend/objects.h" objects)
if(objects MATCHES "struct RecordedCommand")
    message(FATAL_ERROR "objects.h contains command-stream definitions")
endif()

file(READ "${SOURCE_ROOT}/src/backend/device_profile.cpp" device_profile)
if(device_profile MATCHES "probe_int64|momo_int64_probe")
    message(FATAL_ERROR
        "shaderInt64 discovery must use OpenCL capability declarations, not compiler probes")
endif()
foreach(int64_declaration CL_DEVICE_PROFILE FULL_PROFILE EMBEDDED_PROFILE cles_khr_int64 __opencl_c_int64)
    if(NOT device_profile MATCHES "${int64_declaration}")
        message(FATAL_ERROR
            "shaderInt64 discovery is missing ${int64_declaration}")
    endif()
endforeach()
