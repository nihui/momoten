// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "program.h"
#include "cache.h"
#include "objects.h"

#include <cstdio>

namespace momoten_detail {

void print_program_build_log(cl_program program, cl_device_id device)
{
    size_t size = 0;
    if (g_opencl.p_clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG, 0, 0, &size)
            != CL_SUCCESS
        || size == 0)
        return;
    std::vector<char> log(size);
    if (g_opencl.p_clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), 0)
        == CL_SUCCESS)
        fprintf(stderr, "[momoten] OpenCL build log:\n%s\n", log.data());
}

static cl_program find_memory_cached_program(
    VkDevice device, const ProgramCacheKey& key)
{
    const std::pair<uint64_t, uint64_t> memory_key(key.first, key.second);
    std::lock_guard<std::mutex> lock(device->program_cache_mutex);
    const std::map<std::pair<uint64_t, uint64_t>, cl_program>::const_iterator it = device->program_cache.find(memory_key);
    if (it == device->program_cache.end())
        return 0;
    if (g_opencl.p_clRetainProgram(it->second) != CL_SUCCESS)
        return 0;
    return it->second;
}

static cl_program install_memory_cached_program(
    VkDevice device, const ProgramCacheKey& key, cl_program program)
{
    const std::pair<uint64_t, uint64_t> memory_key(key.first, key.second);
    std::lock_guard<std::mutex> lock(device->program_cache_mutex);
    const std::map<std::pair<uint64_t, uint64_t>, cl_program>::const_iterator existing = device->program_cache.find(memory_key);
    if (existing != device->program_cache.end())
    {
        if (g_opencl.p_clRetainProgram(existing->second) == CL_SUCCESS)
        {
            g_opencl.p_clReleaseProgram(program);
            return existing->second;
        }
        return program;
    }

    // Layer tests and dynamic-shape models may create many specialized
    // pipelines over one device lifetime. Rusticl programs can retain tens of
    // megabytes of JIT state each, so keep this cache deliberately bounded.
    // The on-disk binary cache remains available after memory eviction.
    if (device->program_cache_capacity == 0)
        return program;
    while (device->program_cache.size() >= device->program_cache_capacity)
    {
        std::map<std::pair<uint64_t, uint64_t>, cl_program>::iterator evicted = device->program_cache.begin();
        g_opencl.p_clReleaseProgram(evicted->second);
        device->program_cache.erase(evicted);
    }

    if (g_opencl.p_clRetainProgram(program) == CL_SUCCESS)
        device->program_cache[memory_key] = program;
    return program;
}

cl_program create_and_build_program(
    VkDevice device, const std::string& source, cl_int& result)
{
    const ProgramCacheKey key = make_program_cache_key(
        device->physical_device->device, source);
    cl_program memory_cached = find_memory_cached_program(device, key);
    if (memory_cached)
    {
        result = CL_SUCCESS;
        if (debug_enabled())
            fprintf(stderr, "[momoten] OpenCL in-memory program cache hit\n");
        return memory_cached;
    }

    std::vector<unsigned char> binary;
    if (read_program_binary(key, binary))
    {
        const size_t binary_size = binary.size();
        const unsigned char* binary_data = binary.data();
        cl_int binary_status = CL_SUCCESS;
        cl_program program = g_opencl.p_clCreateProgramWithBinary(
            device->context, 1, &device->physical_device->device, &binary_size,
            &binary_data, &binary_status, &result);
        if (program && result == CL_SUCCESS && binary_status == CL_SUCCESS)
        {
            result = g_opencl.p_clBuildProgram(
                program, 1, &device->physical_device->device, 0, 0, 0);
            if (result == CL_SUCCESS)
            {
                if (debug_enabled())
                    fprintf(stderr, "[momoten] OpenCL program cache hit %s\n",
                            key.path.c_str());
                return install_memory_cached_program(device, key, program);
            }
            g_opencl.p_clReleaseProgram(program);
        }
        else if (program)
        {
            g_opencl.p_clReleaseProgram(program);
        }
        if (debug_enabled())
            fprintf(stderr, "[momoten] ignoring invalid OpenCL program cache entry %s\n",
                    key.path.c_str());
        std::remove(key.path.c_str());
    }

    const char* source_data = source.c_str();
    const size_t source_size = source.size();
    cl_program program = g_opencl.p_clCreateProgramWithSource(
        device->context, 1, &source_data, &source_size, &result);
    if (!program)
        return 0;
    result = g_opencl.p_clBuildProgram(
        program, 1, &device->physical_device->device, 0, 0, 0);
    if (result != CL_SUCCESS)
        return program;

    if (!key.path.empty())
    {
        size_t binary_size = 0;
        if (g_opencl.p_clGetProgramInfo(
                program, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size),
                &binary_size, 0)
                == CL_SUCCESS
            && binary_size != 0)
        {
            binary.resize(binary_size);
            unsigned char* binary_data = binary.data();
            if (g_opencl.p_clGetProgramInfo(
                    program, CL_PROGRAM_BINARIES, sizeof(binary_data),
                    &binary_data, 0)
                == CL_SUCCESS)
            {
                write_program_binary(key, binary);
                if (debug_enabled())
                    fprintf(stderr, "[momoten] OpenCL program cache miss %s\n",
                            key.path.c_str());
            }
        }
    }
    return install_memory_cached_program(device, key, program);
}

} // namespace momoten_detail
