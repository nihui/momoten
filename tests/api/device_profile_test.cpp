// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/device_profile.h"
#include "backend/objects.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace momoten_detail {
OpenCLApi g_opencl = {};
} // namespace momoten_detail

struct _cl_program
{
    std::string source;
};

struct _cl_mem
{
    std::vector<cl_ushort> data;
};

static bool native_build;
static bool empty_build;
static size_t subgroup_size;
static size_t preferred_multiple;
static size_t queried_local_size;
static cl_int subgroup_result;
static cl_int preferred_result;
static cl_int context_result;
static cl_int program_result;
static cl_int kernel_result;
static int live_contexts;
static int live_programs;
static int live_kernels;
static bool matching_build_options;
static std::string expected_build_options;
static uint32_t expected_language_version = 100;
static std::string device_profile;
static std::string language_version;
static bool native_requires_modern;
static bool election_build = true;
static bool barrier_build = true;
static bool complete_native_source;
static bool versions_size_invalid;
static bool versions_read_failure;
static cl_device_type device_type = CL_DEVICE_TYPE_CPU;
static int probe_failure;
static int live_queues;
static int live_buffers;
static cl_mem probe_buffers[2];

struct NameVersion
{
    cl_uint version;
    char name[64];
};
static std::vector<NameVersion> language_versions;

static cl_int write_info(const void* value, size_t size, size_t capacity, void* output, size_t* output_size)
{
    if (output_size)
        *output_size = size;
    if (output)
    {
        if (capacity < size)
            return CL_INVALID_VALUE;
        memcpy(output, value, size);
    }
    return CL_SUCCESS;
}

static cl_int CL_API_CALL get_device_info(cl_device_id, cl_device_info parameter, size_t capacity, void* output, size_t* output_size)
{
    if (parameter == CL_DEVICE_TYPE)
        return write_info(&device_type, sizeof(device_type), capacity, output, output_size);
    if (parameter == CL_DEVICE_PROFILE && !device_profile.empty())
        return write_info(device_profile.c_str(), device_profile.size() + 1, capacity, output, output_size);
    if (parameter == 0x103d && !language_version.empty()) // CL_DEVICE_OPENCL_C_VERSION
        return write_info(language_version.c_str(), language_version.size() + 1, capacity, output, output_size);
    if (parameter == 0x1066 && !language_versions.empty()) // CL_DEVICE_OPENCL_C_ALL_VERSIONS
    {
        if (versions_read_failure && output)
            return CL_INVALID_VALUE;
        const size_t size = language_versions.size() * sizeof(NameVersion) - (versions_size_invalid ? 1 : 0);
        return write_info(language_versions.data(), size, capacity, output, output_size);
    }
    return CL_INVALID_VALUE;
}

static cl_context CL_API_CALL create_context(const cl_context_properties*, cl_uint, const cl_device_id*, void(CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int* result)
{
    ++live_contexts;
    *result = context_result;
    return reinterpret_cast<cl_context>(1);
}

static cl_int CL_API_CALL release_context(cl_context)
{
    --live_contexts;
    return CL_SUCCESS;
}

static cl_program CL_API_CALL create_program(cl_context, cl_uint count, const char** sources, const size_t* sizes, cl_int* result)
{
    cl_program program = new _cl_program;
    for (cl_uint i = 0; i < count; i++)
        program->source.append(sources[i], sizes && sizes[i] ? sizes[i] : strlen(sources[i]));
    ++live_programs;
    *result = program_result;
    return program;
}

static cl_int CL_API_CALL build_program(cl_program program, cl_uint, const cl_device_id*, const char* options, void(CL_CALLBACK*)(cl_program, void*), void*)
{
    const std::string actual_options = options ? options : "";
    matching_build_options = matching_build_options && actual_options == expected_build_options;
    const bool requires_subgroups = program->source.find("get_sub_group_size") != std::string::npos;
    if (!requires_subgroups)
        return empty_build ? CL_SUCCESS : CL_BUILD_PROGRAM_FAILURE;
    const bool election = program->source.find("sub_group_elect()") != std::string::npos;
    const bool barrier = program->source.find("sub_group_barrier(") != std::string::npos;
    complete_native_source = election && barrier
                             && program->source.find("get_sub_group_local_id()") != std::string::npos
                             && program->source.find("get_sub_group_id()") != std::string::npos
                             && program->source.find("get_num_sub_groups()") != std::string::npos;
    const bool modern = actual_options == "-cl-std=CL2.0" || actual_options == "-cl-std=CL3.0";
    return native_build && (!native_requires_modern || modern) && (!election || election_build) && (!barrier || barrier_build) ? CL_SUCCESS : CL_BUILD_PROGRAM_FAILURE;
}

static cl_int CL_API_CALL release_program(cl_program program)
{
    delete program;
    --live_programs;
    return CL_SUCCESS;
}

static cl_kernel CL_API_CALL create_kernel(cl_program, const char*, cl_int* result)
{
    ++live_kernels;
    *result = kernel_result;
    return reinterpret_cast<cl_kernel>(1);
}

static cl_int CL_API_CALL release_kernel(cl_kernel)
{
    --live_kernels;
    return CL_SUCCESS;
}

static cl_int CL_API_CALL get_subgroup_info(cl_kernel, cl_device_id, cl_kernel_sub_group_info parameter, size_t input_size, const void* input, size_t capacity, void* output, size_t*)
{
    if (parameter != CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR || input_size != 3 * sizeof(size_t) || capacity != sizeof(size_t))
        return CL_INVALID_VALUE;
    const size_t* local_size = static_cast<const size_t*>(input);
    if (local_size[1] != 1 || local_size[2] != 1)
        return CL_INVALID_VALUE;
    queried_local_size = local_size[0];
    *static_cast<size_t*>(output) = subgroup_size;
    return subgroup_result;
}

static cl_int CL_API_CALL get_workgroup_info(cl_kernel, cl_device_id, cl_kernel_work_group_info parameter, size_t capacity, void* output, size_t*)
{
    if (parameter != CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE || capacity != sizeof(size_t))
        return CL_INVALID_VALUE;
    *static_cast<size_t*>(output) = preferred_multiple;
    return preferred_result;
}

static cl_command_queue CL_API_CALL create_queue(cl_context, cl_device_id, cl_command_queue_properties, cl_int* error)
{
    ++live_queues;
    *error = probe_failure == 1 ? CL_OUT_OF_RESOURCES : CL_SUCCESS;
    return reinterpret_cast<cl_command_queue>(1);
}

static cl_int CL_API_CALL release_queue(cl_command_queue)
{
    --live_queues;
    return CL_SUCCESS;
}

static cl_mem CL_API_CALL create_buffer(cl_context, cl_mem_flags, size_t bytes, void* input, cl_int* error)
{
    cl_mem buffer = new _cl_mem;
    const cl_ushort* data = static_cast<const cl_ushort*>(input);
    buffer->data.assign(data, data + bytes / sizeof(cl_ushort));
    ++live_buffers;
    *error = probe_failure == 2 || (probe_failure == 3 && live_buffers == 2) ? CL_OUT_OF_RESOURCES : CL_SUCCESS;
    return buffer;
}

static cl_int CL_API_CALL release_buffer(cl_mem buffer)
{
    delete buffer;
    --live_buffers;
    return CL_SUCCESS;
}

static cl_int CL_API_CALL set_kernel_arg(cl_kernel, cl_uint index, size_t bytes, const void* value)
{
    if (index >= 2 || bytes != sizeof(cl_mem))
        return CL_INVALID_ARG_VALUE;
    probe_buffers[index] = *static_cast<const cl_mem*>(value);
    return probe_failure == 4 ? CL_INVALID_ARG_VALUE : CL_SUCCESS;
}

static cl_int CL_API_CALL enqueue_kernel(cl_command_queue, cl_kernel, cl_uint dimensions, const size_t*, const size_t* global, const size_t*, cl_uint, const cl_event*, cl_event*)
{
    if (probe_failure == 5 || dimensions != 1 || *global * 4 + 2 != probe_buffers[0]->data.size())
        return CL_INVALID_WORK_GROUP_SIZE;
    for (size_t i = 0; i < *global; i++)
        for (size_t lane = 0; lane < 4; lane++)
            probe_buffers[1]->data[i * 4 + lane] = probe_buffers[0]->data[i * 4 + 3 - lane + 2];
    if (probe_failure == 7)
        probe_buffers[1]->data[3] ^= 1;
    return CL_SUCCESS;
}

static cl_int CL_API_CALL read_buffer(cl_command_queue, cl_mem buffer, cl_bool, size_t offset, size_t bytes, void* output, cl_uint, const cl_event*, cl_event*)
{
    if (probe_failure == 6)
        return CL_OUT_OF_RESOURCES;
    memcpy(output, reinterpret_cast<const char*>(buffer->data.data()) + offset, bytes);
    return CL_SUCCESS;
}

static int check_packed_loads(const char* name, const char* extensions, bool expected)
{
    const momoten_detail::ShaderDeviceProfile profile = momoten_detail::probe_shader_device_profile(
        reinterpret_cast<cl_device_id>(1), extensions, 128, 128, momoten_detail::OpenCLDriverPolicy());
    if (profile.packed_fp16_vector_loads != expected || live_queues || live_buffers || live_contexts || live_programs || live_kernels)
    {
        fprintf(stderr, "device_profile_test: packed half loads: %s\n", name);
        return 1;
    }
    return 0;
}

static int check_profile(const char* name, const char* extensions, momoten::SubgroupMode mode, uint32_t size, size_t max_workgroup = 1024, size_t max_x = 1024, size_t expected_query_local = 1024)
{
    queried_local_size = 0;
    matching_build_options = true;
    complete_native_source = false;
    const momoten_detail::ShaderDeviceProfile profile = momoten_detail::probe_shader_device_profile(
        reinterpret_cast<cl_device_id>(1), extensions, max_workgroup, max_x,
        momoten_detail::OpenCLDriverPolicy());
    if (profile.subgroup_mode != mode || profile.subgroup_size != size || profile.subgroup_operations != VK_SUBGROUP_FEATURE_BASIC_BIT
        || queried_local_size != expected_query_local || !matching_build_options
        || profile.compiler.version != expected_language_version
        || profile.compiler.full_profile != (device_profile == "FULL_PROFILE")
        || (mode == momoten::SubgroupModeNative && !complete_native_source)
        || live_contexts || live_programs || live_kernels)
    {
        fprintf(stderr, "device_profile_test: %s: mode=%d size=%u query-local=%zu live=%d/%d/%d\n", name, profile.subgroup_mode, profile.subgroup_size, queried_local_size, live_contexts, live_programs, live_kernels);
        return 1;
    }
    return 0;
}

static int check_compiler(const char* name, uint32_t version, const char* options)
{
    expected_language_version = version;
    expected_build_options = options;
    const momoten::SubgroupMode mode = version >= 200 ? momoten::SubgroupModeNative : momoten::SubgroupModeEmulatedBasic;
    return check_profile(name, "cl_khr_subgroups cl_khr_subgroup_non_uniform_vote", mode, 64);
}

int main()
{
    momoten_detail::g_opencl.p_clGetDeviceInfo = get_device_info;
    momoten_detail::g_opencl.p_clCreateContext = create_context;
    momoten_detail::g_opencl.p_clReleaseContext = release_context;
    momoten_detail::g_opencl.p_clCreateProgramWithSource = create_program;
    momoten_detail::g_opencl.p_clBuildProgram = build_program;
    momoten_detail::g_opencl.p_clReleaseProgram = release_program;
    momoten_detail::g_opencl.p_clCreateKernel = create_kernel;
    momoten_detail::g_opencl.p_clReleaseKernel = release_kernel;
    momoten_detail::g_opencl.p_clGetKernelSubGroupInfoKHR = get_subgroup_info;
    momoten_detail::g_opencl.p_clGetKernelWorkGroupInfo = get_workgroup_info;

    const char* extensions = "cl_khr_subgroups cl_khr_subgroup_non_uniform_vote";
    const momoten::SubgroupMode emulated = momoten::SubgroupModeEmulatedBasic;
    empty_build = true;
    subgroup_size = 64;
    preferred_multiple = 8;
    int result = check_profile("query works without subgroup builtins", extensions, emulated, 64);
    result |= check_profile("one-dimensional work-item limit", extensions, emulated, 64, 1024, 128, 128);
    result |= check_profile("workgroup invocation limit", extensions, emulated, 64, 128, 1024, 128);
    preferred_result = CL_INVALID_OPERATION;
    result |= check_profile("width query does not require scheduling hint", extensions, emulated, 64);
    preferred_result = CL_SUCCESS;

    native_build = true;
    result |= check_profile("native basic remains available", extensions, momoten::SubgroupModeNative, 64);
    result |= check_profile("missing vote uses emulation", "cl_khr_subgroups", emulated, 64);
    native_build = false;

    subgroup_result = CL_INVALID_OPERATION;
    result |= check_profile("failed width query", extensions, emulated, 8);
    subgroup_result = CL_SUCCESS;
    const size_t invalid_sizes[] = {0, 48, 256};
    for (size_t i = 0; i < sizeof(invalid_sizes) / sizeof(invalid_sizes[0]); i++)
    {
        subgroup_size = invalid_sizes[i];
        result |= check_profile("invalid width query", extensions, emulated, 8);
    }
    subgroup_size = 64;

    momoten_detail::g_opencl.p_clGetKernelSubGroupInfoKHR = 0;
    result |= check_profile("missing optional entry point", extensions, emulated, 8, 1024, 1024, 0);
    momoten_detail::g_opencl.p_clGetKernelSubGroupInfoKHR = get_subgroup_info;
    result |= check_profile("missing extension", "", emulated, 8, 1024, 1024, 0);
    result |= check_profile("extension token boundary", "cl_khr_subgroups_extra", emulated, 8, 1024, 1024, 0);
    preferred_multiple = 96;
    result |= check_profile("legacy scheduling hint rounded down", "", emulated, 64, 1024, 1024, 0);
    result |= check_profile("legacy scheduling hint capped", "", emulated, 32, 32, 32, 0);
    preferred_result = CL_INVALID_OPERATION;
    result |= check_profile("unavailable queries", "", momoten::SubgroupModeSingleton, 1, 1024, 1024, 0);
    empty_build = false;
    result |= check_profile("unavailable compiler", extensions, momoten::SubgroupModeSingleton, 1, 1024, 1024, 0);
    empty_build = true;
    preferred_result = CL_SUCCESS;
    native_build = true;
    native_requires_modern = true;
    device_profile = "FULL_PROFILE";
    language_version = "OpenCL C 1.2 legacy default";
    const NameVersion versions[] = {{(1u << 22) | (2u << 12), "OpenCL C"}, {3u << 22, "OpenCL C"}, {2u << 22, "OpenCL C"}, {4u << 22, "Other Language"}};
    language_versions.assign(versions, versions + 4);
    result |= check_compiler("version list takes precedence over 1.2 version string", 300, "-cl-std=CL3.0");
    device_profile = "EMBEDDED_PROFILE";
    result |= check_compiler("embedded profile also selects modern language", 300, "-cl-std=CL3.0");
    election_build = false;
    result |= check_profile("unsupported election retains width for emulation", extensions, emulated, 64);
    election_build = true;
    barrier_build = false;
    result |= check_profile("unsupported barrier retains width for emulation", extensions, emulated, 64);
    barrier_build = true;
    device_profile = "FULL_PROFILE";
    versions_size_invalid = true;
    result |= check_compiler("malformed version list falls back to version string", 120, "-cl-std=CL1.2");
    versions_size_invalid = false;
    versions_read_failure = true;
    language_version = "OpenCL C 2.0";
    result |= check_compiler("failed version list read falls back to version string", 200, "-cl-std=CL2.0");
    versions_read_failure = false;
    language_versions.clear();
    result |= check_compiler("OpenCL C 2.0 without version list", 200, "-cl-std=CL2.0");
    language_version = "OpenCL C 3.0";
    result |= check_compiler("OpenCL C 3.0 without version list", 300, "-cl-std=CL3.0");
    language_version = "OpenCL C 1.2";
    result |= check_compiler("OpenCL C 1.2", 120, "-cl-std=CL1.2");
    language_version = "OpenCL C 1.1";
    result |= check_compiler("OpenCL C 1.1", 110, "-cl-std=CL1.1");
    language_version = "OpenCL C 1.0";
    result |= check_compiler("OpenCL C 1.0", 100, "");
    language_version = "invalid";
    result |= check_compiler("unknown language uses legacy default", 100, "");
    language_version.clear();
    result |= check_compiler("missing language queries use legacy default", 100, "");
    context_result = CL_INVALID_DEVICE;
    result |= check_profile("failed context creation releases handles", extensions, momoten::SubgroupModeSingleton, 1, 1024, 1024, 0);
    context_result = CL_SUCCESS;
    program_result = CL_INVALID_VALUE;
    result |= check_profile("failed program creation releases handles", extensions, momoten::SubgroupModeSingleton, 1, 1024, 1024, 0);
    program_result = CL_SUCCESS;
    kernel_result = CL_INVALID_KERNEL_NAME;
    result |= check_profile("failed kernel creation releases handles", extensions, momoten::SubgroupModeSingleton, 1, 1024, 1024, 0);
    kernel_result = CL_SUCCESS;
    momoten_detail::g_opencl.p_clCreateCommandQueue = create_queue;
    momoten_detail::g_opencl.p_clReleaseCommandQueue = release_queue;
    momoten_detail::g_opencl.p_clCreateBuffer = create_buffer;
    momoten_detail::g_opencl.p_clReleaseMemObject = release_buffer;
    momoten_detail::g_opencl.p_clSetKernelArg = set_kernel_arg;
    momoten_detail::g_opencl.p_clEnqueueNDRangeKernel = enqueue_kernel;
    momoten_detail::g_opencl.p_clEnqueueReadBuffer = read_buffer;
    result |= check_packed_loads("CPU stays conservative", "cl_khr_fp16", false);
    device_type = CL_DEVICE_TYPE_GPU;
    result |= check_packed_loads("missing fp16", "", false);
    result |= check_packed_loads("verified lane mapping", "cl_khr_fp16", true);
    empty_build = false;
    result |= check_packed_loads("unavailable compiler", "cl_khr_fp16", false);
    empty_build = true;
    for (probe_failure = 1; probe_failure <= 7; probe_failure++)
        result |= check_packed_loads("execution or mapping failure", "cl_khr_fp16", false);

    VkPhysicalDevice_T physical = {};
    VkDevice_T device = {};
    device.physical_device = &physical;
    physical.opencl_device_type = CL_DEVICE_TYPE_GPU;
    physical.shader_profile.packed_fp16_vector_loads = true;
    momoten::TranslationOptions options;
    const cl_uint alignments[] = {8, 16, 24, 32, 48, 64, 128};
    for (size_t i = 0; i < sizeof(alignments) / sizeof(alignments[0]); i++)
    {
        physical.mem_base_alignment_bits = alignments[i];
        momoten_detail::configure_translation_options(&device, options);
        if (options.packed_fp16_vector_loads != (alignments[i] >= 32 && alignments[i] % 32 == 0))
        {
            fprintf(stderr, "device_profile_test: packed load descriptor alignment guard\n");
            result = 1;
        }
    }
    physical.opencl_device_type = CL_DEVICE_TYPE_CPU;
    momoten_detail::configure_translation_options(&device, options);
    if (options.packed_fp16_vector_loads)
        result = 1;
    return result;
}
