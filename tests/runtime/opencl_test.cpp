// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "momoten/spv_to_clc.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

static bool read_spirv(const char* path, std::vector<uint32_t>& words)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        return false;
    const std::streamoff size = stream.tellg();
    if (size < 0 || size % 4 != 0)
        return false;
    words.resize(static_cast<size_t>(size) / 4);
    stream.seekg(0, std::ios::beg);
    return words.empty() || static_cast<bool>(stream.read(reinterpret_cast<char*>(words.data()), size));
}

static int fail(const char* operation, cl_int error)
{
    fprintf(stderr, "opencl_test: %s failed with OpenCL error %d\n", operation, error);
    return 1;
}

static bool device_has_extension(cl_device_id device, const char* extension)
{
    size_t size = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, 0, &size) != CL_SUCCESS || size == 0)
        return false;
    std::vector<char> value(size);
    if (clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, size, value.data(), 0) != CL_SUCCESS)
        return false;
    const std::string padded = " " + std::string(value.data()) + " ";
    return padded.find(" " + std::string(extension) + " ") != std::string::npos;
}

static cl_device_id find_compiler_device(bool require_fp16, bool require_fp64, bool require_integer_dot_product)
{
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, 0, &platform_count) != CL_SUCCESS || platform_count == 0)
        return 0;

    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), 0) != CL_SUCCESS)
        return 0;

    for (cl_uint p = 0; p < platform_count; p++)
    {
        cl_uint device_count = 0;
        cl_int ret = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, 0, &device_count);
        if (ret == CL_DEVICE_NOT_FOUND)
            continue;
        if (ret != CL_SUCCESS || device_count == 0)
            continue;

        std::vector<cl_device_id> devices(device_count);
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, device_count, devices.data(), 0) != CL_SUCCESS)
            continue;

        for (cl_uint d = 0; d < device_count; d++)
        {
            cl_bool compiler_available = CL_FALSE;
            if (clGetDeviceInfo(devices[d], CL_DEVICE_COMPILER_AVAILABLE,
                                sizeof(compiler_available), &compiler_available, 0)
                    == CL_SUCCESS
                && compiler_available == CL_TRUE && (!require_fp16 || device_has_extension(devices[d], "cl_khr_fp16")) && (!require_fp64 || device_has_extension(devices[d], "cl_khr_fp64")) && (!require_integer_dot_product || device_has_extension(devices[d], "cl_khr_integer_dot_product")))
                return devices[d];
        }
    }

    return 0;
}

static void print_build_log(cl_program program, cl_device_id device)
{
    size_t size = 0;
    if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, 0, &size) != CL_SUCCESS || size == 0)
        return;
    std::vector<char> log(size);
    if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), 0) == CL_SUCCESS)
        fprintf(stderr, "%s\n", log.data());
}

static int build_test_kernel(cl_context context, cl_device_id device,
                             const momoten::TranslationResult& translated,
                             cl_program& program, cl_kernel& kernel)
{
    const char* source = translated.source.c_str();
    const size_t source_size = translated.source.size();
    cl_int ret = CL_SUCCESS;
    program = clCreateProgramWithSource(context, 1, &source, &source_size, &ret);
    if (!program)
        return fail("clCreateProgramWithSource", ret);

    ret = clBuildProgram(program, 1, &device, 0, 0, 0);
    if (ret != CL_SUCCESS)
    {
        print_build_log(program, device);
        clReleaseProgram(program);
        program = 0;
        return fail("clBuildProgram", ret);
    }

    kernel = clCreateKernel(program, translated.abi.entry_point.c_str(), &ret);
    if (!kernel)
    {
        clReleaseProgram(program);
        program = 0;
        return fail("clCreateKernel", ret);
    }
    return 0;
}

static int run_fp16_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel,
                           const momoten::TranslationResult& translated,
                           cl_uint address_bits)
{
    if (translated.abi.buffers.size() != 3)
    {
        fprintf(stderr, "opencl_test: fp16 fixture expected three storage buffers\n");
        return 1;
    }

    const uint16_t input_pattern[4] = {0xc000u, 0xbc00u, 0x0000u, 0x3c00u};
    const uint16_t expected_pattern[4] = {0xbe00u, 0xb800u, 0x3800u, 0x3e00u};
    uint16_t input[16];
    uint16_t output[16] = {};
    uint16_t matrix[16] = {};
    for (uint32_t i = 0; i < 16; i++)
    {
        input[i] = input_pattern[i % 4];
        if (i / 4 == i % 4)
            matrix[i] = 0x3c00u; // half(1.0), column-major identity
    }

    cl_int ret = CL_SUCCESS;
    cl_mem buffers[3] = {};
    buffers[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(input), input, &ret);
    if (!buffers[0])
        return fail("clCreateBuffer(fp16 input)", ret);
    buffers[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(output), output, &ret);
    if (!buffers[1])
    {
        clReleaseMemObject(buffers[0]);
        return fail("clCreateBuffer(fp16 output)", ret);
    }
    buffers[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(matrix), matrix, &ret);
    if (!buffers[2])
    {
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[0]);
        return fail("clCreateBuffer(fp16 matrix)", ret);
    }

    const cl_uint offset32 = 0;
    const cl_ulong offset64 = 0;
    const void* offset = address_bits == 64 ? static_cast<const void*>(&offset64) : static_cast<const void*>(&offset32);
    const size_t offset_size = address_bits == 64 ? sizeof(offset64) : sizeof(offset32);
    for (size_t i = 0; i < translated.abi.buffers.size() && ret == CL_SUCCESS; i++)
    {
        const momoten::BufferArgument& argument = translated.abi.buffers[i];
        if (argument.binding >= 3)
        {
            ret = CL_INVALID_ARG_INDEX;
            break;
        }
        ret = clSetKernelArg(kernel, argument.buffer_arg_index,
                             sizeof(buffers[argument.binding]), &buffers[argument.binding]);
        if (ret == CL_SUCCESS)
            ret = clSetKernelArg(kernel, argument.offset_arg_index, offset_size, offset);
    }

    const size_t global_size[3] = {4, 1, 1};
    const size_t local_size[3] = {translated.abi.local_size[0], translated.abi.local_size[1],
                                  translated.abi.local_size[2]};
    if (ret == CL_SUCCESS)
        ret = clEnqueueNDRangeKernel(queue, kernel, 3, 0, global_size, local_size, 0, 0, 0);
    if (ret == CL_SUCCESS)
        ret = clEnqueueReadBuffer(queue, buffers[1], CL_TRUE, 0,
                                  sizeof(output), output, 0, 0, 0);

    int status = 0;
    if (ret != CL_SUCCESS)
        status = fail("fp16 kernel dispatch", ret);
    else
    {
        for (uint32_t i = 0; i < 16; i++)
        {
            if (output[i] != expected_pattern[i % 4])
            {
                fprintf(stderr, "opencl_test: fp16 output[%u]=0x%04x expected=0x%04x\n",
                        i, output[i], expected_pattern[i % 4]);
                status = 1;
            }
        }
    }

    clReleaseMemObject(buffers[2]);
    clReleaseMemObject(buffers[1]);
    clReleaseMemObject(buffers[0]);
    return status;
}

static int run_fp64_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel,
                           const momoten::TranslationResult& translated,
                           cl_uint address_bits)
{
    if (!translated.abi.fp64 || translated.abi.buffers.size() != 2
        || translated.abi.buffers[0].binding != 0
        || translated.abi.buffers[1].binding != 1)
    {
        fprintf(stderr, "opencl_test: fp64 fixture expected input/output storage buffers and an fp64 ABI\n");
        return 1;
    }

    double input[4] = {0.0, 1.0, 2.0, 3.0};
    double output[4] = {};
    cl_int ret = CL_SUCCESS;
    cl_mem buffers[2] = {};
    buffers[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(input), input, &ret);
    if (!buffers[0])
        return fail("clCreateBuffer(fp64 input)", ret);
    buffers[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(output), output, &ret);
    if (!buffers[1])
    {
        clReleaseMemObject(buffers[0]);
        return fail("clCreateBuffer(fp64 output)", ret);
    }

    const cl_uint offset32 = 0;
    const cl_ulong offset64 = 0;
    const void* offset = address_bits == 64 ? static_cast<const void*>(&offset64) : static_cast<const void*>(&offset32);
    const size_t offset_size = address_bits == 64 ? sizeof(offset64) : sizeof(offset32);
    for (size_t i = 0; i < translated.abi.buffers.size() && ret == CL_SUCCESS; i++)
    {
        const momoten::BufferArgument& argument = translated.abi.buffers[i];
        if (argument.binding >= 2)
        {
            ret = CL_INVALID_ARG_INDEX;
            break;
        }
        ret = clSetKernelArg(kernel, argument.buffer_arg_index,
                             sizeof(buffers[argument.binding]), &buffers[argument.binding]);
        if (ret == CL_SUCCESS)
            ret = clSetKernelArg(kernel, argument.offset_arg_index, offset_size, offset);
    }

    const size_t global_size[3] = {4, 1, 1};
    const size_t local_size[3] = {translated.abi.local_size[0], translated.abi.local_size[1],
                                  translated.abi.local_size[2]};
    if (ret == CL_SUCCESS)
        ret = clEnqueueNDRangeKernel(queue, kernel, 3, 0, global_size, local_size, 0, 0, 0);
    if (ret == CL_SUCCESS)
        ret = clEnqueueReadBuffer(queue, buffers[1], CL_TRUE, 0, sizeof(output), output, 0, 0, 0);

    int status = 0;
    if (ret != CL_SUCCESS)
        status = fail("fp64 kernel dispatch", ret);
    else
    {
        const double expected[4] = {3.0, 11.0, 27.0, 51.0};
        for (uint32_t i = 0; i < 4; i++)
        {
            if (output[i] != expected[i])
            {
                fprintf(stderr, "opencl_test: fp64 output[%u]=%.17g expected=%.17g\n",
                        i, output[i], expected[i]);
                status = 1;
            }
        }
    }

    clReleaseMemObject(buffers[1]);
    clReleaseMemObject(buffers[0]);
    return status;
}

static int compile_only_int16_kernel(cl_context, cl_command_queue, cl_kernel,
                                     const momoten::TranslationResult&, cl_uint)
{
    // Reaching the runner means the translated source built successfully and
    // the OpenCL implementation accepted the generated kernel entry point.
    return 0;
}

static int run_atomic_packed_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel,
                                    const momoten::TranslationResult& translated,
                                    cl_uint address_bits)
{
    if (translated.abi.buffers.size() != 1 || translated.abi.buffers[0].binding != 0)
    {
        fprintf(stderr, "opencl_test: packed atomic fixture expected one storage buffer\n");
        return 1;
    }

    cl_int ret = CL_SUCCESS;
    int32_t packed = 0;
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(packed), &packed, &ret);
    if (!buffer)
        return fail("clCreateBuffer(packed atomic)", ret);

    const cl_uint offset32 = 0;
    const cl_ulong offset64 = 0;
    const void* offset = address_bits == 64 ? static_cast<const void*>(&offset64) : static_cast<const void*>(&offset32);
    const size_t offset_size = address_bits == 64 ? sizeof(offset64) : sizeof(offset32);
    const momoten::BufferArgument& argument = translated.abi.buffers[0];
    ret = clSetKernelArg(kernel, argument.buffer_arg_index, sizeof(buffer), &buffer);
    if (ret == CL_SUCCESS)
        ret = clSetKernelArg(kernel, argument.offset_arg_index, offset_size, offset);

    // Many workgroups repeatedly update four different bytes in the same
    // int32 word. This exercises the compare-exchange retry path under real
    // contention, matching ncnn packed-int8 scalar stores.
    const size_t global_size[3] = {1024, 1, 1};
    const size_t local_size[3] = {translated.abi.local_size[0], translated.abi.local_size[1],
                                  translated.abi.local_size[2]};
    if (ret == CL_SUCCESS)
        ret = clEnqueueNDRangeKernel(queue, kernel, 3, 0, global_size, local_size, 0, 0, 0);
    if (ret == CL_SUCCESS)
        ret = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(packed), &packed, 0, 0, 0);

    int status = 0;
    if (ret != CL_SUCCESS)
        status = fail("packed atomic kernel dispatch", ret);
    else if (static_cast<uint32_t>(packed) != 0x04030201u)
    {
        fprintf(stderr, "opencl_test: packed atomic output=0x%08x expected=0x04030201\n",
                static_cast<uint32_t>(packed));
        status = 1;
    }

    clReleaseMemObject(buffer);
    return status;
}

static int run_scalar_16bit_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel,
                                   const momoten::TranslationResult& translated,
                                   cl_uint address_bits)
{
    if (translated.abi.buffers.size() != 6)
    {
        fprintf(stderr, "opencl_test: scalar 16-bit fixture expected six storage buffers\n");
        return 1;
    }

    // Each scalar descriptor begins two bytes into its allocation. The last
    // active element occupies the allocation's final two bytes.
    uint16_t half_data[4] = {0x7bffu, 0x3c00u, 0x4000u, 0x4200u};
    uint16_t ushort_data[4] = {65535u, 32768u, 32769u, 32770u};
    int16_t short_data[4] = {32767, -32768, -32768, -32768};
    uint16_t ushort_vector_data[6] = {65535u, 65535u, 65535u, 65535u, 65535u, 7u};
    uint16_t half_output[4] = {0x7bffu, 0x7bffu, 0x7bffu, 0x7bffu};
    const float untouched = -123.f;
    float output[5] = {untouched, untouched, untouched, untouched, untouched};
    void* data[6] = {half_data, ushort_data, output, half_output, short_data, ushort_vector_data};
    const size_t sizes[6] = {sizeof(half_data), sizeof(ushort_data), sizeof(output), sizeof(half_output), sizeof(short_data), sizeof(ushort_vector_data)};

    cl_int ret = CL_SUCCESS;
    cl_mem buffers[6] = {};
    for (size_t i = 0; i < 6; i++)
    {
        const cl_mem_flags flags = i == 2 || i == 3 ? CL_MEM_READ_WRITE : CL_MEM_READ_ONLY;
        buffers[i] = clCreateBuffer(context, flags | CL_MEM_COPY_HOST_PTR, sizes[i], data[i], &ret);
        if (!buffers[i])
        {
            for (size_t j = 0; j < i; j++)
                clReleaseMemObject(buffers[j]);
            return fail("clCreateBuffer(scalar 16-bit)", ret);
        }
    }

    // Float and u16vec2 descriptors require four-byte alignment; scalar
    // half/ushort/short descriptors deliberately provide only two bytes.
    const cl_uint offsets32[6] = {2, 2, 4, 2, 2, 4};
    const cl_ulong offsets64[6] = {2, 2, 4, 2, 2, 4};
    const size_t offset_size = address_bits == 64 ? sizeof(cl_ulong) : sizeof(cl_uint);
    for (size_t i = 0; i < translated.abi.buffers.size() && ret == CL_SUCCESS; i++)
    {
        const momoten::BufferArgument& argument = translated.abi.buffers[i];
        if (argument.binding >= 6)
        {
            ret = CL_INVALID_ARG_INDEX;
            break;
        }
        ret = clSetKernelArg(kernel, argument.buffer_arg_index, sizeof(buffers[argument.binding]), &buffers[argument.binding]);
        if (ret == CL_SUCCESS)
            ret = clSetKernelArg(kernel, argument.offset_arg_index, offset_size,
                                 address_bits == 64 ? static_cast<const void*>(&offsets64[argument.binding]) : static_cast<const void*>(&offsets32[argument.binding]));
    }

    const size_t global_size[3] = {4, 1, 1};
    const size_t local_size[3] = {translated.abi.local_size[0], translated.abi.local_size[1], translated.abi.local_size[2]};
    if (ret == CL_SUCCESS)
        ret = clEnqueueNDRangeKernel(queue, kernel, 3, 0, global_size, local_size, 0, 0, 0);
    if (ret == CL_SUCCESS)
        ret = clEnqueueReadBuffer(queue, buffers[2], CL_TRUE, 0, sizeof(output), output, 0, 0, 0);
    if (ret == CL_SUCCESS)
        ret = clEnqueueReadBuffer(queue, buffers[3], CL_TRUE, 0, sizeof(half_output), half_output, 0, 0, 0);

    int status = 0;
    if (ret != CL_SUCCESS)
        status = fail("scalar 16-bit kernel dispatch", ret);
    else
    {
        const uint16_t expected_half[3] = {0x4800u, 0x4900u, 0x4a00u};
        for (uint32_t i = 0; i < 3; i++)
        {
            const float expected = static_cast<float>(8 + i * 2);
            if (output[1 + i] != expected)
            {
                fprintf(stderr, "opencl_test: scalar 16-bit output[%u]=%g expected=%g\n", i, output[1 + i], expected);
                status = 1;
            }
            if (half_output[1 + i] != expected_half[i])
            {
                fprintf(stderr, "opencl_test: scalar half output[%u]=0x%04x expected=0x%04x\n", i, half_output[1 + i], expected_half[i]);
                status = 1;
            }
        }
        if (output[0] != untouched || output[4] != untouched || half_output[0] != 0x7bffu)
        {
            fprintf(stderr, "opencl_test: scalar 16-bit kernel modified a descriptor prefix or inactive invocation\n");
            status = 1;
        }
    }

    for (size_t i = 0; i < 6; i++)
        clReleaseMemObject(buffers[i]);
    return status;
}

static int run_subgroup_basic_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel,
                                     const momoten::TranslationResult& translated,
                                     cl_uint address_bits)
{
    if (translated.abi.subgroup_mode != momoten::SubgroupModeEmulatedBasic || translated.abi.subgroup_size != 8 || translated.abi.buffers.size() != 1)
    {
        fprintf(stderr, "opencl_test: subgroup BASIC fixture did not use the emulated ABI\n");
        return 1;
    }

    uint32_t output[4] = {};
    cl_int ret = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(output), output, &ret);
    if (!buffer)
        return fail("clCreateBuffer(subgroup BASIC)", ret);

    const cl_uint offset32 = 0;
    const cl_ulong offset64 = 0;
    const void* offset = address_bits == 64 ? static_cast<const void*>(&offset64) : static_cast<const void*>(&offset32);
    const size_t offset_size = address_bits == 64 ? sizeof(offset64) : sizeof(offset32);
    const momoten::BufferArgument& argument = translated.abi.buffers[0];
    ret = clSetKernelArg(kernel, argument.buffer_arg_index, sizeof(buffer), &buffer);
    if (ret == CL_SUCCESS)
        ret = clSetKernelArg(kernel, argument.offset_arg_index, offset_size, offset);

    const size_t global_size[3] = {4, 1, 1};
    const size_t local_size[3] = {translated.abi.local_size[0], translated.abi.local_size[1],
                                  translated.abi.local_size[2]};
    if (ret == CL_SUCCESS)
        ret = clEnqueueNDRangeKernel(queue, kernel, 3, 0, global_size, local_size, 0, 0, 0);
    if (ret == CL_SUCCESS)
        ret = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(output), output, 0, 0, 0);

    int status = 0;
    if (ret != CL_SUCCESS)
        status = fail("subgroup BASIC kernel dispatch", ret);
    else
    {
        for (uint32_t i = 0; i < 4; i++)
        {
            const uint32_t subgroup_size = translated.abi.subgroup_size;
            const uint32_t invocation_id = i % subgroup_size;
            const uint32_t subgroup_id = i / subgroup_size;
            const uint32_t subgroup_count = (4u + subgroup_size - 1) / subgroup_size;
            const uint32_t expected = subgroup_size + invocation_id * 10u
                                      + subgroup_id * 100u + subgroup_count * 1000u
                                      + (invocation_id == 0 ? 10000u : 0u);
            if (output[i] != expected)
            {
                fprintf(stderr, "opencl_test: subgroup output[%u]=%u expected=%u\n",
                        i, output[i], expected);
                status = 1;
            }
        }
    }

    clReleaseMemObject(buffer);
    return status;
}

static int run_integer_dot_product_kernel(
    cl_context context, cl_command_queue queue, cl_kernel kernel,
    const momoten::TranslationResult& translated, cl_uint address_bits)
{
    if (!translated.abi.integer_dot_product || translated.abi.buffers.size() != 2)
    {
        fprintf(stderr, "opencl_test: integer dot-product fixture has an invalid ABI\n");
        return 1;
    }

    const uint32_t input[12] = {
        0x7f80ff01u, 0x0203feffu, 0x04030201u, 0x08070605u,
        0x80000000u, 0x7fffffffu, 2u, 3u,
        0xffffffffu, 0xfffffffeu, 0x80000001u, 0xfffffffdu};
    const uint32_t expected[14] = {
        0xffffff7fu, 70u, 119u, 12u, 0x80000000u, 0xffffffffu, 0x7fffffffu,
        0x7ffffffdu, 0xffffffffu, 0u, 123u, 0x80000000u, 0xffffffffu, 321u};
    uint32_t output[14] = {};
    cl_int ret = CL_SUCCESS;
    cl_mem buffers[2] = {};
    buffers[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(input), const_cast<uint32_t*>(input), &ret);
    if (!buffers[0])
        return fail("clCreateBuffer(integer dot input)", ret);
    buffers[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(output), output, &ret);
    if (!buffers[1])
    {
        clReleaseMemObject(buffers[0]);
        return fail("clCreateBuffer(integer dot output)", ret);
    }

    const cl_uint offset32 = 0;
    const cl_ulong offset64 = 0;
    const void* offset = address_bits == 64 ? static_cast<const void*>(&offset64) : static_cast<const void*>(&offset32);
    const size_t offset_size = address_bits == 64 ? sizeof(offset64) : sizeof(offset32);
    for (size_t i = 0; i < translated.abi.buffers.size() && ret == CL_SUCCESS; i++)
    {
        const momoten::BufferArgument& argument = translated.abi.buffers[i];
        if (argument.binding >= 2)
        {
            ret = CL_INVALID_ARG_INDEX;
            break;
        }
        ret = clSetKernelArg(kernel, argument.buffer_arg_index,
                             sizeof(buffers[argument.binding]), &buffers[argument.binding]);
        if (ret == CL_SUCCESS)
            ret = clSetKernelArg(kernel, argument.offset_arg_index, offset_size, offset);
    }

    const size_t global_size[3] = {1, 1, 1};
    const size_t local_size[3] = {translated.abi.local_size[0], translated.abi.local_size[1],
                                  translated.abi.local_size[2]};
    if (ret == CL_SUCCESS)
        ret = clEnqueueNDRangeKernel(queue, kernel, 3, 0, global_size, local_size, 0, 0, 0);
    if (ret == CL_SUCCESS)
        ret = clEnqueueReadBuffer(queue, buffers[1], CL_TRUE, 0,
                                  sizeof(output), output, 0, 0, 0);

    int status = 0;
    if (ret != CL_SUCCESS)
        status = fail("integer dot-product kernel dispatch", ret);
    else
    {
        for (uint32_t i = 0; i < 14; i++)
        {
            if (output[i] != expected[i])
            {
                fprintf(stderr,
                        "opencl_test: integer dot output[%u]=0x%08x expected=0x%08x\n",
                        i, output[i], expected[i]);
                status = 1;
            }
        }
    }

    clReleaseMemObject(buffers[1]);
    clReleaseMemObject(buffers[0]);
    return status;
}

static int run_workgroup_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel,
                                const momoten::TranslationResult& translated,
                                cl_uint address_bits, size_t max_physical_size,
                                std::vector<uint32_t>* captured_output = 0)
{
    if (!translated.abi.workgroup_splittable || translated.abi.buffers.size() != 1
        || translated.abi.subgroup_mode != momoten::SubgroupModeEmulatedBasic
        || translated.abi.subgroup_size != 8)
    {
        fprintf(stderr, "opencl_test: workgroup coordinate fixture has an unexpected ABI\n");
        return 1;
    }

    cl_device_id device = 0;
    cl_int ret = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device, 0);
    size_t kernel_limit = 0;
    size_t device_limit = 0;
    size_t compile_size[3] = {};
    cl_uint dimensions = 0;
    if (ret == CL_SUCCESS)
        ret = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                       sizeof(kernel_limit), &kernel_limit, 0);
    if (ret == CL_SUCCESS)
        ret = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                       sizeof(compile_size), compile_size, 0);
    if (ret == CL_SUCCESS)
        ret = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                              sizeof(device_limit), &device_limit, 0);
    if (ret == CL_SUCCESS)
        ret = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                              sizeof(dimensions), &dimensions, 0);
    if (ret != CL_SUCCESS)
        return fail("workgroup limit query", ret);
    if (dimensions < 3)
        return fail("workgroup dimensions", CL_INVALID_WORK_DIMENSION);

    const bool virtual_workgroup = translated.abi.workgroup_mode == momoten::WorkgroupModeVirtual;
    for (size_t d = 0; d < 3; d++)
    {
        const size_t expected = virtual_workgroup ? 0 : translated.abi.local_size[d];
        if (compile_size[d] != expected)
        {
            fprintf(stderr, "opencl_test: compiled workgroup size[%zu]=%zu expected=%zu\n", d, compile_size[d], expected);
            return 1;
        }
    }

    std::vector<size_t> dimension_limits(dimensions, 1);
    ret = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                          dimension_limits.size() * sizeof(size_t), dimension_limits.data(), 0);
    if (ret != CL_SUCCESS)
        return fail("clGetDeviceInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES)", ret);

    const size_t logical_size = static_cast<size_t>(translated.abi.local_size[0]) * translated.abi.local_size[1] * translated.abi.local_size[2];
    size_t local_size[3] = {translated.abi.local_size[0], translated.abi.local_size[1], translated.abi.local_size[2]};
    size_t chunk_count = 1;
    if (virtual_workgroup)
    {
        size_t physical_size = std::min(logical_size, kernel_limit);
        physical_size = std::min(physical_size, device_limit);
        physical_size = std::min(physical_size, dimension_limits[0]);
        physical_size = std::min(physical_size, max_physical_size);
        physical_size -= physical_size % translated.abi.subgroup_size;
        if (physical_size == 0 || physical_size >= logical_size)
        {
            fprintf(stderr, "opencl_test: failed to force a smaller physical workgroup\n");
            return 1;
        }
        local_size[0] = physical_size;
        local_size[1] = 1;
        local_size[2] = 1;
        chunk_count = 1 + (logical_size - 1) / physical_size;
    }
    else if (logical_size > kernel_limit || logical_size > device_limit
             || local_size[0] > dimension_limits[0] || local_size[1] > dimension_limits[1] || local_size[2] > dimension_limits[2])
    {
        fprintf(stderr, "opencl_test: fixed 3D workgroup exceeds this OpenCL device's limits; skipping\n");
        return 77;
    }
    const size_t group_count[3] = {2, 3, 2};
    const size_t invocation_count = logical_size * group_count[0] * group_count[1] * group_count[2];
    const size_t field_count = 8;
    std::vector<uint32_t> output(invocation_count * field_count, 0xffffffffu);

    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   output.size() * sizeof(uint32_t), output.data(), &ret);
    if (!buffer)
        return fail("clCreateBuffer(workgroup coordinates)", ret);

    const cl_uint offset32 = 0;
    const cl_ulong offset64 = 0;
    const void* offset = address_bits == 64 ? static_cast<const void*>(&offset64) : static_cast<const void*>(&offset32);
    const size_t scalar_size = address_bits == 64 ? sizeof(cl_ulong) : sizeof(cl_uint);
    const momoten::BufferArgument& argument = translated.abi.buffers[0];
    ret = clSetKernelArg(kernel, argument.buffer_arg_index, sizeof(buffer), &buffer);
    if (ret == CL_SUCCESS)
        ret = clSetKernelArg(kernel, argument.offset_arg_index, scalar_size, offset);

    const size_t global_size[3] = {group_count[0] * chunk_count * local_size[0], group_count[1] * local_size[1], group_count[2] * local_size[2]};
    if (ret == CL_SUCCESS)
        ret = clEnqueueNDRangeKernel(queue, kernel, 3, 0, global_size, local_size, 0, 0, 0);
    if (ret == CL_SUCCESS)
        ret = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0,
                                  output.size() * sizeof(uint32_t), output.data(), 0, 0, 0);

    int status = 0;
    if (ret != CL_SUCCESS)
        status = fail("workgroup coordinate kernel dispatch", ret);
    else
    {
        for (size_t gz = 0; gz < group_count[2] && status == 0; gz++)
        {
            for (size_t gy = 0; gy < group_count[1] && status == 0; gy++)
            {
                for (size_t gx = 0; gx < group_count[0] && status == 0; gx++)
                {
                    const size_t group_index = gx + group_count[0] * (gy + group_count[1] * gz);
                    for (size_t local_index = 0; local_index < logical_size; local_index++)
                    {
                        const uint32_t sx = translated.abi.local_size[0];
                        const uint32_t sy = translated.abi.local_size[1];
                        const uint32_t sz = translated.abi.local_size[2];
                        const uint32_t lx = static_cast<uint32_t>(local_index % sx);
                        const uint32_t ly = static_cast<uint32_t>((local_index / sx) % sy);
                        const uint32_t lz = static_cast<uint32_t>(local_index / (sx * sy));
                        const size_t base = (group_index * logical_size + local_index) * field_count;
                        const uint32_t expected[8] = {
                            lx | (ly << 8) | (lz << 16),
                            static_cast<uint32_t>(gx | (gy << 8) | (gz << 16)),
                            static_cast<uint32_t>((gx * sx + lx) | ((gy * sy + ly) << 8) | ((gz * sz + lz) << 16)),
                            static_cast<uint32_t>(group_count[0] | (group_count[1] << 8) | (group_count[2] << 16)),
                            static_cast<uint32_t>(translated.abi.subgroup_size | ((local_index % translated.abi.subgroup_size) << 8) | ((local_index / translated.abi.subgroup_size) << 16)),
                            static_cast<uint32_t>(((logical_size + translated.abi.subgroup_size - 1) / translated.abi.subgroup_size) | ((local_index % translated.abi.subgroup_size == 0 ? 1u : 0u) << 16)),
                            static_cast<uint32_t>(local_index),
                            sx | (sy << 8) | (sz << 16)};
                        for (size_t field = 0; field < field_count; field++)
                        {
                            if (output[base + field] != expected[field])
                            {
                                fprintf(stderr, "opencl_test: workgroup output[%zu]=0x%08x expected=0x%08x\n",
                                        base + field, output[base + field], expected[field]);
                                status = 1;
                                break;
                            }
                        }
                        if (status != 0)
                            break;
                    }
                }
            }
        }
    }

    clReleaseMemObject(buffer);
    if (status == 0 && captured_output)
        captured_output->swap(output);
    return status;
}

static int run_workgroup_split_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel,
                                      const momoten::TranslationResult& translated,
                                      cl_uint address_bits)
{
    if (translated.abi.workgroup_mode != momoten::WorkgroupModeVirtual || translated.abi.local_size[0] != 8
        || translated.abi.local_size[1] != 16 || translated.abi.local_size[2] != 3)
    {
        fprintf(stderr, "opencl_test: workgroup split fixture has an unexpected ABI\n");
        return 1;
    }
    return run_workgroup_kernel(context, queue, kernel, translated, address_bits, 256);
}

static int run_workgroup_3d_modes(cl_context context, cl_command_queue queue, cl_device_id device,
                                  cl_kernel kernel, const momoten::TranslationResult& translated,
                                  cl_uint address_bits, const std::vector<uint32_t>& words,
                                  const momoten::TranslationOptions& options)
{
    if (translated.abi.workgroup_mode != momoten::WorkgroupModeDirect || translated.abi.local_size[0] != 8
        || translated.abi.local_size[1] != 4 || translated.abi.local_size[2] != 2)
    {
        fprintf(stderr, "opencl_test: direct 3D workgroup fixture has an unexpected ABI\n");
        return 1;
    }
    std::vector<uint32_t> direct_output;
    const int direct_status = run_workgroup_kernel(context, queue, kernel, translated, address_bits, 64, &direct_output);
    if (direct_status != 0)
        return direct_status;

    momoten::TranslationOptions virtual_options = options;
    virtual_options.workgroup_mode = momoten::WorkgroupModeVirtual;
    momoten::TranslationResult virtual_translated;
    if (!momoten::translate_spirv_to_opencl_c(words.data(), words.size(), virtual_options, virtual_translated))
    {
        fprintf(stderr, "opencl_test: virtual 3D translation failed: %s\n", virtual_translated.diagnostics.c_str());
        return 1;
    }
    if (virtual_translated.abi.workgroup_mode != momoten::WorkgroupModeVirtual)
    {
        fprintf(stderr, "opencl_test: explicit virtual 3D mode was not reflected\n");
        return 1;
    }

    cl_program virtual_program = 0;
    cl_kernel virtual_kernel = 0;
    if (build_test_kernel(context, device, virtual_translated, virtual_program, virtual_kernel))
        return 1;
    std::vector<uint32_t> virtual_output;
    int status = run_workgroup_kernel(context, queue, virtual_kernel, virtual_translated, address_bits, 16, &virtual_output);
    clReleaseKernel(virtual_kernel);
    clReleaseProgram(virtual_program);
    if (status == 0 && direct_output != virtual_output)
    {
        fprintf(stderr, "opencl_test: direct and virtual 3D workgroup coordinates or subgroup results differ\n");
        status = 1;
    }
    return status;
}

typedef int (*OpenCLKernelTestRunner)(
    cl_context, cl_command_queue, cl_kernel,
    const momoten::TranslationResult&, cl_uint);

enum OpenCLTestFeature
{
    OpenCLTestBaseline,
    OpenCLTestFp16,
    OpenCLTestFp64,
    OpenCLTestInt16,
    OpenCLTestAtomicPacked,
    OpenCLTestScalar16Bit,
    OpenCLTestSubgroupBasic,
    OpenCLTestIntegerDotProduct,
    OpenCLTestWorkgroupSplit,
    OpenCLTestWorkgroup3D
};

struct OpenCLTestCase
{
    const char* mode;
    OpenCLTestFeature feature;
    bool requires_fp16;
    bool requires_fp64;
    bool requires_integer_dot_product;
    OpenCLKernelTestRunner runner;
};

static const OpenCLTestCase opencl_test_cases[] = {
    {0, OpenCLTestBaseline, false, false, false, 0},
    {"fp16", OpenCLTestFp16, true, false, false, run_fp16_kernel},
    {"fp64", OpenCLTestFp64, false, true, false, run_fp64_kernel},
    {"int16", OpenCLTestInt16, false, false, false, compile_only_int16_kernel},
    {"atomic-packed", OpenCLTestAtomicPacked, false, false, false,
     run_atomic_packed_kernel},
    {"scalar-16bit", OpenCLTestScalar16Bit, true, false, false,
     run_scalar_16bit_kernel},
    {"subgroup-basic", OpenCLTestSubgroupBasic, false, false, false,
     run_subgroup_basic_kernel},
    {"integer-dot-product", OpenCLTestIntegerDotProduct, false, false, true,
     run_integer_dot_product_kernel},
    {"workgroup-split", OpenCLTestWorkgroupSplit, false, false, false,
     run_workgroup_split_kernel},
    {"workgroup-3d", OpenCLTestWorkgroup3D, false, false, false, 0}};

static const OpenCLTestCase* find_opencl_test_case(int argc, char** argv)
{
    if (argc == 2)
        return &opencl_test_cases[0];
    if (argc != 3)
        return 0;
    for (size_t i = 1;
         i < sizeof(opencl_test_cases) / sizeof(opencl_test_cases[0]); i++)
    {
        if (strcmp(argv[2], opencl_test_cases[i].mode) == 0)
            return &opencl_test_cases[i];
    }
    return 0;
}

int main(int argc, char** argv)
{
    const OpenCLTestCase* test_case = find_opencl_test_case(argc, argv);
    if (!test_case)
    {
        fprintf(stderr, "opencl_test: expected SPIR-V input path and optional fp16, fp64, int16, atomic-packed, scalar-16bit, subgroup-basic, integer-dot-product, workgroup-split, or workgroup-3d mode\n");
        return 1;
    }

    cl_device_id device = find_compiler_device(
        test_case->requires_fp16, test_case->requires_fp64, test_case->requires_integer_dot_product);
    if (!device)
    {
        fprintf(stderr, "opencl_test: no compatible OpenCL device with an online compiler; skipping\n");
        return 77;
    }

    cl_uint address_bits = 0;
    cl_int ret = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(address_bits), &address_bits, 0);
    if (ret != CL_SUCCESS)
        return fail("clGetDeviceInfo(CL_DEVICE_ADDRESS_BITS)", ret);

    std::vector<uint32_t> words;
    if (!read_spirv(argv[1], words))
    {
        fprintf(stderr, "opencl_test: failed to read SPIR-V\n");
        return 1;
    }

    momoten::TranslationOptions options;
    options.address_bits = address_bits;
    if (test_case->feature == OpenCLTestSubgroupBasic || test_case->feature == OpenCLTestWorkgroupSplit || test_case->feature == OpenCLTestWorkgroup3D)
    {
        options.subgroup_mode = momoten::SubgroupModeEmulatedBasic;
        options.subgroup_size = 8;
    }
    if (test_case->feature == OpenCLTestWorkgroupSplit)
        options.workgroup_mode = momoten::WorkgroupModeVirtual;
    if (test_case->feature == OpenCLTestIntegerDotProduct)
    {
        cl_device_integer_dot_product_capabilities_khr capabilities = 0;
        if (clGetDeviceInfo(device, CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR,
                            sizeof(capabilities), &capabilities, 0)
                != CL_SUCCESS
            || (capabilities & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR) == 0)
        {
            fprintf(stderr, "opencl_test: compatible packed integer dot product is unavailable; skipping\n");
            return 77;
        }
        options.integer_dot_product = true;
        options.integer_dot_product_input_4x8bit = (capabilities & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR) != 0;
        options.integer_dot_product_input_4x8bit_packed = true;
    }
    const uint32_t element_count = 8;
    if (test_case->feature == OpenCLTestBaseline)
    {
        momoten::SpecializationValue specialization;
        specialization.constant_id = 0;
        specialization.data.resize(4);
        memcpy(specialization.data.data(), &element_count, 4);
        options.specializations.push_back(specialization);
    }

    momoten::TranslationResult translated;
    if (!momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, translated))
    {
        fprintf(stderr, "opencl_test: translation failed: %s\n", translated.diagnostics.c_str());
        return 1;
    }

    cl_context context = clCreateContext(0, 1, &device, 0, 0, &ret);
    if (!context)
        return fail("clCreateContext", ret);

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &ret);
    if (!queue)
    {
        clReleaseContext(context);
        return fail("clCreateCommandQueue", ret);
    }

    cl_program program = 0;
    cl_kernel kernel = 0;
    if (build_test_kernel(context, device, translated, program, kernel))
    {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    if (test_case->runner || test_case->feature == OpenCLTestWorkgroup3D)
    {
        const int status = test_case->feature == OpenCLTestWorkgroup3D
                               ? run_workgroup_3d_modes(context, queue, device, kernel, translated, address_bits, words, options)
                               : test_case->runner(context, queue, kernel, translated, address_bits);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return status;
    }

    float input[element_count];
    float output[element_count];
    const struct PushConstants
    {
        float scale;
        int32_t bias;
        uint32_t element_count;
    } push_constants = {2.5f, -3, 4};
    if (translated.abi.push_constant_size != sizeof(push_constants))
    {
        fprintf(stderr, "opencl_test: mixed scalar push-constant size is incorrect\n");
        return 1;
    }
    const float untouched = -123.f;
    for (uint32_t i = 0; i < element_count; i++)
    {
        input[i] = static_cast<float>(i * 2) - 3.f;
        output[i] = untouched;
    }

    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(input), input, &ret);
    if (!input_buffer)
        return fail("clCreateBuffer(input)", ret);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(output), output, &ret);
    if (!output_buffer)
        return fail("clCreateBuffer(output)", ret);
    const cl_uint offset32 = 0;
    const cl_ulong offset64 = 0;
    const void* offset = address_bits == 64 ? static_cast<const void*>(&offset64) : static_cast<const void*>(&offset32);
    const size_t offset_size = address_bits == 64 ? sizeof(offset64) : sizeof(offset32);

    ret = clSetKernelArg(kernel, 0, sizeof(input_buffer), &input_buffer);
    if (ret == CL_SUCCESS) ret = clSetKernelArg(kernel, 1, offset_size, offset);
    if (ret == CL_SUCCESS) ret = clSetKernelArg(kernel, 2, sizeof(output_buffer), &output_buffer);
    if (ret == CL_SUCCESS) ret = clSetKernelArg(kernel, 3, offset_size, offset);
    if (ret == CL_SUCCESS) ret = clSetKernelArg(kernel, translated.abi.push_constant_arg_index, sizeof(push_constants), &push_constants);
    if (ret != CL_SUCCESS)
        return fail("clSetKernelArg", ret);

    const size_t global_size[3] = {element_count, 1, 1};
    const size_t local_size[3] = {translated.abi.local_size[0], translated.abi.local_size[1],
                                  translated.abi.local_size[2]};
    ret = clEnqueueNDRangeKernel(queue, kernel, 3, 0, global_size, local_size, 0, 0, 0);
    if (ret != CL_SUCCESS)
        return fail("clEnqueueNDRangeKernel", ret);

    ret = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(output), output, 0, 0, 0);
    if (ret != CL_SUCCESS)
        return fail("clEnqueueReadBuffer", ret);

    int status = 0;
    for (uint32_t i = 0; i < element_count; i++)
    {
        const uint32_t source = i - i % 4 + (3 - i % 4);
        const float expected = i < push_constants.element_count ? rintf(input[source] * push_constants.scale) + static_cast<float>(push_constants.bias) : untouched;
        if (fabsf(output[i] - expected) > 1e-6f)
        {
            fprintf(stderr, "opencl_test: output[%u]=%g expected=%g\n", i, output[i], expected);
            status = 1;
        }
    }

    clReleaseMemObject(output_buffer);
    clReleaseMemObject(input_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return status;
}
