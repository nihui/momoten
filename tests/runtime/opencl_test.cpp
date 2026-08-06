// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "momoten/spv_to_clc.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

static cl_device_id find_compiler_device(bool require_fp16, bool require_integer_dot_product)
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
                && compiler_available == CL_TRUE && (!require_fp16 || device_has_extension(devices[d], "cl_khr_fp16")) && (!require_integer_dot_product || device_has_extension(devices[d], "cl_khr_integer_dot_product")))
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
    const cl_uint size32 = sizeof(input);
    const cl_ulong size64 = sizeof(input);
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
        if (ret == CL_SUCCESS)
            ret = clSetKernelArg(kernel, argument.size_arg_index, offset_size,
                                 address_bits == 64 ? (const void*)&size64 : (const void*)&size32);
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
    const cl_uint size32 = sizeof(packed);
    const cl_ulong size64 = sizeof(packed);
    const void* offset = address_bits == 64 ? static_cast<const void*>(&offset64) : static_cast<const void*>(&offset32);
    const void* size = address_bits == 64 ? static_cast<const void*>(&size64) : static_cast<const void*>(&size32);
    const size_t offset_size = address_bits == 64 ? sizeof(offset64) : sizeof(offset32);
    const momoten::BufferArgument& argument = translated.abi.buffers[0];
    ret = clSetKernelArg(kernel, argument.buffer_arg_index, sizeof(buffer), &buffer);
    if (ret == CL_SUCCESS)
        ret = clSetKernelArg(kernel, argument.offset_arg_index, offset_size, offset);
    if (ret == CL_SUCCESS)
        ret = clSetKernelArg(kernel, argument.size_arg_index, offset_size, size);

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
    if (translated.abi.buffers.size() != 4)
    {
        fprintf(stderr, "opencl_test: scalar 16-bit fixture expected four storage buffers\n");
        return 1;
    }

    uint16_t half_data[16] = {};
    uint16_t ushort_data[16] = {};
    uint16_t half_output[16] = {};
    float output[8] = {};
    const uint16_t half_values[4] = {0x3c00u, 0x4000u, 0x4200u, 0x4400u};
    for (uint32_t i = 0; i < 4; i++)
    {
        // The descriptor starts at element four and the shader deliberately
        // loads from nonzero indices one through four.
        half_data[5 + i] = half_values[i];
        ushort_data[5 + i] = static_cast<uint16_t>((i + 1) * 10);
    }

    cl_int ret = CL_SUCCESS;
    cl_mem buffers[4] = {};
    buffers[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(half_data), half_data, &ret);
    if (!buffers[0])
        return fail("clCreateBuffer(scalar half)", ret);
    buffers[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(ushort_data), ushort_data, &ret);
    if (!buffers[1])
    {
        clReleaseMemObject(buffers[0]);
        return fail("clCreateBuffer(scalar ushort)", ret);
    }
    buffers[2] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(output), output, &ret);
    if (!buffers[2])
    {
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[0]);
        return fail("clCreateBuffer(scalar output)", ret);
    }
    buffers[3] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(half_output), half_output, &ret);
    if (!buffers[3])
    {
        clReleaseMemObject(buffers[2]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[0]);
        return fail("clCreateBuffer(scalar half output)", ret);
    }

    const cl_uint offset32 = 8;
    const cl_ulong offset64 = 8;
    const cl_uint size32 = 24;
    const cl_ulong size64 = 24;
    const void* offset = address_bits == 64 ? static_cast<const void*>(&offset64) : static_cast<const void*>(&offset32);
    const void* size = address_bits == 64 ? static_cast<const void*>(&size64) : static_cast<const void*>(&size32);
    const size_t offset_size = address_bits == 64 ? sizeof(offset64) : sizeof(offset32);
    for (size_t i = 0; i < translated.abi.buffers.size() && ret == CL_SUCCESS; i++)
    {
        const momoten::BufferArgument& argument = translated.abi.buffers[i];
        if (argument.binding >= 4)
        {
            ret = CL_INVALID_ARG_INDEX;
            break;
        }
        ret = clSetKernelArg(kernel, argument.buffer_arg_index,
                             sizeof(buffers[argument.binding]), &buffers[argument.binding]);
        if (ret == CL_SUCCESS)
            ret = clSetKernelArg(kernel, argument.offset_arg_index, offset_size, offset);
        if (ret == CL_SUCCESS)
            ret = clSetKernelArg(kernel, argument.size_arg_index, offset_size, size);
    }

    const size_t global_size[3] = {4, 1, 1};
    const size_t local_size[3] = {translated.abi.local_size[0], translated.abi.local_size[1],
                                  translated.abi.local_size[2]};
    if (ret == CL_SUCCESS)
        ret = clEnqueueNDRangeKernel(queue, kernel, 3, 0, global_size, local_size, 0, 0, 0);
    if (ret == CL_SUCCESS)
        ret = clEnqueueReadBuffer(queue, buffers[2], CL_TRUE, 8,
                                  sizeof(float) * 4, output + 2, 0, 0, 0);
    if (ret == CL_SUCCESS)
        ret = clEnqueueReadBuffer(queue, buffers[3], CL_TRUE, 8,
                                  sizeof(uint16_t) * 4, half_output + 4, 0, 0, 0);

    int status = 0;
    if (ret != CL_SUCCESS)
        status = fail("scalar 16-bit kernel dispatch", ret);
    else
    {
        for (uint32_t i = 0; i < 4; i++)
        {
            const float expected = static_cast<float>((i + 1) * 11);
            const uint16_t expected_half[4] = {0x4980u, 0x4d80u, 0x5020u, 0x5180u};
            if (output[2 + i] != expected)
            {
                fprintf(stderr, "opencl_test: scalar 16-bit output[%u]=%g expected=%g\n",
                        i, output[2 + i], expected);
                status = 1;
            }
            if (half_output[4 + i] != expected_half[i])
            {
                fprintf(stderr, "opencl_test: scalar half output[%u]=0x%04x expected=0x%04x\n",
                        i, half_output[4 + i], expected_half[i]);
                status = 1;
            }
        }
    }

    clReleaseMemObject(buffers[3]);
    clReleaseMemObject(buffers[2]);
    clReleaseMemObject(buffers[1]);
    clReleaseMemObject(buffers[0]);
    return status;
}

static int run_subgroup_basic_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel,
                                     const momoten::TranslationResult& translated,
                                     cl_uint address_bits)
{
    if (translated.abi.subgroup_mode != momoten::SubgroupModeSingleton || translated.abi.subgroup_size != 1 || translated.abi.buffers.size() != 1)
    {
        fprintf(stderr, "opencl_test: subgroup BASIC fixture did not use the singleton ABI\n");
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
    const cl_uint size32 = sizeof(output);
    const cl_ulong size64 = sizeof(output);
    const void* offset = address_bits == 64 ? static_cast<const void*>(&offset64) : static_cast<const void*>(&offset32);
    const void* size = address_bits == 64 ? static_cast<const void*>(&size64) : static_cast<const void*>(&size32);
    const size_t offset_size = address_bits == 64 ? sizeof(offset64) : sizeof(offset32);
    const momoten::BufferArgument& argument = translated.abi.buffers[0];
    ret = clSetKernelArg(kernel, argument.buffer_arg_index, sizeof(buffer), &buffer);
    if (ret == CL_SUCCESS)
        ret = clSetKernelArg(kernel, argument.offset_arg_index, offset_size, offset);
    if (ret == CL_SUCCESS)
        ret = clSetKernelArg(kernel, argument.size_arg_index, offset_size, size);

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
            const uint32_t expected = 14001u + i * 100u;
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
    const cl_uint sizes32[2] = {sizeof(input), sizeof(output)};
    const cl_ulong sizes64[2] = {sizeof(input), sizeof(output)};
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
        if (ret == CL_SUCCESS)
            ret = clSetKernelArg(kernel, argument.size_arg_index, offset_size,
                                 address_bits == 64 ? static_cast<const void*>(&sizes64[argument.binding]) : static_cast<const void*>(&sizes32[argument.binding]));
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

typedef int (*OpenCLKernelTestRunner)(
    cl_context, cl_command_queue, cl_kernel,
    const momoten::TranslationResult&, cl_uint);

enum OpenCLTestFeature
{
    OpenCLTestBaseline,
    OpenCLTestFp16,
    OpenCLTestAtomicPacked,
    OpenCLTestScalar16Bit,
    OpenCLTestSubgroupBasic,
    OpenCLTestIntegerDotProduct
};

struct OpenCLTestCase
{
    const char* mode;
    OpenCLTestFeature feature;
    bool requires_fp16;
    bool requires_integer_dot_product;
    OpenCLKernelTestRunner runner;
};

static const OpenCLTestCase opencl_test_cases[] = {
    {0, OpenCLTestBaseline, false, false, 0},
    {"fp16", OpenCLTestFp16, true, false, run_fp16_kernel},
    {"atomic-packed", OpenCLTestAtomicPacked, false, false,
     run_atomic_packed_kernel},
    {"scalar-16bit", OpenCLTestScalar16Bit, true, false,
     run_scalar_16bit_kernel},
    {"subgroup-basic", OpenCLTestSubgroupBasic, false, false,
     run_subgroup_basic_kernel},
    {"integer-dot-product", OpenCLTestIntegerDotProduct, false, true,
     run_integer_dot_product_kernel}};

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
        fprintf(stderr, "opencl_test: expected SPIR-V input path and optional fp16, atomic-packed, scalar-16bit, subgroup-basic, or integer-dot-product mode\n");
        return 1;
    }

    cl_device_id device = find_compiler_device(
        test_case->requires_fp16, test_case->requires_integer_dot_product);
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

    const char* source = translated.source.c_str();
    const size_t source_size = translated.source.size();
    cl_program program = clCreateProgramWithSource(context, 1, &source, &source_size, &ret);
    if (!program)
    {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return fail("clCreateProgramWithSource", ret);
    }

    ret = clBuildProgram(program, 1, &device, 0, 0, 0);
    if (ret != CL_SUCCESS)
    {
        print_build_log(program, device);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return fail("clBuildProgram", ret);
    }

    cl_kernel kernel = clCreateKernel(program, translated.abi.entry_point.c_str(), &ret);
    if (!kernel)
    {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return fail("clCreateKernel", ret);
    }

    if (test_case->runner)
    {
        const int status = test_case->runner(
            context, queue, kernel, translated, address_bits);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return status;
    }

    float input[element_count];
    float output[element_count];
    const float scale = 2.5f;
    for (uint32_t i = 0; i < element_count; i++)
    {
        input[i] = static_cast<float>(i) - 3.f;
        output[i] = 0.f;
    }

    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(input), input, &ret);
    if (!input_buffer)
        return fail("clCreateBuffer(input)", ret);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(output), output, &ret);
    if (!output_buffer)
        return fail("clCreateBuffer(output)", ret);
    cl_mem push_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(scale), const_cast<float*>(&scale), &ret);
    if (!push_buffer)
        return fail("clCreateBuffer(push constants)", ret);

    const cl_uint offset32 = 0;
    const cl_ulong offset64 = 0;
    const cl_uint buffer_size32 = sizeof(input);
    const cl_ulong buffer_size64 = sizeof(input);
    const void* offset = address_bits == 64 ? static_cast<const void*>(&offset64) : static_cast<const void*>(&offset32);
    const void* buffer_size = address_bits == 64 ? static_cast<const void*>(&buffer_size64) : static_cast<const void*>(&buffer_size32);
    const size_t offset_size = address_bits == 64 ? sizeof(offset64) : sizeof(offset32);

    ret = clSetKernelArg(kernel, 0, sizeof(input_buffer), &input_buffer);
    if (ret == CL_SUCCESS) ret = clSetKernelArg(kernel, 1, offset_size, offset);
    if (ret == CL_SUCCESS) ret = clSetKernelArg(kernel, 2, offset_size, buffer_size);
    if (ret == CL_SUCCESS) ret = clSetKernelArg(kernel, 3, sizeof(output_buffer), &output_buffer);
    if (ret == CL_SUCCESS) ret = clSetKernelArg(kernel, 4, offset_size, offset);
    if (ret == CL_SUCCESS) ret = clSetKernelArg(kernel, 5, offset_size, buffer_size);
    if (ret == CL_SUCCESS) ret = clSetKernelArg(kernel, 6, sizeof(push_buffer), &push_buffer);
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
        const float expected = rintf(input[source] * scale);
        if (fabsf(output[i] - expected) > 1e-6f)
        {
            fprintf(stderr, "opencl_test: output[%u]=%g expected=%g\n", i, output[i], expected);
            status = 1;
        }
    }

    clReleaseMemObject(push_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(input_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return status;
}
