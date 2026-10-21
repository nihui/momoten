// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "buffer_copy.h"
#include "objects.h"
#include "program.h"

namespace momoten_detail {

static cl_int create_buffer_copy_kernel(VkDevice device)
{
    std::string source = device->physical_device->address_bits == 64
                             ? "typedef ulong momo_copy_offset;\n"
                             : "typedef uint momo_copy_offset;\n";
    source += R"(
__kernel void momo_copy_buffer(__global const uchar* source, __global uchar* destination,
                               momo_copy_offset source_offset, momo_copy_offset destination_offset,
                               momo_copy_offset byte_count)
{
    size_t index = get_global_id(0) * 16;
    if (index >= byte_count) return;
    source += source_offset + index;
    destination += destination_offset + index;
    if (byte_count - index >= 16)
        vstore16(vload16(0, source), 0, destination);
    else
        for (uint i = 0; i < (uint)(byte_count - index); i++) destination[i] = source[i];
}
)";

    cl_int result = CL_SUCCESS;
    const std::string build_options = opencl_language_options(device->physical_device->shader_profile.compiler);
    cl_program program = create_and_build_program(device, source, build_options, result);
    if (!program)
        return result == CL_SUCCESS ? CL_INVALID_PROGRAM : result;
    cl_kernel kernel = 0;
    if (result == CL_SUCCESS)
        kernel = g_opencl.p_clCreateKernel(program, "momo_copy_buffer", &result);
    else
        print_program_build_log(program, device->physical_device->device);
    g_opencl.p_clReleaseProgram(program);
    if (result != CL_SUCCESS)
    {
        if (kernel)
            g_opencl.p_clReleaseKernel(kernel);
        return result;
    }
    if (!kernel)
        return CL_INVALID_KERNEL;
    device->buffer_copy_kernel = kernel;
    return CL_SUCCESS;
}

cl_int enqueue_buffer_copy(
    VkDevice device, cl_mem source, cl_mem destination,
    size_t source_offset, size_t destination_offset, size_t byte_count,
    cl_uint wait_count, const cl_event* wait_list, cl_event* event)
{
    if (!device->physical_device->driver_policy.use_buffer_copy_kernel)
        return g_opencl.p_clEnqueueCopyBuffer(
            device->command_queue, source, destination, source_offset, destination_offset,
            byte_count, wait_count, wait_list, event);

    const bool wide_address = device->physical_device->address_bits == 64;
    if (!wide_address && (source_offset > 0xffffffffu || destination_offset > 0xffffffffu || byte_count > 0xffffffffu))
        return CL_INVALID_VALUE;
    if (byte_count == 0)
        return CL_INVALID_VALUE;
    if (!device->buffer_copy_kernel)
    {
        const cl_int result = create_buffer_copy_kernel(device);
        if (result != CL_SUCCESS)
            return result;
    }

    cl_kernel kernel = device->buffer_copy_kernel;
    cl_int result = g_opencl.p_clSetKernelArg(kernel, 0, sizeof(source), &source);
    if (result != CL_SUCCESS)
        return result;
    result = g_opencl.p_clSetKernelArg(kernel, 1, sizeof(destination), &destination);
    if (result != CL_SUCCESS)
        return result;
    const cl_ulong offsets[] = {source_offset, destination_offset, byte_count};
    for (cl_uint i = 0; i < 3; i++)
    {
        const cl_uint narrow_offset = static_cast<cl_uint>(offsets[i]);
        result = g_opencl.p_clSetKernelArg(kernel, i + 2,
                                           wide_address ? sizeof(cl_ulong) : sizeof(cl_uint),
                                           wide_address ? static_cast<const void*>(&offsets[i]) : static_cast<const void*>(&narrow_offset));
        if (result != CL_SUCCESS)
            return result;
    }
    const size_t global_size = byte_count / 16 + (byte_count % 16 != 0);
    return g_opencl.p_clEnqueueNDRangeKernel(
        device->command_queue, kernel, 1, 0, &global_size, 0, wait_count, wait_list, event);
}

} // namespace momoten_detail
