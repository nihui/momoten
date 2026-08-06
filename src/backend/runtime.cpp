// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "runtime.h"
#include "device.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#include <direct.h>
#endif
#include <sys/stat.h>

momoten_detail::HandleTable<VkDeviceMemory, DeviceMemory> g_device_memories;
momoten_detail::HandleTable<VkBuffer, Buffer> g_buffers;
momoten_detail::HandleTable<VkShaderModule, ShaderModule> g_shader_modules;
momoten_detail::HandleTable<VkDescriptorSetLayout, DescriptorSetLayout> g_descriptor_set_layouts;
momoten_detail::HandleTable<VkPipelineLayout, PipelineLayout> g_pipeline_layouts;
momoten_detail::HandleTable<VkPipeline, Pipeline> g_pipelines;
momoten_detail::HandleTable<VkDescriptorUpdateTemplate, DescriptorUpdateTemplate> g_descriptor_update_templates;
momoten_detail::HandleTable<VkPipelineCache, PipelineCache> g_pipeline_caches;
momoten_detail::HandleTable<VkCommandPool, CommandPool> g_command_pools;
momoten_detail::HandleTable<VkFence, Fence> g_fences;

DeviceMemory::~DeviceMemory()
{
    if (memory)
        momoten_detail::g_opencl.p_clReleaseMemObject(memory);
}

Pipeline::~Pipeline()
{
    if (kernel)
        momoten_detail::g_opencl.p_clReleaseKernel(kernel);
    if (program)
        momoten_detail::g_opencl.p_clReleaseProgram(program);
}

CommandPool::~CommandPool()
{
    for (size_t i = 0; i < buffers.size(); i++)
        delete buffers[i];
}

Fence::~Fence()
{
    if (event)
        momoten_detail::g_opencl.p_clReleaseEvent(event);
}

static VkDeviceSize descriptor_range(const std::shared_ptr<Buffer>& buffer,
                                     const DescriptorValue& descriptor)
{
    if (!buffer || descriptor.offset >= buffer->size)
        return 0;
    const VkDeviceSize available = buffer->size - descriptor.offset;
    return descriptor.range == VK_WHOLE_SIZE ? available : std::min(descriptor.range, available);
}

static bool bound_range(const std::shared_ptr<Buffer>& buffer,
                        VkDeviceSize buffer_offset, VkDeviceSize size,
                        std::shared_ptr<DeviceMemory>& memory,
                        size_t& memory_offset, size_t& byte_count)
{
    if (!buffer || !buffer->memory || buffer_offset > buffer->size || size > buffer->size - buffer_offset)
        return false;
    const VkDeviceSize absolute = buffer->memory_offset + buffer_offset;
    if (absolute > buffer->memory->size || size > buffer->memory->size - absolute)
        return false;
    if (absolute > static_cast<VkDeviceSize>(SIZE_MAX) || size > static_cast<VkDeviceSize>(SIZE_MAX))
        return false;
    memory = buffer->memory;
    memory_offset = static_cast<size_t>(absolute);
    byte_count = static_cast<size_t>(size);
    return true;
}

static cl_int upload_shadow(VkDevice device, const std::shared_ptr<DeviceMemory>& memory,
                            size_t offset, size_t size)
{
    if (!memory->host_visible || size == 0)
        return CL_SUCCESS;
    return momoten_detail::g_opencl.p_clEnqueueWriteBuffer(device->command_queue, memory->memory, CL_FALSE,
                                                           offset, size, memory->shadow.data() + offset, 0, 0, 0);
}

static cl_int download_shadow(VkDevice device, const std::shared_ptr<DeviceMemory>& memory,
                              size_t offset, size_t size)
{
    if (!memory->host_visible || size == 0)
        return CL_SUCCESS;
    return momoten_detail::g_opencl.p_clEnqueueReadBuffer(device->command_queue, memory->memory, CL_FALSE,
                                                          offset, size, memory->shadow.data() + offset, 0, 0, 0);
}

cl_int replay_copy(VkDevice device, const RecordedCommand& command)
{
    for (size_t i = 0; i < command.copy_regions.size(); i++)
    {
        const VkBufferCopy& region = command.copy_regions[i];
        std::shared_ptr<DeviceMemory> src_memory;
        std::shared_ptr<DeviceMemory> dst_memory;
        size_t src_offset = 0;
        size_t dst_offset = 0;
        size_t size = 0;
        size_t dst_size = 0;
        if (!bound_range(command.src_buffer, region.srcOffset, region.size,
                         src_memory, src_offset, size)
            || !bound_range(command.dst_buffer, region.dstOffset, region.size,
                            dst_memory, dst_offset, dst_size))
            return CL_INVALID_VALUE;

        cl_int ret = upload_shadow(device, src_memory, src_offset, size);
        if (ret != CL_SUCCESS)
            return ret;
        ret = momoten_detail::g_opencl.p_clEnqueueCopyBuffer(device->command_queue, src_memory->memory, dst_memory->memory,
                                                             src_offset, dst_offset, size, 0, 0, 0);
        if (ret != CL_SUCCESS)
            return ret;
        ret = download_shadow(device, dst_memory, dst_offset, size);
        if (ret != CL_SUCCESS)
            return ret;
    }
    return CL_SUCCESS;
}

cl_int replay_dispatch(VkDevice device, const RecordedCommand& command)
{
    if (!command.pipeline)
        return CL_INVALID_KERNEL;
    // A zero Vulkan group count is a valid no-op, while OpenCL rejects a zero
    // global work size.
    if (command.group_count[0] == 0 || command.group_count[1] == 0 || command.group_count[2] == 0)
        return CL_SUCCESS;
    const std::shared_ptr<Pipeline>& pipeline = command.pipeline;

    for (size_t i = 0; i < pipeline->abi.buffers.size(); i++)
    {
        const momoten::BufferArgument& argument = pipeline->abi.buffers[i];
        const std::map<uint32_t, DescriptorValue>::const_iterator it = command.descriptors.find(argument.binding);
        if (it == command.descriptors.end() || !it->second.buffer || !it->second.buffer->memory)
            return CL_INVALID_MEM_OBJECT;

        const std::shared_ptr<Buffer>& buffer = it->second.buffer;
        const VkDeviceSize range = descriptor_range(buffer, it->second);
        std::shared_ptr<DeviceMemory> memory;
        size_t absolute_offset = 0;
        size_t byte_count = 0;
        if (!bound_range(buffer, it->second.offset, range, memory, absolute_offset, byte_count))
            return CL_INVALID_VALUE;
        if (argument.access != momoten::BufferAccessWriteOnly)
        {
            const cl_int upload_ret = upload_shadow(device, memory, absolute_offset, byte_count);
            if (upload_ret != CL_SUCCESS)
                return upload_ret;
        }

        cl_mem cl_buffer = memory->memory;
        cl_int ret = momoten_detail::g_opencl.p_clSetKernelArg(pipeline->kernel, argument.buffer_arg_index,
                                                               sizeof(cl_buffer), &cl_buffer);
        if (ret != CL_SUCCESS)
            return ret;

        if (pipeline->abi.address_bits == 64)
        {
            const cl_ulong size = static_cast<cl_ulong>(byte_count);
            ret = momoten_detail::g_opencl.p_clSetKernelArg(pipeline->kernel, argument.size_arg_index,
                                                            sizeof(size), &size);
        }
        else
        {
            if (byte_count > 0xffffffffu)
                return CL_INVALID_VALUE;
            const cl_uint size = static_cast<cl_uint>(byte_count);
            ret = momoten_detail::g_opencl.p_clSetKernelArg(pipeline->kernel, argument.size_arg_index,
                                                            sizeof(size), &size);
        }
        if (ret != CL_SUCCESS)
            return ret;

        if (pipeline->abi.address_bits == 64)
        {
            const cl_ulong offset = static_cast<cl_ulong>(absolute_offset);
            ret = momoten_detail::g_opencl.p_clSetKernelArg(pipeline->kernel, argument.offset_arg_index,
                                                            sizeof(offset), &offset);
        }
        else
        {
            if (absolute_offset > 0xffffffffu)
                return CL_INVALID_VALUE;
            const cl_uint offset = static_cast<cl_uint>(absolute_offset);
            ret = momoten_detail::g_opencl.p_clSetKernelArg(pipeline->kernel, argument.offset_arg_index,
                                                            sizeof(offset), &offset);
        }
        if (ret != CL_SUCCESS)
            return ret;
    }

    cl_mem push_buffer = 0;
    if (pipeline->abi.push_constant_arg_index >= 0)
    {
        if (command.push_constants.size() < pipeline->abi.push_constant_size)
            return CL_INVALID_ARG_SIZE;
        cl_int ret = CL_SUCCESS;
        push_buffer = momoten_detail::g_opencl.p_clCreateBuffer(device->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                                pipeline->abi.push_constant_size,
                                                                const_cast<unsigned char*>(command.push_constants.data()), &ret);
        if (!push_buffer)
            return ret;
        ret = momoten_detail::g_opencl.p_clSetKernelArg(pipeline->kernel, pipeline->abi.push_constant_arg_index,
                                                        sizeof(push_buffer), &push_buffer);
        if (ret != CL_SUCCESS)
        {
            momoten_detail::g_opencl.p_clReleaseMemObject(push_buffer);
            return ret;
        }
    }

    size_t global_size[3] = {0, 0, 0};
    size_t local_size[3] = {
        pipeline->opencl_local_size[0],
        pipeline->opencl_local_size[1],
        pipeline->opencl_local_size[2]};
    if (pipeline->abi.workgroup_splittable)
    {
        if (pipeline->workgroup_chunk_count == 0 || local_size[0] == 0
            || command.group_count[0] > SIZE_MAX / pipeline->workgroup_chunk_count)
        {
            if (push_buffer) momoten_detail::g_opencl.p_clReleaseMemObject(push_buffer);
            return CL_INVALID_GLOBAL_WORK_SIZE;
        }
        const size_t physical_group_count_x = static_cast<size_t>(command.group_count[0]) * pipeline->workgroup_chunk_count;
        if (physical_group_count_x > SIZE_MAX / local_size[0])
        {
            if (push_buffer) momoten_detail::g_opencl.p_clReleaseMemObject(push_buffer);
            return CL_INVALID_GLOBAL_WORK_SIZE;
        }
        global_size[0] = physical_group_count_x * local_size[0];
        global_size[1] = command.group_count[1];
        global_size[2] = command.group_count[2];
    }
    else
    {
        for (size_t d = 0; d < 3; d++)
        {
            if (command.group_count[d] != 0 && local_size[d] > SIZE_MAX / command.group_count[d])
            {
                if (push_buffer) momoten_detail::g_opencl.p_clReleaseMemObject(push_buffer);
                return CL_INVALID_GLOBAL_WORK_SIZE;
            }
            global_size[d] = local_size[d] * command.group_count[d];
        }
    }

    cl_int ret = momoten_detail::g_opencl.p_clEnqueueNDRangeKernel(device->command_queue, pipeline->kernel, 3, 0,
                                                                   global_size, local_size, 0, 0, 0);
    if (push_buffer)
        momoten_detail::g_opencl.p_clReleaseMemObject(push_buffer);
    if (ret != CL_SUCCESS)
        return ret;
    for (size_t i = 0; i < pipeline->abi.buffers.size(); i++)
    {
        const momoten::BufferArgument& argument = pipeline->abi.buffers[i];
        if (argument.access == momoten::BufferAccessReadOnly)
            continue;
        const DescriptorValue& descriptor = command.descriptors.find(argument.binding)->second;
        std::shared_ptr<DeviceMemory> memory;
        size_t absolute_offset = 0;
        size_t byte_count = 0;
        if (!bound_range(descriptor.buffer, descriptor.offset,
                         descriptor_range(descriptor.buffer, descriptor),
                         memory, absolute_offset, byte_count))
            return CL_INVALID_VALUE;
        ret = download_shadow(device, memory, absolute_offset, byte_count);
        if (ret != CL_SUCCESS)
            return ret;
    }

    return CL_SUCCESS;
}

void assign_fence_event(const std::shared_ptr<Fence>& fence, cl_event event)
{
    if (!fence)
        return;
    if (event && momoten_detail::g_opencl.p_clRetainEvent(event) != CL_SUCCESS)
        return;
    std::lock_guard<std::mutex> lock(fence->mutex);
    if (fence->event)
        momoten_detail::g_opencl.p_clReleaseEvent(fence->event);
    fence->event = event;
    fence->signaled = false;
}

void retire_pending_submissions(VkDevice device, bool all)
{
    for (size_t i = 0; i < device->pending_submissions.size();)
    {
        PendingSubmission& submission = device->pending_submissions[i];
        cl_int status = CL_COMPLETE;
        const bool complete = all || (momoten_detail::g_opencl.p_clGetEventInfo(submission.event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, 0) == CL_SUCCESS && status == CL_COMPLETE);
        if (!complete)
        {
            i++;
            continue;
        }

        for (size_t b = 0; b < submission.command_buffers.size(); b++)
        {
            VkCommandBuffer buffer = submission.command_buffers[b];
            if (!buffer || buffer->pending_count == 0)
                continue;
            buffer->pending_count--;
            if (buffer->pending_count == 0)
            {
                buffer->state = (buffer->usage & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
                                    ? momoten_detail::COMMAND_BUFFER_STATE_INVALID
                                    : momoten_detail::COMMAND_BUFFER_STATE_EXECUTABLE;
            }
        }
        momoten_detail::g_opencl.p_clReleaseEvent(submission.event);
        device->pending_submissions.erase(device->pending_submissions.begin() + i);
    }
}

bool fence_is_signaled(const std::shared_ptr<Fence>& fence)
{
    if (!fence)
        return false;
    std::lock_guard<std::mutex> lock(fence->mutex);
    if (fence->signaled)
        return true;
    if (!fence->event)
        return false;
    cl_int status = 0;
    if (momoten_detail::g_opencl.p_clGetEventInfo(fence->event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                                  sizeof(status), &status, 0)
            == CL_SUCCESS
        && status == CL_COMPLETE)
    {
        fence->signaled = true;
        return true;
    }
    return false;
}
