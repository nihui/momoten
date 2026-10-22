// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "replay.h"
#include "memory_sync.h"
#include "buffer_copy.h"

#include <algorithm>

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

static size_t fill_repartitioned_workgroup(const size_t global[3], const size_t axis_limits[3],
                                           size_t capacity, bool powers_of_two, size_t local[3])
{
    std::fill(local, local + 3, 1);
    // Cover nearby rows with a narrow x span before extending z or x.
    // Every dimension divides the grid exactly, without padded invocations.
    const size_t order[] = {0, 1, 2, 0};
    for (size_t i = 0; i < 4; i++)
    {
        const size_t d = order[i];
        const size_t extent = global[d] / local[d];
        size_t limit = std::min(capacity, axis_limits[d] / local[d]);
        limit = std::min(limit, extent);
        if (i == 0)
            limit = std::min(limit, static_cast<size_t>(4));
        size_t factor = 1;
        if (powers_of_two)
        {
            while (factor <= limit / 2 && extent % (factor * 2) == 0)
                factor *= 2;
        }
        else
        {
            factor = limit;
            while (extent % factor != 0)
                factor--;
        }
        local[d] *= factor;
        capacity /= factor;
    }
    return local[0] * local[1] * local[2];
}

ReplayState::ReplayState(VkDevice value)
    : device(value)
{
}

cl_int replay_copy(ReplayState& state, const RecordedCommand& command)
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

        cl_int ret = upload_host_ranges(state, src_memory, src_offset, size);
        if (ret != CL_SUCCESS)
            return ret;

        ret = momoten_detail::enqueue_buffer_copy(
            state.device, src_memory->memory, dst_memory->memory,
            src_offset, dst_offset, size, 0, 0, 0);
        if (ret != CL_SUCCESS)
            return ret;
        mark_device_range(state, dst_memory, dst_offset, size);
    }
    return CL_SUCCESS;
}

cl_int replay_barrier(ReplayState&, const RecordedCommand&)
{
    // All commands use the same in-order OpenCL queue. Its implicit memory
    // dependencies already cover the supported compute and buffer barriers.
    // Mapped host memory is synchronized before and after replay.
    return CL_SUCCESS;
}

cl_int replay_dispatch(ReplayState& state, const RecordedCommand& command)
{
    VkDevice device = state.device;
    if (!command.pipeline)
        return CL_INVALID_KERNEL;
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
            const cl_int upload_ret = upload_host_ranges(state, memory, absolute_offset, byte_count);
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

    if (pipeline->abi.push_constant_arg_index >= 0)
    {
        if (command.push_constants.size() < pipeline->abi.push_constant_size)
            return CL_INVALID_ARG_SIZE;
        for (size_t offset = 0; offset < pipeline->abi.push_constant_size; offset += 4)
        {
            const cl_int ret = momoten_detail::g_opencl.p_clSetKernelArg(
                pipeline->kernel, pipeline->abi.push_constant_arg_index + static_cast<cl_uint>(offset / 4),
                4, command.push_constants.data() + offset);
            if (ret != CL_SUCCESS)
                return ret;
        }
    }

    size_t global_size[3] = {0, 0, 0};
    size_t local_size[3] = {
        pipeline->opencl_local_size[0],
        pipeline->opencl_local_size[1],
        pipeline->opencl_local_size[2]};
    if (pipeline->abi.workgroup_mode == momoten::WorkgroupModeVirtual)
    {
        if (pipeline->workgroup_chunk_count == 0 || local_size[0] == 0
            || command.group_count[0] > SIZE_MAX / pipeline->workgroup_chunk_count)
            return CL_INVALID_GLOBAL_WORK_SIZE;
        const size_t physical_group_count_x = static_cast<size_t>(command.group_count[0]) * pipeline->workgroup_chunk_count;
        if (physical_group_count_x > SIZE_MAX / local_size[0])
            return CL_INVALID_GLOBAL_WORK_SIZE;
        global_size[0] = physical_group_count_x * local_size[0];
        global_size[1] = command.group_count[1];
        global_size[2] = command.group_count[2];
    }
    else if (pipeline->abi.workgroup_mode == momoten::WorkgroupModeRepartitioned)
    {
        for (size_t d = 0; d < 3; d++)
        {
            if (device->physical_device->max_work_item_sizes[d] == 0)
                return CL_INVALID_WORK_ITEM_SIZE;
            if (command.group_count[d] != 0 && pipeline->abi.local_size[d] > SIZE_MAX / command.group_count[d])
                return CL_INVALID_GLOBAL_WORK_SIZE;
            global_size[d] = static_cast<size_t>(pipeline->abi.local_size[d]) * command.group_count[d];
            local_size[d] = 1;
        }
        const size_t capacity = pipeline->opencl_workgroup_limit;
        if (capacity == 0)
            return CL_INVALID_WORK_GROUP_SIZE;
        const size_t filled = fill_repartitioned_workgroup(global_size, device->physical_device->max_work_item_sizes,
                                                           capacity, true, local_size);
        if (filled < capacity)
        {
            // Keep established full power-of-two shapes. Consider odd divisors
            // only when they fill more lanes under the same device limits.
            size_t candidate[3];
            if (fill_repartitioned_workgroup(global_size, device->physical_device->max_work_item_sizes,
                                             capacity, false, candidate)
                > filled)
                std::copy(candidate, candidate + 3, local_size);
        }
    }
    else
    {
        for (size_t d = 0; d < 3; d++)
        {
            if (command.group_count[d] != 0 && local_size[d] > SIZE_MAX / command.group_count[d])
                return CL_INVALID_GLOBAL_WORK_SIZE;
            global_size[d] = local_size[d] * command.group_count[d];
        }
    }

    cl_int ret = momoten_detail::g_opencl.p_clEnqueueNDRangeKernel(
        device->command_queue, pipeline->kernel, 3, 0,
        global_size, local_size, 0, 0, 0);
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
        mark_device_range(state, memory, absolute_offset, byte_count);
    }

    return CL_SUCCESS;
}

cl_int finish_replay(ReplayState& state, cl_event& event)
{
    const cl_int readback_ret = finish_mapped_memory_readbacks(state);
    if (readback_ret != CL_SUCCESS)
        return readback_ret;

    // One event covers the entire submission, including mapped readbacks and
    // empty submissions. Intermediate commands need no explicit events.
    VkDevice device = state.device;
    if (device->physical_device->driver_policy.use_copy_completion_event)
    {
        if (!device->completion_buffer)
        {
            cl_uint initial[2] = {};
            cl_int ret = CL_SUCCESS;
            cl_mem buffer = momoten_detail::g_opencl.p_clCreateBuffer(device->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(initial), initial, &ret);
            if (!buffer)
                return ret == CL_SUCCESS ? CL_MEM_OBJECT_ALLOCATION_FAILURE : ret;
            device->completion_buffer = buffer;
        }
        // Non-overlapping regions of one private buffer provide an in-order
        // completion event without a blocking wait or a host readback lifetime.
        return momoten_detail::g_opencl.p_clEnqueueCopyBuffer(device->command_queue, device->completion_buffer, device->completion_buffer, 0, sizeof(cl_uint), sizeof(cl_uint), 0, 0, &event);
    }
    return momoten_detail::g_opencl.p_clEnqueueMarker(
        state.device->command_queue, &event);
}

void abort_replay(ReplayState& state)
{
    state.touched_memories.clear();
}
