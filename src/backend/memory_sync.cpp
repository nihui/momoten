// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "memory_sync.h"
#include "replay.h"

#include <algorithm>

static void remember_touched_memory(ReplayState& state,
                                    const std::shared_ptr<DeviceMemory>& memory)
{
    if (std::find(state.touched_memories.begin(), state.touched_memories.end(),
                  memory)
        == state.touched_memories.end())
        state.touched_memories.push_back(memory);
}

void mark_host_range(const std::shared_ptr<DeviceMemory>& memory,
                     VkDeviceSize offset, VkDeviceSize size)
{
    if (!memory || !memory->host_visible || size == 0)
        return;
    std::lock_guard<std::mutex> lock(memory->mutex);
    memory->range_map.set(offset, size, momoten_detail::MEMORY_AUTHORITY_HOST);
}

void prepare_mapped_memory_for_submit(VkDevice device)
{
    std::vector<std::shared_ptr<DeviceMemory> > allocations;
    {
        std::lock_guard<std::mutex> lock(device->allocation_mutex);
        for (size_t i = 0; i < device->allocations.size();)
        {
            const std::shared_ptr<DeviceMemory> memory = device->allocations[i].lock();
            if (!memory)
            {
                device->allocations.erase(device->allocations.begin() + i);
                continue;
            }
            allocations.push_back(memory);
            i++;
        }
    }

    for (size_t i = 0; i < allocations.size(); i++)
    {
        const std::shared_ptr<DeviceMemory>& memory = allocations[i];
        std::lock_guard<std::mutex> lock(memory->mutex);
        if (memory->mapped)
            memory->range_map.replace(
                memory->mapped_offset, memory->mapped_size,
                momoten_detail::MEMORY_AUTHORITY_SYNCHRONIZED,
                momoten_detail::MEMORY_AUTHORITY_HOST);
    }
}

cl_int upload_host_ranges(ReplayState& state,
                          const std::shared_ptr<DeviceMemory>& memory,
                          size_t offset, size_t size)
{
    if (!memory->host_visible || size == 0)
        return CL_SUCCESS;

    std::vector<momoten_detail::MemoryRangeState> ranges;
    {
        std::lock_guard<std::mutex> lock(memory->mutex);
        ranges = memory->range_map.collect(
            offset, size, momoten_detail::MEMORY_AUTHORITY_HOST);
    }

    for (size_t i = 0; i < ranges.size(); i++)
    {
        const size_t begin = static_cast<size_t>(ranges[i].begin);
        const size_t byte_count = static_cast<size_t>(ranges[i].end - ranges[i].begin);
        cl_event event = 0;
        const cl_uint wait_count = state.tail_event ? 1u : 0u;
        const cl_event* wait_list = state.tail_event ? &state.tail_event : 0;
        const cl_int ret = momoten_detail::g_opencl.p_clEnqueueWriteBuffer(
            state.device->command_queue, memory->memory, CL_FALSE,
            begin, byte_count, memory->shadow.data() + begin,
            wait_count, wait_list, &event);
        if (ret != CL_SUCCESS)
            return ret;
        advance_replay_event(state, event);
        {
            std::lock_guard<std::mutex> lock(memory->mutex);
            memory->range_map.set(
                ranges[i].begin, ranges[i].end - ranges[i].begin,
                momoten_detail::MEMORY_AUTHORITY_SYNCHRONIZED);
        }
    }
    return CL_SUCCESS;
}

static cl_int enqueue_readbacks(ReplayState& state,
                                const std::shared_ptr<DeviceMemory>& memory,
                                VkDeviceSize offset, VkDeviceSize size)
{
    if (!memory->host_visible || size == 0)
        return CL_SUCCESS;

    std::vector<momoten_detail::MemoryRangeState> ranges;
    {
        std::lock_guard<std::mutex> lock(memory->mutex);
        ranges = memory->range_map.collect(
            offset, size, momoten_detail::MEMORY_AUTHORITY_DEVICE);
    }

    for (size_t i = 0; i < ranges.size(); i++)
    {
        const size_t begin = static_cast<size_t>(ranges[i].begin);
        const size_t byte_count = static_cast<size_t>(ranges[i].end - ranges[i].begin);
        cl_event event = 0;
        const cl_uint wait_count = state.tail_event ? 1u : 0u;
        const cl_event* wait_list = state.tail_event ? &state.tail_event : 0;
        const cl_int ret = momoten_detail::g_opencl.p_clEnqueueReadBuffer(
            state.device->command_queue, memory->memory, CL_FALSE,
            begin, byte_count, memory->shadow.data() + begin,
            wait_count, wait_list, &event);
        if (ret != CL_SUCCESS)
            return ret;
        advance_replay_event(state, event);
        {
            std::lock_guard<std::mutex> lock(memory->mutex);
            memory->range_map.set(
                ranges[i].begin, ranges[i].end - ranges[i].begin,
                momoten_detail::MEMORY_AUTHORITY_SYNCHRONIZED);
        }
    }
    return CL_SUCCESS;
}

void mark_device_range(ReplayState& state,
                       const std::shared_ptr<DeviceMemory>& memory,
                       VkDeviceSize offset, VkDeviceSize size)
{
    {
        std::lock_guard<std::mutex> lock(memory->mutex);
        memory->range_map.set(
            offset, size, momoten_detail::MEMORY_AUTHORITY_DEVICE);
    }
    remember_touched_memory(state, memory);
}

cl_int synchronize_memory_to_host(VkDevice device,
                                  const std::shared_ptr<DeviceMemory>& memory,
                                  VkDeviceSize offset, VkDeviceSize size)
{
    if (!memory || !memory->host_visible || size == 0)
        return CL_SUCCESS;

    std::vector<momoten_detail::MemoryRangeState> ranges;
    {
        std::lock_guard<std::mutex> lock(memory->mutex);
        ranges = memory->range_map.collect(
            offset, size, momoten_detail::MEMORY_AUTHORITY_DEVICE);
    }
    for (size_t i = 0; i < ranges.size(); i++)
    {
        const size_t begin = static_cast<size_t>(ranges[i].begin);
        const size_t byte_count = static_cast<size_t>(ranges[i].end - ranges[i].begin);
        const cl_int ret = momoten_detail::g_opencl.p_clEnqueueReadBuffer(
            device->command_queue, memory->memory, CL_TRUE,
            begin, byte_count, memory->shadow.data() + begin, 0, 0, 0);
        if (ret != CL_SUCCESS)
            return ret;
        std::lock_guard<std::mutex> lock(memory->mutex);
        memory->range_map.set(
            ranges[i].begin, ranges[i].end - ranges[i].begin,
            momoten_detail::MEMORY_AUTHORITY_SYNCHRONIZED);
    }
    return CL_SUCCESS;
}

cl_int finish_mapped_memory_readbacks(ReplayState& state)
{
    for (size_t i = 0; i < state.touched_memories.size(); i++)
    {
        const std::shared_ptr<DeviceMemory>& memory = state.touched_memories[i];
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
        {
            std::lock_guard<std::mutex> lock(memory->mutex);
            if (memory->mapped)
            {
                offset = memory->mapped_offset;
                size = memory->mapped_size;
            }
        }
        const cl_int ret = enqueue_readbacks(state, memory, offset, size);
        if (ret != CL_SUCCESS)
            return ret;
    }
    return CL_SUCCESS;
}
