// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/device.h"
#include "backend/memory_sync.h"
#include "backend/objects.h"
#include "vulkan_internal.h"

momoten_detail::HandleTable<VkDeviceMemory, DeviceMemory> g_device_memories;
momoten_detail::HandleTable<VkBuffer, Buffer> g_buffers;

DeviceMemory::~DeviceMemory()
{
    if (memory)
        momoten_detail::g_opencl.p_clReleaseMemObject(memory);
}

namespace momoten_detail {

VkResult impl_allocate_memory(
    VkDevice device, const VkMemoryAllocateInfo* info,
    const VkAllocationCallbacks* allocator, VkDeviceMemory* memory)
{
    if (!device || !info || !memory
        || info->sType != VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (allocator || info->pNext)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (info->allocationSize == 0 || info->memoryTypeIndex >= 2
        || info->allocationSize > static_cast<VkDeviceSize>(SIZE_MAX)
        || info->allocationSize > device->physical_device->max_allocation)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    cl_int ret = CL_SUCCESS;
    cl_mem cl_memory = momoten_detail::g_opencl.p_clCreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                                 static_cast<size_t>(info->allocationSize), 0, &ret);
    if (!cl_memory)
    {
        log_error("clCreateBuffer", ret);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    std::shared_ptr<DeviceMemory> value(new DeviceMemory);
    value->device = device;
    value->size = info->allocationSize;
    value->memory_type_index = info->memoryTypeIndex;
    value->host_visible = info->memoryTypeIndex == 1;
    value->mapped = false;
    value->mapped_offset = 0;
    value->mapped_size = 0;
    value->memory = cl_memory;
    value->range_map.reset(value->size);
    if (value->host_visible)
        value->shadow.resize(static_cast<size_t>(value->size));
    {
        std::lock_guard<std::mutex> lock(device->allocation_mutex);
        device->allocations.push_back(value);
    }
    *memory = g_device_memories.make_handle(value);
    return VK_SUCCESS;
}

void impl_free_memory(
    VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks*)
{
    const std::shared_ptr<DeviceMemory> value = g_device_memories.get_handle(memory);
    if (value && value->device == device)
        g_device_memories.erase_handle(memory);
}

VkResult impl_map_memory(
    VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size,
    VkMemoryMapFlags flags, void** data)
{
    const std::shared_ptr<DeviceMemory> value = g_device_memories.get_handle(memory);
    if (!value || value->device != device || !data || !value->host_visible
        || flags != 0 || offset >= value->size)
        return VK_ERROR_MEMORY_MAP_FAILED;
    const VkDeviceSize mapped_size = size == VK_WHOLE_SIZE
                                         ? value->size - offset
                                         : size;
    if (mapped_size == 0 || mapped_size > value->size - offset
        || offset > static_cast<VkDeviceSize>(SIZE_MAX))
        return VK_ERROR_MEMORY_MAP_FAILED;
    {
        std::lock_guard<std::mutex> lock(value->mutex);
        if (value->mapped)
            return VK_ERROR_MEMORY_MAP_FAILED;
    }

    std::lock_guard<std::mutex> queue_lock(device->queue_mutex);
    const cl_int ret = synchronize_memory_to_host(
        device, value, offset, mapped_size);
    if (ret != CL_SUCCESS)
    {
        log_error("map memory readback", ret);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    {
        std::lock_guard<std::mutex> lock(value->mutex);
        value->mapped = true;
        value->mapped_offset = offset;
        value->mapped_size = mapped_size;
        value->range_map.set(
            offset, mapped_size, momoten_detail::MEMORY_AUTHORITY_HOST);
    }
    *data = value->shadow.data() + static_cast<size_t>(offset);
    return VK_SUCCESS;
}

void impl_unmap_memory(VkDevice device, VkDeviceMemory memory)
{
    const std::shared_ptr<DeviceMemory> value = g_device_memories.get_handle(memory);
    if (!value || value->device != device)
        return;
    std::lock_guard<std::mutex> lock(value->mutex);
    if (!value->mapped)
        return;
    value->range_map.set(
        value->mapped_offset, value->mapped_size,
        momoten_detail::MEMORY_AUTHORITY_HOST);
    value->mapped = false;
    value->mapped_offset = 0;
    value->mapped_size = 0;
}

struct ResolvedMappedRange
{
    std::shared_ptr<DeviceMemory> memory;
    VkDeviceSize offset;
    VkDeviceSize size;
};

static bool resolve_mapped_range(
    VkDevice device, const VkMappedMemoryRange& range,
    ResolvedMappedRange& resolved)
{
    if (range.sType != VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE)
        return false;
    const std::shared_ptr<DeviceMemory> memory = g_device_memories.get_handle(range.memory);
    if (!memory || memory->device != device || !memory->host_visible
        || range.offset >= memory->size)
        return false;
    const VkDeviceSize size = range.size == VK_WHOLE_SIZE
                                  ? memory->size - range.offset
                                  : range.size;
    if (size == 0 || size > memory->size - range.offset)
        return false;

    std::lock_guard<std::mutex> lock(memory->mutex);
    if (!memory->mapped || range.offset < memory->mapped_offset
        || range.offset - memory->mapped_offset > memory->mapped_size
        || size > memory->mapped_size - (range.offset - memory->mapped_offset))
        return false;
    resolved.memory = memory;
    resolved.offset = range.offset;
    resolved.size = size;
    return true;
}

VkResult impl_flush_mapped_memory_ranges(
    VkDevice device, uint32_t count, const VkMappedMemoryRange* ranges)
{
    if (count && !ranges)
        return VK_ERROR_MEMORY_MAP_FAILED;
    std::vector<ResolvedMappedRange> resolved(count);
    for (uint32_t i = 0; i < count; i++)
    {
        if (!resolve_mapped_range(device, ranges[i], resolved[i]))
            return VK_ERROR_MEMORY_MAP_FAILED;
    }
    for (uint32_t i = 0; i < count; i++)
        mark_host_range(resolved[i].memory, resolved[i].offset, resolved[i].size);
    return VK_SUCCESS;
}

VkResult impl_invalidate_mapped_memory_ranges(
    VkDevice device, uint32_t count, const VkMappedMemoryRange* ranges)
{
    if (count && !ranges)
        return VK_ERROR_MEMORY_MAP_FAILED;
    std::vector<ResolvedMappedRange> resolved(count);
    for (uint32_t i = 0; i < count; i++)
    {
        if (!resolve_mapped_range(device, ranges[i], resolved[i]))
            return VK_ERROR_MEMORY_MAP_FAILED;
    }

    std::lock_guard<std::mutex> queue_lock(device->queue_mutex);
    for (uint32_t i = 0; i < count; i++)
    {
        const cl_int ret = synchronize_memory_to_host(
            device, resolved[i].memory, resolved[i].offset, resolved[i].size);
        if (ret != CL_SUCCESS)
        {
            log_error("invalidate memory readback", ret);
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
    }
    return VK_SUCCESS;
}

void impl_get_device_memory_commitment(
    VkDevice device, VkDeviceMemory memory, VkDeviceSize* committed)
{
    if (!committed)
        return;
    const std::shared_ptr<DeviceMemory> value = g_device_memories.get_handle(memory);
    *committed = value && value->device == device ? value->size : 0;
}

VkResult impl_create_buffer(
    VkDevice device, const VkBufferCreateInfo* info,
    const VkAllocationCallbacks* allocator, VkBuffer* buffer)
{
    if (!device || !info || !buffer
        || info->sType != VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO
        || info->size == 0)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (allocator || info->pNext || info->flags
        || info->sharingMode != VK_SHARING_MODE_EXCLUSIVE)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    const VkBufferUsageFlags supported_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                               | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (info->usage == 0 || (info->usage & ~supported_usage) != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    std::shared_ptr<Buffer> value(new Buffer);
    value->device = device;
    value->size = info->size;
    value->usage = info->usage;
    value->memory_offset = 0;
    *buffer = g_buffers.make_handle(value);
    return VK_SUCCESS;
}

void impl_destroy_buffer(
    VkDevice device, VkBuffer buffer, const VkAllocationCallbacks*)
{
    const std::shared_ptr<Buffer> value = g_buffers.get_handle(buffer);
    if (value && value->device == device)
        g_buffers.erase_handle(buffer);
}

void impl_get_buffer_memory_requirements(
    VkDevice device, VkBuffer buffer, VkMemoryRequirements* requirements)
{
    const std::shared_ptr<Buffer> value = g_buffers.get_handle(buffer);
    if (!value || value->device != device || !requirements)
        return;
    requirements->size = value->size;
    requirements->alignment = 16;
    requirements->memoryTypeBits = 3;
}

VkResult impl_bind_buffer_memory(
    VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset)
{
    const std::shared_ptr<Buffer> buffer_value = g_buffers.get_handle(buffer);
    const std::shared_ptr<DeviceMemory> memory_value = g_device_memories.get_handle(memory);
    if (!device || !buffer_value || !memory_value || buffer_value->device != device || memory_value->device != device || offset > memory_value->size || buffer_value->size > memory_value->size - offset)
        return VK_ERROR_MEMORY_MAP_FAILED;
    buffer_value->memory = memory_value;
    buffer_value->memory_offset = offset;
    return VK_SUCCESS;
}

VkResult impl_bind_buffer_memory2(
    VkDevice device, uint32_t bind_info_count, const VkBindBufferMemoryInfo* bind_infos)
{
    if (bind_info_count && !bind_infos)
        return VK_ERROR_INITIALIZATION_FAILED;
    for (uint32_t i = 0; i < bind_info_count; i++)
    {
        const VkResult result = impl_bind_buffer_memory(
            device, bind_infos[i].buffer, bind_infos[i].memory, bind_infos[i].memoryOffset);
        if (result != VK_SUCCESS)
            return result;
    }
    return VK_SUCCESS;
}

void impl_get_buffer_memory_requirements2(
    VkDevice device, const VkBufferMemoryRequirementsInfo2* info,
    VkMemoryRequirements2* requirements)
{
    if (info && requirements)
        impl_get_buffer_memory_requirements(device, info->buffer, &requirements->memoryRequirements);
}

} // namespace momoten_detail
