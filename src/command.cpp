// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/objects.h"
#include "vulkan_internal.h"

#include <cstring>

momoten_detail::HandleTable<VkCommandPool, CommandPool> g_command_pools;

CommandPool::~CommandPool()
{
    for (size_t i = 0; i < buffers.size(); i++)
        delete buffers[i];
}

namespace momoten_detail {

VkResult impl_create_command_pool(
    VkDevice device, const VkCommandPoolCreateInfo* info,
    const VkAllocationCallbacks* allocator, VkCommandPool* pool)
{
    if (!device || !info || !pool
        || info->sType != VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO
        || info->queueFamilyIndex != 0)
        return VK_ERROR_INITIALIZATION_FAILED;
    const VkCommandPoolCreateFlags supported_flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
                                                     | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (allocator || info->pNext || (info->flags & ~supported_flags))
        return VK_ERROR_FEATURE_NOT_PRESENT;
    std::shared_ptr<CommandPool> value(new CommandPool);
    value->device = device;
    *pool = g_command_pools.make_handle(value);
    return VK_SUCCESS;
}

void impl_destroy_command_pool(
    VkDevice device, VkCommandPool pool, const VkAllocationCallbacks*)
{
    const std::shared_ptr<CommandPool> value = g_command_pools.get_handle(pool);
    if (value && value->device == device)
        g_command_pools.erase_handle(pool);
}

VkResult impl_allocate_command_buffers(
    VkDevice device, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* buffers)
{
    const std::shared_ptr<CommandPool> pool = info
                                                  ? g_command_pools.get_handle(info->commandPool)
                                                  : std::shared_ptr<CommandPool>();
    if (!device || !info || !pool || pool->device != device || !buffers
        || info->sType != VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
        || info->commandBufferCount == 0
        || info->level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (info->pNext)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    for (uint32_t i = 0; i < info->commandBufferCount; i++)
    {
        VkCommandBuffer_T* buffer = new VkCommandBuffer_T;
        buffer->device = device;
        buffer->pool = pool;
        buffer->state = momoten_detail::COMMAND_BUFFER_STATE_INITIAL;
        buffer->error = VK_SUCCESS;
        buffer->usage = 0;
        buffer->pending_count = 0;
        pool->buffers.push_back(buffer);
        buffers[i] = buffer;
    }
    return VK_SUCCESS;
}

void impl_free_command_buffers(
    VkDevice device, VkCommandPool pool_handle, uint32_t count, const VkCommandBuffer* buffers)
{
    const std::shared_ptr<CommandPool> pool = g_command_pools.get_handle(pool_handle);
    if (!pool || pool->device != device || !buffers)
        return;
    for (uint32_t i = 0; i < count; i++)
    {
        std::vector<VkCommandBuffer>::iterator it = std::find(pool->buffers.begin(), pool->buffers.end(), buffers[i]);
        if (it != pool->buffers.end())
        {
            if ((*it)->state == momoten_detail::COMMAND_BUFFER_STATE_PENDING)
                continue;
            delete *it;
            pool->buffers.erase(it);
        }
    }
}

static void reset_command_buffer_state(VkCommandBuffer buffer)
{
    buffer->state = momoten_detail::COMMAND_BUFFER_STATE_INITIAL;
    buffer->error = VK_SUCCESS;
    buffer->usage = 0;
    buffer->pending_count = 0;
    buffer->current_pipeline.reset();
    buffer->current_descriptors.clear();
    buffer->current_push_constants.clear();
    buffer->commands.clear();
}

VkResult impl_reset_command_pool(
    VkDevice device, VkCommandPool pool_handle, VkCommandPoolResetFlags)
{
    const std::shared_ptr<CommandPool> pool = g_command_pools.get_handle(pool_handle);
    if (!pool || pool->device != device)
        return VK_ERROR_INITIALIZATION_FAILED;
    for (size_t i = 0; i < pool->buffers.size(); i++)
    {
        if (pool->buffers[i]->state == momoten_detail::COMMAND_BUFFER_STATE_PENDING)
            return VK_ERROR_INITIALIZATION_FAILED;
    }
    for (size_t i = 0; i < pool->buffers.size(); i++)
        reset_command_buffer_state(pool->buffers[i]);
    return VK_SUCCESS;
}

void impl_trim_command_pool(
    VkDevice, VkCommandPool, VkCommandPoolTrimFlags)
{
}

VkResult impl_reset_command_buffer(
    VkCommandBuffer buffer, VkCommandBufferResetFlags)
{
    if (!buffer || buffer->state == momoten_detail::COMMAND_BUFFER_STATE_PENDING)
        return VK_ERROR_INITIALIZATION_FAILED;
    reset_command_buffer_state(buffer);
    return VK_SUCCESS;
}

VkResult impl_begin_command_buffer(
    VkCommandBuffer buffer, const VkCommandBufferBeginInfo* info)
{
    if (!buffer || !info
        || info->sType != VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        || buffer->state == momoten_detail::COMMAND_BUFFER_STATE_RECORDING
        || buffer->state == momoten_detail::COMMAND_BUFFER_STATE_PENDING)
        return VK_ERROR_INITIALIZATION_FAILED;
    const VkCommandBufferUsageFlags supported_flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                                                      | VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    if (info->pNext || info->pInheritanceInfo
        || (info->flags & ~supported_flags))
        return VK_ERROR_FEATURE_NOT_PRESENT;
    reset_command_buffer_state(buffer);
    buffer->state = momoten_detail::COMMAND_BUFFER_STATE_RECORDING;
    buffer->usage = info->flags;
    return VK_SUCCESS;
}

VkResult impl_end_command_buffer(VkCommandBuffer buffer)
{
    if (!buffer)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (buffer->state != momoten_detail::COMMAND_BUFFER_STATE_RECORDING)
        return buffer->error == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : buffer->error;
    if (buffer->error != VK_SUCCESS)
    {
        buffer->state = momoten_detail::COMMAND_BUFFER_STATE_INVALID;
        return buffer->error;
    }
    buffer->state = momoten_detail::COMMAND_BUFFER_STATE_EXECUTABLE;
    return VK_SUCCESS;
}

static bool command_buffer_is_recording(VkCommandBuffer buffer)
{
    return buffer && buffer->state == momoten_detail::COMMAND_BUFFER_STATE_RECORDING;
}

static void record_command_buffer_error(VkCommandBuffer buffer, VkResult error)
{
    if (!buffer || buffer->error != VK_SUCCESS)
        return;
    buffer->error = error;
    if (buffer->state != momoten_detail::COMMAND_BUFFER_STATE_RECORDING)
        buffer->state = momoten_detail::COMMAND_BUFFER_STATE_INVALID;
}

template<typename Function>
static void record_command(VkCommandBuffer buffer, const Function& function)
{
    try
    {
        function();
    }
    catch (const std::bad_alloc&)
    {
        record_command_buffer_error(buffer, VK_ERROR_OUT_OF_HOST_MEMORY);
    }
    catch (...)
    {
        record_command_buffer_error(buffer, VK_ERROR_INITIALIZATION_FAILED);
    }
}

void impl_cmd_bind_pipeline(
    VkCommandBuffer buffer, VkPipelineBindPoint bind_point, VkPipeline pipeline)
{
    if (!command_buffer_is_recording(buffer) || bind_point != VK_PIPELINE_BIND_POINT_COMPUTE)
    {
        record_command_buffer_error(buffer, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    const std::shared_ptr<Pipeline> value = g_pipelines.get_handle(pipeline);
    if (!value || value->device != buffer->device)
    {
        record_command_buffer_error(buffer, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    buffer->current_pipeline = value;
}

void impl_cmd_push_descriptor_set(
    VkCommandBuffer buffer, VkPipelineBindPoint bind_point, VkPipelineLayout layout_handle,
    uint32_t set, uint32_t write_count, const VkWriteDescriptorSet* writes)
{
    record_command(buffer, [&]() {
        const std::shared_ptr<PipelineLayout> layout = g_pipeline_layouts.get_handle(layout_handle);
        if (!command_buffer_is_recording(buffer) || bind_point != VK_PIPELINE_BIND_POINT_COMPUTE || !layout || layout->device != buffer->device || set != 0 || (write_count && !writes))
        {
            record_command_buffer_error(buffer, VK_ERROR_INITIALIZATION_FAILED);
            return;
        }
        for (uint32_t w = 0; w < write_count; w++)
        {
            const VkWriteDescriptorSet& write = writes[w];
            if (write.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || !write.pBufferInfo)
            {
                record_command_buffer_error(buffer, VK_ERROR_FEATURE_NOT_PRESENT);
                return;
            }
            for (uint32_t i = 0; i < write.descriptorCount; i++)
            {
                DescriptorValue value;
                value.buffer = g_buffers.get_handle(write.pBufferInfo[i].buffer);
                if (!value.buffer || value.buffer->device != buffer->device)
                {
                    record_command_buffer_error(buffer, VK_ERROR_INITIALIZATION_FAILED);
                    return;
                }
                value.offset = write.pBufferInfo[i].offset;
                value.range = write.pBufferInfo[i].range;
                buffer->current_descriptors[write.dstBinding + i] = value;
            }
        }
    });
}

void impl_cmd_push_descriptor_set_with_template(
    VkCommandBuffer buffer, VkDescriptorUpdateTemplate update_template_handle,
    VkPipelineLayout layout_handle, uint32_t set, const void* data)
{
    record_command(buffer, [&]() {
        const std::shared_ptr<DescriptorUpdateTemplate> update_template = g_descriptor_update_templates.get_handle(update_template_handle);
        const std::shared_ptr<PipelineLayout> layout = g_pipeline_layouts.get_handle(layout_handle);
        if (!command_buffer_is_recording(buffer) || !update_template || !layout || update_template->device != buffer->device || layout->device != buffer->device || set != 0 || !data)
        {
            record_command_buffer_error(buffer, VK_ERROR_INITIALIZATION_FAILED);
            return;
        }
        const unsigned char* bytes = static_cast<const unsigned char*>(data);
        for (size_t e = 0; e < update_template->entries.size(); e++)
        {
            const VkDescriptorUpdateTemplateEntry& entry = update_template->entries[e];
            if (entry.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                continue;
            for (uint32_t i = 0; i < entry.descriptorCount; i++)
            {
                const size_t stride = entry.stride ? entry.stride : sizeof(VkDescriptorBufferInfo);
                const VkDescriptorBufferInfo* info = reinterpret_cast<const VkDescriptorBufferInfo*>(
                    bytes + entry.offset + stride * i);
                DescriptorValue value;
                value.buffer = g_buffers.get_handle(info->buffer);
                if (!value.buffer || value.buffer->device != buffer->device)
                {
                    record_command_buffer_error(buffer, VK_ERROR_INITIALIZATION_FAILED);
                    return;
                }
                value.offset = info->offset;
                value.range = info->range;
                buffer->current_descriptors[entry.dstBinding + i] = value;
            }
        }
    });
}

void impl_cmd_push_constants(
    VkCommandBuffer buffer, VkPipelineLayout layout_handle, VkShaderStageFlags stage_flags,
    uint32_t offset, uint32_t size, const void* values)
{
    record_command(buffer, [&]() {
        const std::shared_ptr<PipelineLayout> layout = g_pipeline_layouts.get_handle(layout_handle);
        if (!command_buffer_is_recording(buffer) || !layout || layout->device != buffer->device || stage_flags != VK_SHADER_STAGE_COMPUTE_BIT || !values || offset > layout->push_constant_size || size > layout->push_constant_size - offset)
        {
            record_command_buffer_error(buffer, VK_ERROR_INITIALIZATION_FAILED);
            return;
        }
        if (buffer->current_push_constants.size() < layout->push_constant_size)
            buffer->current_push_constants.resize(layout->push_constant_size);
        memcpy(buffer->current_push_constants.data() + offset, values, size);
    });
}

void impl_cmd_dispatch(
    VkCommandBuffer buffer, uint32_t x, uint32_t y, uint32_t z)
{
    record_command(buffer, [&]() {
        if (!command_buffer_is_recording(buffer) || !buffer->current_pipeline)
        {
            record_command_buffer_error(buffer, VK_ERROR_INITIALIZATION_FAILED);
            return;
        }
        RecordedCommand command;
        command.type = RecordedCommand::Dispatch;
        command.pipeline = buffer->current_pipeline;
        command.descriptors = buffer->current_descriptors;
        command.push_constants = buffer->current_push_constants;
        command.group_count[0] = x;
        command.group_count[1] = y;
        command.group_count[2] = z;
        buffer->commands.push_back(command);
    });
}

void impl_cmd_copy_buffer(
    VkCommandBuffer buffer, VkBuffer src, VkBuffer dst,
    uint32_t region_count, const VkBufferCopy* regions)
{
    record_command(buffer, [&]() {
        const std::shared_ptr<Buffer> src_buffer = g_buffers.get_handle(src);
        const std::shared_ptr<Buffer> dst_buffer = g_buffers.get_handle(dst);
        if (!command_buffer_is_recording(buffer) || !src_buffer || !dst_buffer || src_buffer->device != buffer->device || dst_buffer->device != buffer->device || (region_count && !regions))
        {
            record_command_buffer_error(buffer, VK_ERROR_INITIALIZATION_FAILED);
            return;
        }
        RecordedCommand command;
        command.type = RecordedCommand::CopyBuffer;
        command.src_buffer = src_buffer;
        command.dst_buffer = dst_buffer;
        command.copy_regions.assign(regions, regions + region_count);
        buffer->commands.push_back(command);
    });
}

void impl_cmd_pipeline_barrier(
    VkCommandBuffer buffer, VkPipelineStageFlags src_stage_mask,
    VkPipelineStageFlags dst_stage_mask, VkDependencyFlags dependency_flags,
    uint32_t memory_barrier_count, const VkMemoryBarrier* memory_barriers,
    uint32_t buffer_barrier_count, const VkBufferMemoryBarrier* buffer_barriers,
    uint32_t image_barrier_count, const VkImageMemoryBarrier*)
{
    record_command(buffer, [&]() {
        if (!command_buffer_is_recording(buffer)
            || (memory_barrier_count && !memory_barriers)
            || (buffer_barrier_count && !buffer_barriers))
        {
            record_command_buffer_error(buffer, VK_ERROR_INITIALIZATION_FAILED);
            return;
        }
        if (image_barrier_count
            || (dependency_flags & ~VK_DEPENDENCY_BY_REGION_BIT) != 0)
        {
            record_command_buffer_error(buffer, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
        }

        RecordedCommand command;
        command.type = RecordedCommand::Barrier;
        command.src_stage_mask = src_stage_mask;
        command.dst_stage_mask = dst_stage_mask;
        command.memory_barriers.reserve(memory_barrier_count);
        for (uint32_t i = 0; i < memory_barrier_count; i++)
        {
            if (memory_barriers[i].sType
                != VK_STRUCTURE_TYPE_MEMORY_BARRIER)
            {
                record_command_buffer_error(
                    buffer, VK_ERROR_INITIALIZATION_FAILED);
                return;
            }
            RecordedMemoryBarrier barrier;
            barrier.src_access_mask = memory_barriers[i].srcAccessMask;
            barrier.dst_access_mask = memory_barriers[i].dstAccessMask;
            command.memory_barriers.push_back(barrier);
        }

        command.buffer_barriers.reserve(buffer_barrier_count);
        for (uint32_t i = 0; i < buffer_barrier_count; i++)
        {
            const VkBufferMemoryBarrier& source = buffer_barriers[i];
            const std::shared_ptr<Buffer> value = g_buffers.get_handle(source.buffer);
            if (source.sType
                    != VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER
                || !value || value->device != buffer->device
                || source.offset >= value->size
                || (source.srcQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED
                    && source.srcQueueFamilyIndex != 0)
                || (source.dstQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED
                    && source.dstQueueFamilyIndex != 0))
            {
                record_command_buffer_error(
                    buffer, VK_ERROR_INITIALIZATION_FAILED);
                return;
            }
            const VkDeviceSize size = source.size == VK_WHOLE_SIZE
                                          ? value->size - source.offset
                                          : source.size;
            if (size == 0 || size > value->size - source.offset)
            {
                record_command_buffer_error(
                    buffer, VK_ERROR_INITIALIZATION_FAILED);
                return;
            }

            RecordedBufferBarrier barrier;
            barrier.src_access_mask = source.srcAccessMask;
            barrier.dst_access_mask = source.dstAccessMask;
            barrier.buffer = value;
            barrier.offset = source.offset;
            barrier.size = size;
            command.buffer_barriers.push_back(barrier);
        }
        buffer->commands.push_back(command);
    });
}

} // namespace momoten_detail
