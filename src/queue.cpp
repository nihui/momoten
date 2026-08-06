// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/device.h"
#include "backend/memory_sync.h"
#include "backend/objects.h"
#include "backend/replay.h"
#include "backend/runtime.h"
#include "vulkan_internal.h"

namespace momoten_detail {

VkResult impl_queue_submit(
    VkQueue queue, uint32_t submit_count, const VkSubmitInfo* submits, VkFence fence_handle)
{
    if (!queue || (submit_count && !submits))
        return VK_ERROR_INITIALIZATION_FAILED;
    VkDevice device = queue->device;
    std::lock_guard<std::mutex> queue_lock(device->queue_mutex);
    retire_pending_submissions(device, false);

    const std::shared_ptr<Fence> fence = fence_handle
                                             ? g_fences.get_handle(fence_handle)
                                             : std::shared_ptr<Fence>();
    if (fence_handle && (!fence || fence->device != device))
        return VK_ERROR_INITIALIZATION_FAILED;

    std::vector<VkCommandBuffer> submitted_buffers;
    for (uint32_t s = 0; s < submit_count; s++)
    {
        if (submits[s].waitSemaphoreCount != 0 || submits[s].signalSemaphoreCount != 0 || (submits[s].commandBufferCount && !submits[s].pCommandBuffers))
            return submits[s].waitSemaphoreCount || submits[s].signalSemaphoreCount
                       ? VK_ERROR_FEATURE_NOT_PRESENT
                       : VK_ERROR_INITIALIZATION_FAILED;
        for (uint32_t b = 0; b < submits[s].commandBufferCount; b++)
        {
            VkCommandBuffer command_buffer = submits[s].pCommandBuffers[b];
            const bool simultaneous = command_buffer && (command_buffer->usage & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT) != 0;
            if (!command_buffer || command_buffer->device != device || (command_buffer->state != momoten_detail::COMMAND_BUFFER_STATE_EXECUTABLE && !(simultaneous && command_buffer->state == momoten_detail::COMMAND_BUFFER_STATE_PENDING)))
                return VK_ERROR_INITIALIZATION_FAILED;
            submitted_buffers.push_back(command_buffer);
        }
    }

    prepare_mapped_memory_for_submit(device);
    ReplayState replay(device);
    for (uint32_t s = 0; s < submit_count; s++)
    {
        for (uint32_t b = 0; b < submits[s].commandBufferCount; b++)
        {
            VkCommandBuffer command_buffer = submits[s].pCommandBuffers[b];
            for (size_t c = 0; c < command_buffer->commands.size(); c++)
            {
                const RecordedCommand& command = command_buffer->commands[c];
                cl_int ret = CL_SUCCESS;
                if (command.type == RecordedCommand::CopyBuffer)
                    ret = replay_copy(replay, command);
                else if (command.type == RecordedCommand::Dispatch)
                    ret = replay_dispatch(replay, command);
                else
                    ret = replay_barrier(replay, command);
                if (ret != CL_SUCCESS)
                {
                    abort_replay(replay);
                    log_error("command replay", ret);
                    return VK_ERROR_DEVICE_LOST;
                }
            }
        }
    }

    cl_event event = 0;
    cl_int ret = finish_replay(replay, event);
    if (ret != CL_SUCCESS)
    {
        abort_replay(replay);
        log_error("finish command replay", ret);
        return VK_ERROR_DEVICE_LOST;
    }
    ret = momoten_detail::g_opencl.p_clFlush(device->command_queue);
    if (ret != CL_SUCCESS)
    {
        momoten_detail::g_opencl.p_clReleaseEvent(event);
        log_error("clFlush", ret);
        return VK_ERROR_DEVICE_LOST;
    }

    PendingSubmission pending;
    pending.event = event;
    pending.command_buffers.swap(submitted_buffers);
    for (size_t i = 0; i < pending.command_buffers.size(); i++)
    {
        pending.command_buffers[i]->pending_count++;
        pending.command_buffers[i]->state = momoten_detail::COMMAND_BUFFER_STATE_PENDING;
    }
    device->pending_submissions.push_back(pending);
    assign_fence_event(fence, event);
    return VK_SUCCESS;
}

VkResult impl_queue_wait_idle(VkQueue queue)
{
    if (!queue)
        return VK_ERROR_INITIALIZATION_FAILED;
    VkDevice device = queue->device;
    std::lock_guard<std::mutex> queue_lock(device->queue_mutex);
    const cl_int ret = momoten_detail::g_opencl.p_clFinish(device->command_queue);
    if (ret == CL_SUCCESS)
        retire_pending_submissions(device, true);
    return ret == CL_SUCCESS ? VK_SUCCESS : VK_ERROR_DEVICE_LOST;
}

VkResult impl_device_wait_idle(VkDevice device)
{
    if (!device)
        return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard<std::mutex> queue_lock(device->queue_mutex);
    const cl_int ret = momoten_detail::g_opencl.p_clFinish(device->command_queue);
    if (ret == CL_SUCCESS)
        retire_pending_submissions(device, true);
    return ret == CL_SUCCESS ? VK_SUCCESS : VK_ERROR_DEVICE_LOST;
}

} // namespace momoten_detail
