// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/objects.h"
#include "backend/runtime.h"
#include "vulkan_internal.h"

#include <chrono>
#include <thread>

namespace momoten_detail {

VkResult impl_create_fence(
    VkDevice device, const VkFenceCreateInfo* info,
    const VkAllocationCallbacks* allocator, VkFence* fence)
{
    if (!device || !info || !fence
        || info->sType != VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (allocator || info->pNext
        || (info->flags & ~VK_FENCE_CREATE_SIGNALED_BIT))
        return VK_ERROR_FEATURE_NOT_PRESENT;
    std::shared_ptr<Fence> value(new Fence);
    value->device = device;
    value->event = 0;
    value->signaled = (info->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0;
    *fence = g_fences.make_handle(value);
    return VK_SUCCESS;
}

void impl_destroy_fence(
    VkDevice device, VkFence fence, const VkAllocationCallbacks*)
{
    const std::shared_ptr<Fence> value = g_fences.get_handle(fence);
    if (value && value->device == device)
        g_fences.erase_handle(fence);
}

VkResult impl_get_fence_status(VkDevice device, VkFence fence)
{
    const std::shared_ptr<Fence> value = g_fences.get_handle(fence);
    if (!value || value->device != device)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!fence_is_signaled(value))
        return VK_NOT_READY;
    std::lock_guard<std::mutex> queue_lock(device->queue_mutex);
    retire_pending_submissions(device, false);
    return VK_SUCCESS;
}

VkResult impl_reset_fences(
    VkDevice device, uint32_t count, const VkFence* fences)
{
    if (count && !fences)
        return VK_ERROR_INITIALIZATION_FAILED;
    for (uint32_t i = 0; i < count; i++)
    {
        const std::shared_ptr<Fence> fence = g_fences.get_handle(fences[i]);
        if (!fence || fence->device != device)
            return VK_ERROR_INITIALIZATION_FAILED;
        std::lock_guard<std::mutex> lock(fence->mutex);
        if (fence->event)
            momoten_detail::g_opencl.p_clReleaseEvent(fence->event);
        fence->event = 0;
        fence->signaled = false;
    }
    return VK_SUCCESS;
}

VkResult impl_wait_for_fences(
    VkDevice device, uint32_t count, const VkFence* fences, VkBool32 wait_all, uint64_t timeout)
{
    if (count == 0 || !fences)
        return VK_ERROR_INITIALIZATION_FAILED;
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<Fence> > values(count);
    for (uint32_t i = 0; i < count; i++)
    {
        values[i] = g_fences.get_handle(fences[i]);
        if (!values[i] || values[i]->device != device)
            return VK_ERROR_INITIALIZATION_FAILED;
    }
    for (;;)
    {
        uint32_t signaled = 0;
        for (uint32_t i = 0; i < count; i++)
            signaled += fence_is_signaled(values[i]) ? 1u : 0u;
        if ((wait_all && signaled == count) || (!wait_all && signaled != 0))
        {
            std::lock_guard<std::mutex> queue_lock(device->queue_mutex);
            retire_pending_submissions(device, false);
            return VK_SUCCESS;
        }
        if (timeout == 0)
            return VK_TIMEOUT;
        if (timeout != UINT64_MAX)
        {
            const uint64_t elapsed = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count());
            if (elapsed >= timeout)
                return VK_TIMEOUT;
        }
        std::this_thread::yield();
    }
}

} // namespace momoten_detail
