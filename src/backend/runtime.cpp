// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "runtime.h"

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
