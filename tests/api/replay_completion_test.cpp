// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/buffer_copy.h"
#include "backend/memory_sync.h"
#include "backend/replay.h"

#include <cstdio>
#include <cstring>

struct _cl_context
{
};
struct _cl_command_queue
{
};
struct _cl_mem
{
    cl_uint data[2];
};
struct _cl_event
{
};

namespace momoten_detail {
OpenCLApi g_opencl = {};
}

struct MockState
{
    _cl_context context;
    _cl_command_queue queue;
    _cl_mem buffer;
    _cl_event events[4];
    size_t event_count = 0;
    unsigned creates = 0;
    unsigned copies = 0;
    unsigned markers = 0;
    cl_int create_error = CL_SUCCESS;
    cl_int enqueue_error = CL_SUCCESS;
    cl_int readback_error = CL_SUCCESS;
    bool null_buffer = false;
    bool readbacks_finished = false;
};

static MockState* mock;
static int failures;

static bool check(bool condition, const char* message)
{
    if (!condition)
    {
        fprintf(stderr, "replay_completion_test: %s\n", message);
        failures++;
    }
    return condition;
}

static cl_mem CL_API_CALL create_buffer(cl_context context, cl_mem_flags flags, size_t size, void* initial, cl_int* error)
{
    mock->creates++;
    *error = CL_INVALID_VALUE;
    if (!check(context == &mock->context && flags == (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR) && size == sizeof(mock->buffer.data) && initial, "invalid completion buffer allocation"))
        return 0;
    memcpy(mock->buffer.data, initial, size);
    check(mock->buffer.data[0] == 0 && mock->buffer.data[1] == 0, "completion buffer is not initialized");
    *error = mock->create_error;
    return *error == CL_SUCCESS && !mock->null_buffer ? &mock->buffer : 0;
}

static cl_int make_event(cl_event* event)
{
    if (!check(mock->readbacks_finished, "completion was enqueued before successful readbacks")
        || !check(event && mock->event_count < 4, "invalid completion event output"))
        return CL_INVALID_VALUE;
    if (mock->enqueue_error != CL_SUCCESS)
        return mock->enqueue_error;
    *event = &mock->events[mock->event_count++];
    return CL_SUCCESS;
}

static cl_int CL_API_CALL enqueue_copy(cl_command_queue queue, cl_mem source, cl_mem destination, size_t source_offset, size_t destination_offset, size_t size, cl_uint wait_count, const cl_event* wait_list, cl_event* event)
{
    mock->copies++;
    if (!check(queue == &mock->queue && source == &mock->buffer && destination == &mock->buffer, "copy does not use the device's private completion buffer")
        || !check(size == sizeof(cl_uint) && source_offset == 0 && destination_offset == sizeof(cl_uint), "completion copy overlaps or exceeds the buffer")
        || !check(wait_count == 0 && !wait_list, "completion copy has unexpected dependencies"))
        return CL_INVALID_VALUE;
    return make_event(event);
}

static cl_int CL_API_CALL enqueue_marker(cl_command_queue queue, cl_event* event)
{
    mock->markers++;
    if (!check(queue == &mock->queue, "marker uses another queue"))
        return CL_INVALID_COMMAND_QUEUE;
    return make_event(event);
}

cl_int finish_mapped_memory_readbacks(ReplayState&)
{
    mock->readbacks_finished = mock->readback_error == CL_SUCCESS;
    return mock->readback_error;
}

// Other replay operations are linked from the production translation unit but
// are outside these completion-only submissions. Fail if they are invoked.
cl_int upload_host_ranges(ReplayState&, const std::shared_ptr<DeviceMemory>&, size_t, size_t)
{
    check(false, "unexpected upload");
    return CL_INVALID_OPERATION;
}

void mark_device_range(ReplayState&, const std::shared_ptr<DeviceMemory>&, VkDeviceSize, VkDeviceSize)
{
    check(false, "unexpected memory write");
}

namespace momoten_detail {

cl_int enqueue_buffer_copy(VkDevice, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*)
{
    check(false, "completion copy routed through the application copy workaround");
    return CL_INVALID_OPERATION;
}

} // namespace momoten_detail

struct Fixture
{
    MockState state;
    VkPhysicalDevice_T physical = {};
    VkDevice_T device = {};

    explicit Fixture(bool copy_completion = true)
    {
        mock = &state;
        device.physical_device = &physical;
        device.context = &state.context;
        device.command_queue = &state.queue;
        physical.driver_policy.use_copy_completion_event = copy_completion;
        momoten_detail::g_opencl = momoten_detail::OpenCLApi();
        momoten_detail::g_opencl.p_clCreateBuffer = create_buffer;
        momoten_detail::g_opencl.p_clEnqueueCopyBuffer = enqueue_copy;
        momoten_detail::g_opencl.p_clEnqueueMarker = enqueue_marker;
    }

    cl_int submit(cl_event& event)
    {
        state.readbacks_finished = false;
        ReplayState replay(&device);
        return finish_replay(replay, event);
    }
};

static void check_empty_submissions(bool copy_completion)
{
    Fixture f(copy_completion);
    cl_event first = 0;
    cl_event second = 0;
    check(f.submit(first) == CL_SUCCESS && first, "empty submission has no completion event");
    check(f.submit(second) == CL_SUCCESS && second && second != first, "successive submissions share a completion event");
    check(f.state.event_count == 2, "outstanding submissions did not retain distinct events");
    check(f.state.creates == (copy_completion ? 1u : 0u), "completion buffer was not reused or was unnecessarily allocated");
    check(f.device.completion_buffer == (copy_completion ? &f.state.buffer : 0), "incorrect cached completion buffer");
    check(f.state.copies == (copy_completion ? 2u : 0u) && f.state.markers == (copy_completion ? 0u : 2u), "incorrect completion operation");
}

static void check_allocation_failure(cl_int error)
{
    Fixture f;
    f.state.create_error = error;
    f.state.null_buffer = true;
    cl_event event = 0;
    const cl_int expected = error == CL_SUCCESS ? CL_MEM_OBJECT_ALLOCATION_FAILURE : error;
    check(f.submit(event) == expected && !event, "allocation failure was not propagated");
    check(!f.device.completion_buffer && f.state.copies == 0 && f.state.markers == 0, "allocation failure cached a buffer or enqueued completion");
    f.state.create_error = CL_SUCCESS;
    f.state.null_buffer = false;
    check(f.submit(event) == CL_SUCCESS && event && f.state.creates == 2, "failed allocation cannot be retried");
}

static void check_enqueue_failure()
{
    Fixture f;
    f.state.enqueue_error = CL_OUT_OF_RESOURCES;
    cl_event event = 0;
    check(f.submit(event) == CL_OUT_OF_RESOURCES && !event, "copy enqueue failure was not propagated");
    check(f.device.completion_buffer == &f.state.buffer && f.state.creates == 1, "enqueue failure lost the allocated buffer");
    check(f.state.markers == 0 && f.state.event_count == 0, "enqueue failure fell back to a broken marker or produced an event");
    f.state.enqueue_error = CL_SUCCESS;
    check(f.submit(event) == CL_SUCCESS && event && f.state.creates == 1 && f.state.copies == 2, "retry did not reuse the allocated buffer");
}

static void check_readback_failure(bool copy_completion)
{
    Fixture f(copy_completion);
    f.state.readback_error = CL_OUT_OF_RESOURCES;
    cl_event event = 0;
    check(f.submit(event) == CL_OUT_OF_RESOURCES && !event, "readback failure was not propagated");
    check(f.state.creates == 0 && f.state.copies == 0 && f.state.markers == 0, "completion was attempted after readback failure");
    f.state.readback_error = CL_SUCCESS;
    check(f.submit(event) == CL_SUCCESS && event, "readback failure prevents a subsequent submission");
}

int main()
{
    check_empty_submissions(false);
    check_empty_submissions(true);
    check_allocation_failure(CL_OUT_OF_HOST_MEMORY);
    check_allocation_failure(CL_SUCCESS);
    check_enqueue_failure();
    check_readback_failure(false);
    check_readback_failure(true);
    return failures ? 1 : 0;
}
