// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_BUFFER_COPY_H
#define MOMOTEN_BACKEND_BUFFER_COPY_H

#include "../vulkan_internal.h"
#include <CL/cl.h>

namespace momoten_detail {

// The caller holds the queue mutex, including while setting cached kernel arguments.
cl_int enqueue_buffer_copy(
    VkDevice device, cl_mem source, cl_mem destination,
    size_t source_offset, size_t destination_offset, size_t byte_count,
    cl_uint wait_count, const cl_event* wait_list, cl_event* event);

} // namespace momoten_detail

#endif // MOMOTEN_BACKEND_BUFFER_COPY_H
