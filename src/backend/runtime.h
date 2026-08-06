// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_RUNTIME_H
#define MOMOTEN_BACKEND_RUNTIME_H

#include "objects.h"

void assign_fence_event(const std::shared_ptr<Fence>& fence, cl_event event);
bool fence_is_signaled(const std::shared_ptr<Fence>& fence);
void retire_pending_submissions(VkDevice device, bool all);

#endif // MOMOTEN_BACKEND_RUNTIME_H
