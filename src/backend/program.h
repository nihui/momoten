// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_PROGRAM_H
#define MOMOTEN_BACKEND_PROGRAM_H

#include "../vulkan_internal.h"

#include <CL/cl.h>

#include <string>

namespace momoten_detail {

void print_program_build_log(cl_program program, cl_device_id device);
cl_program create_and_build_program(
    VkDevice device, const std::string& source, cl_int& result);

} // namespace momoten_detail

#endif // MOMOTEN_BACKEND_PROGRAM_H
