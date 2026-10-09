// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/device_profile.h"
#include "backend/objects.h"

#include <stdint.h>
#include <stdio.h>

namespace momoten_detail {

OpenCLApi g_opencl = {};

} // namespace momoten_detail

static int check_parameter_size(uint32_t address_bits, size_t required_size)
{
    VkPhysicalDevice_T physical = {};
    physical.address_bits = address_bits;
    physical.max_workgroup_size = 1;
    for (size_t d = 0; d < 3; d++)
        physical.max_work_item_sizes[d] = 1;
    VkDevice_T device = {};
    device.physical_device = &physical;

    momoten::KernelABI abi;
    abi.address_bits = address_bits;
    abi.buffers.resize(1);
    abi.push_constant_arg_index = 3;
    abi.push_constant_size = 12;

    std::string diagnostic;
    physical.max_parameter_size = required_size - 1;
    if (momoten_detail::validate_pipeline_abi(&device, abi, diagnostic)
        || diagnostic.find("CL_DEVICE_MAX_PARAMETER_SIZE") == std::string::npos)
    {
        fprintf(stderr, "parameter_size_test: %u-bit arguments exceeding the parameter limit were not rejected: %s\n", address_bits, diagnostic.c_str());
        return 1;
    }

    diagnostic.clear();
    physical.max_parameter_size = required_size;
    if (!momoten_detail::validate_pipeline_abi(&device, abi, diagnostic))
    {
        fprintf(stderr, "parameter_size_test: %u-bit arguments exactly at the parameter limit were rejected: %s\n", address_bits, diagnostic.c_str());
        return 1;
    }

    diagnostic.clear();
    physical.max_parameter_size = SIZE_MAX;
    abi.push_constant_size = SIZE_MAX;
    if (momoten_detail::validate_pipeline_abi(&device, abi, diagnostic)
        || diagnostic.find("overflow") == std::string::npos)
    {
        fprintf(stderr, "parameter_size_test: %u-bit parameter-size overflow was not rejected: %s\n", address_bits, diagnostic.c_str());
        return 1;
    }

    return 0;
}

int main()
{
    return check_parameter_size(32, sizeof(cl_mem) + 8 + 12)
           || check_parameter_size(64, sizeof(cl_mem) + 16 + 12);
}
