// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_DEVICE_H
#define MOMOTEN_BACKEND_DEVICE_H

#include "../vulkan_internal.h"
#include "opencl_loader.h"

#include <string>
#include <vector>

void log_error(const char* operation, cl_int error);
std::string opencl_device_string(cl_device_id device, cl_device_info parameter);
std::vector<VkExtensionProperties> device_extensions(VkPhysicalDevice physical);
bool has_opencl_extension(VkPhysicalDevice physical, const char* name);
const std::vector<VkExtensionProperties>& instance_extensions();
bool supports_device_extension(VkPhysicalDevice physical, const char* name);
std::vector<VkPhysicalDevice> discover_opencl_devices(VkInstance instance);

#endif // MOMOTEN_BACKEND_DEVICE_H
