// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#if defined(__ANDROID__)

#include "hwvulkan_abi.h"

#include <errno.h>
#include <string.h>

namespace {

int close_device(MomotenHwDevice*)
{
    return 0;
}

int open_device(const MomotenHwModule* module, const char* id,
                MomotenHwDevice** device);

MomotenHwModuleMethods module_methods = {open_device};

} // namespace

// AOSP's Vulkan Loader looks up the exact C symbol "HMI" and verifies that
// the module ID equals "vulkan". This is separate from desktop ICD JSON.
extern "C" {
__attribute__((visibility("default"))) MomotenHwVulkanModule HMI = {
    {
        MOMOTEN_HW_MODULE_TAG,
        MOMOTEN_HW_MODULE_VERSION_0_1,
        MOMOTEN_HW_HAL_API_VERSION,
        MOMOTEN_HWVULKAN_MODULE_ID,
        "momoten experimental Vulkan over OpenCL",
        "nihui",
        &module_methods,
        0,
        {}
    }
};
}

namespace {

MomotenHwVulkanDevice vulkan_device = {
    {
        MOMOTEN_HW_DEVICE_TAG,
        MOMOTEN_HW_DEVICE_VERSION_0_1,
        &HMI.common,
        {},
        close_device
    },
    vkEnumerateInstanceExtensionProperties,
    vkCreateInstance,
    vkGetInstanceProcAddr
};

int open_device(const MomotenHwModule* module, const char* id,
                MomotenHwDevice** device)
{
    if (!device)
        return -EINVAL;
    *device = 0;
    if (module != &HMI.common || !id || strcmp(id, MOMOTEN_HWVULKAN_DEVICE_0) != 0)
        return -ENOENT;

    *device = &vulkan_device.common;
    return 0;
}

} // namespace

#endif // __ANDROID__
