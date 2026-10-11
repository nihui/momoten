// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_ANDROID_HWVULKAN_ABI_H
#define MOMOTEN_ANDROID_HWVULKAN_ABI_H

// Private Android Vulkan HAL ABI, not part of the public NDK.
//
// Mirrors the binary layouts of AOSP hardware/hardware.h and
// frameworks/native/vulkan/include/hardware/hwvulkan.h (HAL 0.1).
// NDK builds cannot include those platform-private headers. Keep this
// adapter isolated from the Vulkan driver and check both pointer sizes.
//
// https://android.googlesource.com/platform/hardware/libhardware/
// https://android.googlesource.com/platform/frameworks/native/

#include <stdint.h>
#include <stddef.h>
#include <vulkan/vulkan.h>

#define MOMOTEN_HW_MAKE_TAG(a, b, c, d) (((a) << 24) | ((b) << 16) | ((c) << 8) | (d))
#define MOMOTEN_HW_MODULE_TAG MOMOTEN_HW_MAKE_TAG('H', 'W', 'M', 'T')
#define MOMOTEN_HW_DEVICE_TAG MOMOTEN_HW_MAKE_TAG('H', 'W', 'D', 'T')
#define MOMOTEN_HW_MODULE_VERSION_0_1 0x0001
#define MOMOTEN_HW_HAL_API_VERSION 0x0100
#define MOMOTEN_HW_DEVICE_VERSION_0_1 0x00010000
#define MOMOTEN_HWVULKAN_DEVICE_0 "vk0"
#define MOMOTEN_HWVULKAN_MODULE_ID "vulkan"

struct MomotenHwModule;
struct MomotenHwDevice;

struct MomotenHwModuleMethods
{
    int (*open)(const MomotenHwModule* module, const char* id, MomotenHwDevice** device);
};

struct MomotenHwModule
{
    uint32_t tag;
    uint16_t module_api_version;
    uint16_t hal_api_version;
    const char* id;
    const char* name;
    const char* author;
    MomotenHwModuleMethods* methods;
    void* dso;
    uint32_t reserved[25];
};

struct MomotenHwDevice
{
    uint32_t tag;
    uint32_t version;
    MomotenHwModule* module;
    uint32_t reserved[12];
    int (*close)(MomotenHwDevice* device);
};

struct MomotenHwVulkanModule
{
    MomotenHwModule common;
};

struct MomotenHwVulkanDevice
{
    MomotenHwDevice common;
    PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties;
    PFN_vkCreateInstance CreateInstance;
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
};

static_assert(sizeof(MomotenHwModule) == (sizeof(void*) == 8 ? 152 : 128),
              "Unexpected Android hw_module_t ABI size");
static_assert(sizeof(MomotenHwDevice) == (sizeof(void*) == 8 ? 72 : 64),
              "Unexpected Android hw_device_t ABI size");
static_assert(offsetof(MomotenHwVulkanDevice, EnumerateInstanceExtensionProperties)
                  == sizeof(MomotenHwDevice),
              "Unexpected Android hwvulkan_device_t ABI layout");

#endif // MOMOTEN_ANDROID_HWVULKAN_ABI_H
