// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/device.h"
#include "backend/objects.h"
#include "backend/runtime.h"
#include "vulkan_internal.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static size_t memory_program_cache_capacity()
{
    const size_t default_capacity = 8;
    const char* value = getenv("MOMOTEN_MEMORY_CACHE_ENTRIES");
    if (!value || !value[0])
        return default_capacity;

    errno = 0;
    char* end = 0;
    const unsigned long long parsed = strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max()))
    {
        if (momoten_detail::debug_enabled())
            fprintf(stderr, "[momoten] ignoring invalid MOMOTEN_MEMORY_CACHE_ENTRIES=%s\n", value);
        return default_capacity;
    }
    return static_cast<size_t>(parsed);
}

namespace momoten_detail {

static bool extension_enabled(const VkDeviceCreateInfo* create_info,
                              const char* extension_name)
{
    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++)
    {
        if (strcmp(create_info->ppEnabledExtensionNames[i], extension_name) == 0)
            return true;
    }
    return false;
}

static VkResult enable_core_features(
    VkPhysicalDevice physical, const VkPhysicalDeviceFeatures& requested,
    bool& fp64)
{
    VkPhysicalDeviceFeatures remaining = requested;
    fp64 = false;
    if (remaining.shaderFloat64)
    {
        if (!physical->shader_profile.fp64)
            return VK_ERROR_FEATURE_NOT_PRESENT;
        fp64 = true;
        remaining.shaderFloat64 = VK_FALSE;
    }

    const VkBool32* features =
        reinterpret_cast<const VkBool32*>(&remaining);
    for (size_t i = 0;
         i < sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32); i++)
    {
        if (features[i] != VK_FALSE)
            return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    return VK_SUCCESS;
}

VkResult impl_create_device(
    VkPhysicalDevice physical, const VkDeviceCreateInfo* create_info,
    const VkAllocationCallbacks* allocator, VkDevice* device)
{
    if (!physical || !create_info || !device
        || create_info->sType != VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
        return VK_ERROR_INITIALIZATION_FAILED;
    if ((create_info->enabledExtensionCount && !create_info->ppEnabledExtensionNames)
        || (create_info->enabledLayerCount && !create_info->ppEnabledLayerNames))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (allocator || create_info->flags)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->enabledLayerCount)
        return VK_ERROR_LAYER_NOT_PRESENT;
    momoten_detail::EnabledShaderProfile enabled_shader_profile;
    bool core_features_specified = create_info->pEnabledFeatures != 0;
    if (create_info->pEnabledFeatures)
    {
        const VkResult result = enable_core_features(
            physical, *create_info->pEnabledFeatures, enabled_shader_profile.fp64);
        if (result != VK_SUCCESS)
            return result;
    }
    if (create_info->queueCreateInfoCount != 1
        || !create_info->pQueueCreateInfos)
        return VK_ERROR_INITIALIZATION_FAILED;
    const VkDeviceQueueCreateInfo& queue_info = create_info->pQueueCreateInfos[0];
    if (queue_info.sType != VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
        || queue_info.queueFamilyIndex != 0 || queue_info.queueCount != 1
        || !queue_info.pQueuePriorities)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (queue_info.pNext || queue_info.flags)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (!(queue_info.pQueuePriorities[0] >= 0.f
          && queue_info.pQueuePriorities[0] <= 1.f))
        return VK_ERROR_INITIALIZATION_FAILED;
    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++)
    {
        if (!create_info->ppEnabledExtensionNames[i])
            return VK_ERROR_INITIALIZATION_FAILED;
        if (!supports_device_extension(physical, create_info->ppEnabledExtensionNames[i]))
            return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    const VkBaseInStructure* feature = reinterpret_cast<const VkBaseInStructure*>(create_info->pNext);
    while (feature)
    {
        if (feature->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
        {
            if (core_features_specified)
                return VK_ERROR_INITIALIZATION_FAILED;
            const VkPhysicalDeviceFeatures2* features2 = reinterpret_cast<const VkPhysicalDeviceFeatures2*>(feature);
            const VkResult result = enable_core_features(
                physical, features2->features, enabled_shader_profile.fp64);
            if (result != VK_SUCCESS)
                return result;
            core_features_specified = true;
        }
        else if (feature->sType
                 == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES)
        {
            const VkPhysicalDevice16BitStorageFeatures* storage = reinterpret_cast<const VkPhysicalDevice16BitStorageFeatures*>(feature);
            if ((storage->storageBuffer16BitAccess
                 && !physical->shader_profile.fp16)
                || storage->uniformAndStorageBuffer16BitAccess
                || storage->storagePushConstant16
                || storage->storageInputOutput16)
                return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        else if (feature->sType
                 == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES)
        {
            const VkPhysicalDeviceShaderFloat16Int8Features* arithmetic = reinterpret_cast<const VkPhysicalDeviceShaderFloat16Int8Features*>(feature);
            if ((arithmetic->shaderFloat16 && !physical->shader_profile.fp16)
                || arithmetic->shaderInt8)
                return VK_ERROR_FEATURE_NOT_PRESENT;
            if (arithmetic->shaderFloat16
                && !extension_enabled(
                    create_info, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME))
                return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
        else if (feature->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES)
        {
            const VkPhysicalDeviceShaderIntegerDotProductFeatures* dot = reinterpret_cast<const VkPhysicalDeviceShaderIntegerDotProductFeatures*>(feature);
            if (dot->shaderIntegerDotProduct)
            {
                if (!physical->shader_profile.integer_dot_product.supported || !supports_device_extension(physical, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME))
                    return VK_ERROR_FEATURE_NOT_PRESENT;
                if (!extension_enabled(
                        create_info,
                        VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME))
                    return VK_ERROR_EXTENSION_NOT_PRESENT;
                enabled_shader_profile.integer_dot_product = true;
            }
        }
        else
        {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        feature = feature->pNext;
    }

    if (!core_features_specified)
    {
        // ncnn queries core shader features but creates the device with a null
        // pEnabledFeatures pointer. Preserve that established simplevk path.
        enabled_shader_profile.fp64 = physical->shader_profile.fp64;
    }

    cl_int ret = CL_SUCCESS;
    cl_context context = momoten_detail::g_opencl.p_clCreateContext(0, 1, &physical->device, 0, 0, &ret);
    if (!context)
    {
        log_error("clCreateContext", ret);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    cl_command_queue queue = momoten_detail::g_opencl.p_clCreateCommandQueue(context, physical->device, 0, &ret);
    if (!queue)
    {
        log_error("clCreateCommandQueue", ret);
        momoten_detail::g_opencl.p_clReleaseContext(context);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkDevice_T* value = new VkDevice_T;
    value->physical_device = physical;
    value->api_version = physical->owner->api_version;
    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++)
        value->enabled_extensions.push_back(create_info->ppEnabledExtensionNames[i]);
    value->context = context;
    value->command_queue = queue;
    value->program_cache_capacity = memory_program_cache_capacity();
    value->enabled_shader_profile = enabled_shader_profile;
    value->queue = new VkQueue_T;
    value->queue->device = value;
    *device = value;
    return VK_SUCCESS;
}

void impl_destroy_device(VkDevice device, const VkAllocationCallbacks*)
{
    if (!device)
        return;
    {
        std::lock_guard<std::mutex> queue_lock(device->queue_mutex);
        if (momoten_detail::g_opencl.p_clFinish(device->command_queue) == CL_SUCCESS)
            retire_pending_submissions(device, true);
    }
    for (std::map<std::pair<uint64_t, uint64_t>, cl_program>::iterator it = device->program_cache.begin();
         it != device->program_cache.end(); ++it)
        momoten_detail::g_opencl.p_clReleaseProgram(it->second);
    device->program_cache.clear();
    delete device->queue;
    momoten_detail::g_opencl.p_clReleaseCommandQueue(device->command_queue);
    momoten_detail::g_opencl.p_clReleaseContext(device->context);
    delete device;
}

void impl_get_device_queue(
    VkDevice device, uint32_t family, uint32_t index, VkQueue* queue)
{
    if (queue)
        *queue = device && family == 0 && index == 0 ? device->queue : VK_NULL_HANDLE;
}

void impl_get_device_queue2(
    VkDevice device, const VkDeviceQueueInfo2* info, VkQueue* queue)
{
    if (!queue)
        return;
    *queue = info && info->flags == 0
                 ? (device && info->queueFamilyIndex == 0 && info->queueIndex == 0
                        ? device->queue
                        : VK_NULL_HANDLE)
                 : VK_NULL_HANDLE;
}

} // namespace momoten_detail
