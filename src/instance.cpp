// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/device.h"
#include "backend/objects.h"
#include "vulkan_internal.h"

#include <cstdio>
#include <cstring>

namespace momoten_detail {

VkResult impl_enumerate_instance_extension_properties(
    const char* layer_name, uint32_t* count, VkExtensionProperties* properties)
{
    if (layer_name)
        return VK_ERROR_LAYER_NOT_PRESENT;
    return enumerate_values(instance_extensions(), count, properties);
}

VkResult impl_enumerate_instance_layer_properties(
    uint32_t* count, VkLayerProperties* properties)
{
    const std::vector<VkLayerProperties> empty;
    return enumerate_values(empty, count, properties);
}

VkResult impl_enumerate_instance_version(uint32_t* version)
{
    if (!version)
        return VK_ERROR_INITIALIZATION_FAILED;
    *version = VK_API_VERSION_1_1;
    return VK_SUCCESS;
}

VkResult impl_create_instance(
    const VkInstanceCreateInfo* create_info, const VkAllocationCallbacks* allocator,
    VkInstance* instance)
{
    static std::once_flag non_conformant_warning_once;
    std::call_once(non_conformant_warning_once, []() {
        fprintf(stderr, "WARNING: momoten is not a conformant Vulkan implementation, testing use only.\n");
    });

    if (!create_info || !instance
        || create_info->sType != VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
        || (create_info->enabledExtensionCount && !create_info->ppEnabledExtensionNames)
        || (create_info->enabledLayerCount && !create_info->ppEnabledLayerNames))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (allocator || create_info->pNext || create_info->flags)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->enabledLayerCount)
        return VK_ERROR_LAYER_NOT_PRESENT;
    if (create_info->pApplicationInfo)
    {
        if (create_info->pApplicationInfo->sType
            != VK_STRUCTURE_TYPE_APPLICATION_INFO)
            return VK_ERROR_INITIALIZATION_FAILED;
        if (create_info->pApplicationInfo->pNext)
            return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++)
    {
        if (!create_info->ppEnabledExtensionNames[i])
            return VK_ERROR_INITIALIZATION_FAILED;
        const std::vector<VkExtensionProperties>& extensions = instance_extensions();
        bool supported = false;
        for (size_t e = 0; e < extensions.size(); e++)
            supported |= strcmp(create_info->ppEnabledExtensionNames[i], extensions[e].extensionName) == 0;
        if (!supported)
            return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    VkInstance_T* value = new VkInstance_T;
    const uint32_t requested_api_version = create_info->pApplicationInfo && create_info->pApplicationInfo->apiVersion
                                               ? create_info->pApplicationInfo->apiVersion
                                               : VK_API_VERSION_1_0;
    if (VK_VERSION_MAJOR(requested_api_version) != 1 || VK_VERSION_MINOR(requested_api_version) > 1)
    {
        delete value;
        return VK_ERROR_INCOMPATIBLE_DRIVER;
    }
    value->api_version = requested_api_version;
    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++)
        value->enabled_extensions.push_back(create_info->ppEnabledExtensionNames[i]);
    value->physical_devices = discover_opencl_devices(value);
    if (value->physical_devices.empty())
    {
        delete value;
        return VK_ERROR_INCOMPATIBLE_DRIVER;
    }
    *instance = value;
    return VK_SUCCESS;
}

void impl_destroy_instance(
    VkInstance instance, const VkAllocationCallbacks*)
{
    if (!instance)
        return;
    for (size_t i = 0; i < instance->physical_devices.size(); i++)
        delete instance->physical_devices[i];
    delete instance;
}

VkResult impl_enumerate_physical_devices(
    VkInstance instance, uint32_t* count, VkPhysicalDevice* devices)
{
    if (!instance)
        return VK_ERROR_INITIALIZATION_FAILED;
    return enumerate_values(instance->physical_devices, count, devices);
}

void impl_get_physical_device_features(
    VkPhysicalDevice physical, VkPhysicalDeviceFeatures* features)
{
    if (features)
    {
        memset(features, 0, sizeof(*features));
        features->shaderInt64 = physical && physical->shader_profile.int64 ? VK_TRUE : VK_FALSE;
        features->shaderFloat64 = physical && physical->shader_profile.fp64 ? VK_TRUE : VK_FALSE;
    }
}

void impl_get_physical_device_features2(
    VkPhysicalDevice physical, VkPhysicalDeviceFeatures2* features)
{
    if (!features)
        return;
    impl_get_physical_device_features(physical, &features->features);
    const VkBool32 fp16 = physical && physical->shader_profile.fp16 ? VK_TRUE : VK_FALSE;
    VkBaseOutStructure* next = reinterpret_cast<VkBaseOutStructure*>(features->pNext);
    while (next)
    {
        switch (next->sType)
        {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES:
        {
            VkPhysicalDevice16BitStorageFeatures* storage = reinterpret_cast<VkPhysicalDevice16BitStorageFeatures*>(next);
            storage->storageBuffer16BitAccess = fp16;
            // momoten currently maps descriptor-backed storage buffers only;
            // uniform and push-constant 16-bit storage are not implemented.
            storage->uniformAndStorageBuffer16BitAccess = VK_FALSE;
            storage->storagePushConstant16 = VK_FALSE;
            storage->storageInputOutput16 = VK_FALSE;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES:
        {
            VkPhysicalDeviceShaderFloat16Int8Features* arithmetic = reinterpret_cast<VkPhysicalDeviceShaderFloat16Int8Features*>(next);
            arithmetic->shaderFloat16 = fp16;
            arithmetic->shaderInt8 = VK_FALSE;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES:
        {
            VkPhysicalDeviceShaderIntegerDotProductFeatures* dot = reinterpret_cast<VkPhysicalDeviceShaderIntegerDotProductFeatures*>(next);
            dot->shaderIntegerDotProduct = physical && physical->shader_profile.integer_dot_product.supported ? VK_TRUE : VK_FALSE;
            break;
        }
        default:
            break;
        }
        next = next->pNext;
    }
}

void impl_get_physical_device_properties(
    VkPhysicalDevice physical, VkPhysicalDeviceProperties* properties)
{
    if (!physical || !properties)
        return;
    memset(properties, 0, sizeof(*properties));
    properties->apiVersion = VK_API_VERSION_1_1;
    properties->driverVersion = VK_MAKE_VERSION(0, 1, 0);
    properties->vendorID = physical->vendor_id;
    properties->deviceID = physical->device_id;
    properties->deviceType = physical->vulkan_device_type;
    const std::string device_name = "momoten experimental non-conformant: " + physical->name;
    strncpy(properties->deviceName, device_name.c_str(), VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
    properties->deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1] = '\0';

    VkPhysicalDeviceLimits& limits = properties->limits;
    limits.maxPushConstantsSize = 256;
    const size_t bytes_per_buffer_argument = sizeof(cl_mem) + 2 * (physical->address_bits == 64 ? sizeof(cl_ulong) : sizeof(cl_uint));
    const size_t parameter_budget = physical->max_parameter_size > limits.maxPushConstantsSize
                                        ? physical->max_parameter_size - limits.maxPushConstantsSize
                                        : 0;
    const uint32_t max_storage_buffers = static_cast<uint32_t>(
        std::min<size_t>(16, parameter_budget / bytes_per_buffer_argument));
    limits.maxStorageBufferRange = static_cast<uint32_t>(
        std::min<cl_ulong>(physical->max_allocation, 0xffffffffull));
    limits.maxMemoryAllocationCount = 4096;
    limits.maxBoundDescriptorSets = 1;
    limits.maxPerStageDescriptorStorageBuffers = max_storage_buffers;
    limits.maxPerStageResources = max_storage_buffers;
    limits.maxDescriptorSetStorageBuffers = max_storage_buffers;
    // Workgroup storage/barriers are not translated by the baseline profile,
    // even when the selected OpenCL device has local memory.
    limits.maxComputeSharedMemorySize = static_cast<uint32_t>(std::min<cl_ulong>(
        physical->local_memory, static_cast<cl_ulong>(UINT32_MAX)));
    limits.maxComputeWorkGroupCount[0] = 65535;
    limits.maxComputeWorkGroupCount[1] = 65535;
    limits.maxComputeWorkGroupCount[2] = 65535;
    limits.maxComputeWorkGroupInvocations = static_cast<uint32_t>(
        std::min<size_t>(physical->max_compute_workgroup_invocations, 0xffffffffu));
    for (size_t i = 0; i < 3; i++)
        limits.maxComputeWorkGroupSize[i] = static_cast<uint32_t>(
            std::min<size_t>(physical->max_work_item_sizes[i], 0xffffffffu));
    limits.minMemoryMapAlignment = sizeof(void*);
    limits.minStorageBufferOffsetAlignment = std::max<VkDeviceSize>(1, physical->mem_base_alignment_bits / 8);
    limits.optimalBufferCopyOffsetAlignment = 1;
    limits.optimalBufferCopyRowPitchAlignment = 1;
    limits.nonCoherentAtomSize = 1;
}

void impl_get_physical_device_properties2(
    VkPhysicalDevice physical, VkPhysicalDeviceProperties2* properties)
{
    if (!properties)
        return;
    impl_get_physical_device_properties(physical, &properties->properties);

    VkBaseOutStructure* extension = static_cast<VkBaseOutStructure*>(properties->pNext);
    while (extension)
    {
        if (extension->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES)
        {
            VkPhysicalDeviceDriverProperties* driver = reinterpret_cast<VkPhysicalDeviceDriverProperties*>(extension);
            driver->driverID = static_cast<VkDriverId>(0);
            strncpy(driver->driverName, "momoten experimental non-conformant",
                    VK_MAX_DRIVER_NAME_SIZE - 1);
            driver->driverName[VK_MAX_DRIVER_NAME_SIZE - 1] = '\0';
            const std::string driver_info = "momoten experimental non-conformant; OpenCL backend: " + physical->driver;
            strncpy(driver->driverInfo, driver_info.c_str(), VK_MAX_DRIVER_INFO_SIZE - 1);
            driver->driverInfo[VK_MAX_DRIVER_INFO_SIZE - 1] = '\0';
            driver->conformanceVersion.major = 0;
            driver->conformanceVersion.minor = 0;
            driver->conformanceVersion.subminor = 0;
            driver->conformanceVersion.patch = 0;
        }
        else if (extension->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES)
        {
            VkPhysicalDeviceSubgroupProperties* subgroup = reinterpret_cast<VkPhysicalDeviceSubgroupProperties*>(extension);
            subgroup->subgroupSize = physical->shader_profile.subgroup_size;
            subgroup->supportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
            subgroup->supportedOperations = physical->shader_profile.subgroup_operations;
            subgroup->quadOperationsInAllStages = VK_FALSE;
        }
        else if (extension->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES)
        {
            VkPhysicalDeviceShaderIntegerDotProductProperties* dot = reinterpret_cast<VkPhysicalDeviceShaderIntegerDotProductProperties*>(extension);
            void* next = dot->pNext;
            const VkStructureType type = dot->sType;
            memset(dot, 0, sizeof(*dot));
            dot->sType = type;
            dot->pNext = next;

            const momoten_detail::IntegerDotProductProfile& profile = physical->shader_profile.integer_dot_product;
            if (profile.supported)
            {
                const momoten_detail::IntegerDotProductAccelerationProfile& a8 = profile.acceleration_8bit;
                const momoten_detail::IntegerDotProductAccelerationProfile& ap = profile.acceleration_4x8bit_packed;
                dot->integerDotProduct8BitUnsignedAccelerated = profile.input_4x8bit && a8.unsigned_accelerated;
                dot->integerDotProduct8BitSignedAccelerated = profile.input_4x8bit && a8.signed_accelerated;
                dot->integerDotProduct8BitMixedSignednessAccelerated = profile.input_4x8bit && a8.mixed_signedness_accelerated;
                dot->integerDotProduct4x8BitPackedUnsignedAccelerated = ap.unsigned_accelerated;
                dot->integerDotProduct4x8BitPackedSignedAccelerated = ap.signed_accelerated;
                dot->integerDotProduct4x8BitPackedMixedSignednessAccelerated = ap.mixed_signedness_accelerated;
                dot->integerDotProductAccumulatingSaturating8BitUnsignedAccelerated = profile.input_4x8bit && a8.accumulating_saturating_unsigned_accelerated;
                dot->integerDotProductAccumulatingSaturating8BitSignedAccelerated = profile.input_4x8bit && a8.accumulating_saturating_signed_accelerated;
                dot->integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated = profile.input_4x8bit && a8.accumulating_saturating_mixed_signedness_accelerated;
                dot->integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated = ap.accumulating_saturating_unsigned_accelerated;
                dot->integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated = ap.accumulating_saturating_signed_accelerated;
                dot->integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated = ap.accumulating_saturating_mixed_signedness_accelerated;
            }
        }
        extension = extension->pNext;
    }
}

void impl_get_physical_device_queue_family_properties(
    VkPhysicalDevice, uint32_t* count, VkQueueFamilyProperties* properties)
{
    if (!count)
        return;
    if (!properties)
    {
        *count = 1;
        return;
    }
    if (*count == 0)
        return;
    memset(&properties[0], 0, sizeof(properties[0]));
    properties[0].queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    properties[0].queueCount = 1;
    properties[0].timestampValidBits = 0;
    properties[0].minImageTransferGranularity.width = 1;
    properties[0].minImageTransferGranularity.height = 1;
    properties[0].minImageTransferGranularity.depth = 1;
    *count = 1;
}

void impl_get_physical_device_queue_family_properties2(
    VkPhysicalDevice physical, uint32_t* count, VkQueueFamilyProperties2* properties)
{
    if (!count)
        return;
    if (!properties)
    {
        impl_get_physical_device_queue_family_properties(physical, count, 0);
        return;
    }
    VkQueueFamilyProperties core;
    uint32_t core_count = 1;
    impl_get_physical_device_queue_family_properties(physical, &core_count, &core);
    if (*count != 0 && core_count != 0)
    {
        properties[0].queueFamilyProperties = core;
        *count = 1;
    }
}

void impl_get_physical_device_memory_properties(
    VkPhysicalDevice physical, VkPhysicalDeviceMemoryProperties* properties)
{
    if (!physical || !properties)
        return;
    memset(properties, 0, sizeof(*properties));
    properties->memoryHeapCount = 1;
    properties->memoryHeaps[0].size = physical->global_memory;
    properties->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    properties->memoryTypeCount = 2;
    properties->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    properties->memoryTypes[0].heapIndex = 0;
    properties->memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    properties->memoryTypes[1].heapIndex = 0;
}

void impl_get_physical_device_memory_properties2(
    VkPhysicalDevice physical, VkPhysicalDeviceMemoryProperties2* properties)
{
    if (properties)
        impl_get_physical_device_memory_properties(physical, &properties->memoryProperties);
}

void impl_get_physical_device_format_properties(
    VkPhysicalDevice, VkFormat, VkFormatProperties* properties)
{
    if (properties) memset(properties, 0, sizeof(*properties));
}

void impl_get_physical_device_format_properties2(
    VkPhysicalDevice physical, VkFormat format, VkFormatProperties2* properties)
{
    if (properties)
        impl_get_physical_device_format_properties(physical, format, &properties->formatProperties);
}

VkResult impl_get_physical_device_image_format_properties(
    VkPhysicalDevice, VkFormat, VkImageType, VkImageTiling, VkImageUsageFlags,
    VkImageCreateFlags, VkImageFormatProperties*)
{
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
}

VkResult impl_get_physical_device_image_format_properties2(
    VkPhysicalDevice physical, const VkPhysicalDeviceImageFormatInfo2* info,
    VkImageFormatProperties2* properties)
{
    if (!info || !properties)
        return VK_ERROR_INITIALIZATION_FAILED;
    return impl_get_physical_device_image_format_properties(
        physical, info->format, info->type, info->tiling, info->usage, info->flags,
        &properties->imageFormatProperties);
}

VkResult impl_enumerate_device_extension_properties(
    VkPhysicalDevice physical, const char* layer_name, uint32_t* count, VkExtensionProperties* properties)
{
    if (layer_name)
        return VK_ERROR_LAYER_NOT_PRESENT;
    return enumerate_values(device_extensions(physical), count, properties);
}

VkResult impl_enumerate_device_layer_properties(
    VkPhysicalDevice, uint32_t* count, VkLayerProperties* properties)
{
    const std::vector<VkLayerProperties> empty;
    return enumerate_values(empty, count, properties);
}

} // namespace momoten_detail
