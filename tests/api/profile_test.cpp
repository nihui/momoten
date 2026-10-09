// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <vector>

static int check_result(const char* operation, VkResult result)
{
    if (result == VK_SUCCESS)
        return 0;
    fprintf(stderr, "profile_test: %s failed with Vulkan error %d\n", operation, result);
    return 1;
}

static int expect_result(const char* operation, VkResult result,
                         VkResult expected)
{
    if (result == expected)
        return 0;
    fprintf(stderr,
            "profile_test: %s returned %d instead of expected Vulkan error %d\n",
            operation, result, expected);
    return 1;
}

int main()
{
    if (vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance") == 0 || vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateDevice") != 0 || vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkQueueSubmit") != 0)
    {
        fprintf(stderr, "profile_test: invalid global proc scope\n");
        return 1;
    }

    PFN_vkEnumerateInstanceVersion enumerate_instance_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    uint32_t api_version = 0;
    if (!enumerate_instance_version || enumerate_instance_version(&api_version) != VK_SUCCESS || api_version != VK_API_VERSION_1_1)
    {
        fprintf(stderr, "profile_test: Vulkan 1.1 instance version is inconsistent\n");
        return 1;
    }

    VkApplicationInfo application_info = {};
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.pApplicationName = "momoten_profile_test";
    application_info.apiVersion = VK_API_VERSION_1_1;
    const char* instance_extensions[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application_info;
    instance_info.enabledExtensionCount = 1;
    instance_info.ppEnabledExtensionNames = instance_extensions;

    VkInstance instance = VK_NULL_HANDLE;
    VkInstanceCreateInfo invalid_instance_info = instance_info;
    invalid_instance_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    if (expect_result("vkCreateInstance(invalid sType)",
                      vkCreateInstance(&invalid_instance_info, 0, &instance),
                      VK_ERROR_INITIALIZATION_FAILED))
        return 1;
    VkBaseInStructure unsupported_instance_chain = {};
    unsupported_instance_chain.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    invalid_instance_info = instance_info;
    invalid_instance_info.pNext = &unsupported_instance_chain;
    if (expect_result("vkCreateInstance(unsupported pNext)",
                      vkCreateInstance(&invalid_instance_info, 0, &instance),
                      VK_ERROR_FEATURE_NOT_PRESENT))
        return 1;
    VkAllocationCallbacks unsupported_allocator = {};
    if (expect_result("vkCreateInstance(custom allocator)",
                      vkCreateInstance(&instance_info, &unsupported_allocator,
                                       &instance),
                      VK_ERROR_FEATURE_NOT_PRESENT))
        return 1;
    VkResult result = vkCreateInstance(&instance_info, 0, &instance);
    if (result == VK_ERROR_INCOMPATIBLE_DRIVER)
        return 77;
    if (check_result("vkCreateInstance", result))
        return 1;

    const char* instance_aliases[][2] = {
        {"vkGetPhysicalDeviceFeatures2", "vkGetPhysicalDeviceFeatures2KHR"},
        {"vkGetPhysicalDeviceProperties2", "vkGetPhysicalDeviceProperties2KHR"},
        {"vkGetPhysicalDeviceQueueFamilyProperties2", "vkGetPhysicalDeviceQueueFamilyProperties2KHR"},
        {"vkGetPhysicalDeviceMemoryProperties2", "vkGetPhysicalDeviceMemoryProperties2KHR"}};
    for (size_t i = 0; i < sizeof(instance_aliases) / sizeof(instance_aliases[0]); i++)
    {
        if (vkGetInstanceProcAddr(instance, instance_aliases[i][0]) == 0 || vkGetInstanceProcAddr(instance, instance_aliases[i][0]) != vkGetInstanceProcAddr(instance, instance_aliases[i][1]))
        {
            fprintf(stderr, "profile_test: instance core/KHR alias mismatch for %s\n",
                    instance_aliases[i][0]);
            return 1;
        }
    }
    if (vkGetInstanceProcAddr(instance, "vkCreateImage") != 0)
    {
        fprintf(stderr, "profile_test: unsupported image API was exposed\n");
        return 1;
    }

    uint32_t physical_device_count = 0;
    result = vkEnumeratePhysicalDevices(instance, &physical_device_count, 0);
    if (result != VK_SUCCESS || physical_device_count == 0)
        return check_result("vkEnumeratePhysicalDevices(count)", result);
    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    result = vkEnumeratePhysicalDevices(
        instance, &physical_device_count, physical_devices.data());
    if (check_result("vkEnumeratePhysicalDevices", result))
        return 1;
    VkPhysicalDevice physical_device = physical_devices[0];

    VkPhysicalDeviceFeatures core_features = {};
    vkGetPhysicalDeviceFeatures(physical_device, &core_features);

    uint32_t device_extension_count = 0;
    result = vkEnumerateDeviceExtensionProperties(
        physical_device, 0, &device_extension_count, 0);
    if (check_result("vkEnumerateDeviceExtensionProperties(count)", result))
        return 1;
    std::vector<VkExtensionProperties> available_device_extensions(device_extension_count);
    result = vkEnumerateDeviceExtensionProperties(
        physical_device, 0, &device_extension_count,
        available_device_extensions.data());
    if (check_result("vkEnumerateDeviceExtensionProperties", result))
        return 1;
    bool driver_properties_supported = false;
    bool integer_dot_product_supported = false;
    bool float_controls_supported = false;
    bool float_controls2_supported = false;
    bool fp16_supported = false;
    for (size_t i = 0; i < available_device_extensions.size(); i++)
    {
        if (strcmp(available_device_extensions[i].extensionName,
                   VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME)
            == 0)
        {
            driver_properties_supported = true;
            if (available_device_extensions[i].specVersion != VK_KHR_DRIVER_PROPERTIES_SPEC_VERSION)
            {
                fprintf(stderr, "profile_test: driver properties extension version is inconsistent\n");
                return 1;
            }
        }
        else if (strcmp(available_device_extensions[i].extensionName,
                        VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)
                 == 0)
        {
            integer_dot_product_supported = true;
            if (available_device_extensions[i].specVersion != VK_KHR_SHADER_INTEGER_DOT_PRODUCT_SPEC_VERSION)
            {
                fprintf(stderr, "profile_test: integer dot-product extension version is inconsistent\n");
                return 1;
            }
        }
        else if (strcmp(available_device_extensions[i].extensionName, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME) == 0)
        {
            float_controls_supported = true;
            if (available_device_extensions[i].specVersion != VK_KHR_SHADER_FLOAT_CONTROLS_SPEC_VERSION)
            {
                fprintf(stderr, "profile_test: float-controls extension version is inconsistent\n");
                return 1;
            }
        }
        else if (strcmp(available_device_extensions[i].extensionName, VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME) == 0)
        {
            float_controls2_supported = true;
            if (available_device_extensions[i].specVersion != VK_KHR_SHADER_FLOAT_CONTROLS_2_SPEC_VERSION)
            {
                fprintf(stderr, "profile_test: float-controls2 extension version is inconsistent\n");
                return 1;
            }
        }
        else if (strcmp(available_device_extensions[i].extensionName, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) == 0)
        {
            fp16_supported = true;
        }
    }
    if (!driver_properties_supported)
    {
        fprintf(stderr, "profile_test: VK_KHR_driver_properties is unavailable\n");
        return 1;
    }
    if (float_controls2_supported && !float_controls_supported)
    {
        fprintf(stderr, "profile_test: float-controls2 dependency is unavailable\n");
        return 1;
    }

    VkPhysicalDeviceShaderIntegerDotProductFeatures integer_dot_features = {};
    integer_dot_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    VkPhysicalDeviceShaderFloatControls2FeaturesKHR float_controls2_features = {};
    float_controls2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR;
    integer_dot_features.pNext = &float_controls2_features;
    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &integer_dot_features;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    if (features2.features.shaderFloat64 != core_features.shaderFloat64)
    {
        fprintf(stderr, "profile_test: core and Features2 shaderFloat64 reports disagree\n");
        return 1;
    }
    if (features2.features.shaderInt64 != core_features.shaderInt64)
    {
        fprintf(stderr, "profile_test: core and Features2 shaderInt64 reports disagree\n");
        return 1;
    }
    if ((integer_dot_features.shaderIntegerDotProduct == VK_TRUE) != integer_dot_product_supported)
    {
        fprintf(stderr, "profile_test: integer dot-product extension and feature disagree\n");
        return 1;
    }
    if ((float_controls2_features.shaderFloatControls2 == VK_TRUE) != float_controls2_supported)
    {
        fprintf(stderr, "profile_test: float-controls2 extension and feature disagree\n");
        return 1;
    }

    VkImageFormatProperties image_properties = {};
    result = vkGetPhysicalDeviceImageFormatProperties(
        physical_device,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_TYPE_3D,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        0,
        &image_properties);
    if (result != VK_ERROR_FORMAT_NOT_SUPPORTED)
    {
        fprintf(stderr, "profile_test: image storage was not rejected\n");
        return 1;
    }

    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    if (properties.apiVersion != VK_API_VERSION_1_1)
    {
        fprintf(stderr, "profile_test: physical device is not Vulkan 1.1\n");
        return 1;
    }
    if (!strstr(properties.deviceName, "experimental")
        || !strstr(properties.deviceName, "non-conformant"))
    {
        fprintf(stderr, "profile_test: physical device name does not disclose non-conformant status\n");
        return 1;
    }
    if (properties.deviceID == 0
        || properties.deviceType < VK_PHYSICAL_DEVICE_TYPE_OTHER
        || properties.deviceType > VK_PHYSICAL_DEVICE_TYPE_CPU)
    {
        fprintf(stderr, "profile_test: OpenCL physical-device identity is invalid\n");
        return 1;
    }

    VkPhysicalDeviceDriverProperties driver_properties = {};
    driver_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceSubgroupProperties subgroup = {};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceShaderIntegerDotProductProperties integer_dot_properties = {};
    integer_dot_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
    VkPhysicalDeviceFloatControlsProperties float_controls_properties = {};
    float_controls_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
    integer_dot_properties.pNext = float_controls_supported ? &float_controls_properties : 0;
    driver_properties.pNext = &subgroup;
    subgroup.pNext = &integer_dot_properties;
    VkPhysicalDeviceProperties2 properties2 = {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &driver_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    if (!strstr(driver_properties.driverName, "experimental")
        || !strstr(driver_properties.driverName, "non-conformant")
        || !strstr(driver_properties.driverInfo, "experimental")
        || !strstr(driver_properties.driverInfo, "non-conformant")
        || driver_properties.conformanceVersion.major != 0
        || driver_properties.conformanceVersion.minor != 0
        || driver_properties.conformanceVersion.subminor != 0
        || driver_properties.conformanceVersion.patch != 0)
    {
        fprintf(stderr, "profile_test: driver properties do not disclose zero-conformance status\n");
        return 1;
    }
    if (subgroup.subgroupSize == 0 || subgroup.subgroupSize > 128 || (subgroup.subgroupSize & (subgroup.subgroupSize - 1)) != 0 || subgroup.supportedStages != VK_SHADER_STAGE_COMPUTE_BIT || subgroup.supportedOperations != VK_SUBGROUP_FEATURE_BASIC_BIT || subgroup.quadOperationsInAllStages != VK_FALSE)
    {
        fprintf(stderr, "profile_test: Vulkan subgroup BASIC profile is inconsistent\n");
        return 1;
    }
    if (integer_dot_properties.integerDotProduct16BitUnsignedAccelerated || integer_dot_properties.integerDotProduct16BitSignedAccelerated || integer_dot_properties.integerDotProduct16BitMixedSignednessAccelerated || integer_dot_properties.integerDotProduct32BitUnsignedAccelerated || integer_dot_properties.integerDotProduct32BitSignedAccelerated || integer_dot_properties.integerDotProduct32BitMixedSignednessAccelerated || integer_dot_properties.integerDotProduct64BitUnsignedAccelerated || integer_dot_properties.integerDotProduct64BitSignedAccelerated || integer_dot_properties.integerDotProduct64BitMixedSignednessAccelerated || integer_dot_properties.integerDotProductAccumulatingSaturating16BitUnsignedAccelerated || integer_dot_properties.integerDotProductAccumulatingSaturating16BitSignedAccelerated || integer_dot_properties.integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated || integer_dot_properties.integerDotProductAccumulatingSaturating32BitUnsignedAccelerated || integer_dot_properties.integerDotProductAccumulatingSaturating32BitSignedAccelerated || integer_dot_properties.integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated || integer_dot_properties.integerDotProductAccumulatingSaturating64BitUnsignedAccelerated || integer_dot_properties.integerDotProductAccumulatingSaturating64BitSignedAccelerated || integer_dot_properties.integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated)
    {
        fprintf(stderr, "profile_test: software-only integer dot widths were reported as accelerated\n");
        return 1;
    }
    if (float_controls_supported && (float_controls_properties.denormBehaviorIndependence != VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL || float_controls_properties.roundingModeIndependence != VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL || float_controls_properties.shaderDenormFlushToZeroFloat16 || float_controls_properties.shaderDenormFlushToZeroFloat32 || float_controls_properties.shaderDenormFlushToZeroFloat64 || float_controls_properties.shaderRoundingModeRTZFloat16 || float_controls_properties.shaderRoundingModeRTZFloat32 || float_controls_properties.shaderRoundingModeRTZFloat64 || (!fp16_supported && (float_controls_properties.shaderSignedZeroInfNanPreserveFloat16 || float_controls_properties.shaderDenormPreserveFloat16 || float_controls_properties.shaderRoundingModeRTEFloat16)) || (!core_features.shaderFloat64 && (float_controls_properties.shaderSignedZeroInfNanPreserveFloat64 || float_controls_properties.shaderDenormPreserveFloat64 || float_controls_properties.shaderRoundingModeRTEFloat64))))
    {
        fprintf(stderr, "profile_test: float-controls properties are inconsistent with the shader profile\n");
        return 1;
    }
    if (float_controls2_supported && (!float_controls_properties.shaderSignedZeroInfNanPreserveFloat32 || !float_controls_properties.shaderRoundingModeRTEFloat32 || (fp16_supported && (!float_controls_properties.shaderSignedZeroInfNanPreserveFloat16 || !float_controls_properties.shaderRoundingModeRTEFloat16)) || (core_features.shaderFloat64 && (!float_controls_properties.shaderSignedZeroInfNanPreserveFloat64 || !float_controls_properties.shaderRoundingModeRTEFloat64))))
    {
        fprintf(stderr, "profile_test: float-controls2 lacks strict floating-point support\n");
        return 1;
    }

    const float priority = 1.f;
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = 0;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkDeviceCreateInfo minimal_device_info = {};
    minimal_device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    minimal_device_info.queueCreateInfoCount = 1;
    minimal_device_info.pQueueCreateInfos = &queue_info;
    VkDevice minimal_device = VK_NULL_HANDLE;
    VkDeviceCreateInfo invalid_device_info = minimal_device_info;
    invalid_device_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    if (expect_result("vkCreateDevice(invalid sType)",
                      vkCreateDevice(physical_device, &invalid_device_info, 0,
                                     &minimal_device),
                      VK_ERROR_INITIALIZATION_FAILED))
        return 1;
    VkBaseInStructure unsupported_device_chain = {};
    unsupported_device_chain.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    invalid_device_info = minimal_device_info;
    invalid_device_info.pNext = &unsupported_device_chain;
    if (expect_result("vkCreateDevice(unsupported pNext)",
                      vkCreateDevice(physical_device, &invalid_device_info, 0,
                                     &minimal_device),
                      VK_ERROR_FEATURE_NOT_PRESENT)
        || expect_result("vkCreateDevice(custom allocator)",
                         vkCreateDevice(physical_device, &minimal_device_info,
                                        &unsupported_allocator, &minimal_device),
                         VK_ERROR_FEATURE_NOT_PRESENT))
        return 1;
    result = vkCreateDevice(physical_device, &minimal_device_info, 0, &minimal_device);
    if (check_result("vkCreateDevice(minimal)", result))
        return 1;
    if (vkGetDeviceProcAddr(VK_NULL_HANDLE, "vkQueueSubmit") != 0 || vkGetDeviceProcAddr(minimal_device, "vkCmdPushDescriptorSetKHR") != 0 || vkGetDeviceProcAddr(minimal_device, "vkTrimCommandPoolKHR") != 0 || vkGetDeviceProcAddr(minimal_device, "vkGetDescriptorSetLayoutSupportKHR") != 0 || vkGetDeviceProcAddr(minimal_device, "vkBindBufferMemory2") == 0 || vkGetDeviceProcAddr(minimal_device, "vkBindBufferMemory2KHR") != 0)
    {
        fprintf(stderr, "profile_test: device proc extension gating is inconsistent\n");
        return 1;
    }
    vkDestroyDevice(minimal_device, 0);

    if (float_controls2_supported)
    {
        VkPhysicalDeviceShaderFloatControls2FeaturesKHR requested_float_controls2 = float_controls2_features;
        requested_float_controls2.pNext = 0;
        VkDeviceCreateInfo float_controls2_device_info = minimal_device_info;
        float_controls2_device_info.pNext = &requested_float_controls2;
        if (expect_result("vkCreateDevice(float-controls2 without extension)",
                          vkCreateDevice(physical_device, &float_controls2_device_info, 0, &minimal_device),
                          VK_ERROR_EXTENSION_NOT_PRESENT))
            return 1;
        const char* float_controls2_extensions[] = {
            VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME,
            VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME};
        float_controls2_device_info.enabledExtensionCount = 1;
        float_controls2_device_info.ppEnabledExtensionNames = float_controls2_extensions;
        if (expect_result("vkCreateDevice(float-controls2 without dependency)",
                          vkCreateDevice(physical_device, &float_controls2_device_info, 0, &minimal_device),
                          VK_ERROR_EXTENSION_NOT_PRESENT))
            return 1;
        float_controls2_device_info.enabledExtensionCount = 2;
        requested_float_controls2.shaderFloatControls2 = VK_FALSE;
        result = vkCreateDevice(physical_device, &float_controls2_device_info, 0, &minimal_device);
        if (check_result("vkCreateDevice(float-controls2 disabled)", result))
            return 1;
        vkDestroyDevice(minimal_device, 0);
    }

    if (core_features.shaderFloat64)
    {
        VkPhysicalDeviceFeatures requested_features = {};
        requested_features.shaderFloat64 = VK_TRUE;
        VkDeviceCreateInfo fp64_device_info = minimal_device_info;
        fp64_device_info.pEnabledFeatures = &requested_features;
        VkDevice fp64_device = VK_NULL_HANDLE;
        result = vkCreateDevice(physical_device, &fp64_device_info, 0, &fp64_device);
        if (check_result("vkCreateDevice(shaderFloat64)", result))
            return 1;
        vkDestroyDevice(fp64_device, 0);
    }

    if (core_features.shaderInt64)
    {
        VkPhysicalDeviceFeatures requested_features = {};
        requested_features.shaderInt64 = VK_TRUE;
        VkDeviceCreateInfo int64_device_info = minimal_device_info;
        int64_device_info.pEnabledFeatures = &requested_features;
        VkDevice int64_device = VK_NULL_HANDLE;
        result = vkCreateDevice(physical_device, &int64_device_info, 0, &int64_device);
        if (check_result("vkCreateDevice(shaderInt64)", result))
            return 1;
        vkDestroyDevice(int64_device, 0);
    }

    std::vector<const char*> device_extensions;
    device_extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_MAINTENANCE1_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
    if (float_controls2_supported)
    {
        device_extensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
        device_extensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME);
    }
    if (integer_dot_product_supported)
        device_extensions.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
    VkDeviceCreateInfo device_info = minimal_device_info;
    integer_dot_features.pNext = float_controls2_supported ? &float_controls2_features : 0;
    device_info.pNext = integer_dot_product_supported ? static_cast<const void*>(&integer_dot_features) : (float_controls2_supported ? static_cast<const void*>(&float_controls2_features) : 0);
    device_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    device_info.ppEnabledExtensionNames = device_extensions.data();
    VkDevice device = VK_NULL_HANDLE;
    result = vkCreateDevice(physical_device, &device_info, 0, &device);
    if (check_result("vkCreateDevice", result))
        return 1;

    VkMemoryAllocateInfo invalid_memory_info = {};
    invalid_memory_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    invalid_memory_info.pNext = &unsupported_device_chain;
    invalid_memory_info.allocationSize = 16;
    invalid_memory_info.memoryTypeIndex = 0;
    VkDeviceMemory invalid_memory = VK_NULL_HANDLE;
    if (expect_result("vkAllocateMemory(unsupported pNext)",
                      vkAllocateMemory(device, &invalid_memory_info, 0,
                                       &invalid_memory),
                      VK_ERROR_FEATURE_NOT_PRESENT))
        return 1;

    VkBufferCreateInfo invalid_buffer_info = {};
    invalid_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    invalid_buffer_info.size = 16;
    invalid_buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    invalid_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer invalid_buffer = VK_NULL_HANDLE;
    if (expect_result("vkCreateBuffer(unsupported usage)",
                      vkCreateBuffer(device, &invalid_buffer_info, 0,
                                     &invalid_buffer),
                      VK_ERROR_FEATURE_NOT_PRESENT))
        return 1;

    const uint32_t invalid_cache_data = 0;
    VkPipelineCacheCreateInfo invalid_cache_info = {};
    invalid_cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    invalid_cache_info.initialDataSize = sizeof(invalid_cache_data);
    invalid_cache_info.pInitialData = &invalid_cache_data;
    VkPipelineCache invalid_cache = VK_NULL_HANDLE;
    if (expect_result("vkCreatePipelineCache(unsupported initial data)",
                      vkCreatePipelineCache(device, &invalid_cache_info, 0,
                                            &invalid_cache),
                      VK_ERROR_FEATURE_NOT_PRESENT))
        return 1;

    const char* device_aliases[][2] = {
        {"vkCreateDescriptorUpdateTemplate", "vkCreateDescriptorUpdateTemplateKHR"},
        {"vkDestroyDescriptorUpdateTemplate", "vkDestroyDescriptorUpdateTemplateKHR"},
        {"vkTrimCommandPool", "vkTrimCommandPoolKHR"},
        {"vkGetDescriptorSetLayoutSupport", "vkGetDescriptorSetLayoutSupportKHR"}};
    for (size_t i = 0; i < sizeof(device_aliases) / sizeof(device_aliases[0]); i++)
    {
        if (vkGetDeviceProcAddr(device, device_aliases[i][0]) == 0 || vkGetDeviceProcAddr(device, device_aliases[i][0]) != vkGetDeviceProcAddr(device, device_aliases[i][1]))
        {
            fprintf(stderr, "profile_test: device core/KHR alias mismatch for %s\n",
                    device_aliases[i][0]);
            return 1;
        }
    }

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, 0, 0, &queue);
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (check_result("vkCreateCommandPool", vkCreateCommandPool(device, &pool_info, 0, &pool)))
        return 1;
    VkCommandBufferAllocateInfo allocate_info = {};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.commandPool = pool;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    if (check_result("vkAllocateCommandBuffers",
                     vkAllocateCommandBuffers(device, &allocate_info, &command_buffer)))
        return 1;

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (check_result("vkBeginCommandBuffer(invalid)",
                     vkBeginCommandBuffer(command_buffer, &begin_info)))
        return 1;
    vkCmdDispatch(command_buffer, 1, 1, 1);
    if (vkEndCommandBuffer(command_buffer) == VK_SUCCESS)
    {
        fprintf(stderr, "profile_test: invalid vkCmdDispatch was silently accepted\n");
        return 1;
    }
    if (check_result("vkResetCommandBuffer(invalid)",
                     vkResetCommandBuffer(command_buffer, 0)))
        return 1;

    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (check_result("vkBeginCommandBuffer(one-time)",
                     vkBeginCommandBuffer(command_buffer, &begin_info))
        || check_result("vkEndCommandBuffer(one-time)", vkEndCommandBuffer(command_buffer)))
        return 1;
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    if (check_result("vkCreateFence", vkCreateFence(device, &fence_info, 0, &fence)) || check_result("vkQueueSubmit", vkQueueSubmit(queue, 1, &submit_info, fence)))
        return 1;
    if (vkResetCommandBuffer(command_buffer, 0) == VK_SUCCESS)
    {
        fprintf(stderr, "profile_test: pending command buffer was reset\n");
        return 1;
    }
    if (check_result("vkWaitForFences",
                     vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX))
        || check_result("vkResetCommandBuffer(completed)",
                        vkResetCommandBuffer(command_buffer, 0)))
        return 1;

    vkDestroyFence(device, fence, 0);
    vkDestroyCommandPool(device, pool, 0);
    vkDestroyDevice(device, 0);
    vkDestroyInstance(instance, 0);
    return 0;
}
