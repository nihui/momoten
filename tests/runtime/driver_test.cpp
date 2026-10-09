// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include <vulkan/vulkan.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <vector>

struct PushConstants
{
    float scale;
    int32_t bias;
    uint32_t element_count;
};

static bool read_spirv(const char* path, std::vector<uint32_t>& words)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        return false;
    const std::streamoff size = stream.tellg();
    if (size < 0 || size % 4 != 0)
        return false;
    words.resize(static_cast<size_t>(size) / 4);
    stream.seekg(0, std::ios::beg);
    return words.empty() || static_cast<bool>(stream.read(reinterpret_cast<char*>(words.data()), size));
}

static int fail(const char* operation, VkResult result)
{
    fprintf(stderr, "driver_test: %s failed with Vulkan error %d\n", operation, result);
    return 1;
}

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 4)
    {
        fprintf(stderr, "driver_test: expected a SPIR-V path and optional subgroup and integer-dot SPIR-V paths\n");
        return 1;
    }

    std::vector<uint32_t> words;
    if (!read_spirv(argv[1], words))
    {
        fprintf(stderr, "driver_test: failed to read SPIR-V\n");
        return 1;
    }
    std::vector<uint32_t> subgroup_words;
    if (argc == 3 && !read_spirv(argv[2], subgroup_words))
    {
        fprintf(stderr, "driver_test: failed to read subgroup SPIR-V\n");
        return 1;
    }
    if (argc == 4 && !read_spirv(argv[2], subgroup_words))
    {
        fprintf(stderr, "driver_test: failed to read subgroup SPIR-V\n");
        return 1;
    }
    std::vector<uint32_t> integer_dot_words;
    if (argc == 4 && !read_spirv(argv[3], integer_dot_words))
    {
        fprintf(stderr, "driver_test: failed to read integer dot-product SPIR-V\n");
        return 1;
    }

    VkApplicationInfo application = {};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "momoten_driver_test";
    application.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&instance_info, 0, &instance);
    if (result == VK_ERROR_INCOMPATIBLE_DRIVER)
        return 77;
    if (result != VK_SUCCESS)
        return fail("vkCreateInstance", result);

    uint32_t physical_count = 0;
    result = vkEnumeratePhysicalDevices(instance, &physical_count, 0);
    if (result != VK_SUCCESS || physical_count == 0)
        return fail("vkEnumeratePhysicalDevices(count)", result);
    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    result = vkEnumeratePhysicalDevices(instance, &physical_count, physical_devices.data());
    if (result != VK_SUCCESS)
        return fail("vkEnumeratePhysicalDevices", result);
    VkPhysicalDevice physical_device = physical_devices[0];

    uint32_t extension_count = 0;
    result = vkEnumerateDeviceExtensionProperties(physical_device, 0, &extension_count, 0);
    if (result != VK_SUCCESS)
        return fail("vkEnumerateDeviceExtensionProperties(count)", result);
    std::vector<VkExtensionProperties> extension_properties(extension_count);
    result = vkEnumerateDeviceExtensionProperties(
        physical_device, 0, &extension_count, extension_properties.data());
    if (result != VK_SUCCESS)
        return fail("vkEnumerateDeviceExtensionProperties", result);
    bool reports_fp16_storage = false;
    bool reports_fp16_arithmetic = false;
    bool reports_integer_dot_product = false;
    for (uint32_t i = 0; i < extension_count; i++)
    {
        reports_fp16_storage = reports_fp16_storage || strcmp(extension_properties[i].extensionName, VK_KHR_16BIT_STORAGE_EXTENSION_NAME) == 0;
        reports_fp16_arithmetic = reports_fp16_arithmetic || strcmp(extension_properties[i].extensionName, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) == 0;
        reports_integer_dot_product = reports_integer_dot_product || strcmp(extension_properties[i].extensionName, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) == 0;
    }

    VkPhysicalDeviceShaderFloat16Int8Features fp16_arithmetic = {};
    fp16_arithmetic.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    VkPhysicalDeviceShaderIntegerDotProductFeatures integer_dot_feature = {};
    integer_dot_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    fp16_arithmetic.pNext = &integer_dot_feature;
    VkPhysicalDevice16BitStorageFeatures fp16_storage = {};
    fp16_storage.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    fp16_storage.pNext = &fp16_arithmetic;
    VkPhysicalDeviceFeatures2 features = {};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &fp16_storage;
    vkGetPhysicalDeviceFeatures2KHR(physical_device, &features);
    if (reports_fp16_storage != (fp16_storage.storageBuffer16BitAccess == VK_TRUE) || reports_fp16_arithmetic != (fp16_arithmetic.shaderFloat16 == VK_TRUE) || reports_fp16_storage != reports_fp16_arithmetic || fp16_storage.uniformAndStorageBuffer16BitAccess || fp16_storage.storagePushConstant16 || fp16_storage.storageInputOutput16 || fp16_arithmetic.shaderInt8 || reports_integer_dot_product != (integer_dot_feature.shaderIntegerDotProduct == VK_TRUE))
    {
        fprintf(stderr, "driver_test: fp16 Vulkan extensions/features are inconsistent\n");
        return 1;
    }

    const float queue_priority = 1.f;
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = 0;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;
    std::vector<const char*> extensions;
    extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    extensions.push_back(VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME);
    extensions.push_back(VK_KHR_MAINTENANCE1_EXTENSION_NAME);
    extensions.push_back(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
    if (reports_integer_dot_product)
        extensions.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pNext = reports_integer_dot_product ? &integer_dot_feature : 0;
    device_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    device_info.ppEnabledExtensionNames = extensions.data();

    VkDevice device = VK_NULL_HANDLE;
    result = vkCreateDevice(physical_device, &device_info, 0, &device);
    if (result != VK_SUCCESS)
        return fail("vkCreateDevice", result);
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, 0, 0, &queue);

    const VkDeviceSize allocation_size = 256;
    const VkDeviceSize bind_offset = 64;
    const VkDeviceSize descriptor_offset = 16;
    const VkDeviceSize data_size = 64;

    VkMemoryAllocateInfo allocation_info = {};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = allocation_size;
    allocation_info.memoryTypeIndex = 1;
    VkDeviceMemory input_memory = VK_NULL_HANDLE;
    VkDeviceMemory output_memory = VK_NULL_HANDLE;
    result = vkAllocateMemory(device, &allocation_info, 0, &input_memory);
    if (result != VK_SUCCESS) return fail("vkAllocateMemory(input)", result);
    result = vkAllocateMemory(device, &allocation_info, 0, &output_memory);
    if (result != VK_SUCCESS) return fail("vkAllocateMemory(output)", result);

    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = descriptor_offset + data_size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer input_buffer = VK_NULL_HANDLE;
    VkBuffer output_buffer = VK_NULL_HANDLE;
    result = vkCreateBuffer(device, &buffer_info, 0, &input_buffer);
    if (result != VK_SUCCESS) return fail("vkCreateBuffer(input)", result);
    result = vkCreateBuffer(device, &buffer_info, 0, &output_buffer);
    if (result != VK_SUCCESS) return fail("vkCreateBuffer(output)", result);
    result = vkBindBufferMemory(device, input_buffer, input_memory, bind_offset);
    if (result != VK_SUCCESS) return fail("vkBindBufferMemory(input)", result);
    result = vkBindBufferMemory(device, output_buffer, output_memory, bind_offset);
    if (result != VK_SUCCESS) return fail("vkBindBufferMemory(output)", result);

    void* input_mapping = 0;
    void* output_mapping = 0;
    result = vkMapMemory(device, input_memory, 0, VK_WHOLE_SIZE, 0, &input_mapping);
    if (result != VK_SUCCESS) return fail("vkMapMemory(input)", result);
    result = vkMapMemory(device, output_memory, 0, VK_WHOLE_SIZE, 0, &output_mapping);
    if (result != VK_SUCCESS) return fail("vkMapMemory(output)", result);
    float* input = reinterpret_cast<float*>(static_cast<unsigned char*>(input_mapping) + bind_offset + descriptor_offset);
    float* output = reinterpret_cast<float*>(static_cast<unsigned char*>(output_mapping) + bind_offset + descriptor_offset);
    VkDescriptorSetLayoutBinding bindings[2] = {};
    for (uint32_t i = 0; i < 2; i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo descriptor_layout_info = {};
    descriptor_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    descriptor_layout_info.bindingCount = 2;
    descriptor_layout_info.pBindings = bindings;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    result = vkCreateDescriptorSetLayout(device, &descriptor_layout_info, 0, &descriptor_layout);
    if (result != VK_SUCCESS) return fail("vkCreateDescriptorSetLayout", result);

    VkPushConstantRange push_range = {};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.size = sizeof(PushConstants);
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    result = vkCreatePipelineLayout(device, &pipeline_layout_info, 0, &pipeline_layout);
    if (result != VK_SUCCESS) return fail("vkCreatePipelineLayout", result);

    VkShaderModuleCreateInfo module_info = {};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = words.size() * sizeof(uint32_t);
    module_info.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    result = vkCreateShaderModule(device, &module_info, 0, &module);
    if (result != VK_SUCCESS) return fail("vkCreateShaderModule", result);

    const uint32_t specialization_value = 8;
    VkSpecializationMapEntry specialization_entry = {};
    specialization_entry.constantID = 0;
    specialization_entry.size = sizeof(specialization_value);
    VkSpecializationInfo specialization_info = {};
    specialization_info.mapEntryCount = 1;
    specialization_info.pMapEntries = &specialization_entry;
    specialization_info.dataSize = sizeof(specialization_value);
    specialization_info.pData = &specialization_value;
    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.stage.pSpecializationInfo = &specialization_info;
    pipeline_info.layout = pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, 0, &pipeline);
    if (result != VK_SUCCESS) return fail("vkCreateComputePipelines", result);
    VkPipeline duplicate_pipeline = VK_NULL_HANDLE;
    result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, 0, &duplicate_pipeline);
    if (result != VK_SUCCESS) return fail("vkCreateComputePipelines(duplicate)", result);

    VkShaderModule subgroup_module = VK_NULL_HANDLE;
    VkPipeline subgroup_pipeline = VK_NULL_HANDLE;
    if (!subgroup_words.empty())
    {
        VkShaderModuleCreateInfo subgroup_module_info = module_info;
        subgroup_module_info.codeSize = subgroup_words.size() * sizeof(uint32_t);
        subgroup_module_info.pCode = subgroup_words.data();
        result = vkCreateShaderModule(device, &subgroup_module_info, 0, &subgroup_module);
        if (result != VK_SUCCESS) return fail("vkCreateShaderModule(subgroup)", result);
        VkComputePipelineCreateInfo subgroup_pipeline_info = pipeline_info;
        subgroup_pipeline_info.stage.module = subgroup_module;
        subgroup_pipeline_info.stage.pSpecializationInfo = 0;
        result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                                          &subgroup_pipeline_info, 0,
                                          &subgroup_pipeline);
        if (result != VK_SUCCESS) return fail("vkCreateComputePipelines(subgroup)", result);
    }

    VkShaderModule integer_dot_module = VK_NULL_HANDLE;
    VkPipeline integer_dot_pipeline = VK_NULL_HANDLE;
    if (!integer_dot_words.empty() && reports_integer_dot_product)
    {
        VkShaderModuleCreateInfo integer_dot_module_info = module_info;
        integer_dot_module_info.codeSize = integer_dot_words.size() * sizeof(uint32_t);
        integer_dot_module_info.pCode = integer_dot_words.data();
        result = vkCreateShaderModule(device, &integer_dot_module_info, 0,
                                      &integer_dot_module);
        if (result != VK_SUCCESS)
            return fail("vkCreateShaderModule(integer dot)", result);
        VkComputePipelineCreateInfo integer_dot_pipeline_info = pipeline_info;
        integer_dot_pipeline_info.stage.module = integer_dot_module;
        integer_dot_pipeline_info.stage.pSpecializationInfo = 0;
        result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                                          &integer_dot_pipeline_info, 0,
                                          &integer_dot_pipeline);
        if (result != VK_SUCCESS)
            return fail("vkCreateComputePipelines(integer dot)", result);
    }

    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    result = vkCreateCommandPool(device, &pool_info, 0, &pool);
    if (result != VK_SUCCESS) return fail("vkCreateCommandPool", result);
    VkCommandBufferAllocateInfo command_allocate_info = {};
    command_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_allocate_info.commandPool = pool;
    command_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_allocate_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    result = vkAllocateCommandBuffers(device, &command_allocate_info, &command_buffer);
    if (result != VK_SUCCESS) return fail("vkAllocateCommandBuffers", result);
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    result = vkBeginCommandBuffer(command_buffer, &begin_info);
    if (result != VK_SUCCESS) return fail("vkBeginCommandBuffer", result);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    VkDescriptorBufferInfo descriptor_infos[2] = {};
    descriptor_infos[0].buffer = input_buffer;
    descriptor_infos[0].offset = descriptor_offset;
    descriptor_infos[0].range = data_size;
    descriptor_infos[1].buffer = output_buffer;
    descriptor_infos[1].offset = descriptor_offset;
    descriptor_infos[1].range = data_size;
    VkWriteDescriptorSet writes[2] = {};
    for (uint32_t i = 0; i < 2; i++)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &descriptor_infos[i];
    }
    vkCmdPushDescriptorSetKHR(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipeline_layout, 0, 2, writes);
    const PushConstants push_constants = {3.f, -3, 8};
    vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push_constants), &push_constants);
    vkCmdDispatch(command_buffer, 2, 1, 1);
    VkMemoryBarrier dispatch_barrier = {};
    dispatch_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    dispatch_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    dispatch_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &dispatch_barrier, 0, 0, 0, 0);

    // Record a second dispatch with both descriptors and push constants
    // changed. Each dispatch must retain its own immutable Vulkan state
    // snapshot until queue submission.
    descriptor_infos[0].buffer = output_buffer;
    descriptor_infos[1].buffer = input_buffer;
    vkCmdPushDescriptorSetKHR(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipeline_layout, 0, 2, writes);
    const float second_scale = 2.f;
    const int32_t second_bias = 5;
    vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(second_scale), &second_scale);
    // Update only the signed bias field, preserving the unsigned element count
    // from the initial full-block update.
    vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       offsetof(PushConstants, bias), sizeof(second_bias), &second_bias);
    vkCmdDispatch(command_buffer, 2, 1, 1);
    result = vkEndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS) return fail("vkEndCommandBuffer", result);

    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    result = vkCreateFence(device, &fence_info, 0, &fence);
    if (result != VK_SUCCESS) return fail("vkCreateFence", result);
    if (vkGetFenceStatus(device, fence) != VK_NOT_READY)
        return fail("vkGetFenceStatus(unsignaled)", VK_ERROR_UNKNOWN);
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    int status = 0;
    // Submit the same recorded dispatches again after restoring input data.
    // Kernel arguments must be replayed from each dispatch snapshot every time.
    for (uint32_t submission = 0; submission < 2; submission++)
    {
        for (uint32_t i = 0; i < 8; i++)
        {
            input[i] = static_cast<float>(i) - 2.f;
            output[i] = 0.f;
        }
        if (submission != 0)
        {
            result = vkResetFences(device, 1, &fence);
            if (result != VK_SUCCESS) return fail("vkResetFences(resubmit)", result);
        }
        result = vkQueueSubmit(queue, 1, &submit_info, fence);
        if (result != VK_SUCCESS) return fail("vkQueueSubmit", result);
        result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        if (result != VK_SUCCESS) return fail("vkWaitForFences", result);
        if (vkGetFenceStatus(device, fence) != VK_SUCCESS)
            return fail("vkGetFenceStatus(signaled)", VK_ERROR_UNKNOWN);

        for (uint32_t i = 0; i < 8; i++)
        {
            const uint32_t source = i - i % 4 + (3 - i % 4);
            const float expected = rintf((static_cast<float>(source) - 2.f) * push_constants.scale) + static_cast<float>(push_constants.bias);
            if (fabsf(output[i] - expected) > 1e-6f)
            {
                fprintf(stderr, "driver_test: submission %u output[%u]=%g expected=%g\n", submission, i, output[i], expected);
                status = 1;
            }

            const float first = rintf((static_cast<float>(i) - 2.f) * push_constants.scale) + static_cast<float>(push_constants.bias);
            const float expected_second = rintf(first * second_scale) + static_cast<float>(second_bias);
            if (fabsf(input[i] - expected_second) > 1e-6f)
            {
                fprintf(stderr, "driver_test: submission %u second-dispatch input[%u]=%g expected=%g\n", submission, i, input[i], expected_second);
                status = 1;
            }
        }
    }

    if (integer_dot_pipeline)
    {
        const uint32_t integer_input[12] = {
            0x7f80ff01u, 0x0203feffu, 0x04030201u, 0x08070605u,
            0x80000000u, 0x7fffffffu, 2u, 3u,
            0xffffffffu, 0xfffffffeu, 0x80000001u, 0xfffffffdu};
        const uint32_t integer_expected[14] = {
            0xffffff7fu, 70u, 119u, 12u, 0x80000000u, 0xffffffffu, 0x7fffffffu,
            0x7ffffffdu, 0xffffffffu, 0u, 123u, 0x80000000u, 0xffffffffu, 321u};
        uint32_t* integer_input_mapping = reinterpret_cast<uint32_t*>(input);
        uint32_t* integer_output_mapping = reinterpret_cast<uint32_t*>(output);
        memcpy(integer_input_mapping, integer_input, sizeof(integer_input));
        memset(integer_output_mapping, 0, sizeof(integer_expected));

        result = vkResetFences(device, 1, &fence);
        if (result != VK_SUCCESS) return fail("vkResetFences(integer dot)", result);
        result = vkResetCommandBuffer(command_buffer, 0);
        if (result != VK_SUCCESS) return fail("vkResetCommandBuffer(integer dot)", result);
        result = vkBeginCommandBuffer(command_buffer, &begin_info);
        if (result != VK_SUCCESS) return fail("vkBeginCommandBuffer(integer dot)", result);
        descriptor_infos[0].buffer = input_buffer;
        descriptor_infos[1].buffer = output_buffer;
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          integer_dot_pipeline);
        vkCmdPushDescriptorSetKHR(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  pipeline_layout, 0, 2, writes);
        vkCmdDispatch(command_buffer, 1, 1, 1);
        result = vkEndCommandBuffer(command_buffer);
        if (result != VK_SUCCESS) return fail("vkEndCommandBuffer(integer dot)", result);
        result = vkQueueSubmit(queue, 1, &submit_info, fence);
        if (result != VK_SUCCESS) return fail("vkQueueSubmit(integer dot)", result);
        result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        if (result != VK_SUCCESS) return fail("vkWaitForFences(integer dot)", result);
        for (uint32_t i = 0; i < 14; i++)
        {
            if (integer_output_mapping[i] != integer_expected[i])
            {
                fprintf(stderr,
                        "driver_test: integer dot output[%u]=0x%08x expected=0x%08x\n",
                        i, integer_output_mapping[i], integer_expected[i]);
                status = 1;
            }
        }
        // Restore the first-dispatch output before the generic copy-path
        // checks below reuse these buffers.
        for (uint32_t i = 0; i < 8; i++)
        {
            const uint32_t source = i - i % 4 + (3 - i % 4);
            output[i] = rintf((static_cast<float>(source) - 2.f) * push_constants.scale) + static_cast<float>(push_constants.bias);
        }
    }

    // Reuse the command buffer and fence for a device copy. This covers the
    // ncnn staging path, command-buffer reset, repeated submission and coherent
    // shadow readback without explicit flush/invalidate calls.
    result = vkResetFences(device, 1, &fence);
    if (result != VK_SUCCESS) return fail("vkResetFences(copy)", result);
    result = vkResetCommandBuffer(command_buffer, 0);
    if (result != VK_SUCCESS) return fail("vkResetCommandBuffer(copy)", result);
    result = vkBeginCommandBuffer(command_buffer, &begin_info);
    if (result != VK_SUCCESS) return fail("vkBeginCommandBuffer(copy)", result);
    VkBufferCopy copy_region = {};
    copy_region.srcOffset = descriptor_offset;
    copy_region.dstOffset = descriptor_offset;
    copy_region.size = data_size;
    vkCmdCopyBuffer(command_buffer, output_buffer, input_buffer, 1, &copy_region);
    result = vkEndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS) return fail("vkEndCommandBuffer(copy)", result);
    result = vkQueueSubmit(queue, 1, &submit_info, fence);
    if (result != VK_SUCCESS) return fail("vkQueueSubmit(copy)", result);
    result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) return fail("vkWaitForFences(copy)", result);
    for (uint32_t i = 0; i < 8; i++)
    {
        const uint32_t source = i - i % 4 + (3 - i % 4);
        const float expected = rintf((static_cast<float>(source) - 2.f) * push_constants.scale) + static_cast<float>(push_constants.bias);
        if (fabsf(input[i] - expected) > 1e-6f)
        {
            fprintf(stderr, "driver_test: copied input[%u]=%g expected=%g\n", i, input[i], expected);
            status = 1;
        }
    }

    // OpenCL 1.0 clEnqueueMarker provides a completion event for an empty
    // Vulkan submission as well.
    result = vkResetFences(device, 1, &fence);
    if (result != VK_SUCCESS) return fail("vkResetFences(empty)", result);
    result = vkQueueSubmit(queue, 0, 0, fence);
    if (result != VK_SUCCESS) return fail("vkQueueSubmit(empty)", result);
    result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) return fail("vkWaitForFences(empty)", result);

    vkDestroyFence(device, fence, 0);
    vkDestroyCommandPool(device, pool, 0);
    vkDestroyPipeline(device, duplicate_pipeline, 0);
    vkDestroyPipeline(device, subgroup_pipeline, 0);
    vkDestroyPipeline(device, integer_dot_pipeline, 0);
    vkDestroyPipeline(device, pipeline, 0);
    vkDestroyShaderModule(device, subgroup_module, 0);
    vkDestroyShaderModule(device, integer_dot_module, 0);
    vkDestroyShaderModule(device, module, 0);
    vkDestroyPipelineLayout(device, pipeline_layout, 0);
    vkDestroyDescriptorSetLayout(device, descriptor_layout, 0);
    vkDestroyBuffer(device, output_buffer, 0);
    vkDestroyBuffer(device, input_buffer, 0);
    vkFreeMemory(device, output_memory, 0);
    vkFreeMemory(device, input_memory, 0);
    vkDestroyDevice(device, 0);
    vkDestroyInstance(instance, 0);
    return status;
}
