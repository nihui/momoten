// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

// This file is intentionally included multiple times. The includer must define
// MOMOTEN_RESULT_PROC, MOMOTEN_VOID_PROC, and MOMOTEN_PFN_PROC. Entries whose
// metadata scope is PROC_SCOPE_NONE remain directly exported but are excluded
// from Vulkan proc lookup, preserving the historical driver ABI.

MOMOTEN_PFN_PROC(vkGetInstanceProcAddr, impl_get_instance_proc_addr,
                 (VkInstance instance, const char* name), (instance, name),
                 PROC_SCOPE_GLOBAL, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkEnumerateInstanceExtensionProperties, impl_enumerate_instance_extension_properties,
                    (const char* layer_name, uint32_t* count, VkExtensionProperties* properties),
                    (layer_name, count, properties), PROC_SCOPE_GLOBAL, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkEnumerateInstanceLayerProperties, impl_enumerate_instance_layer_properties,
                    (uint32_t* count, VkLayerProperties* properties), (count, properties),
                    PROC_SCOPE_GLOBAL, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkEnumerateInstanceVersion, impl_enumerate_instance_version,
                    (uint32_t* version), (version), PROC_SCOPE_GLOBAL, 0, VK_API_VERSION_1_1)
MOMOTEN_RESULT_PROC(vkCreateInstance, impl_create_instance,
                    (const VkInstanceCreateInfo* create_info, const VkAllocationCallbacks* allocator, VkInstance* instance),
                    (create_info, allocator, instance), PROC_SCOPE_GLOBAL, 0, VK_API_VERSION_1_0)

MOMOTEN_VOID_PROC(vkDestroyInstance, impl_destroy_instance,
                  (VkInstance instance, const VkAllocationCallbacks* allocator), (instance, allocator),
                  PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkEnumeratePhysicalDevices, impl_enumerate_physical_devices,
                    (VkInstance instance, uint32_t* count, VkPhysicalDevice* devices), (instance, count, devices),
                    PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceFeatures, impl_get_physical_device_features,
                  (VkPhysicalDevice physical, VkPhysicalDeviceFeatures* features), (physical, features),
                  PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceFeatures2, impl_get_physical_device_features2,
                  (VkPhysicalDevice physical, VkPhysicalDeviceFeatures2* features), (physical, features),
                  PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_1)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceFeatures2KHR, impl_get_physical_device_features2,
                  (VkPhysicalDevice physical, VkPhysicalDeviceFeatures2* features), (physical, features),
                  PROC_SCOPE_INSTANCE, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, 0)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceProperties, impl_get_physical_device_properties,
                  (VkPhysicalDevice physical, VkPhysicalDeviceProperties* properties), (physical, properties),
                  PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceProperties2, impl_get_physical_device_properties2,
                  (VkPhysicalDevice physical, VkPhysicalDeviceProperties2* properties), (physical, properties),
                  PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_1)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceProperties2KHR, impl_get_physical_device_properties2,
                  (VkPhysicalDevice physical, VkPhysicalDeviceProperties2* properties), (physical, properties),
                  PROC_SCOPE_INSTANCE, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, 0)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceQueueFamilyProperties, impl_get_physical_device_queue_family_properties,
                  (VkPhysicalDevice physical, uint32_t* count, VkQueueFamilyProperties* properties),
                  (physical, count, properties), PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceQueueFamilyProperties2, impl_get_physical_device_queue_family_properties2,
                  (VkPhysicalDevice physical, uint32_t* count, VkQueueFamilyProperties2* properties),
                  (physical, count, properties), PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_1)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceQueueFamilyProperties2KHR, impl_get_physical_device_queue_family_properties2,
                  (VkPhysicalDevice physical, uint32_t* count, VkQueueFamilyProperties2* properties),
                  (physical, count, properties), PROC_SCOPE_INSTANCE,
                  VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, 0)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceMemoryProperties, impl_get_physical_device_memory_properties,
                  (VkPhysicalDevice physical, VkPhysicalDeviceMemoryProperties* properties), (physical, properties),
                  PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceMemoryProperties2, impl_get_physical_device_memory_properties2,
                  (VkPhysicalDevice physical, VkPhysicalDeviceMemoryProperties2* properties), (physical, properties),
                  PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_1)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceMemoryProperties2KHR, impl_get_physical_device_memory_properties2,
                  (VkPhysicalDevice physical, VkPhysicalDeviceMemoryProperties2* properties), (physical, properties),
                  PROC_SCOPE_INSTANCE, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, 0)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceFormatProperties, impl_get_physical_device_format_properties,
                  (VkPhysicalDevice physical, VkFormat format, VkFormatProperties* properties),
                  (physical, format, properties), PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceFormatProperties2, impl_get_physical_device_format_properties2,
                  (VkPhysicalDevice physical, VkFormat format, VkFormatProperties2* properties),
                  (physical, format, properties), PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_1)
MOMOTEN_VOID_PROC(vkGetPhysicalDeviceFormatProperties2KHR, impl_get_physical_device_format_properties2,
                  (VkPhysicalDevice physical, VkFormat format, VkFormatProperties2* properties),
                  (physical, format, properties), PROC_SCOPE_INSTANCE,
                  VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, 0)
MOMOTEN_RESULT_PROC(vkGetPhysicalDeviceImageFormatProperties, impl_get_physical_device_image_format_properties,
                    (VkPhysicalDevice physical, VkFormat format, VkImageType type, VkImageTiling tiling,
                     VkImageUsageFlags usage, VkImageCreateFlags flags, VkImageFormatProperties* properties),
                    (physical, format, type, tiling, usage, flags, properties),
                    PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkGetPhysicalDeviceImageFormatProperties2, impl_get_physical_device_image_format_properties2,
                    (VkPhysicalDevice physical, const VkPhysicalDeviceImageFormatInfo2* info,
                     VkImageFormatProperties2* properties),
                    (physical, info, properties),
                    PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_1)
MOMOTEN_RESULT_PROC(vkGetPhysicalDeviceImageFormatProperties2KHR, impl_get_physical_device_image_format_properties2,
                    (VkPhysicalDevice physical, const VkPhysicalDeviceImageFormatInfo2* info,
                     VkImageFormatProperties2* properties),
                    (physical, info, properties),
                    PROC_SCOPE_INSTANCE, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, 0)
MOMOTEN_RESULT_PROC(vkEnumerateDeviceExtensionProperties, impl_enumerate_device_extension_properties,
                    (VkPhysicalDevice physical, const char* layer_name, uint32_t* count,
                     VkExtensionProperties* properties),
                    (physical, layer_name, count, properties),
                    PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkEnumerateDeviceLayerProperties, impl_enumerate_device_layer_properties,
                    (VkPhysicalDevice physical, uint32_t* count, VkLayerProperties* properties),
                    (physical, count, properties), PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkCreateDevice, impl_create_device,
                    (VkPhysicalDevice physical, const VkDeviceCreateInfo* create_info,
                     const VkAllocationCallbacks* allocator, VkDevice* device),
                    (physical, create_info, allocator, device), PROC_SCOPE_INSTANCE, 0, VK_API_VERSION_1_0)

MOMOTEN_PFN_PROC(vkGetDeviceProcAddr, impl_get_device_proc_addr,
                 (VkDevice device, const char* name), (device, name), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkDestroyDevice, impl_destroy_device,
                  (VkDevice device, const VkAllocationCallbacks* allocator), (device, allocator),
                  PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkGetDeviceQueue, impl_get_device_queue,
                  (VkDevice device, uint32_t family, uint32_t index, VkQueue* queue),
                  (device, family, index, queue), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkGetDeviceQueue2, impl_get_device_queue2,
                  (VkDevice device, const VkDeviceQueueInfo2* info, VkQueue* queue), (device, info, queue),
                  PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_1)
MOMOTEN_RESULT_PROC(vkAllocateMemory, impl_allocate_memory,
                    (VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* allocator,
                     VkDeviceMemory* memory),
                    (device, info, allocator, memory),
                    PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkFreeMemory, impl_free_memory,
                  (VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* allocator),
                  (device, memory, allocator), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkMapMemory, impl_map_memory,
                    (VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size,
                     VkMemoryMapFlags flags, void** data),
                    (device, memory, offset, size, flags, data),
                    PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkUnmapMemory, impl_unmap_memory,
                  (VkDevice device, VkDeviceMemory memory), (device, memory),
                  PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkFlushMappedMemoryRanges, impl_flush_mapped_memory_ranges,
                    (VkDevice device, uint32_t count, const VkMappedMemoryRange* ranges), (device, count, ranges),
                    PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkInvalidateMappedMemoryRanges, impl_invalidate_mapped_memory_ranges,
                    (VkDevice device, uint32_t count, const VkMappedMemoryRange* ranges), (device, count, ranges),
                    PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkGetDeviceMemoryCommitment, impl_get_device_memory_commitment,
                  (VkDevice device, VkDeviceMemory memory, VkDeviceSize* committed), (device, memory, committed),
                  PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkCreateBuffer, impl_create_buffer,
                    (VkDevice device, const VkBufferCreateInfo* info, const VkAllocationCallbacks* allocator,
                     VkBuffer* buffer),
                    (device, info, allocator, buffer), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkDestroyBuffer, impl_destroy_buffer,
                  (VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* allocator),
                  (device, buffer, allocator), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkGetBufferMemoryRequirements, impl_get_buffer_memory_requirements,
                  (VkDevice device, VkBuffer buffer, VkMemoryRequirements* requirements),
                  (device, buffer, requirements), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkGetBufferMemoryRequirements2, impl_get_buffer_memory_requirements2,
                  (VkDevice device, const VkBufferMemoryRequirementsInfo2* info, VkMemoryRequirements2* requirements),
                  (device, info, requirements), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_1)
MOMOTEN_VOID_PROC(vkGetBufferMemoryRequirements2KHR, impl_get_buffer_memory_requirements2,
                  (VkDevice device, const VkBufferMemoryRequirementsInfo2* info, VkMemoryRequirements2* requirements),
                  (device, info, requirements), PROC_SCOPE_NONE, 0, 0)
MOMOTEN_RESULT_PROC(vkBindBufferMemory, impl_bind_buffer_memory,
                    (VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset),
                    (device, buffer, memory, offset), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkBindBufferMemory2, impl_bind_buffer_memory2,
                    (VkDevice device, uint32_t count, const VkBindBufferMemoryInfo* infos), (device, count, infos),
                    PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_1)
MOMOTEN_RESULT_PROC(vkBindBufferMemory2KHR, impl_bind_buffer_memory2,
                    (VkDevice device, uint32_t count, const VkBindBufferMemoryInfo* infos), (device, count, infos),
                    PROC_SCOPE_NONE, 0, 0)
MOMOTEN_RESULT_PROC(vkCreateDescriptorSetLayout, impl_create_descriptor_set_layout,
                    (VkDevice device, const VkDescriptorSetLayoutCreateInfo* info,
                     const VkAllocationCallbacks* allocator, VkDescriptorSetLayout* layout),
                    (device, info, allocator, layout), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkDestroyDescriptorSetLayout, impl_destroy_descriptor_set_layout,
                  (VkDevice device, VkDescriptorSetLayout layout, const VkAllocationCallbacks* allocator),
                  (device, layout, allocator), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkGetDescriptorSetLayoutSupport, impl_get_descriptor_set_layout_support,
                  (VkDevice device, const VkDescriptorSetLayoutCreateInfo* info, VkDescriptorSetLayoutSupport* support),
                  (device, info, support), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_1)
MOMOTEN_VOID_PROC(vkGetDescriptorSetLayoutSupportKHR, impl_get_descriptor_set_layout_support,
                  (VkDevice device, const VkDescriptorSetLayoutCreateInfo* info, VkDescriptorSetLayoutSupport* support),
                  (device, info, support), PROC_SCOPE_DEVICE, VK_KHR_MAINTENANCE3_EXTENSION_NAME, 0)
MOMOTEN_RESULT_PROC(vkCreatePipelineLayout, impl_create_pipeline_layout,
                    (VkDevice device, const VkPipelineLayoutCreateInfo* info,
                     const VkAllocationCallbacks* allocator, VkPipelineLayout* layout),
                    (device, info, allocator, layout), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkDestroyPipelineLayout, impl_destroy_pipeline_layout,
                  (VkDevice device, VkPipelineLayout layout, const VkAllocationCallbacks* allocator),
                  (device, layout, allocator), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkCreateShaderModule, impl_create_shader_module,
                    (VkDevice device, const VkShaderModuleCreateInfo* info,
                     const VkAllocationCallbacks* allocator, VkShaderModule* module),
                    (device, info, allocator, module), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkDestroyShaderModule, impl_destroy_shader_module,
                  (VkDevice device, VkShaderModule module, const VkAllocationCallbacks* allocator),
                  (device, module, allocator), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkCreateComputePipelines, impl_create_compute_pipelines,
                    (VkDevice device, VkPipelineCache pipeline_cache, uint32_t count,
                     const VkComputePipelineCreateInfo* infos, const VkAllocationCallbacks* allocator,
                     VkPipeline* pipelines),
                    (device, pipeline_cache, count, infos, allocator, pipelines),
                    PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkDestroyPipeline, impl_destroy_pipeline,
                  (VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* allocator),
                  (device, pipeline, allocator), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkCreateDescriptorUpdateTemplate, impl_create_descriptor_update_template,
                    (VkDevice device, const VkDescriptorUpdateTemplateCreateInfo* info,
                     const VkAllocationCallbacks* allocator, VkDescriptorUpdateTemplate* update_template),
                    (device, info, allocator, update_template), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_1)
MOMOTEN_RESULT_PROC(vkCreateDescriptorUpdateTemplateKHR, impl_create_descriptor_update_template,
                    (VkDevice device, const VkDescriptorUpdateTemplateCreateInfo* info,
                     const VkAllocationCallbacks* allocator, VkDescriptorUpdateTemplate* update_template),
                    (device, info, allocator, update_template), PROC_SCOPE_DEVICE,
                    VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME, 0)
MOMOTEN_VOID_PROC(vkDestroyDescriptorUpdateTemplate, impl_destroy_descriptor_update_template,
                  (VkDevice device, VkDescriptorUpdateTemplate update_template,
                   const VkAllocationCallbacks* allocator),
                  (device, update_template, allocator),
                  PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_1)
MOMOTEN_VOID_PROC(vkDestroyDescriptorUpdateTemplateKHR, impl_destroy_descriptor_update_template,
                  (VkDevice device, VkDescriptorUpdateTemplate update_template,
                   const VkAllocationCallbacks* allocator),
                  (device, update_template, allocator),
                  PROC_SCOPE_DEVICE, VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME, 0)
MOMOTEN_RESULT_PROC(vkCreateCommandPool, impl_create_command_pool,
                    (VkDevice device, const VkCommandPoolCreateInfo* info, const VkAllocationCallbacks* allocator,
                     VkCommandPool* pool),
                    (device, info, allocator, pool),
                    PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkDestroyCommandPool, impl_destroy_command_pool,
                  (VkDevice device, VkCommandPool pool, const VkAllocationCallbacks* allocator),
                  (device, pool, allocator), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkAllocateCommandBuffers, impl_allocate_command_buffers,
                    (VkDevice device, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* buffers),
                    (device, info, buffers), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkFreeCommandBuffers, impl_free_command_buffers,
                  (VkDevice device, VkCommandPool pool, uint32_t count, const VkCommandBuffer* buffers),
                  (device, pool, count, buffers), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkResetCommandPool, impl_reset_command_pool,
                    (VkDevice device, VkCommandPool pool, VkCommandPoolResetFlags flags), (device, pool, flags),
                    PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkTrimCommandPool, impl_trim_command_pool,
                  (VkDevice device, VkCommandPool pool, VkCommandPoolTrimFlags flags), (device, pool, flags),
                  PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_1)
MOMOTEN_VOID_PROC(vkTrimCommandPoolKHR, impl_trim_command_pool,
                  (VkDevice device, VkCommandPool pool, VkCommandPoolTrimFlags flags), (device, pool, flags),
                  PROC_SCOPE_DEVICE, VK_KHR_MAINTENANCE1_EXTENSION_NAME, 0)
MOMOTEN_RESULT_PROC(vkResetCommandBuffer, impl_reset_command_buffer,
                    (VkCommandBuffer buffer, VkCommandBufferResetFlags flags), (buffer, flags),
                    PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkBeginCommandBuffer, impl_begin_command_buffer,
                    (VkCommandBuffer buffer, const VkCommandBufferBeginInfo* info), (buffer, info),
                    PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkEndCommandBuffer, impl_end_command_buffer,
                    (VkCommandBuffer buffer), (buffer), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkCmdBindPipeline, impl_cmd_bind_pipeline,
                  (VkCommandBuffer buffer, VkPipelineBindPoint bind_point, VkPipeline pipeline),
                  (buffer, bind_point, pipeline), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkCmdPushDescriptorSetKHR, impl_cmd_push_descriptor_set,
                  (VkCommandBuffer buffer, VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                   uint32_t set, uint32_t write_count, const VkWriteDescriptorSet* writes),
                  (buffer, bind_point, layout, set, write_count, writes), PROC_SCOPE_DEVICE,
                  VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, 0)
MOMOTEN_VOID_PROC(vkCmdPushDescriptorSetWithTemplateKHR, impl_cmd_push_descriptor_set_with_template,
                  (VkCommandBuffer buffer, VkDescriptorUpdateTemplate update_template, VkPipelineLayout layout,
                   uint32_t set, const void* data),
                  (buffer, update_template, layout, set, data),
                  PROC_SCOPE_DEVICE, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, 0)
MOMOTEN_VOID_PROC(vkCmdPushConstants, impl_cmd_push_constants,
                  (VkCommandBuffer buffer, VkPipelineLayout layout, VkShaderStageFlags stage_flags,
                   uint32_t offset, uint32_t size, const void* values),
                  (buffer, layout, stage_flags, offset, size, values), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkCmdDispatch, impl_cmd_dispatch,
                  (VkCommandBuffer buffer, uint32_t x, uint32_t y, uint32_t z), (buffer, x, y, z),
                  PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkCmdCopyBuffer, impl_cmd_copy_buffer,
                  (VkCommandBuffer buffer, VkBuffer src, VkBuffer dst, uint32_t region_count,
                   const VkBufferCopy* regions),
                  (buffer, src, dst, region_count, regions),
                  PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkCmdPipelineBarrier, impl_cmd_pipeline_barrier,
                  (VkCommandBuffer buffer, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                   VkDependencyFlags dependency_flags, uint32_t memory_barrier_count,
                   const VkMemoryBarrier* memory_barriers, uint32_t buffer_barrier_count,
                   const VkBufferMemoryBarrier* buffer_barriers, uint32_t image_barrier_count,
                   const VkImageMemoryBarrier* image_barriers),
                  (buffer, src_stage, dst_stage, dependency_flags, memory_barrier_count, memory_barriers,
                   buffer_barrier_count, buffer_barriers, image_barrier_count, image_barriers),
                  PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkQueueSubmit, impl_queue_submit,
                    (VkQueue queue, uint32_t submit_count, const VkSubmitInfo* submits, VkFence fence),
                    (queue, submit_count, submits, fence), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkQueueWaitIdle, impl_queue_wait_idle,
                    (VkQueue queue), (queue), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkDeviceWaitIdle, impl_device_wait_idle,
                    (VkDevice device), (device), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkCreateFence, impl_create_fence,
                    (VkDevice device, const VkFenceCreateInfo* info, const VkAllocationCallbacks* allocator,
                     VkFence* fence),
                    (device, info, allocator, fence), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkDestroyFence, impl_destroy_fence,
                  (VkDevice device, VkFence fence, const VkAllocationCallbacks* allocator),
                  (device, fence, allocator), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkGetFenceStatus, impl_get_fence_status,
                    (VkDevice device, VkFence fence), (device, fence), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkResetFences, impl_reset_fences,
                    (VkDevice device, uint32_t count, const VkFence* fences), (device, count, fences),
                    PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkWaitForFences, impl_wait_for_fences,
                    (VkDevice device, uint32_t count, const VkFence* fences, VkBool32 wait_all, uint64_t timeout),
                    (device, count, fences, wait_all, timeout), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkCreatePipelineCache, impl_create_pipeline_cache,
                    (VkDevice device, const VkPipelineCacheCreateInfo* info,
                     const VkAllocationCallbacks* allocator, VkPipelineCache* cache),
                    (device, info, allocator, cache), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_VOID_PROC(vkDestroyPipelineCache, impl_destroy_pipeline_cache,
                  (VkDevice device, VkPipelineCache cache, const VkAllocationCallbacks* allocator),
                  (device, cache, allocator), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkGetPipelineCacheData, impl_get_pipeline_cache_data,
                    (VkDevice device, VkPipelineCache cache, size_t* size, void* data),
                    (device, cache, size, data), PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
MOMOTEN_RESULT_PROC(vkMergePipelineCaches, impl_merge_pipeline_caches,
                    (VkDevice device, VkPipelineCache cache, uint32_t source_count,
                     const VkPipelineCache* source_caches),
                    (device, cache, source_count, source_caches),
                    PROC_SCOPE_DEVICE, 0, VK_API_VERSION_1_0)
