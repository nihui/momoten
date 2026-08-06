// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_OBJECTS_H
#define MOMOTEN_BACKEND_OBJECTS_H

#include "../vulkan_internal.h"
#include "command_stream.h"
#include "device_profile.h"
#include "memory_state.h"
#include "opencl_loader.h"
#include "momoten/spv_to_clc.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

struct VkInstance_T
{
    uint32_t api_version;
    std::vector<std::string> enabled_extensions;
    std::vector<VkPhysicalDevice> physical_devices;
};

struct VkPhysicalDevice_T
{
    VkInstance owner;
    cl_platform_id platform;
    cl_device_id device;
    std::string name;
    std::string vendor;
    std::string driver;
    std::string extensions;
    cl_device_type opencl_device_type;
    cl_uint vendor_id;
    uint32_t device_id;
    VkPhysicalDeviceType vulkan_device_type;
    cl_uint address_bits;
    cl_ulong global_memory;
    cl_ulong max_allocation;
    size_t max_workgroup_size;
    size_t max_compute_workgroup_invocations;
    size_t max_work_item_sizes[3];
    size_t max_parameter_size;
    cl_ulong local_memory;
    cl_uint max_compute_units;
    cl_uint mem_base_alignment_bits;
    momoten_detail::ShaderDeviceProfile shader_profile;
};

struct VkQueue_T;

struct PendingSubmission
{
    cl_event event;
    std::vector<VkCommandBuffer> command_buffers;
};

struct VkDevice_T
{
    VkPhysicalDevice physical_device;
    uint32_t api_version;
    std::vector<std::string> enabled_extensions;
    cl_context context;
    cl_command_queue command_queue;
    VkQueue queue;
    std::mutex queue_mutex;
    std::vector<PendingSubmission> pending_submissions;
    std::mutex allocation_mutex;
    std::vector<std::weak_ptr<struct DeviceMemory> > allocations;
    std::mutex program_cache_mutex;
    std::map<std::pair<uint64_t, uint64_t>, cl_program> program_cache;
    size_t program_cache_capacity;
    momoten_detail::EnabledShaderProfile enabled_shader_profile;
};

struct VkQueue_T
{
    VkDevice device;
};

struct DeviceMemory
{
    ~DeviceMemory();

    VkDevice device;
    VkDeviceSize size;
    uint32_t memory_type_index;
    bool host_visible;
    bool mapped;
    VkDeviceSize mapped_offset;
    VkDeviceSize mapped_size;
    cl_mem memory;
    std::vector<unsigned char> shadow;
    std::mutex mutex;
    momoten_detail::MemoryRangeMap range_map;
};

struct Buffer
{
    VkDevice device;
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    std::shared_ptr<DeviceMemory> memory;
    VkDeviceSize memory_offset;
};

struct ShaderModule
{
    VkDevice device;
    std::vector<uint32_t> words;
};

struct DescriptorSetLayout
{
    VkDevice device;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
};

struct PipelineLayout
{
    VkDevice device;
    std::shared_ptr<DescriptorSetLayout> descriptor_layout;
    uint32_t push_constant_size;
};

struct Pipeline
{
    ~Pipeline();

    VkDevice device;
    std::shared_ptr<PipelineLayout> layout;
    cl_program program;
    cl_kernel kernel;
    momoten::KernelABI abi;
    size_t opencl_local_size[3];
    size_t workgroup_chunk_count;
};

struct DescriptorUpdateTemplate
{
    VkDevice device;
    std::vector<VkDescriptorUpdateTemplateEntry> entries;
};

struct PipelineCache
{
    VkDevice device;
};

struct VkCommandBuffer_T;

struct CommandPool
{
    ~CommandPool();

    VkDevice device;
    std::vector<VkCommandBuffer> buffers;
};

struct VkCommandBuffer_T
{
    VkDevice device;
    std::weak_ptr<CommandPool> pool;
    momoten_detail::CommandBufferState state;
    VkResult error;
    VkCommandBufferUsageFlags usage;
    uint32_t pending_count;
    std::shared_ptr<Pipeline> current_pipeline;
    std::map<uint32_t, DescriptorValue> current_descriptors;
    std::vector<unsigned char> current_push_constants;
    std::vector<RecordedCommand> commands;
};

struct Fence
{
    ~Fence();

    VkDevice device;
    std::mutex mutex;
    cl_event event;
    bool signaled;
};

extern momoten_detail::HandleTable<VkDeviceMemory, DeviceMemory> g_device_memories;
extern momoten_detail::HandleTable<VkBuffer, Buffer> g_buffers;
extern momoten_detail::HandleTable<VkShaderModule, ShaderModule> g_shader_modules;
extern momoten_detail::HandleTable<VkDescriptorSetLayout, DescriptorSetLayout> g_descriptor_set_layouts;
extern momoten_detail::HandleTable<VkPipelineLayout, PipelineLayout> g_pipeline_layouts;
extern momoten_detail::HandleTable<VkPipeline, Pipeline> g_pipelines;
extern momoten_detail::HandleTable<VkDescriptorUpdateTemplate, DescriptorUpdateTemplate> g_descriptor_update_templates;
extern momoten_detail::HandleTable<VkPipelineCache, PipelineCache> g_pipeline_caches;
extern momoten_detail::HandleTable<VkCommandPool, CommandPool> g_command_pools;
extern momoten_detail::HandleTable<VkFence, Fence> g_fences;

#endif // MOMOTEN_BACKEND_OBJECTS_H
