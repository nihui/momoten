// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/objects.h"
#include "backend/pipeline_build.h"
#include "vulkan_internal.h"

momoten_detail::HandleTable<VkShaderModule, ShaderModule> g_shader_modules;
momoten_detail::HandleTable<VkDescriptorSetLayout, DescriptorSetLayout> g_descriptor_set_layouts;
momoten_detail::HandleTable<VkPipelineLayout, PipelineLayout> g_pipeline_layouts;
momoten_detail::HandleTable<VkPipeline, Pipeline> g_pipelines;
momoten_detail::HandleTable<VkDescriptorUpdateTemplate, DescriptorUpdateTemplate> g_descriptor_update_templates;
momoten_detail::HandleTable<VkPipelineCache, PipelineCache> g_pipeline_caches;

Pipeline::~Pipeline()
{
    if (kernel)
        momoten_detail::g_opencl.p_clReleaseKernel(kernel);
    if (program)
        momoten_detail::g_opencl.p_clReleaseProgram(program);
}

namespace momoten_detail {

static bool descriptor_set_layout_supported(
    const VkDescriptorSetLayoutCreateInfo* info)
{
    if (!info || info->sType != VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
        || info->pNext
        || info->flags != VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR
        || (info->bindingCount && !info->pBindings))
        return false;
    for (uint32_t i = 0; i < info->bindingCount; i++)
    {
        const VkDescriptorSetLayoutBinding& binding = info->pBindings[i];
        if (binding.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            || binding.descriptorCount != 1
            || binding.stageFlags != VK_SHADER_STAGE_COMPUTE_BIT
            || binding.pImmutableSamplers)
            return false;
        for (uint32_t j = 0; j < i; j++)
        {
            if (info->pBindings[j].binding == binding.binding)
                return false;
        }
    }
    return true;
}

VkResult impl_create_descriptor_set_layout(
    VkDevice device, const VkDescriptorSetLayoutCreateInfo* info,
    const VkAllocationCallbacks* allocator, VkDescriptorSetLayout* layout)
{
    if (!device || !info || !layout)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (allocator || !descriptor_set_layout_supported(info))
        return VK_ERROR_FEATURE_NOT_PRESENT;
    std::shared_ptr<DescriptorSetLayout> value(new DescriptorSetLayout);
    value->device = device;
    if (info->bindingCount)
        value->bindings.assign(info->pBindings,
                               info->pBindings + info->bindingCount);
    *layout = g_descriptor_set_layouts.make_handle(value);
    return VK_SUCCESS;
}

void impl_destroy_descriptor_set_layout(
    VkDevice device, VkDescriptorSetLayout layout, const VkAllocationCallbacks*)
{
    const std::shared_ptr<DescriptorSetLayout> value = g_descriptor_set_layouts.get_handle(layout);
    if (value && value->device == device)
        g_descriptor_set_layouts.erase_handle(layout);
}

void impl_get_descriptor_set_layout_support(
    VkDevice, const VkDescriptorSetLayoutCreateInfo* info, VkDescriptorSetLayoutSupport* support)
{
    if (!support)
        return;
    support->supported = descriptor_set_layout_supported(info) ? VK_TRUE : VK_FALSE;
}

VkResult impl_create_pipeline_layout(
    VkDevice device, const VkPipelineLayoutCreateInfo* info,
    const VkAllocationCallbacks* allocator, VkPipelineLayout* layout)
{
    if (!device || !info || !layout
        || info->sType != VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
        return VK_ERROR_INITIALIZATION_FAILED;
    if ((info->setLayoutCount && !info->pSetLayouts)
        || (info->pushConstantRangeCount && !info->pPushConstantRanges))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (allocator || info->pNext || info->flags || info->setLayoutCount > 1)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    const std::shared_ptr<DescriptorSetLayout> descriptor_layout = info->setLayoutCount && info->pSetLayouts
                                                                       ? g_descriptor_set_layouts.get_handle(info->pSetLayouts[0])
                                                                       : std::shared_ptr<DescriptorSetLayout>();
    if ((info->setLayoutCount && (!descriptor_layout || descriptor_layout->device != device)) || (info->pushConstantRangeCount && !info->pPushConstantRanges))
        return VK_ERROR_INITIALIZATION_FAILED;
    std::shared_ptr<PipelineLayout> value(new PipelineLayout);
    value->device = device;
    value->descriptor_layout = descriptor_layout;
    value->push_constant_size = 0;
    for (uint32_t i = 0; i < info->pushConstantRangeCount; i++)
    {
        if (info->pPushConstantRanges[i].stageFlags != VK_SHADER_STAGE_COMPUTE_BIT
            || info->pPushConstantRanges[i].size == 0
            || (info->pPushConstantRanges[i].offset & 3u) != 0
            || (info->pPushConstantRanges[i].size & 3u) != 0
            || info->pPushConstantRanges[i].offset > 256
            || info->pPushConstantRanges[i].size
                   > 256 - info->pPushConstantRanges[i].offset)
        {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        value->push_constant_size = std::max(value->push_constant_size,
                                             info->pPushConstantRanges[i].offset + info->pPushConstantRanges[i].size);
    }
    *layout = g_pipeline_layouts.make_handle(value);
    return VK_SUCCESS;
}

void impl_destroy_pipeline_layout(
    VkDevice device, VkPipelineLayout layout, const VkAllocationCallbacks*)
{
    const std::shared_ptr<PipelineLayout> value = g_pipeline_layouts.get_handle(layout);
    if (value && value->device == device)
        g_pipeline_layouts.erase_handle(layout);
}

VkResult impl_create_shader_module(
    VkDevice device, const VkShaderModuleCreateInfo* info,
    const VkAllocationCallbacks* allocator, VkShaderModule* module)
{
    if (!device || !info || !module
        || info->sType != VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
        || !info->pCode || info->codeSize < 20 || info->codeSize % 4 != 0
        || info->pCode[0] != 0x07230203u)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (allocator || info->pNext || info->flags)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    std::shared_ptr<ShaderModule> value(new ShaderModule);
    value->device = device;
    value->words.assign(info->pCode, info->pCode + info->codeSize / 4);
    *module = g_shader_modules.make_handle(value);
    return VK_SUCCESS;
}

void impl_destroy_shader_module(
    VkDevice device, VkShaderModule module, const VkAllocationCallbacks*)
{
    const std::shared_ptr<ShaderModule> value = g_shader_modules.get_handle(module);
    if (value && value->device == device)
        g_shader_modules.erase_handle(module);
}

VkResult impl_create_compute_pipelines(
    VkDevice device, VkPipelineCache pipeline_cache, uint32_t count,
    const VkComputePipelineCreateInfo* infos,
    const VkAllocationCallbacks* allocator, VkPipeline* pipelines)
{
    return build_compute_pipelines(
        device, pipeline_cache, count, infos, allocator, pipelines);
}

void impl_destroy_pipeline(
    VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks*)
{
    const std::shared_ptr<Pipeline> value = g_pipelines.get_handle(pipeline);
    if (value && value->device == device)
        g_pipelines.erase_handle(pipeline);
}

VkResult impl_create_descriptor_update_template(
    VkDevice device, const VkDescriptorUpdateTemplateCreateInfo* info,
    const VkAllocationCallbacks* allocator,
    VkDescriptorUpdateTemplate* update_template)
{
    if (!device || !info || !update_template
        || info->sType
               != VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO
        || (info->descriptorUpdateEntryCount
            && !info->pDescriptorUpdateEntries))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (allocator || info->pNext || info->flags
        || info->templateType
               != VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR
        || info->set != 0
        || info->pipelineBindPoint != VK_PIPELINE_BIND_POINT_COMPUTE)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    const std::shared_ptr<PipelineLayout> pipeline_layout = g_pipeline_layouts.get_handle(info->pipelineLayout);
    if (!pipeline_layout || pipeline_layout->device != device)
        return VK_ERROR_INITIALIZATION_FAILED;
    for (uint32_t i = 0; i < info->descriptorUpdateEntryCount; i++)
    {
        const VkDescriptorUpdateTemplateEntry& entry = info->pDescriptorUpdateEntries[i];
        if (entry.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            || entry.descriptorCount != 1 || entry.dstArrayElement != 0)
            return VK_ERROR_FEATURE_NOT_PRESENT;
        bool binding_found = false;
        if (pipeline_layout->descriptor_layout)
        {
            const std::vector<VkDescriptorSetLayoutBinding>& bindings = pipeline_layout->descriptor_layout->bindings;
            for (size_t b = 0; b < bindings.size(); b++)
            {
                if (bindings[b].binding == entry.dstBinding
                    && bindings[b].descriptorType
                           == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                    && bindings[b].descriptorCount == 1)
                {
                    binding_found = true;
                    break;
                }
            }
        }
        if (!binding_found)
            return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    std::shared_ptr<DescriptorUpdateTemplate> value(new DescriptorUpdateTemplate);
    value->device = device;
    if (info->descriptorUpdateEntryCount)
        value->entries.assign(
            info->pDescriptorUpdateEntries,
            info->pDescriptorUpdateEntries + info->descriptorUpdateEntryCount);
    *update_template = g_descriptor_update_templates.make_handle(value);
    return VK_SUCCESS;
}

void impl_destroy_descriptor_update_template(
    VkDevice device, VkDescriptorUpdateTemplate update_template, const VkAllocationCallbacks*)
{
    const std::shared_ptr<DescriptorUpdateTemplate> value = g_descriptor_update_templates.get_handle(update_template);
    if (value && value->device == device)
        g_descriptor_update_templates.erase_handle(update_template);
}

VkResult impl_create_pipeline_cache(
    VkDevice device, const VkPipelineCacheCreateInfo* info,
    const VkAllocationCallbacks* allocator, VkPipelineCache* cache)
{
    if (!device || !info || !cache
        || info->sType != VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (allocator || info->pNext || info->flags || info->initialDataSize
        || info->pInitialData)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    std::shared_ptr<PipelineCache> value(new PipelineCache);
    value->device = device;
    *cache = g_pipeline_caches.make_handle(value);
    return VK_SUCCESS;
}

void impl_destroy_pipeline_cache(
    VkDevice device, VkPipelineCache cache, const VkAllocationCallbacks*)
{
    const std::shared_ptr<PipelineCache> value = g_pipeline_caches.get_handle(cache);
    if (value && value->device == device)
        g_pipeline_caches.erase_handle(cache);
}

VkResult impl_get_pipeline_cache_data(
    VkDevice, VkPipelineCache, size_t* size, void*)
{
    if (!size)
        return VK_ERROR_INITIALIZATION_FAILED;
    *size = 0;
    return VK_SUCCESS;
}

VkResult impl_merge_pipeline_caches(
    VkDevice, VkPipelineCache, uint32_t, const VkPipelineCache*)
{
    return VK_SUCCESS;
}

} // namespace momoten_detail
