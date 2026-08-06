// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/cache.h"
#include "backend/device.h"
#include "backend/objects.h"
#include "backend/program.h"
#include "vulkan_internal.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

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

static void dump_pipeline_artifacts(const std::shared_ptr<ShaderModule>& module,
                                    const momoten::TranslationOptions& options,
                                    const momoten::TranslationResult& translated)
{
    const char* directory_value = getenv("MOMOTEN_DUMP_DIR");
    if (!directory_value || !directory_value[0])
        directory_value = getenv("NCNN_MOMOTEN_DUMP_DIR");
    if (!directory_value || !directory_value[0])
        return;
    const std::string directory(directory_value);
    if (!ensure_cache_directory(directory))
    {
        fprintf(stderr, "[momoten] cannot create dump directory %s\n", directory.c_str());
        return;
    }

    uint64_t hash = 1469598103934665603ull;
    hash = fnv1a64(module->words.data(), module->words.size() * sizeof(uint32_t), hash);
    hash = fnv1a64_string(options.entry_point, hash);
    hash = fnv1a64(&options.subgroup_mode, sizeof(options.subgroup_mode), hash);
    hash = fnv1a64(&options.subgroup_size, sizeof(options.subgroup_size), hash);
    hash = fnv1a64(&options.integer_dot_product,
                   sizeof(options.integer_dot_product), hash);
    hash = fnv1a64(&options.integer_dot_product_input_4x8bit,
                   sizeof(options.integer_dot_product_input_4x8bit), hash);
    hash = fnv1a64(&options.integer_dot_product_input_4x8bit_packed,
                   sizeof(options.integer_dot_product_input_4x8bit_packed), hash);
    for (size_t i = 0; i < options.specializations.size(); i++)
    {
        hash = fnv1a64(&options.specializations[i].constant_id,
                       sizeof(options.specializations[i].constant_id), hash);
        hash = fnv1a64(options.specializations[i].data.data(),
                       options.specializations[i].data.size(), hash);
    }

    std::ostringstream base;
    base << directory;
    if (directory[directory.size() - 1] != '/' && directory[directory.size() - 1] != '\\')
        base << '/';
    base << "momo-" << std::hex << std::setfill('0') << std::setw(16) << hash;

    std::ofstream spirv((base.str() + ".spv").c_str(), std::ios::binary | std::ios::trunc);
    spirv.write(reinterpret_cast<const char*>(module->words.data()),
                module->words.size() * sizeof(uint32_t));

    std::ofstream source((base.str() + ".cl").c_str(), std::ios::binary | std::ios::trunc);
    source.write(translated.source.data(), translated.source.size());

    std::ofstream abi((base.str() + ".abi.txt").c_str(), std::ios::trunc);
    abi << "entry_point=" << translated.abi.entry_point << '\n'
        << "address_bits=" << translated.abi.address_bits << '\n'
        << "local_size=" << translated.abi.local_size[0] << ','
        << translated.abi.local_size[1] << ',' << translated.abi.local_size[2] << '\n'
        << "subgroup_mode="
        << (translated.abi.subgroup_mode == momoten::SubgroupModeNative
                ? "native"
            : translated.abi.subgroup_mode == momoten::SubgroupModeEmulatedBasic
                ? "emulated-basic"
                : "singleton")
        << '\n'
        << "subgroup_size=" << translated.abi.subgroup_size << '\n'
        << "integer_dot_product=" << translated.abi.integer_dot_product << '\n'
        << "workgroup_splittable=" << translated.abi.workgroup_splittable << '\n'
        << "push_constant_arg=" << translated.abi.push_constant_arg_index << '\n'
        << "push_constant_size=" << translated.abi.push_constant_size << '\n';
    if (!translated.diagnostics.empty())
        abi << "diagnostics=" << translated.diagnostics << '\n';
    for (size_t i = 0; i < options.specializations.size(); i++)
    {
        abi << "specialization[" << i << "].constant_id="
            << options.specializations[i].constant_id << '\n';
        abi << "specialization[" << i << "].data=";
        for (size_t j = 0; j < options.specializations[i].data.size(); j++)
            abi << std::hex << std::setfill('0') << std::setw(2)
                << static_cast<unsigned>(options.specializations[i].data[j]);
        abi << std::dec << '\n';
    }
    for (size_t i = 0; i < translated.abi.buffers.size(); i++)
    {
        const momoten::BufferArgument& buffer = translated.abi.buffers[i];
        abi << "buffer[" << i << "].set=" << buffer.descriptor_set << '\n'
            << "buffer[" << i << "].binding=" << buffer.binding << '\n'
            << "buffer[" << i << "].buffer_arg=" << buffer.buffer_arg_index << '\n'
            << "buffer[" << i << "].offset_arg=" << buffer.offset_arg_index << '\n'
            << "buffer[" << i << "].size_arg=" << buffer.size_arg_index << '\n'
            << "buffer[" << i << "].access=" << static_cast<unsigned>(buffer.access) << '\n';
    }
    for (size_t i = 0; i < translated.abi.required_extensions.size(); i++)
        abi << "required_extension[" << i << "]="
            << translated.abi.required_extensions[i] << '\n';
    fprintf(stderr, "[momoten] dumped pipeline artifacts to %s.{spv,cl,abi.txt}\n",
            base.str().c_str());
}

VkResult impl_create_compute_pipelines(
    VkDevice device, VkPipelineCache pipeline_cache, uint32_t count, const VkComputePipelineCreateInfo* infos,
    const VkAllocationCallbacks* allocator, VkPipeline* pipelines)
{
    if (!device || count == 0 || !infos || !pipelines)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (pipeline_cache)
    {
        const std::shared_ptr<PipelineCache> cache = g_pipeline_caches.get_handle(pipeline_cache);
        if (!cache || cache->device != device)
            return VK_ERROR_INITIALIZATION_FAILED;
    }

    for (uint32_t i = 0; i < count; i++)
        pipelines[i] = VK_NULL_HANDLE;

    for (uint32_t i = 0; i < count; i++)
    {
        const VkComputePipelineCreateInfo& info = infos[i];
        const std::shared_ptr<ShaderModule> module = g_shader_modules.get_handle(info.stage.module);
        const std::shared_ptr<PipelineLayout> layout = g_pipeline_layouts.get_handle(info.layout);
        if (info.sType != VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
            || info.stage.sType
                   != VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
            || info.stage.stage != VK_SHADER_STAGE_COMPUTE_BIT || !module
            || !layout || !info.stage.pName || !info.stage.pName[0]
            || module->device != device || layout->device != device)
            return VK_ERROR_INITIALIZATION_FAILED;
        if (info.pNext || info.flags || info.stage.pNext || info.stage.flags)
            return VK_ERROR_FEATURE_NOT_PRESENT;

        momoten::TranslationOptions options;
        options.entry_point = info.stage.pName;
        momoten_detail::configure_translation_options(device, options);
        if (info.stage.pSpecializationInfo)
        {
            const VkSpecializationInfo& specialization = *info.stage.pSpecializationInfo;
            if ((specialization.mapEntryCount && !specialization.pMapEntries)
                || (specialization.dataSize && !specialization.pData))
                return VK_ERROR_INITIALIZATION_FAILED;
            for (uint32_t e = 0; e < specialization.mapEntryCount; e++)
            {
                const VkSpecializationMapEntry& entry = specialization.pMapEntries[e];
                if (entry.size == 0 || !specialization.pData
                    || entry.offset > specialization.dataSize
                    || entry.size > specialization.dataSize - entry.offset)
                    return VK_ERROR_INITIALIZATION_FAILED;
                momoten::SpecializationValue value;
                value.constant_id = entry.constantID;
                const unsigned char* data = static_cast<const unsigned char*>(specialization.pData) + entry.offset;
                value.data.assign(data, data + entry.size);
                options.specializations.push_back(value);
            }
        }

        momoten::TranslationResult translated;
        const bool translation_succeeded = momoten::translate_spirv_to_opencl_c(
            module->words.data(), module->words.size(), options, translated);
        size_t requested_workgroup_size = 1;
        if (translation_succeeded)
        {
            for (size_t d = 0; d < 3; d++)
            {
                if (translated.abi.local_size[d] == 0 || requested_workgroup_size > SIZE_MAX / translated.abi.local_size[d])
                {
                    fprintf(stderr, "[momoten] translated workgroup invocation count overflows size_t\n");
                    return VK_ERROR_FEATURE_NOT_PRESENT;
                }
                requested_workgroup_size *= translated.abi.local_size[d];
            }
        }
        dump_pipeline_artifacts(module, options, translated);
        if (!translation_succeeded)
        {
            fprintf(stderr, "[momoten] SPIR-V translation failed: %s\n", translated.diagnostics.c_str());
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }

        if (translated.abi.push_constant_size > layout->push_constant_size)
        {
            fprintf(stderr, "[momoten] shader push-constant block exceeds the pipeline layout\n");
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        for (size_t b = 0; b < translated.abi.buffers.size(); b++)
        {
            bool binding_found = false;
            if (layout->descriptor_layout)
            {
                const std::vector<VkDescriptorSetLayoutBinding>& bindings = layout->descriptor_layout->bindings;
                for (size_t j = 0; j < bindings.size(); j++)
                {
                    if (bindings[j].binding == translated.abi.buffers[b].binding && bindings[j].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && bindings[j].descriptorCount == 1)
                    {
                        binding_found = true;
                        break;
                    }
                }
            }
            if (!binding_found)
            {
                fprintf(stderr, "[momoten] SPIR-V descriptor binding %u is absent from the pipeline layout\n",
                        translated.abi.buffers[b].binding);
                return VK_ERROR_FEATURE_NOT_PRESENT;
            }
        }

        std::string abi_diagnostic;
        if (!momoten_detail::validate_pipeline_abi(
                device, translated.abi, abi_diagnostic))
        {
            fprintf(stderr, "[momoten] pipeline ABI is not supported by this OpenCL device: %s\n",
                    abi_diagnostic.c_str());
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }

        cl_int ret = CL_SUCCESS;
        cl_program program = momoten_detail::create_and_build_program(
            device, translated.source, ret);
        if (!program)
        {
            log_error("OpenCL program creation", ret);
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        if (ret != CL_SUCCESS)
        {
            momoten_detail::print_program_build_log(
                program, device->physical_device->device);
            if (momoten_detail::debug_enabled())
                fprintf(stderr, "[momoten] generated source:\n%s\n", translated.source.c_str());
            momoten_detail::g_opencl.p_clReleaseProgram(program);
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        cl_kernel kernel = momoten_detail::g_opencl.p_clCreateKernel(program, translated.abi.entry_point.c_str(), &ret);
        if (!kernel)
        {
            log_error("clCreateKernel", ret);
            momoten_detail::g_opencl.p_clReleaseProgram(program);
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        size_t kernel_workgroup_size = 0;
        cl_ulong kernel_local_memory = 0;
        if (momoten_detail::g_opencl.p_clGetKernelWorkGroupInfo(kernel, device->physical_device->device,
                                                                CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernel_workgroup_size),
                                                                &kernel_workgroup_size, 0)
                != CL_SUCCESS
            || momoten_detail::g_opencl.p_clGetKernelWorkGroupInfo(kernel, device->physical_device->device,
                                                                   CL_KERNEL_LOCAL_MEM_SIZE, sizeof(kernel_local_memory),
                                                                   &kernel_local_memory, 0)
                   != CL_SUCCESS)
        {
            momoten_detail::g_opencl.p_clReleaseKernel(kernel);
            momoten_detail::g_opencl.p_clReleaseProgram(program);
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        size_t opencl_local_size[3] = {
            translated.abi.local_size[0], translated.abi.local_size[1],
            translated.abi.local_size[2]};
        size_t workgroup_chunk_count = 1;
        if (translated.abi.workgroup_splittable)
        {
            const size_t physical_limit = std::min(
                std::min(kernel_workgroup_size,
                         device->physical_device->max_compute_workgroup_invocations),
                device->physical_device->max_work_item_sizes[0]);
            size_t physical_workgroup_size = std::min(requested_workgroup_size, physical_limit);
            if (requested_workgroup_size > physical_workgroup_size && translated.abi.subgroup_size > 1)
                physical_workgroup_size -= physical_workgroup_size % translated.abi.subgroup_size;
            if (physical_workgroup_size == 0)
            {
                fprintf(stderr, "[momoten] cannot split logical workgroup=%zu into a physical OpenCL workgroup: kernel maximum=%zu, profile maximum=%zu, x-dimension maximum=%zu, subgroup size=%u\n",
                        requested_workgroup_size, kernel_workgroup_size,
                        device->physical_device->max_compute_workgroup_invocations,
                        device->physical_device->max_work_item_sizes[0],
                        translated.abi.subgroup_size);
                momoten_detail::g_opencl.p_clReleaseKernel(kernel);
                momoten_detail::g_opencl.p_clReleaseProgram(program);
                return VK_ERROR_FEATURE_NOT_PRESENT;
            }
            opencl_local_size[0] = physical_workgroup_size;
            opencl_local_size[1] = 1;
            opencl_local_size[2] = 1;
            workgroup_chunk_count = 1 + (requested_workgroup_size - 1) / physical_workgroup_size;
            if (momoten_detail::debug_enabled() && workgroup_chunk_count > 1)
                fprintf(stderr, "[momoten] split logical workgroup=%zu (%u,%u,%u) into %zu OpenCL workgroups of %zu work-items\n",
                        requested_workgroup_size, translated.abi.local_size[0],
                        translated.abi.local_size[1], translated.abi.local_size[2],
                        workgroup_chunk_count, physical_workgroup_size);
        }

        const bool workgroup_too_large = !translated.abi.workgroup_splittable && requested_workgroup_size > kernel_workgroup_size;
        if (workgroup_too_large || kernel_local_memory > device->physical_device->local_memory)
        {
            fprintf(stderr, "[momoten] built kernel exceeds OpenCL limits: requested workgroup=%zu (%u,%u,%u), kernel maximum=%zu, kernel local memory=%llu, device local memory=%llu, splittable=%u\n",
                    requested_workgroup_size,
                    translated.abi.local_size[0], translated.abi.local_size[1],
                    translated.abi.local_size[2], kernel_workgroup_size,
                    static_cast<unsigned long long>(kernel_local_memory),
                    static_cast<unsigned long long>(device->physical_device->local_memory),
                    translated.abi.workgroup_splittable ? 1u : 0u);
            momoten_detail::g_opencl.p_clReleaseKernel(kernel);
            momoten_detail::g_opencl.p_clReleaseProgram(program);
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        if (translated.abi.subgroup_mode == momoten::SubgroupModeNative)
        {
            const size_t local_size[3] = {
                opencl_local_size[0], opencl_local_size[1], opencl_local_size[2]};
            size_t kernel_subgroup_size = 0;
            if (!momoten_detail::g_opencl.p_clGetKernelSubGroupInfoKHR || momoten_detail::g_opencl.p_clGetKernelSubGroupInfoKHR(kernel, device->physical_device->device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR, sizeof(local_size), local_size, sizeof(kernel_subgroup_size), &kernel_subgroup_size, 0) != CL_SUCCESS || kernel_subgroup_size != translated.abi.subgroup_size)
            {
                fprintf(stderr, "[momoten] native OpenCL subgroup size does not match the advertised Vulkan subgroup size\n");
                momoten_detail::g_opencl.p_clReleaseKernel(kernel);
                momoten_detail::g_opencl.p_clReleaseProgram(program);
                return VK_ERROR_FEATURE_NOT_PRESENT;
            }
        }

        std::shared_ptr<Pipeline> pipeline(new Pipeline());
        pipeline->device = device;
        pipeline->layout = layout;
        pipeline->program = program;
        pipeline->kernel = kernel;
        pipeline->abi = translated.abi;
        pipeline->opencl_local_size[0] = opencl_local_size[0];
        pipeline->opencl_local_size[1] = opencl_local_size[1];
        pipeline->opencl_local_size[2] = opencl_local_size[2];
        pipeline->workgroup_chunk_count = workgroup_chunk_count;
        pipeline->source.swap(translated.source);
        pipelines[i] = g_pipelines.make_handle(pipeline);
    }
    return VK_SUCCESS;
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
