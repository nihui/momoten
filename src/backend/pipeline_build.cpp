// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "pipeline_build.h"
#include "cache.h"
#include "device.h"
#include "objects.h"
#include "program.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace momoten_detail {

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
        << "int64=" << translated.abi.int64 << '\n'
        << "fp64=" << translated.abi.fp64 << '\n'
        << "integer_dot_product=" << translated.abi.integer_dot_product << '\n'
        << "float_controls2=" << translated.abi.float_controls2 << '\n'
        << "fast_math_flags=" << translated.abi.floating_point.fast_math_flags << '\n'
        << "floating_point_widths=" << translated.abi.floating_point.widths << '\n'
        << "denorm_preserve_widths=" << translated.abi.floating_point.denorm_preserve_widths << '\n'
        << "round_to_nearest_widths=" << translated.abi.floating_point.round_to_nearest_widths << '\n'
        << "signed_zero_inf_nan_preserve_widths=" << translated.abi.floating_point.signed_zero_inf_nan_preserve_widths << '\n'
        << "requires_builtin_accuracy=" << translated.abi.floating_point.requires_builtin_accuracy << '\n'
        << "opencl_build_options=" << opencl_build_options(translated.abi, module->device->physical_device->shader_profile) << '\n'
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

VkResult build_compute_pipelines(
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
        const std::string build_options = opencl_build_options(translated.abi, device->physical_device->shader_profile);
        cl_program program = momoten_detail::create_and_build_program(
            device, translated.source, build_options, ret);
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
        pipelines[i] = g_pipelines.make_handle(pipeline);
    }
    return VK_SUCCESS;
}

} // namespace momoten_detail
