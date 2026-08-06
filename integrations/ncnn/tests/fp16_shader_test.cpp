// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "gpu.h"
#include "layer_shader_type.h"
#include "option.h"
#include "momoten/spv_to_clc.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <string>
#include <vector>

static int compile_absval(bool fp16_arithmetic)
{
    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = true;
    opt.use_fp16_uniform = false;
    opt.use_fp16_arithmetic = fp16_arithmetic;
    opt.use_shader_local_memory = false;
    opt.use_subgroup_ops = false;
    opt.use_cooperative_matrix = false;

    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(ncnn::LayerShaderType::absval, opt, spirv) != 0)
    {
        fprintf(stderr, "ncnn_fp16_shader_test: ncnn failed to compile AbsVal fp16 shader\n");
        return 1;
    }

    momoten::TranslationOptions translation_options;
    momoten::SpecializationValue n;
    n.constant_id = 0;
    n.data.resize(sizeof(uint32_t));
    const uint32_t element_count = 16;
    memcpy(n.data.data(), &element_count, sizeof(element_count));
    translation_options.specializations.push_back(n);

    momoten::TranslationResult translated;
    if (!momoten::translate_spirv_to_opencl_c(
            spirv.data(), spirv.size(), translation_options, translated))
    {
        fprintf(stderr, "ncnn_fp16_shader_test: AbsVal translation failed: %s\n",
                translated.diagnostics.c_str());
        return 1;
    }

    if (translated.abi.required_extensions.size() != 1 || translated.abi.required_extensions[0] != "cl_khr_fp16" || translated.source.find("#pragma OPENCL EXTENSION cl_khr_fp16 : enable") == std::string::npos || translated.source.find("__global half4 *") == std::string::npos)
    {
        fprintf(stderr, "ncnn_fp16_shader_test: AbsVal fp16 storage lowering is incomplete\n");
        return 1;
    }

    const bool has_half_arithmetic = translated.source.find("half4 v =") != std::string::npos;
    if (has_half_arithmetic != fp16_arithmetic)
    {
        fprintf(stderr, "ncnn_fp16_shader_test: AbsVal arithmetic precision mismatch\n");
        return 1;
    }
    return 0;
}

static void add_u32_specialization(momoten::TranslationOptions& options,
                                   uint32_t id, uint32_t value)
{
    momoten::SpecializationValue specialization;
    specialization.constant_id = id;
    specialization.data.resize(sizeof(value));
    memcpy(specialization.data.data(), &value, sizeof(value));
    options.specializations.push_back(specialization);
}

static int compile_innerproduct_pack4()
{
    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = true;
    opt.use_fp16_uniform = false;
    opt.use_fp16_arithmetic = true;
    opt.use_shader_local_memory = false;
    opt.use_subgroup_ops = false;
    opt.use_cooperative_matrix = false;

    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(
            ncnn::LayerShaderType::innerproduct_pack4, opt, spirv)
        != 0)
    {
        fprintf(stderr, "ncnn_fp16_shader_test: ncnn failed to compile InnerProduct pack4 fp16 shader\n");
        return 1;
    }

    momoten::TranslationOptions translation_options;
    // Preserve the loop, f16mat4 storage load and half4-times-matrix path.
    add_u32_specialization(translation_options, 0, 1); // bias_term
    add_u32_specialization(translation_options, 1, 0); // activation_type
    add_u32_specialization(translation_options, 2, 0); // activation_param_0 bits
    add_u32_specialization(translation_options, 3, 0); // activation_param_1 bits
    const uint32_t shapes[] = {1, 4, 1, 1, 4, 1, 1, 1, 1, 1};
    for (uint32_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        add_u32_specialization(translation_options, 4 + i, shapes[i]);

    momoten::TranslationResult translated;
    if (!momoten::translate_spirv_to_opencl_c(
            spirv.data(), spirv.size(), translation_options, translated))
    {
        fprintf(stderr, "ncnn_fp16_shader_test: InnerProduct pack4 translation failed: %s\n",
                translated.diagnostics.c_str());
        return 1;
    }
    if (translated.source.find("__global momo_hmat4 *") == std::string::npos || translated.source.find("half4(dot(") == std::string::npos)
    {
        fprintf(stderr, "ncnn_fp16_shader_test: InnerProduct f16mat4 lowering was not exercised\n");
        return 1;
    }
    return 0;
}

static int scan_all_fp16_shaders()
{
    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = true;
    opt.use_fp16_uniform = false;
    opt.use_fp16_arithmetic = true;
    opt.use_shader_local_memory = false;
    opt.use_subgroup_ops = false;
    opt.use_cooperative_matrix = false;

    int translated_count = 0;
    int failed_count = 0;
    for (int shader = 0; shader < ncnn::LayerShaderType::vulkan_activation; shader++)
    {
        std::vector<uint32_t> spirv;
        if (ncnn::compile_spirv_module(shader, opt, spirv) != 0)
        {
            fprintf(stderr, "fp16 scan shader=%d ncnn_compile_failed\n", shader);
            failed_count++;
            continue;
        }
        momoten::TranslationOptions options;
        momoten::TranslationResult translated;
        if (!momoten::translate_spirv_to_opencl_c(
                spirv.data(), spirv.size(), options, translated))
        {
            fprintf(stderr, "fp16 scan shader=%d %s\n", shader, translated.diagnostics.c_str());
            failed_count++;
            continue;
        }
        translated_count++;
    }
    fprintf(stderr, "fp16 scan translated=%d failed=%d total=%d\n",
            translated_count, failed_count, ncnn::LayerShaderType::vulkan_activation);
    return failed_count == 0 ? 0 : 1;
}

int main()
{
    if (ncnn::get_gpu_count() == 0)
        return 77;
    if (getenv("NCNN_MOMOTEN_SCAN_FP16"))
        return scan_all_fp16_shaders();
    return compile_absval(false) || compile_absval(true) || compile_innerproduct_pack4();
}
