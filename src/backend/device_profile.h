// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_DEVICE_PROFILE_H
#define MOMOTEN_BACKEND_DEVICE_PROFILE_H

#include "../vulkan_internal.h"
#include "opencl_loader.h"
#include "momoten/spv_to_clc.h"

#include <cstdint>
#include <string>

namespace momoten_detail {

struct IntegerDotProductAccelerationProfile
{
    bool unsigned_accelerated;
    bool signed_accelerated;
    bool mixed_signedness_accelerated;
    bool accumulating_saturating_unsigned_accelerated;
    bool accumulating_saturating_signed_accelerated;
    bool accumulating_saturating_mixed_signedness_accelerated;

    IntegerDotProductAccelerationProfile();
};

struct IntegerDotProductProfile
{
    bool supported;
    bool input_4x8bit;
    bool input_4x8bit_packed;
    IntegerDotProductAccelerationProfile acceleration_8bit;
    IntegerDotProductAccelerationProfile acceleration_4x8bit_packed;

    IntegerDotProductProfile();
};

struct ShaderDeviceProfile
{
    bool fp16;
    bool int64;
    bool fp64;
    bool float_controls2;
    uint32_t denorm_preserve_widths;
    uint32_t round_to_nearest_widths;
    uint32_t signed_zero_inf_nan_preserve_widths;
    // Modern FULL_PROFILE math guarantees are required for OpenCL mad and
    // unsafe math. Empty on older compilers and embedded profiles.
    std::string relaxed_math_standard;
    momoten::SubgroupMode subgroup_mode;
    uint32_t subgroup_size;
    uint32_t subgroup_operations;
    IntegerDotProductProfile integer_dot_product;

    ShaderDeviceProfile();
};

struct EnabledShaderProfile
{
    bool int64;
    bool fp64;
    bool integer_dot_product;
    bool float_controls2;

    EnabledShaderProfile();
};

ShaderDeviceProfile probe_shader_device_profile(
    cl_device_id device, const std::string& extensions,
    size_t max_workgroup_size, size_t max_work_item_size_x);

void configure_translation_options(
    VkDevice device, momoten::TranslationOptions& options);
bool validate_pipeline_abi(
    VkDevice device, const momoten::KernelABI& abi, std::string& diagnostic);
std::string opencl_build_options(const momoten::KernelABI& abi, const ShaderDeviceProfile& profile);

} // namespace momoten_detail

#endif // MOMOTEN_BACKEND_DEVICE_PROFILE_H
