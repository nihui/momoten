// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_DEVICE_PROFILE_H
#define MOMOTEN_BACKEND_DEVICE_PROFILE_H

#include "../vulkan_internal.h"
#include "opencl_loader.h"
#include "driver_policy.h"
#include "momoten/spv_to_clc.h"

#include <cstdint>
#include <string>

namespace momoten_detail {

struct OpenCLCompilerProfile
{
    // OpenCL C version, using 100, 110, 120, 200 and 300.
    uint32_t version;
    bool full_profile;

    OpenCLCompilerProfile();
};

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
    momoten::IntegerDotProductBackend backend;
    bool input_4x8bit;
    bool input_4x8bit_packed;
    IntegerDotProductAccelerationProfile acceleration_8bit;
    IntegerDotProductAccelerationProfile acceleration_4x8bit_packed;

    IntegerDotProductProfile();
};

struct ShaderDeviceProfile
{
    OpenCLCompilerProfile compiler;
    bool byte_addressable_store;
    bool fp16;
    bool packed_fp16_vector_loads;
    bool int64;
    bool fp64;
    bool float_controls2;
    uint32_t denorm_preserve_widths;
    uint32_t round_to_nearest_widths;
    uint32_t signed_zero_inf_nan_preserve_widths;
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
    size_t max_workgroup_size, size_t max_work_item_size_x,
    const OpenCLDriverPolicy& policy);

void configure_translation_options(
    VkDevice device, momoten::TranslationOptions& options);
bool resolve_workgroup_invocations(
    const momoten::KernelABI& abi, size_t& invocations, std::string& diagnostic);
bool validate_workgroup_limits(
    VkDevice device, const momoten::KernelABI& abi, size_t invocations, std::string& diagnostic);
bool validate_pipeline_abi(
    VkDevice device, const momoten::KernelABI& abi, std::string& diagnostic);
std::string opencl_language_options(const OpenCLCompilerProfile& profile);
std::string opencl_build_options(const momoten::KernelABI& abi, const ShaderDeviceProfile& profile);

} // namespace momoten_detail

#endif // MOMOTEN_BACKEND_DEVICE_PROFILE_H
