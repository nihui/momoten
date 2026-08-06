// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_DEVICE_PROFILE_H
#define MOMOTEN_BACKEND_DEVICE_PROFILE_H

#include "../vulkan_internal.h"
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
    momoten::SubgroupMode subgroup_mode;
    uint32_t subgroup_size;
    uint32_t subgroup_operations;
    IntegerDotProductProfile integer_dot_product;

    ShaderDeviceProfile();
};

struct EnabledShaderProfile
{
    bool integer_dot_product;

    EnabledShaderProfile();
};

void configure_translation_options(
    VkDevice device, momoten::TranslationOptions& options);
bool validate_pipeline_abi(
    VkDevice device, const momoten::KernelABI& abi, std::string& diagnostic);

} // namespace momoten_detail

#endif // MOMOTEN_BACKEND_DEVICE_PROFILE_H
