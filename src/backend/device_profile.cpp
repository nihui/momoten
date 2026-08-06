// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "device_profile.h"
#include "objects.h"

#include <sstream>

namespace momoten_detail {

IntegerDotProductAccelerationProfile::IntegerDotProductAccelerationProfile()
    : unsigned_accelerated(false), signed_accelerated(false), mixed_signedness_accelerated(false), accumulating_saturating_unsigned_accelerated(false), accumulating_saturating_signed_accelerated(false), accumulating_saturating_mixed_signedness_accelerated(false)
{
}

IntegerDotProductProfile::IntegerDotProductProfile()
    : supported(false), input_4x8bit(false), input_4x8bit_packed(false)
{
}

ShaderDeviceProfile::ShaderDeviceProfile()
    : fp16(false), subgroup_mode(momoten::SubgroupModeSingleton), subgroup_size(1), subgroup_operations(VK_SUBGROUP_FEATURE_BASIC_BIT)
{
}

EnabledShaderProfile::EnabledShaderProfile()
    : integer_dot_product(false)
{
}

void configure_translation_options(
    VkDevice device, momoten::TranslationOptions& options)
{
    const ShaderDeviceProfile& physical = device->physical_device->shader_profile;
    const EnabledShaderProfile& enabled = device->enabled_shader_profile;
    options.address_bits = device->physical_device->address_bits;
    options.subgroup_mode = physical.subgroup_mode;
    options.subgroup_size = physical.subgroup_size;
    options.integer_dot_product = enabled.integer_dot_product;
    options.integer_dot_product_input_4x8bit = enabled.integer_dot_product && physical.integer_dot_product.input_4x8bit;
    options.integer_dot_product_input_4x8bit_packed = enabled.integer_dot_product && physical.integer_dot_product.input_4x8bit_packed;
}

bool validate_pipeline_abi(
    VkDevice device, const momoten::KernelABI& abi, std::string& diagnostic)
{
    const VkPhysicalDevice physical = device->physical_device;
    const ShaderDeviceProfile& profile = physical->shader_profile;
    if (abi.subgroup_mode != profile.subgroup_mode || abi.subgroup_size != profile.subgroup_size)
    {
        diagnostic = "translated subgroup ABI does not match the physical device profile";
        return false;
    }
    if (abi.integer_dot_product && !device->enabled_shader_profile.integer_dot_product)
    {
        diagnostic = "shader integer dot product was not enabled when the Vulkan device was created";
        return false;
    }
    for (size_t i = 0; i < abi.required_extensions.size(); i++)
    {
        const std::string padded_extensions = " " + physical->extensions + " ";
        const std::string required = " " + abi.required_extensions[i] + " ";
        if (padded_extensions.find(required) == std::string::npos)
        {
            diagnostic = "generated kernel requires unavailable OpenCL extension " + abi.required_extensions[i];
            return false;
        }
    }

    size_t parameter_size = 0;
    const size_t offset_size = abi.address_bits == 64 ? sizeof(cl_ulong) : sizeof(cl_uint);
    for (size_t i = 0; i < abi.buffers.size(); i++)
    {
        if (parameter_size > SIZE_MAX - sizeof(cl_mem) - offset_size * 2)
        {
            diagnostic = "kernel parameter-size calculation overflowed";
            return false;
        }
        parameter_size += sizeof(cl_mem) + offset_size * 2;
    }
    if (abi.push_constant_arg_index >= 0)
    {
        if (parameter_size > SIZE_MAX - sizeof(cl_mem))
        {
            diagnostic = "kernel parameter-size calculation overflowed";
            return false;
        }
        parameter_size += sizeof(cl_mem);
    }
    if (parameter_size > physical->max_parameter_size)
    {
        std::ostringstream message;
        message << "generated kernel arguments require " << parameter_size
                << " bytes, exceeding CL_DEVICE_MAX_PARAMETER_SIZE="
                << physical->max_parameter_size;
        diagnostic = message.str();
        return false;
    }

    size_t invocations = 1;
    for (size_t d = 0; d < 3; d++)
    {
        if (abi.local_size[d] == 0 || invocations > SIZE_MAX / abi.local_size[d])
        {
            diagnostic = "resolved local size is zero or its invocation count overflows";
            return false;
        }
        if (!abi.workgroup_splittable && abi.local_size[d] > physical->max_work_item_sizes[d])
        {
            diagnostic = "non-splittable local size exceeds CL_DEVICE_MAX_WORK_ITEM_SIZES";
            return false;
        }
        invocations *= abi.local_size[d];
    }
    if (!abi.workgroup_splittable && invocations > physical->max_workgroup_size)
    {
        std::ostringstream message;
        message << "resolved workgroup has " << invocations
                << " work-items, exceeding CL_DEVICE_MAX_WORK_GROUP_SIZE="
                << physical->max_workgroup_size;
        diagnostic = message.str();
        return false;
    }
    return true;
}

} // namespace momoten_detail
