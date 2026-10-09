// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "device_profile.h"
#include "objects.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <vector>

namespace momoten_detail {

// OpenCL 3.0 declarations used opportunistically without raising the OpenCL
// 1.0 compile target. Older devices reject the property query.
static const cl_device_info MOMOTEN_CL_DEVICE_OPENCL_C_FEATURES = 0x106f;
static const size_t MOMOTEN_CL_NAME_VERSION_MAX_NAME_SIZE = 64;

static bool extension_string_contains(const std::string& extensions, const char* name)
{
    const std::string padded = " " + extensions + " ";
    const std::string required = " " + std::string(name) + " ";
    return padded.find(required) != std::string::npos;
}

static std::string device_info_string(cl_device_id device, cl_device_info parameter)
{
    size_t size = 0;
    if (momoten_detail::g_opencl.p_clGetDeviceInfo(device, parameter, 0, 0, &size) != CL_SUCCESS || size == 0)
        return std::string();

    std::vector<char> value(size);
    if (momoten_detail::g_opencl.p_clGetDeviceInfo(device, parameter, size, value.data(), 0) != CL_SUCCESS)
        return std::string();

    value.back() = '\0';
    return std::string(value.data());
}

struct OpenCLCNameVersion
{
    cl_uint version;
    char name[MOMOTEN_CL_NAME_VERSION_MAX_NAME_SIZE];
};

static bool has_opencl_c_feature(cl_device_id device, const char* required_name)
{
    size_t size = 0;
    if (momoten_detail::g_opencl.p_clGetDeviceInfo(device, MOMOTEN_CL_DEVICE_OPENCL_C_FEATURES, 0, 0, &size) != CL_SUCCESS || size == 0 || size % sizeof(OpenCLCNameVersion) != 0)
        return false;

    std::vector<OpenCLCNameVersion> features(size / sizeof(OpenCLCNameVersion));
    if (momoten_detail::g_opencl.p_clGetDeviceInfo(device, MOMOTEN_CL_DEVICE_OPENCL_C_FEATURES, size, features.data(), 0) != CL_SUCCESS)
        return false;

    for (size_t i = 0; i < features.size(); i++)
    {
        if (strncmp(features[i].name, required_name, sizeof(features[i].name)) == 0)
            return true;
    }
    return false;
}

static bool declares_int64(cl_device_id device, const std::string& extensions)
{
    const std::string profile = device_info_string(device, CL_DEVICE_PROFILE);
    if (profile == "FULL_PROFILE")
        return true;

    if (profile == "EMBEDDED_PROFILE" && extension_string_contains(extensions, "cles_khr_int64"))
        return true;

    return has_opencl_c_feature(device, "__opencl_c_int64");
}

static uint32_t probe_opencl_subgroup_size(cl_device_id device,
                                           const std::string& extensions,
                                           size_t max_workgroup_size,
                                           size_t max_work_item_size_x)
{
    if (!extension_string_contains(extensions, "cl_khr_subgroups") || !momoten_detail::g_opencl.p_clGetKernelSubGroupInfoKHR)
        return 0;

    static const char source[] = "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
                                 "__kernel void momo_subgroup_probe(__global uint* output)\n"
                                 "{\n"
                                 "    output[get_global_id(0)] = get_sub_group_size();\n"
                                 "}\n";

    cl_int ret = CL_SUCCESS;
    cl_context context = momoten_detail::g_opencl.p_clCreateContext(
        0, 1, &device, 0, 0, &ret);
    if (!context || ret != CL_SUCCESS)
        return 0;

    const char* source_pointer = source;
    const size_t source_size = sizeof(source) - 1;
    cl_program program = momoten_detail::g_opencl.p_clCreateProgramWithSource(
        context, 1, &source_pointer, &source_size, &ret);
    if (!program || ret != CL_SUCCESS)
    {
        momoten_detail::g_opencl.p_clReleaseContext(context);
        return 0;
    }

    ret = momoten_detail::g_opencl.p_clBuildProgram(program, 1, &device, 0, 0, 0);
    cl_kernel kernel = ret == CL_SUCCESS ? momoten_detail::g_opencl.p_clCreateKernel(program, "momo_subgroup_probe", &ret) : 0;

    size_t subgroup_size = 0;
    if (kernel && ret == CL_SUCCESS)
    {
        const size_t local_size[3] = {
            std::max<size_t>(1, std::min(max_workgroup_size, max_work_item_size_x)), 1, 1};
        ret = momoten_detail::g_opencl.p_clGetKernelSubGroupInfoKHR(
            kernel, device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR,
            sizeof(local_size), local_size, sizeof(subgroup_size), &subgroup_size, 0);
    }

    if (kernel)
        momoten_detail::g_opencl.p_clReleaseKernel(kernel);
    momoten_detail::g_opencl.p_clReleaseProgram(program);
    momoten_detail::g_opencl.p_clReleaseContext(context);

    // Vulkan subgroupSize is restricted to a power of two no greater than
    // 128. A failed or unusual query lets the caller use the emulated fallback.
    if (ret != CL_SUCCESS || subgroup_size == 0 || subgroup_size > 128 || (subgroup_size & (subgroup_size - 1)) != 0)
        return 0;
    return static_cast<uint32_t>(subgroup_size);
}

static uint32_t probe_emulated_basic_subgroup_size(
    cl_device_id device, size_t max_workgroup_size)
{
    // This value selects a logical subgroup implemented by momoten; it is not
    // reported as a native OpenCL subgroup contract. The kernel scheduling
    // hint gives the emulation a portable, performance-oriented width without
    // relying on vendor names or device-specific queries.
    static const char source[] = "__kernel void momo_workgroup_probe(void) {}\n";

    cl_int ret = CL_SUCCESS;
    cl_context context = momoten_detail::g_opencl.p_clCreateContext(
        0, 1, &device, 0, 0, &ret);
    if (!context || ret != CL_SUCCESS)
        return 0;

    const char* source_pointer = source;
    const size_t source_size = sizeof(source) - 1;
    cl_program program = momoten_detail::g_opencl.p_clCreateProgramWithSource(
        context, 1, &source_pointer, &source_size, &ret);
    if (!program || ret != CL_SUCCESS)
    {
        momoten_detail::g_opencl.p_clReleaseContext(context);
        return 0;
    }

    ret = momoten_detail::g_opencl.p_clBuildProgram(program, 1, &device, 0, 0, 0);
    cl_kernel kernel = ret == CL_SUCCESS ? momoten_detail::g_opencl.p_clCreateKernel(program, "momo_workgroup_probe", &ret) : 0;

    size_t preferred_multiple = 0;
    if (kernel && ret == CL_SUCCESS)
    {
        ret = momoten_detail::g_opencl.p_clGetKernelWorkGroupInfo(
            kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
            sizeof(preferred_multiple), &preferred_multiple, 0);
    }

    if (kernel)
        momoten_detail::g_opencl.p_clReleaseKernel(kernel);
    momoten_detail::g_opencl.p_clReleaseProgram(program);
    momoten_detail::g_opencl.p_clReleaseContext(context);

    const size_t candidate = std::min<size_t>(preferred_multiple, std::min<size_t>(max_workgroup_size, 128));
    if (ret != CL_SUCCESS || candidate <= 1)
        return 0;

    uint32_t logical_size = 1;
    while (logical_size <= candidate / 2)
        logical_size <<= 1;
    return logical_size;
}

static bool probe_integer_dot_product(
    cl_device_id device, const std::string& extensions,
    momoten_detail::IntegerDotProductProfile& profile)
{
    profile = momoten_detail::IntegerDotProductProfile();
    cl_device_integer_dot_product_capabilities_khr capabilities = 0;
    cl_device_integer_dot_product_acceleration_properties_khr acceleration_8bit;
    cl_device_integer_dot_product_acceleration_properties_khr acceleration_packed;
    memset(&acceleration_8bit, 0, sizeof(acceleration_8bit));
    memset(&acceleration_packed, 0, sizeof(acceleration_packed));

    if (!extension_string_contains(extensions, "cl_khr_integer_dot_product") || momoten_detail::g_opencl.p_clGetDeviceInfo(device, CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR, sizeof(capabilities), &capabilities, 0) != CL_SUCCESS)
        return false;

    const bool supports_8bit = (capabilities & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR) != 0;
    const bool supports_packed = (capabilities & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR) != 0;
    profile.input_4x8bit = supports_8bit;
    profile.input_4x8bit_packed = supports_packed;
    // Packed 4x8 is the representation used by Vulkan shaders when shaderInt8
    // is disabled, and is the path ncnn selects for its int8 kernels.
    if (!supports_packed)
        return false;

    if (supports_8bit)
    {
        if (momoten_detail::g_opencl.p_clGetDeviceInfo(
                device, CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR,
                sizeof(acceleration_8bit), &acceleration_8bit, 0)
            != CL_SUCCESS)
            memset(&acceleration_8bit, 0, sizeof(acceleration_8bit));
    }
    if (momoten_detail::g_opencl.p_clGetDeviceInfo(
            device, CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR,
            sizeof(acceleration_packed), &acceleration_packed, 0)
        != CL_SUCCESS)
        memset(&acceleration_packed, 0, sizeof(acceleration_packed));

    momoten_detail::IntegerDotProductAccelerationProfile& a8 = profile.acceleration_8bit;
    a8.unsigned_accelerated = acceleration_8bit.unsigned_accelerated == CL_TRUE;
    a8.signed_accelerated = acceleration_8bit.signed_accelerated == CL_TRUE;
    a8.mixed_signedness_accelerated = acceleration_8bit.mixed_signedness_accelerated == CL_TRUE;
    a8.accumulating_saturating_unsigned_accelerated = acceleration_8bit.accumulating_saturating_unsigned_accelerated == CL_TRUE;
    a8.accumulating_saturating_signed_accelerated = acceleration_8bit.accumulating_saturating_signed_accelerated == CL_TRUE;
    a8.accumulating_saturating_mixed_signedness_accelerated = acceleration_8bit.accumulating_saturating_mixed_signedness_accelerated == CL_TRUE;

    momoten_detail::IntegerDotProductAccelerationProfile& ap = profile.acceleration_4x8bit_packed;
    ap.unsigned_accelerated = acceleration_packed.unsigned_accelerated == CL_TRUE;
    ap.signed_accelerated = acceleration_packed.signed_accelerated == CL_TRUE;
    ap.mixed_signedness_accelerated = acceleration_packed.mixed_signedness_accelerated == CL_TRUE;
    ap.accumulating_saturating_unsigned_accelerated = acceleration_packed.accumulating_saturating_unsigned_accelerated == CL_TRUE;
    ap.accumulating_saturating_signed_accelerated = acceleration_packed.accumulating_saturating_signed_accelerated == CL_TRUE;
    ap.accumulating_saturating_mixed_signedness_accelerated = acceleration_packed.accumulating_saturating_mixed_signedness_accelerated == CL_TRUE;

    std::string source = "#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable\n"
                         "__kernel void momo_integer_dot_product_probe(__global uint* output, uint a, uint b)\n"
                         "{\n"
                         "    output[0] = as_uint(dot_4x8packed_ss_int(a, b));\n"
                         "    output[1] = dot_4x8packed_uu_uint(a, b);\n"
                         "    output[2] = as_uint(dot_4x8packed_su_int(a, b));\n"
                         "    output[3] = as_uint(dot_4x8packed_us_int(a, b));\n"
                         "    output[4] = as_uint(dot_acc_sat_4x8packed_ss_int(a, b, as_int(a)));\n"
                         "    output[5] = dot_acc_sat_4x8packed_uu_uint(a, b, a);\n"
                         "    output[6] = as_uint(dot_acc_sat_4x8packed_su_int(a, b, as_int(a)));\n"
                         "    output[7] = as_uint(dot_acc_sat_4x8packed_us_int(a, b, as_int(a)));\n";
    if (supports_8bit)
    {
        source += "    char4 sa = as_char4(a);\n"
                  "    char4 sb = as_char4(b);\n"
                  "    uchar4 ua = as_uchar4(a);\n"
                  "    uchar4 ub = as_uchar4(b);\n"
                  "    output[8] = as_uint(dot(sa, sb));\n"
                  "    output[9] = dot(ua, ub);\n"
                  "    output[10] = as_uint(dot(sa, ub));\n"
                  "    output[11] = as_uint(dot(ua, sb));\n"
                  "    output[12] = as_uint(dot_acc_sat(sa, sb, as_int(a)));\n"
                  "    output[13] = dot_acc_sat(ua, ub, a);\n"
                  "    output[14] = as_uint(dot_acc_sat(sa, ub, as_int(a)));\n"
                  "    output[15] = as_uint(dot_acc_sat(ua, sb, as_int(a)));\n";
    }
    source += "}\n";

    cl_int ret = CL_SUCCESS;
    cl_context context = momoten_detail::g_opencl.p_clCreateContext(
        0, 1, &device, 0, 0, &ret);
    if (!context || ret != CL_SUCCESS)
        return false;

    const char* source_pointer = source.c_str();
    const size_t source_size = source.size();
    cl_program program = momoten_detail::g_opencl.p_clCreateProgramWithSource(
        context, 1, &source_pointer, &source_size, &ret);
    if (program && ret == CL_SUCCESS)
        ret = momoten_detail::g_opencl.p_clBuildProgram(program, 1, &device, 0, 0, 0);

    if (program)
        momoten_detail::g_opencl.p_clReleaseProgram(program);
    momoten_detail::g_opencl.p_clReleaseContext(context);
    profile.supported = program && ret == CL_SUCCESS;
    return profile.supported;
}

ShaderDeviceProfile probe_shader_device_profile(
    cl_device_id device, const std::string& extensions,
    size_t max_workgroup_size, size_t max_work_item_size_x)
{
    ShaderDeviceProfile profile;
    profile.fp16 = extension_string_contains(extensions, "cl_khr_fp16");
    profile.int64 = declares_int64(device, extensions);

    cl_device_fp_config double_config = 0;
    profile.fp64 = extension_string_contains(extensions, "cl_khr_fp64")
                   && g_opencl.p_clGetDeviceInfo(
                          device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(double_config),
                          &double_config, 0)
                          == CL_SUCCESS
                   && double_config != 0;

    const uint32_t opencl_subgroup_size = probe_opencl_subgroup_size(
        device, extensions, max_workgroup_size, max_work_item_size_x);
    const bool native_basic = opencl_subgroup_size != 0
                              && extension_string_contains(
                                  extensions, "cl_khr_subgroup_non_uniform_vote");
    if (native_basic)
    {
        profile.subgroup_mode = momoten::SubgroupModeNative;
        profile.subgroup_size = opencl_subgroup_size;
    }
    else
    {
        uint32_t emulated_subgroup_size = opencl_subgroup_size;
        if (emulated_subgroup_size == 0)
            emulated_subgroup_size = probe_emulated_basic_subgroup_size(
                device, max_workgroup_size);
        if (emulated_subgroup_size != 0)
        {
            profile.subgroup_mode = momoten::SubgroupModeEmulatedBasic;
            profile.subgroup_size = emulated_subgroup_size;
        }
    }

    probe_integer_dot_product(
        device, extensions, profile.integer_dot_product);
    return profile;
}

IntegerDotProductAccelerationProfile::IntegerDotProductAccelerationProfile()
    : unsigned_accelerated(false), signed_accelerated(false), mixed_signedness_accelerated(false), accumulating_saturating_unsigned_accelerated(false), accumulating_saturating_signed_accelerated(false), accumulating_saturating_mixed_signedness_accelerated(false)
{
}

IntegerDotProductProfile::IntegerDotProductProfile()
    : supported(false), input_4x8bit(false), input_4x8bit_packed(false)
{
}

ShaderDeviceProfile::ShaderDeviceProfile()
    : fp16(false), int64(false), fp64(false), subgroup_mode(momoten::SubgroupModeSingleton), subgroup_size(1), subgroup_operations(VK_SUBGROUP_FEATURE_BASIC_BIT)
{
}

EnabledShaderProfile::EnabledShaderProfile()
    : int64(false), fp64(false), integer_dot_product(false)
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
    if (abi.fp64 && !device->enabled_shader_profile.fp64)
    {
        diagnostic = "shader float64 was not enabled when the Vulkan device was created";
        return false;
    }
    if (abi.int64 && !device->enabled_shader_profile.int64)
    {
        diagnostic = "shader int64 was not enabled when the Vulkan device was created";
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
        if (parameter_size > SIZE_MAX - abi.push_constant_size)
        {
            diagnostic = "kernel parameter-size calculation overflowed";
            return false;
        }
        parameter_size += abi.push_constant_size;
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

std::string opencl_build_options(const momoten::KernelABI& abi)
{
    // Keep the restricted fp64 path conservative. Several otherwise capable
    // online compilers become unstable while optimizing long double-precision
    // arithmetic chains. This core option changes optimization only, not the
    // shader's floating-point semantics.
    return abi.fp64 ? "-cl-opt-disable" : std::string();
}

} // namespace momoten_detail
