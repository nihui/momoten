// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "device_profile.h"
#include "objects.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

namespace momoten_detail {

// OpenCL 3.0 declarations used opportunistically without raising the OpenCL
// 1.0 compile target. Older devices reject the property query.
static const cl_device_info MOMOTEN_CL_DEVICE_OPENCL_C_FEATURES = 0x106f;
static const cl_device_info MOMOTEN_CL_DEVICE_OPENCL_C_ALL_VERSIONS = 0x1066;
static const cl_device_info MOMOTEN_CL_DEVICE_OPENCL_C_VERSION = 0x103d;
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

static OpenCLCompilerProfile probe_opencl_compiler_profile(cl_device_id device)
{
    OpenCLCompilerProfile profile;
    profile.full_profile = device_info_string(device, CL_DEVICE_PROFILE) == "FULL_PROFILE";

    size_t size = 0;
    if (g_opencl.p_clGetDeviceInfo(device, MOMOTEN_CL_DEVICE_OPENCL_C_ALL_VERSIONS, 0, 0, &size) == CL_SUCCESS && size && size % sizeof(OpenCLCNameVersion) == 0)
    {
        std::vector<OpenCLCNameVersion> versions(size / sizeof(OpenCLCNameVersion));
        if (g_opencl.p_clGetDeviceInfo(device, MOMOTEN_CL_DEVICE_OPENCL_C_ALL_VERSIONS, size, versions.data(), 0) == CL_SUCCESS)
        {
            uint32_t selected = 0;
            for (size_t i = 0; i < versions.size(); i++)
            {
                if (strncmp(versions[i].name, "OpenCL C", sizeof(versions[i].name)) != 0)
                    continue;
                const uint32_t major = versions[i].version >> 22;
                const uint32_t minor = (versions[i].version >> 12) & 0x3ffu;
                const uint32_t version = major * 100 + minor * 10;
                if ((major == 1 && minor <= 2) || ((major == 2 || major == 3) && minor == 0))
                    selected = std::max(selected, version);
            }
            if (selected)
            {
                profile.version = selected;
                return profile;
            }
        }
    }

    // OpenCL 1.0 lacks this query and -cl-std. Later implementations report
    // their language version here even without the OpenCL 3.0 version list.
    const std::string version = device_info_string(device, MOMOTEN_CL_DEVICE_OPENCL_C_VERSION);
    unsigned int major = 0;
    unsigned int minor = 0;
    if (sscanf(version.c_str(), "OpenCL C %u.%u", &major, &minor) == 2)
    {
        if (major == 3 || major == 2)
            profile.version = major * 100;
        else if (major == 1)
            profile.version = 100 + std::min(minor, 2u) * 10;
    }
    return profile;
}

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

static uint32_t query_opencl_subgroup_size(cl_kernel kernel, cl_device_id device, size_t max_workgroup_size, size_t max_work_item_size_x)
{
    if (!momoten_detail::g_opencl.p_clGetKernelSubGroupInfoKHR)
        return 0;

    const size_t local_size[3] = {
        std::max<size_t>(1, std::min(max_workgroup_size, max_work_item_size_x)), 1, 1};
    size_t subgroup_size = 0;
    const cl_int ret = momoten_detail::g_opencl.p_clGetKernelSubGroupInfoKHR(
        kernel, device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR,
        sizeof(local_size), local_size, sizeof(subgroup_size), &subgroup_size, 0);

    // Vulkan subgroupSize is restricted to a power of two no greater than
    // 128. A failed or unusual query lets the caller use the emulated fallback.
    if (ret != CL_SUCCESS || subgroup_size == 0 || subgroup_size > 128 || (subgroup_size & (subgroup_size - 1)) != 0)
        return 0;
    return static_cast<uint32_t>(subgroup_size);
}

// Capability probes share the selected language dialect and one context.
// Keep compiler options independent of shader-specific math permissions.
class OpenCLCapabilityProbe
{
public:
    OpenCLCapabilityProbe(cl_device_id device_id, const OpenCLCompilerProfile& compiler)
        : device(device_id), options(opencl_language_options(compiler)), context(0), kernel(0)
    {
        cl_int ret = CL_SUCCESS;
        context = g_opencl.p_clCreateContext(0, 1, &device, 0, 0, &ret);
        if (context && ret != CL_SUCCESS)
        {
            g_opencl.p_clReleaseContext(context);
            context = 0;
        }
    }

    ~OpenCLCapabilityProbe()
    {
        if (kernel)
            g_opencl.p_clReleaseKernel(kernel);
        if (context)
            g_opencl.p_clReleaseContext(context);
    }

    cl_kernel build(const std::string& source, const char* name)
    {
        if (kernel)
        {
            g_opencl.p_clReleaseKernel(kernel);
            kernel = 0;
        }
        if (!context)
            return 0;

        const char* source_pointer = source.c_str();
        const size_t source_size = source.size();
        cl_int ret = CL_SUCCESS;
        cl_program program = g_opencl.p_clCreateProgramWithSource(context, 1, &source_pointer, &source_size, &ret);
        if (!program)
            return 0;
        if (ret == CL_SUCCESS)
            ret = g_opencl.p_clBuildProgram(program, 1, &device, options.empty() ? 0 : options.c_str(), 0, 0);
        if (ret == CL_SUCCESS)
            kernel = g_opencl.p_clCreateKernel(program, name, &ret);
        g_opencl.p_clReleaseProgram(program);
        if (kernel && ret != CL_SUCCESS)
        {
            g_opencl.p_clReleaseKernel(kernel);
            kernel = 0;
        }
        return kernel;
    }

private:
    OpenCLCapabilityProbe(const OpenCLCapabilityProbe&) = delete;
    OpenCLCapabilityProbe& operator=(const OpenCLCapabilityProbe&) = delete;

    cl_device_id device;
    std::string options;
    cl_context context;
    cl_kernel kernel;
};

static uint32_t probe_native_subgroup_size(OpenCLCapabilityProbe& probe, cl_device_id device, const std::string& extensions, size_t max_workgroup_size, size_t max_work_item_size_x)
{
    if (!extension_string_contains(extensions, "cl_khr_subgroups") || !extension_string_contains(extensions, "cl_khr_subgroup_non_uniform_vote") || !g_opencl.p_clGetKernelSubGroupInfoKHR)
        return 0;

    // Compile every builtin emitted by native BASIC lowering, including the
    // vote extension's election and both local and global memory barriers.
    static const char source[] = "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
                                 "#pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_vote : enable\n"
                                 "__kernel void momo_subgroup_probe(__global uint4* output)\n"
                                 "{\n"
                                 "    uint elected = sub_group_elect();\n"
                                 "    sub_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n"
                                 "    output[get_global_id(0)] = (uint4)(get_sub_group_size(), get_sub_group_local_id(), get_sub_group_id(), get_num_sub_groups() + elected);\n"
                                 "}\n";
    cl_kernel kernel = probe.build(source, "momo_subgroup_probe");
    return kernel ? query_opencl_subgroup_size(kernel, device, max_workgroup_size, max_work_item_size_x) : 0;
}

static uint32_t probe_emulated_subgroup_size(OpenCLCapabilityProbe& probe, cl_device_id device, const std::string& extensions, size_t max_workgroup_size, size_t max_work_item_size_x)
{
    // A width query does not require subgroup builtins. It can still guide
    // emulation when native BASIC operations are unavailable. The scheduling
    // hint is a last resort and need not equal the hardware subgroup width.
    cl_kernel kernel = probe.build("__kernel void momo_workgroup_probe(void) {}\n", "momo_workgroup_probe");
    if (!kernel)
        return 0;

    size_t preferred_size = 0;
    if (extension_string_contains(extensions, "cl_khr_subgroups"))
        preferred_size = query_opencl_subgroup_size(kernel, device, max_workgroup_size, max_work_item_size_x);
    if (preferred_size == 0 && g_opencl.p_clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(preferred_size), &preferred_size, 0) != CL_SUCCESS)
        return 0;

    const size_t candidate = std::min<size_t>(preferred_size, std::min<size_t>(max_workgroup_size, 128));
    if (candidate <= 1)
        return 0;

    uint32_t logical_size = 1;
    while (logical_size <= candidate / 2)
        logical_size <<= 1;
    return logical_size;
}

static bool probe_integer_dot_product(
    OpenCLCapabilityProbe& probe, cl_device_id device, const std::string& extensions,
    momoten_detail::IntegerDotProductProfile& profile, const OpenCLDriverPolicy& policy)
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
    ap.accumulating_saturating_unsigned_accelerated = !policy.split_packed_dot_acc_sat && acceleration_packed.accumulating_saturating_unsigned_accelerated == CL_TRUE;
    ap.accumulating_saturating_signed_accelerated = !policy.split_packed_dot_acc_sat && acceleration_packed.accumulating_saturating_signed_accelerated == CL_TRUE;
    ap.accumulating_saturating_mixed_signedness_accelerated = !policy.split_packed_dot_acc_sat && acceleration_packed.accumulating_saturating_mixed_signedness_accelerated == CL_TRUE;

    std::string source = "#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable\n"
                         "__kernel void momo_integer_dot_product_probe(__global uint* output, uint a, uint b)\n"
                         "{\n"
                         "    output[0] = as_uint(dot_4x8packed_ss_int(a, b));\n"
                         "    output[1] = dot_4x8packed_uu_uint(a, b);\n"
                         "    output[2] = as_uint(dot_4x8packed_su_int(a, b));\n"
                         "    output[3] = as_uint(dot_4x8packed_us_int(a, b));\n";
    if (policy.split_packed_dot_acc_sat)
    {
        source += "    output[4] = as_uint(add_sat(dot_4x8packed_ss_int(a, b), as_int(a)));\n"
                  "    output[5] = add_sat(dot_4x8packed_uu_uint(a, b), a);\n"
                  "    output[6] = as_uint(add_sat(dot_4x8packed_su_int(a, b), as_int(a)));\n"
                  "    output[7] = as_uint(add_sat(dot_4x8packed_us_int(a, b), as_int(a)));\n";
    }
    else
    {
        source += "    output[4] = as_uint(dot_acc_sat_4x8packed_ss_int(a, b, as_int(a)));\n"
                  "    output[5] = dot_acc_sat_4x8packed_uu_uint(a, b, a);\n";
        source += policy.swap_mixed_dot_acc_sat
                      ? "    output[6] = as_uint(dot_acc_sat_4x8packed_us_int(b, a, as_int(a)));\n"
                      : "    output[6] = as_uint(dot_acc_sat_4x8packed_su_int(a, b, as_int(a)));\n";
        source += "    output[7] = as_uint(dot_acc_sat_4x8packed_us_int(a, b, as_int(a)));\n";
    }
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
                  "    output[13] = dot_acc_sat(ua, ub, a);\n";
        source += policy.swap_mixed_dot_acc_sat
                      ? "    output[14] = as_uint(dot_acc_sat(ub, sa, as_int(a)));\n"
                      : "    output[14] = as_uint(dot_acc_sat(sa, ub, as_int(a)));\n";
        source += "    output[15] = as_uint(dot_acc_sat(ua, sb, as_int(a)));\n";
    }
    source += "}\n";

    profile.supported = probe.build(source, "momo_integer_dot_product_probe") != 0;
    return profile.supported;
}

ShaderDeviceProfile probe_shader_device_profile(
    cl_device_id device, const std::string& extensions,
    size_t max_workgroup_size, size_t max_work_item_size_x, const OpenCLDriverPolicy& policy)
{
    ShaderDeviceProfile profile;
    profile.compiler = probe_opencl_compiler_profile(device);
    OpenCLCapabilityProbe probe(device, profile.compiler);
    profile.fp16 = extension_string_contains(extensions, "cl_khr_fp16");
    profile.int64 = declares_int64(device, extensions);

    cl_device_fp_config double_config = 0;
    profile.fp64 = extension_string_contains(extensions, "cl_khr_fp64")
                   && g_opencl.p_clGetDeviceInfo(
                          device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(double_config),
                          &double_config, 0)
                          == CL_SUCCESS
                   && double_config != 0;

    cl_device_fp_config single_config = 0;
    g_opencl.p_clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(single_config), &single_config, 0);
    cl_device_fp_config half_config = 0;
    if (profile.fp16)
        g_opencl.p_clGetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG, sizeof(half_config), &half_config, 0);
    const cl_device_fp_config configs[] = {half_config, single_config, profile.fp64 ? double_config : 0};
    for (uint32_t i = 0; i < 3; i++)
    {
        if (configs[i] & CL_FP_DENORM)
            profile.denorm_preserve_widths |= 1u << i;
        if (configs[i] & CL_FP_ROUND_TO_NEAREST)
            profile.round_to_nearest_widths |= 1u << i;
        if (configs[i] & CL_FP_INF_NAN)
            profile.signed_zero_inf_nan_preserve_widths |= 1u << i;
    }
    const cl_device_fp_config strict_config = CL_FP_INF_NAN | CL_FP_ROUND_TO_NEAREST;
    profile.float_controls2 = (single_config & strict_config) == strict_config
                              && (!profile.fp16 || (half_config & strict_config) == strict_config)
                              && (!profile.fp64 || (double_config & strict_config) == strict_config);
    const uint32_t opencl_subgroup_size = probe_native_subgroup_size(
        probe, device, extensions, max_workgroup_size, max_work_item_size_x);
    if (opencl_subgroup_size != 0)
    {
        profile.subgroup_mode = momoten::SubgroupModeNative;
        profile.subgroup_size = opencl_subgroup_size;
    }
    else
    {
        const uint32_t emulated_subgroup_size = probe_emulated_subgroup_size(
            probe, device, extensions, max_workgroup_size, max_work_item_size_x);
        if (emulated_subgroup_size != 0)
        {
            profile.subgroup_mode = momoten::SubgroupModeEmulatedBasic;
            profile.subgroup_size = emulated_subgroup_size;
        }
    }

    probe_integer_dot_product(
        probe, device, extensions, profile.integer_dot_product, policy);
    return profile;
}

OpenCLCompilerProfile::OpenCLCompilerProfile()
    : version(100), full_profile(false)
{
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
    : fp16(false), int64(false), fp64(false), float_controls2(true), denorm_preserve_widths(0), round_to_nearest_widths(0), signed_zero_inf_nan_preserve_widths(0), subgroup_mode(momoten::SubgroupModeSingleton), subgroup_size(1), subgroup_operations(VK_SUBGROUP_FEATURE_BASIC_BIT)
{
}

EnabledShaderProfile::EnabledShaderProfile()
    : int64(false), fp64(false), integer_dot_product(false), float_controls2(false)
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
    options.integer_dot_product_split_packed_acc_sat = enabled.integer_dot_product && device->physical_device->driver_policy.split_packed_dot_acc_sat;
    options.integer_dot_product_swap_mixed_acc_sat = enabled.integer_dot_product && device->physical_device->driver_policy.swap_mixed_dot_acc_sat;
}

bool resolve_workgroup_invocations(
    const momoten::KernelABI& abi, size_t& invocations, std::string& diagnostic)
{
    invocations = 1;
    for (size_t d = 0; d < 3; d++)
    {
        if (abi.local_size[d] == 0 || invocations > SIZE_MAX / abi.local_size[d])
        {
            diagnostic = "resolved local size is zero or its invocation count overflows";
            return false;
        }
        invocations *= abi.local_size[d];
    }
    return true;
}

bool validate_workgroup_limits(
    VkDevice device, const momoten::KernelABI& abi, size_t invocations, std::string& diagnostic)
{
    if (abi.workgroup_mode != momoten::WorkgroupModeDirect)
        return true;

    const VkPhysicalDevice physical = device->physical_device;
    for (size_t d = 0; d < 3; d++)
    {
        if (abi.local_size[d] > physical->max_work_item_sizes[d])
        {
            diagnostic = "direct local size exceeds CL_DEVICE_MAX_WORK_ITEM_SIZES";
            return false;
        }
    }
    if (invocations > physical->max_workgroup_size)
    {
        std::ostringstream message;
        message << "direct workgroup has " << invocations
                << " work-items, exceeding CL_DEVICE_MAX_WORK_GROUP_SIZE="
                << physical->max_workgroup_size;
        diagnostic = message.str();
        return false;
    }
    if (invocations > physical->max_compute_workgroup_invocations)
    {
        std::ostringstream message;
        message << "direct workgroup has " << invocations
                << " work-items, exceeding the device profile maximum="
                << physical->max_compute_workgroup_invocations;
        diagnostic = message.str();
        return false;
    }
    return true;
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
    if (abi.float_controls2 && !device->enabled_shader_profile.float_controls2)
    {
        diagnostic = "shader float controls 2 was not enabled when the Vulkan device was created";
        return false;
    }
    if (abi.floating_point.denorm_preserve_widths & ~profile.denorm_preserve_widths)
    {
        diagnostic = "SPIR-V DenormPreserve requires unavailable native OpenCL denormal support";
        return false;
    }
    if (abi.floating_point.round_to_nearest_widths & ~profile.round_to_nearest_widths)
    {
        diagnostic = "SPIR-V RoundingModeRTE requires unavailable native OpenCL round-to-nearest support";
        return false;
    }
    if (abi.floating_point.signed_zero_inf_nan_preserve_widths & ~profile.signed_zero_inf_nan_preserve_widths)
    {
        diagnostic = "SPIR-V SignedZeroInfNanPreserve requires unavailable native OpenCL infinity and NaN support";
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
        if (parameter_size > SIZE_MAX - sizeof(cl_mem) - offset_size)
        {
            diagnostic = "kernel parameter-size calculation overflowed";
            return false;
        }
        parameter_size += sizeof(cl_mem) + offset_size;
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

    size_t invocations = 0;
    return resolve_workgroup_invocations(abi, invocations, diagnostic)
           && validate_workgroup_limits(device, abi, invocations, diagnostic);
}

std::string opencl_language_options(const OpenCLCompilerProfile& profile)
{
    switch (profile.version)
    {
    case 110: return "-cl-std=CL1.1";
    case 120: return "-cl-std=CL1.2";
    case 200: return "-cl-std=CL2.0";
    case 300: return "-cl-std=CL3.0";
    default: return std::string(); // OpenCL 1.0 has no -cl-std option.
    }
}

static void append_opencl_build_option(std::string& options, const char* option)
{
    if (!options.empty())
        options += ' ';
    options += option;
}

std::string opencl_build_options(const momoten::KernelABI& abi, const ShaderDeviceProfile& profile)
{
    std::string options = opencl_language_options(profile.compiler);
    // Keep the restricted fp64 path conservative. Several otherwise capable
    // online compilers become unstable while optimizing long double-precision
    // arithmetic chains. Retain the selected language even on this path.
    if (abi.fp64)
    {
        append_opencl_build_option(options, "-cl-opt-disable");
        return options;
    }

    const momoten::FloatingPointControls& controls = abi.floating_point;
    // Modern FULL_PROFILE math guarantees are required for OpenCL mad and
    // unsafe math. Language selection itself also applies to embedded devices.
    const bool relaxed_math = profile.compiler.full_profile && profile.compiler.version >= 200;
    // The seven SPIR-V permissions cover arithmetic transforms, but not
    // approximate builtin functions or a required DenormPreserve mode.
    // OpenCL half mad accuracy is not sufficient for Vulkan half arithmetic.
    if (relaxed_math && controls.widths == momoten::FloatingPointWidth32 && controls.fast_math_flags == 0x7000fu && !controls.denorm_preserve_widths && !controls.requires_builtin_accuracy)
    {
        append_opencl_build_option(options, "-cl-fast-relaxed-math");
        return options;
    }
    if ((controls.fast_math_flags & 3u) == 3u) // NotNaN | NotInf
        append_opencl_build_option(options, "-cl-finite-math-only");
    if (controls.fast_math_flags & 4u) // NSZ
        append_opencl_build_option(options, "-cl-no-signed-zeros");
    if (relaxed_math && (controls.fast_math_flags & 0x10000u) && !(controls.widths & momoten::FloatingPointWidth16)) // AllowContract
        append_opencl_build_option(options, "-cl-mad-enable");
    return options;
}

} // namespace momoten_detail
