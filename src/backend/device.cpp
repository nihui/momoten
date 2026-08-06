// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "device.h"
#include "objects.h"

#include <cstdio>
#include <cstring>

// OpenCL 1.1 query used opportunistically without raising the OpenCL 1.0
// compile target. OpenCL 1.0 devices reject it and take the conservative path.
static const cl_device_info MOMOTEN_CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035;

void log_error(const char* operation, cl_int error)
{
    fprintf(stderr, "[momoten] %s failed with OpenCL error %d\n", operation, error);
}

std::string opencl_device_string(cl_device_id device, cl_device_info parameter)
{
    size_t size = 0;
    if (momoten_detail::g_opencl.p_clGetDeviceInfo(device, parameter, 0, 0, &size) != CL_SUCCESS || size == 0)
        return std::string();
    std::vector<char> value(size);
    if (momoten_detail::g_opencl.p_clGetDeviceInfo(device, parameter, size, value.data(), 0) != CL_SUCCESS)
        return std::string();
    return std::string(value.data());
}

static bool extension_string_contains(const std::string& extensions, const char* name)
{
    const std::string padded = " " + extensions + " ";
    const std::string required = " " + std::string(name) + " ";
    return padded.find(required) != std::string::npos;
}

static uint32_t probe_native_subgroup_size(cl_device_id device,
                                           const std::string& extensions,
                                           size_t max_workgroup_size,
                                           size_t max_work_item_size_x)
{
    if (!extension_string_contains(extensions, "cl_khr_subgroups") || !extension_string_contains(extensions, "cl_khr_subgroup_non_uniform_vote") || !momoten_detail::g_opencl.p_clGetKernelSubGroupInfoKHR)
        return 0;

    static const char source[] = "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
                                 "#pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_vote : enable\n"
                                 "__kernel void momo_subgroup_probe(__global uint* output)\n"
                                 "{\n"
                                 "    if (sub_group_elect())\n"
                                 "        output[get_sub_group_id()] = get_sub_group_size();\n"
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
    // 128. A failed or unusual probe is kept on the exact singleton path.
    if (ret != CL_SUCCESS || subgroup_size == 0 || subgroup_size > 128 || (subgroup_size & (subgroup_size - 1)) != 0)
        return 0;
    return static_cast<uint32_t>(subgroup_size);
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

template<typename T>
static T get_device_value(cl_device_id device, cl_device_info parameter, T fallback)
{
    T value = fallback;
    if (momoten_detail::g_opencl.p_clGetDeviceInfo(device, parameter, sizeof(value), &value, 0) != CL_SUCCESS)
        return fallback;
    return value;
}

static VkExtensionProperties make_extension(const char* name, uint32_t version)
{
    VkExtensionProperties property = {};
    strncpy(property.extensionName, name, VK_MAX_EXTENSION_NAME_SIZE - 1);
    property.specVersion = version;
    return property;
}

static uint32_t stable_device_id(cl_uint vendor_id, cl_device_type device_type,
                                 const std::string& vendor,
                                 const std::string& name)
{
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < sizeof(vendor_id); i++)
    {
        hash ^= (vendor_id >> (i * 8)) & 0xffu;
        hash *= 16777619u;
    }
    for (size_t i = 0; i < sizeof(device_type); i++)
    {
        hash ^= static_cast<uint64_t>(device_type) >> (i * 8) & 0xffu;
        hash *= 16777619u;
    }
    for (size_t i = 0; i < vendor.size(); i++)
    {
        hash ^= static_cast<unsigned char>(vendor[i]);
        hash *= 16777619u;
    }
    hash ^= 0xffu;
    hash *= 16777619u;
    for (size_t i = 0; i < name.size(); i++)
    {
        hash ^= static_cast<unsigned char>(name[i]);
        hash *= 16777619u;
    }
    return hash ? hash : 1u;
}

static VkPhysicalDeviceType vulkan_device_type(cl_device_type device_type,
                                               cl_bool host_unified_memory)
{
    if (device_type & CL_DEVICE_TYPE_GPU)
    {
        return host_unified_memory == CL_TRUE
                   ? VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
                   : VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    }
    if (device_type & CL_DEVICE_TYPE_CPU)
        return VK_PHYSICAL_DEVICE_TYPE_CPU;
    return VK_PHYSICAL_DEVICE_TYPE_OTHER;
}

bool has_opencl_extension(VkPhysicalDevice physical, const char* name)
{
    if (!physical || !name || !name[0])
        return false;
    const std::string padded_extensions = " " + physical->extensions + " ";
    const std::string required = " " + std::string(name) + " ";
    return padded_extensions.find(required) != std::string::npos;
}

std::vector<VkExtensionProperties> device_extensions(VkPhysicalDevice physical)
{
    std::vector<VkExtensionProperties> extensions = {
        make_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_SPEC_VERSION),
        make_extension(VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME, VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_SPEC_VERSION),
        make_extension(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME, VK_KHR_DRIVER_PROPERTIES_SPEC_VERSION),
        make_extension(VK_KHR_MAINTENANCE1_EXTENSION_NAME, VK_KHR_MAINTENANCE1_SPEC_VERSION),
        make_extension(VK_KHR_MAINTENANCE3_EXTENSION_NAME, VK_KHR_MAINTENANCE3_SPEC_VERSION)};
    if (physical && physical->shader_profile.fp16)
    {
        extensions.push_back(make_extension(VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
                                            VK_KHR_16BIT_STORAGE_SPEC_VERSION));
        extensions.push_back(make_extension(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
                                            VK_KHR_SHADER_FLOAT16_INT8_SPEC_VERSION));
    }
    if (physical && physical->shader_profile.integer_dot_product.supported)
        extensions.push_back(make_extension(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME,
                                            VK_KHR_SHADER_INTEGER_DOT_PRODUCT_SPEC_VERSION));
    return extensions;
}

const std::vector<VkExtensionProperties>& instance_extensions()
{
    static const std::vector<VkExtensionProperties> extensions = {
        make_extension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
                       VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_SPEC_VERSION)};
    return extensions;
}

bool supports_device_extension(VkPhysicalDevice physical, const char* name)
{
    const std::vector<VkExtensionProperties> extensions = device_extensions(physical);
    for (size_t i = 0; i < extensions.size(); i++)
    {
        if (strcmp(extensions[i].extensionName, name) == 0)
            return true;
    }
    return false;
}

std::vector<VkPhysicalDevice> discover_opencl_devices(VkInstance instance)
{
    std::vector<VkPhysicalDevice> result;
    cl_uint platform_count = 0;
    if (momoten_detail::momo_clGetPlatformIDs(0, 0, &platform_count) != CL_SUCCESS || platform_count == 0)
        return result;

    std::vector<cl_platform_id> platforms(platform_count);
    if (momoten_detail::momo_clGetPlatformIDs(platform_count, platforms.data(), 0) != CL_SUCCESS)
        return result;

    for (cl_uint p = 0; p < platform_count; p++)
    {
        cl_uint device_count = 0;
        cl_int ret = momoten_detail::g_opencl.p_clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, 0, &device_count);
        if (ret == CL_DEVICE_NOT_FOUND)
            continue;
        if (ret != CL_SUCCESS || device_count == 0)
            continue;

        std::vector<cl_device_id> devices(device_count);
        if (momoten_detail::g_opencl.p_clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, device_count, devices.data(), 0) != CL_SUCCESS)
            continue;

        for (cl_uint d = 0; d < device_count; d++)
        {
            const cl_bool device_available = get_device_value<cl_bool>(devices[d], CL_DEVICE_AVAILABLE, CL_FALSE);
            const cl_bool compiler_available = get_device_value<cl_bool>(devices[d], CL_DEVICE_COMPILER_AVAILABLE, CL_FALSE);
            const cl_bool little_endian = get_device_value<cl_bool>(devices[d], CL_DEVICE_ENDIAN_LITTLE, CL_FALSE);
            const cl_device_type device_type = get_device_value<cl_device_type>(devices[d], CL_DEVICE_TYPE, 0);
            const cl_bool host_unified_memory = get_device_value<cl_bool>(
                devices[d], MOMOTEN_CL_DEVICE_HOST_UNIFIED_MEMORY, CL_FALSE);
            const cl_uint vendor_id = get_device_value<cl_uint>(devices[d], CL_DEVICE_VENDOR_ID, 0);
            const cl_uint address_bits = get_device_value<cl_uint>(devices[d], CL_DEVICE_ADDRESS_BITS, 0);
            const cl_uint work_item_dimensions = get_device_value<cl_uint>(devices[d], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 0);
            if (device_available != CL_TRUE || compiler_available != CL_TRUE || little_endian != CL_TRUE || (address_bits != 32 && address_bits != 64) || work_item_dimensions < 3)
                continue;

            VkPhysicalDevice_T* physical = new VkPhysicalDevice_T;
            physical->owner = instance;
            physical->platform = platforms[p];
            physical->device = devices[d];
            physical->name = opencl_device_string(devices[d], CL_DEVICE_NAME);
            physical->vendor = opencl_device_string(devices[d], CL_DEVICE_VENDOR);
            physical->driver = opencl_device_string(devices[d], CL_DRIVER_VERSION);
            physical->extensions = opencl_device_string(devices[d], CL_DEVICE_EXTENSIONS);
            physical->opencl_device_type = device_type;
            physical->vendor_id = vendor_id;
            physical->device_id = stable_device_id(
                vendor_id, device_type, physical->vendor, physical->name);
            physical->vulkan_device_type = vulkan_device_type(
                device_type, host_unified_memory);
            physical->address_bits = address_bits;
            physical->global_memory = get_device_value<cl_ulong>(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, 0);
            physical->max_allocation = get_device_value<cl_ulong>(devices[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, 0);
            physical->max_workgroup_size = get_device_value<size_t>(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, 1);
            physical->max_parameter_size = get_device_value<size_t>(devices[d], CL_DEVICE_MAX_PARAMETER_SIZE, 0);
            physical->local_memory = get_device_value<cl_ulong>(devices[d], CL_DEVICE_LOCAL_MEM_SIZE, 0);
            physical->max_compute_units = get_device_value<cl_uint>(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, 1);
            physical->mem_base_alignment_bits = get_device_value<cl_uint>(devices[d], CL_DEVICE_MEM_BASE_ADDR_ALIGN, 128);
            std::vector<size_t> work_item_sizes(work_item_dimensions, 1);
            if (momoten_detail::g_opencl.p_clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                                           work_item_sizes.size() * sizeof(size_t), work_item_sizes.data(), 0)
                != CL_SUCCESS)
            {
                delete physical;
                continue;
            }
            for (size_t i = 0; i < 3; i++)
                physical->max_work_item_sizes[i] = work_item_sizes[i];
            physical->shader_profile = momoten_detail::ShaderDeviceProfile();
            physical->shader_profile.fp16 = extension_string_contains(physical->extensions, "cl_khr_fp16");
            const uint32_t native_subgroup_size = probe_native_subgroup_size(
                devices[d], physical->extensions, physical->max_workgroup_size,
                physical->max_work_item_sizes[0]);
            if (native_subgroup_size != 0)
            {
                physical->shader_profile.subgroup_mode = momoten::SubgroupModeNative;
                physical->shader_profile.subgroup_size = native_subgroup_size;
            }
            probe_integer_dot_product(
                devices[d], physical->extensions,
                physical->shader_profile.integer_dot_product);
            result.push_back(physical);
        }
    }
    return result;
}
