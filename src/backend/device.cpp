// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "device.h"
#include "objects.h"

#include <cstdio>
#include <cstring>

// OpenCL 1.1 query used opportunistically without raising the OpenCL 1.0
// compile target. OpenCL 1.0 devices reject it and take the conservative path.
static const cl_device_info MOMOTEN_CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035;

// OpenCL exposes a device-wide maximum, but CL_KERNEL_WORK_GROUP_SIZE may be
// lower after translating a register-heavy SPIR-V kernel. Vulkan applications
// need a useful limit before pipeline compilation, so the restricted momoten
// compute profile advertises a conservative value that ncnn can honor while
// selecting and specializing its local size.
static const size_t MOMOTEN_MAX_COMPUTE_WORKGROUP_INVOCATIONS = 256;

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
            physical->max_compute_workgroup_invocations = std::min(
                physical->max_workgroup_size,
                MOMOTEN_MAX_COMPUTE_WORKGROUP_INVOCATIONS);
            if (momoten_detail::debug_enabled() && physical->max_compute_workgroup_invocations != physical->max_workgroup_size)
                fprintf(stderr, "[momoten] compute profile caps %s workgroup invocations at %zu (OpenCL device maximum %zu)\n",
                        physical->name.c_str(),
                        physical->max_compute_workgroup_invocations,
                        physical->max_workgroup_size);
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
            physical->shader_profile = momoten_detail::probe_shader_device_profile(
                devices[d], physical->extensions,
                physical->max_workgroup_size,
                physical->max_work_item_sizes[0]);
            if (momoten_detail::debug_enabled())
            {
                const char* mode = physical->shader_profile.subgroup_mode
                                           == momoten::SubgroupModeNative
                                       ? "native"
                                   : physical->shader_profile.subgroup_mode
                                           == momoten::SubgroupModeEmulatedBasic
                                       ? "emulated-basic"
                                       : "singleton";
                fprintf(stderr, "[momoten] subgroup profile for %s: %s, size %u\n",
                        physical->name.c_str(), mode,
                        physical->shader_profile.subgroup_size);
            }
            result.push_back(physical);
        }
    }
    return result;
}
