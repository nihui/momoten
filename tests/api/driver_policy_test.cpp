// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/device_profile.h"
#include "backend/objects.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace momoten_detail {
OpenCLApi g_opencl = {};
} // namespace momoten_detail

struct _cl_program
{
    std::string source;
};

static bool vector_support;
static bool default_build_options;
static std::string dot_probe_source;

static cl_int write_info(const void* value, size_t size, size_t capacity, void* output, size_t* output_size)
{
    if (output_size)
        *output_size = size;
    if (output)
    {
        if (capacity < size)
            return CL_INVALID_VALUE;
        memcpy(output, value, size);
    }
    return CL_SUCCESS;
}

static cl_int CL_API_CALL get_device_info(cl_device_id, cl_device_info parameter, size_t capacity, void* output, size_t* output_size)
{
    const char* value = 0;
    if (parameter == CL_DEVICE_PROFILE)
        value = "FULL_PROFILE";
    if (value)
        return write_info(value, strlen(value) + 1, capacity, output, output_size);
    if (parameter == CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR)
    {
        cl_device_integer_dot_product_capabilities_khr capabilities = CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR;
        if (vector_support)
            capabilities |= CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR;
        return write_info(&capabilities, sizeof(capabilities), capacity, output, output_size);
    }
    if (parameter == CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR || parameter == CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR)
    {
        const cl_device_integer_dot_product_acceleration_properties_khr acceleration = {CL_TRUE, CL_TRUE, CL_TRUE, CL_TRUE, CL_TRUE, CL_TRUE};
        return write_info(&acceleration, sizeof(acceleration), capacity, output, output_size);
    }
    return CL_INVALID_VALUE;
}

static cl_context CL_API_CALL create_context(const cl_context_properties*, cl_uint, const cl_device_id*, void(CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int* result)
{
    *result = CL_SUCCESS;
    return reinterpret_cast<cl_context>(1);
}

static cl_int CL_API_CALL release_context(cl_context)
{
    return CL_SUCCESS;
}

static cl_program CL_API_CALL create_program(cl_context, cl_uint count, const char** sources, const size_t* sizes, cl_int* result)
{
    cl_program program = new _cl_program;
    for (cl_uint i = 0; i < count; i++)
        program->source.append(sources[i], sizes && sizes[i] ? sizes[i] : strlen(sources[i]));
    *result = CL_SUCCESS;
    return program;
}

static cl_int CL_API_CALL build_program(cl_program program, cl_uint, const cl_device_id*, const char* options, void(CL_CALLBACK*)(cl_program, void*), void*)
{
    default_build_options = default_build_options && (!options || !options[0]);
    if (program->source.find("momo_integer_dot_product_probe") == std::string::npos)
        return CL_BUILD_PROGRAM_FAILURE;
    dot_probe_source = program->source;
    return CL_SUCCESS;
}

static cl_int CL_API_CALL release_program(cl_program program)
{
    delete program;
    return CL_SUCCESS;
}

static int check_profile(const char* name, const char* vendor, const char* driver, bool supports_vector, bool workaround, bool swap = false)
{
    vector_support = supports_vector;
    default_build_options = true;
    dot_probe_source.clear();
    const momoten_detail::OpenCLDriverPolicy policy = momoten_detail::opencl_driver_policy(vendor, driver);
    const momoten_detail::ShaderDeviceProfile profile = momoten_detail::probe_shader_device_profile(
        reinterpret_cast<cl_device_id>(1), "cl_khr_integer_dot_product", 1, 1,
        policy);
    const momoten_detail::IntegerDotProductProfile& dot = profile.integer_dot_product;
    const momoten_detail::IntegerDotProductAccelerationProfile& packed = dot.acceleration_4x8bit_packed;
    if (!dot.supported || !dot.input_4x8bit_packed || dot.input_4x8bit != supports_vector
        || !default_build_options
        || !packed.signed_accelerated || !packed.unsigned_accelerated || !packed.mixed_signedness_accelerated
        || packed.accumulating_saturating_signed_accelerated != !workaround
        || packed.accumulating_saturating_unsigned_accelerated != !workaround
        || packed.accumulating_saturating_mixed_signedness_accelerated != !workaround
        || dot.acceleration_8bit.accumulating_saturating_signed_accelerated != supports_vector
        || dot.acceleration_8bit.accumulating_saturating_mixed_signedness_accelerated != supports_vector)
    {
        fprintf(stderr, "driver_policy_test: %s has an incorrect capability or acceleration profile\n", name);
        return 1;
    }
    const char* suffixes[] = {"ss_int", "uu_uint", "su_int", "us_int"};
    for (size_t i = 0; i < 4; i++)
    {
        const std::string suffix = suffixes[i];
        const bool native = dot_probe_source.find("dot_acc_sat_4x8packed_" + suffix + "(") != std::string::npos;
        const bool split = dot_probe_source.find("add_sat(dot_4x8packed_" + suffix + "(") != std::string::npos;
        if (native != (!workaround && !(swap && i == 2)) || split != workaround)
        {
            fprintf(stderr, "driver_policy_test: %s probes the wrong packed saturation path\n", name);
            return 1;
        }
    }
    if ((dot_probe_source.find("dot_acc_sat(sa, sb,") != std::string::npos) != supports_vector
        || (dot_probe_source.find("dot_acc_sat(sa, ub,") != std::string::npos) != (supports_vector && !swap)
        || (dot_probe_source.find("dot_acc_sat(ub, sa, as_int(a))") != std::string::npos) != (supports_vector && swap)
        || (dot_probe_source.find("dot_acc_sat_4x8packed_us_int(b, a, as_int(a))") != std::string::npos) != swap)
    {
        fprintf(stderr, "driver_policy_test: %s changed the vector saturation path\n", name);
        return 1;
    }

    VkPhysicalDevice_T physical = {};
    physical.shader_profile = profile;
    physical.driver_policy = policy;
    VkDevice_T device = {};
    device.physical_device = &physical;
    device.enabled_shader_profile.integer_dot_product = true;
    momoten::TranslationOptions options;
    momoten_detail::configure_translation_options(&device, options);
    if (!options.integer_dot_product || !options.integer_dot_product_input_4x8bit_packed
        || options.integer_dot_product_split_packed_acc_sat != workaround
        || options.integer_dot_product_swap_mixed_acc_sat != swap)
    {
        fprintf(stderr, "driver_policy_test: %s did not propagate the workaround to translation\n", name);
        return 1;
    }
    device.enabled_shader_profile.integer_dot_product = false;
    momoten_detail::configure_translation_options(&device, options);
    if (options.integer_dot_product || options.integer_dot_product_split_packed_acc_sat || options.integer_dot_product_swap_mixed_acc_sat)
    {
        fprintf(stderr, "driver_policy_test: %s retained a disabled integer-dot profile\n", name);
        return 1;
    }
    return 0;
}

static int check_copy_policy(const char* vendor, const char* driver, bool use_kernel)
{
    const momoten_detail::OpenCLDriverPolicy policy = momoten_detail::opencl_driver_policy(vendor, driver);
    if (policy.use_buffer_copy_kernel != use_kernel || policy.split_packed_dot_acc_sat || policy.swap_mixed_dot_acc_sat)
    {
        fprintf(stderr, "driver_policy_test: incorrect copy policy for %s / %s\n", vendor, driver);
        return 1;
    }
    return 0;
}

int main()
{
    momoten_detail::g_opencl.p_clGetDeviceInfo = get_device_info;
    momoten_detail::g_opencl.p_clCreateContext = create_context;
    momoten_detail::g_opencl.p_clReleaseContext = release_context;
    momoten_detail::g_opencl.p_clCreateProgramWithSource = create_program;
    momoten_detail::g_opencl.p_clBuildProgram = build_program;
    momoten_detail::g_opencl.p_clReleaseProgram = release_program;

    const char* broken_driver = "OpenCL 3.0 QUALCOMM build: 0800.73.1 Compiler E031.47.18.50";
    const char* mixed_driver = "OpenCL 3.0 QUALCOMM build: 0676.73 Compiler E031.41.03.62";
    return check_profile("affected driver", "QUALCOMM", broken_driver, true, true)
           || check_profile("packed-only inputs", "QUALCOMM", broken_driver, false, true)
           || check_profile("other vendor", "another vendor", broken_driver, true, false)
           || check_profile("other release", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.74.1 Compiler E031.47.18.50", true, false)
           || check_profile("other compiler", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.73.1 Compiler E031.47.18.51", true, false)
           || check_profile("mixed saturation driver", "QUALCOMM", mixed_driver, true, false, true)
           || check_profile("mixed saturation packed-only", "QUALCOMM", mixed_driver, false, false, true)
           || check_profile("mixed saturation other vendor", "another vendor", mixed_driver, true, false)
           || check_profile("mixed saturation other release", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0676.74 Compiler E031.41.03.62", true, false)
           || check_profile("mixed saturation other compiler", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0676.73 Compiler E031.41.03.63", true, false)
           || check_profile("missing identity", "", "", true, false)
           || check_copy_policy("Imagination Technologies", "1.10@5130912", true)
           || check_copy_policy("Imagination Technologies", "1.11@5425693", true)
           || check_copy_policy("Imagination Technologies", "1.11@5425694", false)
           || check_copy_policy("another vendor", "1.11@5425693", false)
           || check_copy_policy("", "", false);
}
