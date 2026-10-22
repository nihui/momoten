// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/device_profile.h"
#include "backend/objects.h"
#include "translator/integer_dot_product.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>

namespace momoten_detail {
OpenCLApi g_opencl = {};
} // namespace momoten_detail

struct _cl_program
{
    std::string source;
};

struct _cl_mem
{
    std::vector<uint32_t> words;
};

static bool vector_support;
static bool khr_capabilities_available = true;
static bool khr_acceleration_available = true;
static std::set<momoten::IntegerDotProductBackend> rejected_backends;
static const momoten::IntegerDotProductBackend dot_backends[] = {
    momoten::IntegerDotProductKhr,
    momoten::IntegerDotProductArm,
    momoten::IntegerDotProductArmAccumulate,
    momoten::IntegerDotProductArmAccumulateSaturate,
    momoten::IntegerDotProductQcom};

enum ProbeFailure
{
    ProbeSuccess,
    FailQueue,
    FailInputBuffer,
    FailOutputBuffer,
    FailKernelArgument,
    FailKernelEnqueue,
    FailBufferRead
};
static int corrupted_result = -1;
static int corrupted_backend = -1;
static ProbeFailure probe_failure = ProbeSuccess;
static int live_buffers;
static int live_queues;
static cl_mem probe_buffers[2];
static std::vector<momoten::IntegerDotProductBackend> attempted_backends;
static bool matching_build_options;
static const char* language_version;
static const char* expected_build_options = "";
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
    if (parameter == 0x103d) // CL_DEVICE_OPENCL_C_VERSION
        value = language_version;
    if (value)
        return write_info(value, strlen(value) + 1, capacity, output, output_size);
    if (parameter == CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR)
    {
        if (!khr_capabilities_available)
            return CL_INVALID_VALUE;
        cl_device_integer_dot_product_capabilities_khr capabilities = CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR;
        if (vector_support)
            capabilities |= CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR;
        return write_info(&capabilities, sizeof(capabilities), capacity, output, output_size);
    }
    if (parameter == CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR || parameter == CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR)
    {
        if (!khr_acceleration_available)
            return CL_INVALID_VALUE;
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
    matching_build_options = matching_build_options && std::string(options ? options : "") == expected_build_options;
    if (program->source.find("momo_integer_dot_product_probe") == std::string::npos)
        return CL_BUILD_PROGRAM_FAILURE;
    dot_probe_source = program->source;
    for (momoten::IntegerDotProductBackend backend : dot_backends)
    {
        const std::string pragma = "#pragma OPENCL EXTENSION " + std::string(momoten::integer_dot_product_extension(backend)) + " : enable";
        if (program->source.find(pragma) != std::string::npos)
        {
            attempted_backends.push_back(backend);
            return rejected_backends.count(backend) ? CL_BUILD_PROGRAM_FAILURE : CL_SUCCESS;
        }
    }
    return CL_BUILD_PROGRAM_FAILURE;
}

static cl_int CL_API_CALL release_program(cl_program program)
{
    delete program;
    return CL_SUCCESS;
}

static cl_kernel CL_API_CALL create_kernel(cl_program, const char*, cl_int* result)
{
    *result = CL_SUCCESS;
    return reinterpret_cast<cl_kernel>(1);
}

static cl_int CL_API_CALL release_kernel(cl_kernel)
{
    return CL_SUCCESS;
}

static cl_command_queue CL_API_CALL create_queue(cl_context, cl_device_id, cl_command_queue_properties, cl_int* error)
{
    ++live_queues;
    *error = probe_failure == FailQueue ? CL_OUT_OF_RESOURCES : CL_SUCCESS;
    return reinterpret_cast<cl_command_queue>(1);
}

static cl_int CL_API_CALL release_queue(cl_command_queue)
{
    --live_queues;
    return CL_SUCCESS;
}

static cl_mem CL_API_CALL create_buffer(cl_context, cl_mem_flags, size_t bytes, void* data, cl_int* error)
{
    cl_mem buffer = new _cl_mem;
    const uint32_t* words = static_cast<const uint32_t*>(data);
    buffer->words.assign(words, words + bytes / sizeof(uint32_t));
    ++live_buffers;
    *error = probe_failure == FailInputBuffer || (probe_failure == FailOutputBuffer && live_buffers == 2) ? CL_OUT_OF_RESOURCES : CL_SUCCESS;
    return buffer;
}

static cl_int CL_API_CALL release_buffer(cl_mem buffer)
{
    delete buffer;
    --live_buffers;
    return CL_SUCCESS;
}

static cl_int CL_API_CALL set_kernel_arg(cl_kernel, cl_uint index, size_t size, const void* value)
{
    if (index >= 2 || size != sizeof(cl_mem) || probe_failure == FailKernelArgument)
        return CL_INVALID_ARG_VALUE;
    probe_buffers[index] = *static_cast<const cl_mem*>(value);
    return CL_SUCCESS;
}

static cl_int CL_API_CALL enqueue_kernel(cl_command_queue, cl_kernel, cl_uint dimensions, const size_t*, const size_t* global, const size_t*, cl_uint, const cl_event*, cl_event*)
{
    if (probe_failure == FailKernelEnqueue || dimensions != 1 || *global * 4 != probe_buffers[0]->words.size() || *global * 6 != probe_buffers[1]->words.size())
        return CL_INVALID_WORK_GROUP_SIZE;
    for (size_t i = 0; i < *global; i++)
    {
        const uint32_t* input = &probe_buffers[0]->words[i * 4];
        uint32_t* output = &probe_buffers[1]->words[i * 6];
        for (int sign = 0; sign < 3; sign++)
        {
            int64_t sum = 0;
            for (unsigned lane = 0; lane < 4; lane++)
            {
                int a = (input[0] >> (lane * 8)) & 255u;
                int b = (input[1] >> (lane * 8)) & 255u;
                if (sign != 1 && a >= 128)
                    a -= 256;
                if (sign == 0 && b >= 128)
                    b -= 256;
                sum += a * b;
            }
            output[sign] = static_cast<uint32_t>(sum);
            int32_t signed_accumulator;
            memcpy(&signed_accumulator, &input[2], sizeof(signed_accumulator));
            const int64_t accumulator = sign == 1 ? int64_t(input[2]) : int64_t(signed_accumulator);
            const int64_t minimum = sign == 1 ? 0 : -2147483648LL;
            const int64_t maximum = sign == 1 ? 4294967295LL : 2147483647LL;
            output[sign + 3] = static_cast<uint32_t>(std::min(maximum, std::max(minimum, sum + accumulator)));
        }
    }
    if (corrupted_result >= 0 && (corrupted_backend < 0 || corrupted_backend == attempted_backends.back()))
        probe_buffers[1]->words[corrupted_result] ^= 1;
    return CL_SUCCESS;
}

static cl_int CL_API_CALL read_buffer(cl_command_queue, cl_mem buffer, cl_bool, size_t offset, size_t bytes, void* output, cl_uint, const cl_event*, cl_event*)
{
    if (probe_failure == FailBufferRead)
        return CL_OUT_OF_RESOURCES;
    memcpy(output, reinterpret_cast<const char*>(buffer->words.data()) + offset, bytes);
    return CL_SUCCESS;
}

enum DotAcceleration
{
    AcceleratedUnsigned = 1u << 0,
    AcceleratedSigned = 1u << 1,
    AcceleratedMixed = 1u << 2,
    AcceleratedSaturatingUnsigned = 1u << 3,
    AcceleratedSaturatingSigned = 1u << 4,
    AcceleratedSaturatingMixed = 1u << 5
};

static unsigned acceleration_mask(const momoten_detail::IntegerDotProductAccelerationProfile& a)
{
    return (a.unsigned_accelerated ? AcceleratedUnsigned : 0)
           | (a.signed_accelerated ? AcceleratedSigned : 0)
           | (a.mixed_signedness_accelerated ? AcceleratedMixed : 0)
           | (a.accumulating_saturating_unsigned_accelerated ? AcceleratedSaturatingUnsigned : 0)
           | (a.accumulating_saturating_signed_accelerated ? AcceleratedSaturatingSigned : 0)
           | (a.accumulating_saturating_mixed_signedness_accelerated ? AcceleratedSaturatingMixed : 0);
}

static int check_vendor_profile(const char* name, const char* extensions, bool supported, momoten::IntegerDotProductBackend backend, unsigned mask, const std::vector<momoten::IntegerDotProductBackend>& attempts)
{
    attempted_backends.clear();
    matching_build_options = true;
    momoten_detail::OpenCLDriverPolicy policy;
    // KHR-only workarounds must never affect vendor lowering.
    policy.split_packed_dot_acc_sat = true;
    policy.swap_mixed_dot_acc_sat = true;
    const momoten_detail::ShaderDeviceProfile profile = momoten_detail::probe_shader_device_profile(
        reinterpret_cast<cl_device_id>(1), extensions, 1, 1, policy);
    const momoten_detail::IntegerDotProductProfile& dot = profile.integer_dot_product;
    if (dot.supported != supported || dot.backend != backend
        || dot.input_4x8bit != supported || dot.input_4x8bit_packed != supported
        || acceleration_mask(dot.acceleration_4x8bit_packed) != mask
        || acceleration_mask(dot.acceleration_8bit) != mask
        || attempted_backends != attempts || !matching_build_options || live_buffers || live_queues)
    {
        fprintf(stderr, "driver_policy_test: %s: backend=%d supported=%d acceleration=%u/%u\n", name, dot.backend, dot.supported, acceleration_mask(dot.acceleration_4x8bit_packed), acceleration_mask(dot.acceleration_8bit));
        return 1;
    }
    VkPhysicalDevice_T physical = {};
    physical.shader_profile = profile;
    physical.driver_policy = policy;
    VkDevice_T device = {};
    device.physical_device = &physical;
    device.enabled_shader_profile.integer_dot_product = supported;
    momoten::TranslationOptions options;
    momoten_detail::configure_translation_options(&device, options);
    if (options.integer_dot_product != supported || options.integer_dot_product_backend != backend
        || options.integer_dot_product_split_packed_acc_sat || options.integer_dot_product_swap_mixed_acc_sat)
        return 1;
    device.enabled_shader_profile.integer_dot_product = false;
    momoten_detail::configure_translation_options(&device, options);
    return options.integer_dot_product || options.integer_dot_product_backend != momoten::IntegerDotProductKhr;
}

static int check_vendor_profiles()
{
    using namespace momoten;
    const char* arm = "cl_arm_integer_dot_product_int8";
    const char* acc = "cl_arm_integer_dot_product_accumulate_int8";
    const char* sat = "cl_arm_integer_dot_product_accumulate_saturate_int8";
    const char* qcom = "cl_qcom_dot_product8";
    const char* all = "cl_khr_integer_dot_product cl_arm_integer_dot_product_int8 cl_arm_integer_dot_product_accumulate_int8 cl_arm_integer_dot_product_accumulate_saturate_int8 cl_qcom_dot_product8";
    language_version = "OpenCL C 1.2";
    expected_build_options = "-cl-std=CL1.2";
    vector_support = true;
    attempted_backends.clear();
    const momoten_detail::ShaderDeviceProfile preferred = momoten_detail::probe_shader_device_profile(
        reinterpret_cast<cl_device_id>(1), all, 1, 1, momoten_detail::OpenCLDriverPolicy());
    if (!preferred.integer_dot_product.supported || preferred.integer_dot_product.backend != IntegerDotProductKhr
        || attempted_backends != std::vector<IntegerDotProductBackend>{IntegerDotProductKhr})
        return 1;
    khr_acceleration_available = false;
    attempted_backends.clear();
    const momoten_detail::ShaderDeviceProfile unaccelerated = momoten_detail::probe_shader_device_profile(
        reinterpret_cast<cl_device_id>(1), all, 1, 1, momoten_detail::OpenCLDriverPolicy());
    if (!unaccelerated.integer_dot_product.supported || unaccelerated.integer_dot_product.backend != IntegerDotProductKhr
        || acceleration_mask(unaccelerated.integer_dot_product.acceleration_8bit)
        || acceleration_mask(unaccelerated.integer_dot_product.acceleration_4x8bit_packed)
        || attempted_backends != std::vector<IntegerDotProductBackend>{IntegerDotProductKhr})
        return 1;
    khr_acceleration_available = true;
    const unsigned arm_acceleration = AcceleratedUnsigned | AcceleratedSigned;
    const unsigned arm_saturating_acceleration = arm_acceleration | AcceleratedSaturatingUnsigned | AcceleratedSaturatingSigned;
    const unsigned qcom_acceleration = AcceleratedUnsigned | AcceleratedMixed | AcceleratedSaturatingMixed;
    int result = 0;
    result |= check_vendor_profile("ARM dot", arm, true, IntegerDotProductArm, arm_acceleration, {IntegerDotProductArm});
    result |= check_vendor_profile("ARM accumulate only", acc, true, IntegerDotProductArmAccumulate, arm_acceleration, {IntegerDotProductArmAccumulate});
    result |= check_vendor_profile("ARM saturate only", sat, true, IntegerDotProductArmAccumulateSaturate, arm_saturating_acceleration, {IntegerDotProductArmAccumulateSaturate});
    result |= check_vendor_profile("QCOM signedness", qcom, true, IntegerDotProductQcom, qcom_acceleration, {IntegerDotProductQcom});
    result |= check_vendor_profile("extension name boundaries", "cl_arm_integer_dot_product_int8_extra cl_qcom_dot_product8_extra", false, IntegerDotProductKhr, 0, {});
    result |= check_vendor_profile("INT16 is not INT8", "cl_arm_integer_dot_product_accumulate_int16", false, IntegerDotProductKhr, 0, {});
    khr_capabilities_available = false;
    result |= check_vendor_profile("KHR query failure", all, true, IntegerDotProductArmAccumulateSaturate, arm_saturating_acceleration, {IntegerDotProductArmAccumulateSaturate});
    khr_capabilities_available = true;
    rejected_backends.insert(IntegerDotProductKhr);
    result |= check_vendor_profile("KHR compile failure", all, true, IntegerDotProductArmAccumulateSaturate, arm_saturating_acceleration, {IntegerDotProductKhr, IntegerDotProductArmAccumulateSaturate});
    corrupted_backend = IntegerDotProductArmAccumulateSaturate;
    corrupted_result = 5;
    result |= check_vendor_profile("ARM saturate arithmetic failure", all, true, IntegerDotProductArm, arm_acceleration, {IntegerDotProductKhr, IntegerDotProductArmAccumulateSaturate, IntegerDotProductArm});
    corrupted_backend = -1;
    corrupted_result = -1;
    rejected_backends.insert(IntegerDotProductArmAccumulateSaturate);
    result |= check_vendor_profile("ARM saturate compile failure", all, true, IntegerDotProductArm, arm_acceleration, {IntegerDotProductKhr, IntegerDotProductArmAccumulateSaturate, IntegerDotProductArm});
    rejected_backends.insert(IntegerDotProductArm);
    result |= check_vendor_profile("ARM dot compile failure", all, true, IntegerDotProductArmAccumulate, arm_acceleration, {IntegerDotProductKhr, IntegerDotProductArmAccumulateSaturate, IntegerDotProductArm, IntegerDotProductArmAccumulate});
    rejected_backends.insert(IntegerDotProductArmAccumulate);
    result |= check_vendor_profile("all ARM candidates fail", all, true, IntegerDotProductQcom, qcom_acceleration, {IntegerDotProductKhr, IntegerDotProductArmAccumulateSaturate, IntegerDotProductArm, IntegerDotProductArmAccumulate, IntegerDotProductQcom});
    rejected_backends.insert(IntegerDotProductQcom);
    result |= check_vendor_profile("all candidates fail", all, false, IntegerDotProductKhr, 0, {IntegerDotProductKhr, IntegerDotProductArmAccumulateSaturate, IntegerDotProductArm, IntegerDotProductArmAccumulate, IntegerDotProductQcom});
    rejected_backends.clear();
    for (corrupted_result = 0; corrupted_result < 6; corrupted_result++)
        result |= check_vendor_profile("incorrect arithmetic", qcom, false, IntegerDotProductKhr, 0, {IntegerDotProductQcom});
    corrupted_result = -1;
    for (ProbeFailure failure : {FailQueue, FailInputBuffer, FailOutputBuffer, FailKernelArgument, FailKernelEnqueue, FailBufferRead})
    {
        probe_failure = failure;
        result |= check_vendor_profile("runtime probe failure", arm, false, IntegerDotProductKhr, 0, {IntegerDotProductArm});
    }
    probe_failure = ProbeSuccess;
    language_version = 0;
    expected_build_options = "";
    return result;
}

static int check_profile(const char* name, const char* vendor, const char* driver, bool supports_vector, bool workaround, bool swap = false)
{
    vector_support = supports_vector;
    matching_build_options = true;
    dot_probe_source.clear();
    const momoten_detail::OpenCLDriverPolicy policy = momoten_detail::opencl_driver_policy(vendor, driver);
    const momoten_detail::ShaderDeviceProfile profile = momoten_detail::probe_shader_device_profile(
        reinterpret_cast<cl_device_id>(1), "cl_khr_integer_dot_product", 1, 1,
        policy);
    const momoten_detail::IntegerDotProductProfile& dot = profile.integer_dot_product;
    const momoten_detail::IntegerDotProductAccelerationProfile& packed = dot.acceleration_4x8bit_packed;
    if (!dot.supported || !dot.input_4x8bit_packed || dot.input_4x8bit != supports_vector
        || !matching_build_options
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
    momoten_detail::g_opencl.p_clCreateKernel = create_kernel;
    momoten_detail::g_opencl.p_clReleaseKernel = release_kernel;
    momoten_detail::g_opencl.p_clCreateCommandQueue = create_queue;
    momoten_detail::g_opencl.p_clReleaseCommandQueue = release_queue;
    momoten_detail::g_opencl.p_clCreateBuffer = create_buffer;
    momoten_detail::g_opencl.p_clReleaseMemObject = release_buffer;
    momoten_detail::g_opencl.p_clSetKernelArg = set_kernel_arg;
    momoten_detail::g_opencl.p_clEnqueueNDRangeKernel = enqueue_kernel;
    momoten_detail::g_opencl.p_clEnqueueReadBuffer = read_buffer;
    if (check_vendor_profiles())
        return 1;

    const char* broken_driver = "OpenCL 3.0 QUALCOMM build: 0800.73.1 Compiler E031.47.18.50";
    const char* samsung_driver = "OpenCL 3.0 QUALCOMM build: 0800.64.7 Compiler E031.47.18.50";
    const char* honor_driver = "OpenCL 3.0 QUALCOMM build: 0800.74 Compiler E031.47.18.51";
    const char* mixed_driver = "OpenCL 3.0 QUALCOMM build: 0676.73 Compiler E031.41.03.62";
    const char* oppo_driver = "OpenCL 3.0 QUALCOMM build: commit unknown Compiler E031.41.03.57";
    language_version = "OpenCL C 3.0";
    expected_build_options = "-cl-std=CL3.0";
    if (check_profile("CL3.0 integer dot uses the selected language", "QUALCOMM", honor_driver, true, true))
        return 1;
    if (check_profile("OPPO Adreno 740 CL3.0 mixed saturation", "QUALCOMM", oppo_driver, true, false, true))
        return 1;
    language_version = "OpenCL C 2.0";
    expected_build_options = "-cl-std=CL2.0";
    if (check_profile("CL2.0 integer dot uses the selected language", "QUALCOMM", mixed_driver, true, false, true))
        return 1;
    language_version = 0;
    expected_build_options = "";
    return check_profile("affected driver", "QUALCOMM", broken_driver, true, true)
           || check_profile("packed-only inputs", "QUALCOMM", broken_driver, false, true)
           || check_profile("other vendor", "another vendor", broken_driver, true, false)
           || check_profile("same compiler other release", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.74.1 Compiler E031.47.18.50", true, true)
           || check_profile("same compiler other OpenCL version", "QUALCOMM", "OpenCL 2.0 QUALCOMM build: 0800.73.1 Compiler E031.47.18.50", true, true)
           || check_profile("compiler followed by metadata", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.73.1 Compiler E031.47.18.50 extra", true, true)
           || check_profile("other compiler", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.73.1 Compiler E031.47.18.52", true, false)
           || check_profile("compiler version prefix", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.73.1 Compiler E031.47.18.500", true, false)
           || check_profile("compiler version suffix", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.73.1 Compiler E031.47.18.50-custom", true, false)
           || check_profile("Samsung Adreno 830 driver", "QUALCOMM", samsung_driver, true, true)
           || check_profile("Samsung packed-only inputs", "QUALCOMM", samsung_driver, false, true)
           || check_profile("Samsung other vendor", "another vendor", samsung_driver, true, false)
           || check_profile("Samsung same compiler other release", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.64.8 Compiler E031.47.18.50", true, true)
           || check_profile("Samsung other compiler", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.64.7 Compiler E031.47.18.52", true, false)
           || check_profile("Honor Adreno 830 compiler", "QUALCOMM", honor_driver, true, true)
           || check_profile("Honor packed-only inputs", "QUALCOMM", honor_driver, false, true)
           || check_profile("Honor other vendor", "another vendor", honor_driver, true, false)
           || check_profile("Honor same compiler other release", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.75 Compiler E031.47.18.51", true, true)
           || check_profile("Honor compiler version prefix", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.74 Compiler E031.47.18.510", true, false)
           || check_profile("OPPO Adreno 740 packed-only", "QUALCOMM", oppo_driver, false, false, true)
           || check_profile("OPPO other vendor", "another vendor", oppo_driver, true, false)
           || check_profile("OPPO compiler followed by metadata", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: commit unknown Compiler E031.41.03.57 extra", true, false, true)
           || check_profile("OPPO compiler version prefix", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: commit unknown Compiler E031.41.03.570", true, false)
           || check_profile("OPPO compiler version suffix", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: commit unknown Compiler E031.41.03.57-custom", true, false)
           || check_profile("mixed saturation driver", "QUALCOMM", mixed_driver, true, false, true)
           || check_profile("mixed saturation packed-only", "QUALCOMM", mixed_driver, false, false, true)
           || check_profile("mixed saturation other vendor", "another vendor", mixed_driver, true, false)
           || check_profile("mixed saturation same compiler other release", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0676.74 Compiler E031.41.03.62", true, false, true)
           || check_profile("mixed saturation other compiler", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0676.73 Compiler E031.41.03.63", true, false)
           || check_profile("mixed saturation compiler version prefix", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0676.73 Compiler E031.41.03.620", true, false)
           || check_profile("missing compiler", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.73.1", true, false)
           || check_profile("empty compiler", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.73.1 Compiler ", true, false)
           || check_profile("other compiler field", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: 0800.73.1 OtherCompiler E031.47.18.50", true, false)
           || check_profile("compiler version outside field", "QUALCOMM", "OpenCL 3.0 QUALCOMM build: E031.47.18.50 Compiler E031.47.18.52", true, false)
           || check_profile("missing identity", "", "", true, false)
           || check_copy_policy("Imagination Technologies", "1.10@5130912", true)
           || check_copy_policy("Imagination Technologies", "1.11@5425693", true)
           || check_copy_policy("Imagination Technologies", "1.11@5425694", false)
           || check_copy_policy("another vendor", "1.11@5425693", false)
           || check_copy_policy("", "", false);
}
