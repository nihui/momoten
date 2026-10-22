// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "momoten/spv_to_clc.h"
#include "translator/integer_dot_product.h"
#include "../translator/integer_dot_product_fixture.h"

#include <CL/cl.h>
#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

// Model only the vendor builtins. The generated SPIR-V lowering, signedness
// corrections, saturation and buffer ABI run unchanged on a real CL compiler.
static std::string emulated_builtins(momoten::IntegerDotProductBackend backend)
{
    std::string source = "int test_ss(char4 a, char4 b) { int4 p = convert_int4(a) * convert_int4(b); return p.x + p.y + p.z + p.w; }\n"
                         "uint test_uu(uchar4 a, uchar4 b) { uint4 p = convert_uint4(a) * convert_uint4(b); return p.x + p.y + p.z + p.w; }\n"
                         "int test_su(char4 a, uchar4 b) { int4 p = convert_int4(a) * convert_int4(b); return p.x + p.y + p.z + p.w; }\n";
    if (backend == momoten::IntegerDotProductQcom)
        return source
               + "int qcom_dot8_acc(uint a, uint b, int c) { return add_sat(test_su(as_char4(a), as_uchar4(b)), c); }\n"
                 "int qcom_udot8_acc(uint a, uint b, int c) { return add_sat((int)test_uu(as_uchar4(a), as_uchar4(b)), c); }\n";
    return source
           + "int __attribute__((overloadable)) arm_dot(char4 a, char4 b) { return test_ss(a, b); }\n"
             "uint __attribute__((overloadable)) arm_dot(uchar4 a, uchar4 b) { return test_uu(a, b); }\n"
             "int __attribute__((overloadable)) arm_dot_acc(char4 a, char4 b, int c) { return as_int(as_uint(test_ss(a, b)) + as_uint(c)); }\n"
             "uint __attribute__((overloadable)) arm_dot_acc(uchar4 a, uchar4 b, uint c) { return test_uu(a, b) + c; }\n"
             "int __attribute__((overloadable)) arm_dot_acc_sat(char4 a, char4 b, int c) { return add_sat(test_ss(a, b), c); }\n"
             "uint __attribute__((overloadable)) arm_dot_acc_sat(uchar4 a, uchar4 b, uint c) { return add_sat(test_uu(a, b), c); }\n";
}

struct Data
{
    std::vector<uint32_t> input;
    std::vector<uint32_t> expected;

    void append(uint32_t a, uint32_t b, uint32_t accumulator)
    {
        input.insert(input.end(), {a, b, accumulator, 0});
        int64_t dots[4] = {};
        for (unsigned lane = 0; lane < 4; lane++)
        {
            const int ua = (a >> (lane * 8)) & 255u;
            const int ub = (b >> (lane * 8)) & 255u;
            const int sa = ua >= 128 ? ua - 256 : ua;
            const int sb = ub >= 128 ? ub - 256 : ub;
            dots[0] += sa * sb;
            dots[1] += ua * ub;
            dots[2] += sa * ub;
            dots[3] += ua * sb;
        }
        int32_t signed_accumulator;
        memcpy(&signed_accumulator, &accumulator, sizeof(accumulator));
        for (unsigned i = 0; i < 4; i++)
            expected.push_back(static_cast<uint32_t>(dots[i]));
        for (unsigned i = 0; i < 4; i++)
        {
            const int64_t sum = dots[i] + (i == 1 ? int64_t(accumulator) : int64_t(signed_accumulator));
            const int64_t minimum = i == 1 ? 0 : -2147483648LL;
            const int64_t maximum = i == 1 ? 4294967295LL : 2147483647LL;
            expected.push_back(static_cast<uint32_t>(std::min(maximum, std::max(minimum, sum))));
        }
        expected.push_back(static_cast<uint32_t>(dots[0]) + accumulator);
    }

    Data()
    {
        const uint32_t boundaries[] = {0, 1, 0xffffffffu, 0x80000000u, 0x80000001u, 0x80008000u, 0x7fffffffu, 0x7ffffffeu, 0x7fff8000u, 0xffff0000u};
        const uint32_t packed[] = {0, 0x01010101u, 0x7f7f7f7fu, 0x80808080u, 0xffffffffu, 0xff7f8001u, 0x807f00ffu};
        for (uint32_t a : packed)
            for (uint32_t b : packed)
                for (uint32_t c : boundaries)
                    append(a, b, c);
        // Every pair of byte values occurs in every lane, with asymmetric
        // neighboring lanes and alternating accumulators near saturation.
        for (uint32_t a = 0; a < 256; a++)
            for (uint32_t b = 0; b < 256; b++)
            {
                const uint32_t va = a | (((a + 17) & 255u) << 8) | ((255u - a) << 16) | ((a ^ 128u) << 24);
                const uint32_t vb = b | ((b ^ 128u) << 8) | (((b + 13) & 255u) << 16) | ((255u - b) << 24);
                append(va, vb, boundaries[(a * 256 + b) % 10]);
            }
        uint32_t random = 0x12345678u;
        for (unsigned i = 0; i < 4096; i++)
        {
            uint32_t values[3];
            for (uint32_t& value : values)
            {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                value = random;
            }
            append(values[0], values[1], values[2]);
        }
    }
};

static bool has_extension(cl_device_id device, const char* extension)
{
    size_t size = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, 0, &size) != CL_SUCCESS || !size)
        return false;
    std::vector<char> text(size);
    if (clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, size, text.data(), 0) != CL_SUCCESS)
        return false;
    return (" " + std::string(text.data()) + " ").find(" " + std::string(extension) + " ") != std::string::npos;
}

struct Runtime
{
    cl_context context = 0;
    cl_command_queue queue = 0;
    cl_mem buffers[2] = {};
    cl_program program = 0;
    cl_kernel kernel = 0;

    ~Runtime()
    {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        for (cl_mem buffer : buffers)
            if (buffer) clReleaseMemObject(buffer);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

static int run(cl_device_id device, momoten::IntegerDotProductBackend backend, bool packed, unsigned address_bits, bool native, const Data& data)
{
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
    std::vector<uint32_t> words;
    if (!tools.Assemble(integer_dot_product_fixture(packed), &words) || !tools.Validate(words))
        return 1;
    momoten::TranslationOptions options;
    options.address_bits = address_bits;
    options.integer_dot_product = true;
    options.integer_dot_product_backend = backend;
    options.integer_dot_product_input_4x8bit = true;
    options.integer_dot_product_input_4x8bit_packed = true;
    momoten::TranslationResult translated;
    if (!momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, translated))
    {
        fprintf(stderr, "vendor_integer_dot_product_test: %s\n", translated.diagnostics.c_str());
        return 1;
    }
    if (!native)
    {
        const std::string pragma = "#pragma OPENCL EXTENSION " + std::string(momoten::integer_dot_product_extension(backend)) + " : enable";
        const size_t position = translated.source.find(pragma);
        if (position == std::string::npos)
            return 1;
        translated.source.erase(position, pragma.size());
        translated.source.insert(0, emulated_builtins(backend));
    }

    Runtime runtime;
    cl_int error = CL_SUCCESS;
    runtime.context = clCreateContext(0, 1, &device, 0, 0, &error);
    if (!runtime.context || error != CL_SUCCESS)
        return 1;
    runtime.queue = clCreateCommandQueue(runtime.context, device, 0, &error);
    if (!runtime.queue || error != CL_SUCCESS)
        return 1;
    const char* source = translated.source.c_str();
    runtime.program = clCreateProgramWithSource(runtime.context, 1, &source, 0, &error);
    if (!runtime.program || error != CL_SUCCESS)
        return 1;
    // Emulation uses the portable baseline. Native tests use the reported
    // language dialect, since vendor overloads can depend on that selection.
    const char* build_options = 0;
    if (native)
    {
        char version[128] = {};
        unsigned major = 0, minor = 0;
        clGetDeviceInfo(device, 0x103d, sizeof(version), version, 0); // CL_DEVICE_OPENCL_C_VERSION
        if (sscanf(version, "OpenCL C %u.%u", &major, &minor) == 2)
            build_options = major >= 3 ? "-cl-std=CL3.0" : major == 2 ? "-cl-std=CL2.0"
                                                       : minor >= 2   ? "-cl-std=CL1.2"
                                                       : minor == 1   ? "-cl-std=CL1.1"
                                                                      : 0;
    }
    error = clBuildProgram(runtime.program, 1, &device, build_options, 0, 0);
    if (error != CL_SUCCESS)
    {
        size_t size = 0;
        clGetProgramBuildInfo(runtime.program, device, CL_PROGRAM_BUILD_LOG, 0, 0, &size);
        std::vector<char> log(size + 1, 0);
        clGetProgramBuildInfo(runtime.program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), 0);
        fprintf(stderr, "vendor_integer_dot_product_test: backend=%d packed=%d: build %d: %s\n", backend, packed, error, log.data());
        return 1;
    }
    runtime.kernel = clCreateKernel(runtime.program, translated.abi.entry_point.c_str(), &error);
    if (!runtime.kernel || error != CL_SUCCESS)
        return 1;
    std::vector<uint32_t> output(data.expected.size(), 0xa5a5a5a5u);
    runtime.buffers[0] = clCreateBuffer(runtime.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data.input.size() * sizeof(uint32_t), const_cast<uint32_t*>(data.input.data()), &error);
    if (!runtime.buffers[0] || error != CL_SUCCESS)
        return 1;
    runtime.buffers[1] = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, output.size() * sizeof(uint32_t), output.data(), &error);
    if (!runtime.buffers[1] || error != CL_SUCCESS)
        return 1;
    const cl_uint offset32 = 0;
    const cl_ulong offset64 = 0;
    for (const momoten::BufferArgument& argument : translated.abi.buffers)
    {
        if (argument.binding >= 2)
            return 1;
        error = clSetKernelArg(runtime.kernel, argument.buffer_arg_index, sizeof(cl_mem), &runtime.buffers[argument.binding]);
        if (error == CL_SUCCESS)
            error = clSetKernelArg(runtime.kernel, argument.offset_arg_index, address_bits / 8, address_bits == 64 ? static_cast<const void*>(&offset64) : static_cast<const void*>(&offset32));
        if (error != CL_SUCCESS)
            return 1;
    }
    const size_t count = data.input.size() / 4;
    const size_t local = 1;
    error = clEnqueueNDRangeKernel(runtime.queue, runtime.kernel, 1, 0, &count, &local, 0, 0, 0);
    if (error == CL_SUCCESS)
        error = clEnqueueReadBuffer(runtime.queue, runtime.buffers[1], CL_TRUE, 0, output.size() * sizeof(uint32_t), output.data(), 0, 0, 0);
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "vendor_integer_dot_product_test: dispatch %d\n", error);
        return 1;
    }
    for (size_t i = 0; i < output.size(); i++)
        if (output[i] != data.expected[i])
        {
            const uint32_t* input = &data.input[i / 9 * 4];
            fprintf(stderr, "vendor_integer_dot_product_test: backend=%d packed=%d bits=%u row=%zu operation=%zu a=%08x b=%08x acc=%08x actual=%08x expected=%08x\n", backend, packed, address_bits, i / 9, i % 9, input[0], input[1], input[2], output[i], data.expected[i]);
            return 1;
        }
    printf("backend=%d packed=%d address_bits=%u %s: %zu results passed\n", backend, packed, address_bits, native ? "native" : "emulated builtins", output.size());
    return 0;
}

int main(int argc, char** argv)
{
    const bool native = argc == 2 && strcmp(argv[1], "native") == 0;
    cl_uint count = 0;
    if (clGetPlatformIDs(0, 0, &count) != CL_SUCCESS || !count)
        return 77;
    std::vector<cl_platform_id> platforms(count);
    if (clGetPlatformIDs(count, platforms.data(), 0) != CL_SUCCESS)
        return 1;
    const Data data;
    unsigned tested = 0;
    for (cl_platform_id platform : platforms)
    {
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, 0, &count) != CL_SUCCESS || !count)
            continue;
        std::vector<cl_device_id> devices(count);
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, count, devices.data(), 0) != CL_SUCCESS)
            return 1;
        for (cl_device_id device : devices)
        {
            cl_bool compiler = CL_FALSE;
            if (clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler), &compiler, 0) != CL_SUCCESS || !compiler)
                continue;
            char profile[32] = {};
            clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profile), profile, 0);
            const bool int64 = strcmp(profile, "FULL_PROFILE") == 0 || has_extension(device, "cles_khr_int64");
            for (int b = momoten::IntegerDotProductArm; b <= momoten::IntegerDotProductQcom; b++)
            {
                const momoten::IntegerDotProductBackend backend = static_cast<momoten::IntegerDotProductBackend>(b);
                if (native && !has_extension(device, momoten::integer_dot_product_extension(backend)))
                    continue;
                for (bool packed : {false, true})
                    for (unsigned bits : {32u, 64u})
                    {
                        if (bits == 64 && !int64)
                            continue;
                        if (run(device, backend, packed, bits, native, data))
                            return 1;
                        tested++;
                    }
            }
            if (!native && tested)
                return 0;
        }
    }
    return tested ? 0 : 77;
}
