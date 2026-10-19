// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>

#include <cstdio>
#include <cstring>
#include <vector>

// This SDK-only probe recognizes two independently reproduced Apple CPU
// restrictions. It must never turn an unrelated failure into a test skip.
// Exit 0: run the full suite; 77: both known restrictions; 1: unexpected failure.
static bool check(cl_int result, const char* operation)
{
    if (result == CL_SUCCESS)
        return true;
    std::fprintf(stderr, "apple_cpu_probe: %s failed: %d\n", operation, result);
    return false;
}

static bool print_build_log(cl_program program, cl_device_id device)
{
    size_t size = 0;
    if (!check(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                     0, 0, &size), "build log size"))
        return false;
    if (size == 0)
        return true;
    std::vector<char> log(size + 1, 0);
    if (!check(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                     size, log.data(), 0), "build log"))
        return false;
    if (log[0])
        std::fprintf(stderr, "apple_cpu_probe: build log: %s\n", log.data());
    return true;
}

static int run_case(cl_context context, cl_command_queue queue, cl_device_id device,
                    cl_mem output, unsigned test)
{
    const char* names[] = {"no-barrier control", "barrier", "struct plus barrier"};
    const char* sources[] = {
        "__kernel __attribute__((reqd_work_group_size(4,1,1))) void k(__global uint*p) "
        "{p[get_global_id(0)] = get_local_id(0);}",
        "__kernel __attribute__((reqd_work_group_size(4,1,1))) void k(__global uint*p) "
        "{__local uint a[4]; uint x=get_local_id(0); a[x]=x; "
        "barrier(CLK_LOCAL_MEM_FENCE); p[get_global_id(0)]=a[3-x];}",
        "typedef struct {float scale; int bias; uint count;} Push; "
        "__kernel __attribute__((reqd_work_group_size(4,1,1))) "
        "void k(__global uint*p, Push push) {__local uint a[4]; uint x=get_local_id(0); "
        "if(get_global_id(0)<push.count) {a[x]=x; barrier(CLK_LOCAL_MEM_FENCE); "
        "p[get_global_id(0)]=a[3-x];}}"
    };
    cl_program program = 0;
    cl_kernel kernel = 0;
    int status = 1;
    do
    {
        cl_int result = CL_SUCCESS;
        program = clCreateProgramWithSource(context, 1, &sources[test], 0, &result);
        if (!check(result, "clCreateProgramWithSource") || !program)
            break;
        result = clBuildProgram(program, 1, &device, 0, 0, 0);
        std::printf("apple_cpu_probe: %s build=%d\n", names[test], result);
        if (!print_build_log(program, device) || !check(result, "clBuildProgram"))
            break;
        kernel = clCreateKernel(program, "k", &result);
        if (!check(result, "clCreateKernel") || !kernel)
            break;

        cl_uint arguments = 0;
        size_t maximum = 0, required[3] = {};
        if (!check(clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(arguments),
                                    &arguments, 0), "kernel argument count")
            || !check(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                                sizeof(maximum), &maximum, 0), "kernel maximum workgroup")
            || !check(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                                sizeof(required), required, 0), "kernel required workgroup"))
            break;
        std::printf("apple_cpu_probe: %s args=%u max=%zu required=%zu,%zu,%zu\n",
                    names[test], arguments, maximum, required[0], required[1], required[2]);
        const bool required_matches = required[0] == 4 && required[1] == 1 && required[2] == 1;
        const bool required_missing = required[0] == 0 && required[1] == 0 && required[2] == 0;
        if (test != 2 && !required_matches)
            break;

        // Reset between cases so stale output cannot make a failed kernel pass.
        const cl_uint sentinel[4] = {99, 99, 99, 99};
        if (!check(clEnqueueWriteBuffer(queue, output, CL_TRUE, 0, sizeof(sentinel),
                                        sentinel, 0, 0, 0), "reset output"))
            break;
        result = clSetKernelArg(kernel, 0, sizeof(output), &output);
        std::printf("apple_cpu_probe: %s arg0(size=%zu)=%d\n", names[test], sizeof(output), result);
        if (!check(result, "set buffer argument"))
            break;
        if (test == 2)
        {
            struct Push { cl_float scale; cl_int bias; cl_uint count; };
            static_assert(sizeof(Push) == 12, "probe requires the original 12-byte struct ABI");
            const Push push = {2.5f, -3, 4};
            result = clSetKernelArg(kernel, 1, sizeof(push), &push);
            std::printf("apple_cpu_probe: %s arg1(size=%zu)=%d\n", names[test], sizeof(push), result);
            if (!check(result, "set struct argument"))
                break;
        }
        const size_t global[3] = {4, 1, 1}, local[3] = {4, 1, 1};
        result = clEnqueueNDRangeKernel(queue, kernel, 3, 0, global, local, 0, 0, 0);
        std::printf("apple_cpu_probe: %s enqueue=%d\n", names[test], result);
        if (test == 1 && arguments == 1 && maximum == 1 && result == CL_INVALID_WORK_GROUP_SIZE)
        {
            std::puts("apple_cpu_probe: confirmed barrier restriction: one argument, maximum=1, enqueue=-54");
            status = 77;
            break;
        }
        if (test == 2 && arguments == 3 && maximum == 1 && required_missing
            && result == CL_INVALID_KERNEL_ARGS)
        {
            std::puts("apple_cpu_probe: confirmed struct ABI restriction: two declared arguments reported as three, maximum=1, required=0,0,0, both setters succeeded, enqueue=-52");
            status = 77;
            break;
        }
        if (!check(result, "clEnqueueNDRangeKernel") || maximum < 4 || !required_matches
            || arguments != (test == 2 ? 2u : 1u))
            break;
        cl_uint values[4] = {};
        if (!check(clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(values),
                                       values, 0, 0, 0), "read output"))
            break;
        std::printf("apple_cpu_probe: %s output=%u,%u,%u,%u\n",
                    names[test], values[0], values[1], values[2], values[3]);
        bool correct = true;
        for (unsigned i = 0; i < 4; ++i)
            correct = correct && values[i] == (test == 0 ? i : 3u - i);
        if (!correct)
            break;
        status = 0;
    } while (false);
    if (kernel && !check(clReleaseKernel(kernel), "clReleaseKernel"))
        status = 1;
    if (program && !check(clReleaseProgram(program), "clReleaseProgram"))
        status = 1;
    if (status == 1)
        std::fprintf(stderr, "apple_cpu_probe: unexpected %s result; refusing capability skip\n", names[test]);
    return status;
}

int main()
{
    // Match the tests' first ALL device, without selecting a different CPU from
    // a multi-device machine. Nonmatching topologies must run the real suite.
    cl_uint count = 0;
    if (!check(clGetPlatformIDs(0, 0, &count), "platform count"))
        return 1;
    if (count != 1)
    {
        std::puts("apple_cpu_probe: not exactly one platform; run full suite");
        return 0;
    }
    cl_platform_id platform = 0;
    char vendor[256] = {};
    if (!check(clGetPlatformIDs(1, &platform, 0), "platform")
        || !check(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, 0), "platform vendor"))
        return 1;
    if (std::strcmp(vendor, "Apple") != 0)
    {
        std::puts("apple_cpu_probe: not an Apple platform; run full suite");
        return 0;
    }
    if (!check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, 0, &count), "ALL device count"))
        return 1;
    if (count != 1)
    {
        std::puts("apple_cpu_probe: not exactly one ALL device; run full suite");
        return 0;
    }
    cl_device_id device = 0;
    cl_device_type type = 0;
    char name[256] = {};
    if (!check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, 0), "ALL device")
        || !check(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, 0), "device type")
        || !check(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, 0), "device name"))
        return 1;
    if (type != CL_DEVICE_TYPE_CPU)
    {
        std::puts("apple_cpu_probe: selected device is not CPU; run full suite");
        return 0;
    }
    std::printf("apple_cpu_probe: Apple CPU device: %s\n", name);

    cl_int result = CL_SUCCESS;
    cl_context context = clCreateContext(0, 1, &device, 0, 0, &result);
    if (!check(result, "clCreateContext") || !context)
        return 1;
    cl_command_queue queue = 0;
    cl_mem output = 0;
    int status = 1;
    do
    {
        queue = clCreateCommandQueue(context, device, 0, &result);
        if (!check(result, "clCreateCommandQueue") || !queue)
            break;
        output = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * sizeof(cl_uint), 0, &result);
        if (!check(result, "clCreateBuffer") || !output)
            break;
        if (run_case(context, queue, device, output, 0) != 0)
            break;
        const int barrier = run_case(context, queue, device, output, 1);
        const int structure = run_case(context, queue, device, output, 2);
        if (barrier == 0 && structure == 0)
            status = 0;
        else if (barrier == 77 && structure == 77)
            status = 77;
        else
            std::fputs("apple_cpu_probe: mixed or unknown capabilities; refusing capability skip\n", stderr);
    } while (false);
    if (output && !check(clReleaseMemObject(output), "clReleaseMemObject"))
        status = 1;
    if (queue && !check(clReleaseCommandQueue(queue), "clReleaseCommandQueue"))
        status = 1;
    if (!check(clReleaseContext(context), "clReleaseContext"))
        status = 1;
    std::printf("apple_cpu_probe: result=%d\n", status);
    return status;
}
