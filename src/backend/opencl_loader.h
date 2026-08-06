// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_OPENCL_LOADER_H
#define MOMOTEN_OPENCL_LOADER_H

#include <CL/cl_icd.h>

namespace momoten_detail {

// Keep this optional OpenCL 1.2 loader entry point usable when momoten is
// compiled with CL_TARGET_OPENCL_VERSION=100.
typedef void*(CL_API_CALL* momo_clGetExtensionFunctionAddressForPlatform_fn)(
    cl_platform_id platform, const char* function_name);

#define MOMOTEN_OPENCL_FUNCTIONS(F) \
    F(clGetDeviceIDs)               \
    F(clGetDeviceInfo)              \
    F(clCreateContext)              \
    F(clReleaseContext)             \
    F(clCreateCommandQueue)         \
    F(clReleaseCommandQueue)        \
    F(clCreateBuffer)               \
    F(clReleaseMemObject)           \
    F(clCreateProgramWithSource)    \
    F(clCreateProgramWithBinary)    \
    F(clRetainProgram)              \
    F(clReleaseProgram)             \
    F(clBuildProgram)               \
    F(clGetProgramInfo)             \
    F(clGetProgramBuildInfo)        \
    F(clCreateKernel)               \
    F(clReleaseKernel)              \
    F(clSetKernelArg)               \
    F(clGetKernelWorkGroupInfo)     \
    F(clGetEventInfo)               \
    F(clRetainEvent)                \
    F(clReleaseEvent)               \
    F(clFlush)                      \
    F(clFinish)                     \
    F(clEnqueueReadBuffer)          \
    F(clEnqueueWriteBuffer)         \
    F(clEnqueueCopyBuffer)          \
    F(clEnqueueNDRangeKernel)       \
    F(clEnqueueMarker)

struct OpenCLApi
{
#define MOMOTEN_DECLARE_OPENCL_FUNCTION(name) name##_fn p_##name;
    MOMOTEN_OPENCL_FUNCTIONS(MOMOTEN_DECLARE_OPENCL_FUNCTION)
#undef MOMOTEN_DECLARE_OPENCL_FUNCTION
    // Optional entry points. They deliberately stay outside the required
    // function list so an OpenCL 1.0 implementation remains loadable.
    momo_clGetExtensionFunctionAddressForPlatform_fn p_clGetExtensionFunctionAddressForPlatform;
    clGetKernelSubGroupInfoKHR_fn p_clGetKernelSubGroupInfoKHR;
};

extern OpenCLApi g_opencl;

bool debug_enabled();
cl_int momo_clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms,
                             cl_uint* num_platforms);

} // namespace momoten_detail

#endif // MOMOTEN_OPENCL_LOADER_H
