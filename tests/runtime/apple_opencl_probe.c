/* Copyright 2026 nihui
 * SPDX-License-Identifier: Apache-2.0
 */

/* Confirm a hosted Apple ARM64 environment limitation independently of ctypes
 * and momoten. This does not claim that hardware is absent or works correctly.
 * Any other result must leave the Python runtime smoke test failing.
 */
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
    cl_platform_id platform = NULL;
    cl_uint count = 0;
    cl_int result = clGetPlatformIDs(1, &platform, &count);
    if (result != CL_SUCCESS || count != 1)
        return 1;

    char name[128] = {0};
    if (clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), name, NULL) != CL_SUCCESS
        || strcmp(name, "Apple") != 0)
        return 1;

    cl_uint device_count = 0;
    cl_int count_result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
    cl_device_id device = NULL;
    cl_int buffer_result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &device_count);
    printf("Native SDK Apple OpenCL probe: count-only=%d, buffer+count=%d\n",
           count_result, buffer_result);
    return count_result == CL_INVALID_VALUE && buffer_result == CL_INVALID_VALUE ? 78 : 1;
}
