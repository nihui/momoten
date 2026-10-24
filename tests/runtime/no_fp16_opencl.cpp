// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include <CL/cl.h>
#include <dlfcn.h>
#include <cstring>
#include <string>
#include <vector>

typedef cl_int(CL_API_CALL* GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);

// retain the OpenCL loader dependency for dlsym-based forwarding
extern "C" GetPlatformIDs momoten_test_opencl_anchor()
{
    return &clGetPlatformIDs;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo(cl_device_id device, cl_device_info key, size_t size, void* output, size_t* required)
{
    typedef cl_int(CL_API_CALL * GetInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
    static GetInfo get_info = reinterpret_cast<GetInfo>(dlsym(RTLD_NEXT, "clGetDeviceInfo"));
    if (!get_info)
        return CL_INVALID_OPERATION;
    if (key != CL_DEVICE_EXTENSIONS)
        return get_info(device, key, size, output, required);
    size_t bytes = 0;
    cl_int status = get_info(device, key, 0, 0, &bytes);
    if (status != CL_SUCCESS)
        return status;
    std::vector<char> extensions(bytes);
    status = get_info(device, key, bytes, extensions.data(), 0);
    if (status != CL_SUCCESS)
        return status;
    std::string filtered = " " + std::string(extensions.data()) + " ";
    const std::string excluded = " cl_khr_fp16 ";
    const size_t offset = filtered.find(excluded);
    if (offset != std::string::npos)
        filtered.erase(offset, excluded.size() - 1);
    if (required)
        *required = filtered.size() + 1;
    if (output)
    {
        if (size < filtered.size() + 1)
            return CL_INVALID_VALUE;
        std::memcpy(output, filtered.c_str(), filtered.size() + 1);
    }
    return CL_SUCCESS;
}
