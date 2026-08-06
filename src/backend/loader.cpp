// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "opencl_loader.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dirent.h>
#include <dlfcn.h>
#endif
#include <sys/stat.h>

namespace momoten_detail {

bool debug_enabled()
{
    const char* value = getenv("MOMOTEN_DEBUG");
    if (!value || !value[0])
        value = getenv("NCNN_MOMOTEN_DEBUG");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

#if defined(_WIN32)
typedef HMODULE OpenCLLibrary;
#else
typedef void* OpenCLLibrary;
#endif

OpenCLApi g_opencl = {};
static OpenCLLibrary g_opencl_library = 0;
static std::vector<cl_platform_id> g_opencl_platforms;
static std::string g_opencl_driver_path;
static std::once_flag g_opencl_once;
static bool g_opencl_loaded = false;

static std::string trim_opencl_path(const std::string& value)
{
    size_t begin = 0;
    while (begin < value.size() && (value[begin] == ' ' || value[begin] == '\t' || value[begin] == '\r' || value[begin] == '\n'))
        begin++;
    size_t end = value.size();
    while (end > begin && (value[end - 1] == ' ' || value[end - 1] == '\t' || value[end - 1] == '\r' || value[end - 1] == '\n'))
        end--;
    if (end - begin >= 2 && ((value[begin] == '"' && value[end - 1] == '"') || (value[begin] == '\'' && value[end - 1] == '\'')))
    {
        begin++;
        end--;
    }
    return value.substr(begin, end - begin);
}

static bool has_suffix(const std::string& value, const char* suffix)
{
    const size_t suffix_size = strlen(suffix);
    return value.size() >= suffix_size && value.compare(value.size() - suffix_size, suffix_size, suffix) == 0;
}

static std::string opencl_library_from_icd(const std::string& icd_path)
{
    std::ifstream stream(icd_path.c_str());
    std::string line;
    while (std::getline(stream, line))
    {
        line = trim_opencl_path(line);
        if (line.empty() || line[0] == '#')
            continue;

        const bool absolute = line[0] == '/' || line[0] == '\\' || (line.size() > 1 && line[1] == ':');
        if (!absolute && (line.find('/') != std::string::npos || line.find('\\') != std::string::npos))
        {
            const size_t slash = icd_path.find_last_of("/\\");
            if (slash != std::string::npos)
                line = icd_path.substr(0, slash + 1) + line;
        }
        return line;
    }
    return std::string();
}

static void append_opencl_candidate(std::vector<std::string>& candidates,
                                    const std::string& value)
{
    std::string path = trim_opencl_path(value);
    if (path.empty())
        return;
    if (has_suffix(path, ".icd"))
        path = opencl_library_from_icd(path);
    if (!path.empty() && std::find(candidates.begin(), candidates.end(), path) == candidates.end())
        candidates.push_back(path);
}

#if !defined(_WIN32)
static void append_opencl_icd_directory(std::vector<std::string>& candidates,
                                        const std::string& directory)
{
    DIR* dir = opendir(directory.c_str());
    if (!dir)
        return;
    std::vector<std::string> icd_files;
    for (dirent* entry = readdir(dir); entry; entry = readdir(dir))
    {
        const std::string name(entry->d_name);
        if (name != "." && name != ".." && has_suffix(name, ".icd"))
            icd_files.push_back(directory + "/" + name);
    }
    closedir(dir);
    std::sort(icd_files.begin(), icd_files.end());
    for (size_t i = 0; i < icd_files.size(); i++)
        append_opencl_candidate(candidates, icd_files[i]);
}
#else
static void append_opencl_registry_candidates(std::vector<std::string>& candidates,
                                              REGSAM view)
{
    HKEY key = 0;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Khronos\\OpenCL\\Vendors",
                      0, KEY_READ | view, &key)
        != ERROR_SUCCESS)
        return;
    for (DWORD index = 0;; index++)
    {
        char name[1024];
        DWORD name_size = sizeof(name);
        DWORD type = 0;
        DWORD enabled = 1;
        DWORD enabled_size = sizeof(enabled);
        const LONG result = RegEnumValueA(key, index, name, &name_size, 0, &type,
                                          reinterpret_cast<BYTE*>(&enabled), &enabled_size);
        if (result == ERROR_NO_MORE_ITEMS)
            break;
        if (result == ERROR_SUCCESS && type == REG_DWORD && enabled == 0)
            append_opencl_candidate(candidates, std::string(name, name_size));
    }
    RegCloseKey(key);
}
#endif

static std::vector<std::string> opencl_driver_candidates()
{
    std::vector<std::string> candidates;
    const char* explicit_driver = getenv("MOMOTEN_OPENCL_DRIVER");
    if (!explicit_driver || !explicit_driver[0])
        explicit_driver = getenv("NCNN_MOMOTEN_OPENCL_DRIVER");
    if (!explicit_driver || !explicit_driver[0])
        explicit_driver = getenv("NCNN_OPENCL_DRIVER");
    if (explicit_driver && explicit_driver[0])
        append_opencl_candidate(candidates, explicit_driver);

#if defined(_WIN32)
#if defined(_WIN64)
    append_opencl_registry_candidates(candidates, KEY_WOW64_64KEY);
    const char* known[] = {"nvopencl64.dll", "amdocl64.dll", "IntelOpenCL64.dll",
                           "intelocl64.dll", "qcomopencl.dll", "libOpenCL.dll"};
#else
    append_opencl_registry_candidates(candidates, KEY_WOW64_32KEY);
    const char* known[] = {"nvopencl.dll", "amdocl.dll", "IntelOpenCL32.dll",
                           "intelocl32.dll", "qcomopencl.dll", "libOpenCL.dll"};
#endif
#elif defined(__ANDROID__)
    const char* icd_location = getenv("OCL_ICD_VENDORS");
    if (icd_location && icd_location[0])
    {
        struct stat status;
        if (stat(icd_location, &status) == 0 && (status.st_mode & S_IFDIR) != 0)
            append_opencl_icd_directory(candidates, icd_location);
        else
            append_opencl_candidate(candidates, icd_location);
    }
    append_opencl_icd_directory(candidates, "/vendor/etc/OpenCL/vendors");
    append_opencl_icd_directory(candidates, "/system/etc/OpenCL/vendors");
    append_opencl_icd_directory(candidates, "/etc/OpenCL/vendors");
#if defined(__LP64__)
    const char* known[] = {
        "/vendor/lib64/libOpenCL.so", "/vendor/lib64/libOpenCL-pixel.so",
        "/vendor/lib64/libGLES_mali.so", "/vendor/lib64/libmali.so",
        "/vendor/lib64/libPVROCL.so", "/odm/lib64/libOpenCL.so",
        "/system/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/egl/libGLES_mali.so",
        "libOpenCL.so", "libOpenCL-pixel.so", "libGLES_mali.so", "libmali.so", "libPVROCL.so"};
#else
    const char* known[] = {
        "/vendor/lib/libOpenCL.so", "/vendor/lib/libOpenCL-pixel.so",
        "/vendor/lib/egl/libGLES_mali.so", "/vendor/lib/libmali.so",
        "/vendor/lib/libPVROCL.so", "/odm/lib/libOpenCL.so",
        "/system/vendor/lib/libOpenCL.so", "/system/vendor/lib/egl/libGLES_mali.so",
        "libOpenCL.so", "libOpenCL-pixel.so", "libGLES_mali.so", "libmali.so", "libPVROCL.so"};
#endif
#elif defined(__APPLE__)
    const char* known[] = {"/System/Library/Frameworks/OpenCL.framework/OpenCL"};
#else
    const char* icd_location = getenv("OCL_ICD_VENDORS");
    if (icd_location && icd_location[0])
    {
        struct stat status;
        if (stat(icd_location, &status) == 0 && (status.st_mode & S_IFDIR) != 0)
            append_opencl_icd_directory(candidates, icd_location);
        else
            append_opencl_candidate(candidates, icd_location);
    }
    append_opencl_icd_directory(candidates, "/etc/OpenCL/vendors");
    append_opencl_icd_directory(candidates, "/usr/local/etc/OpenCL/vendors");
    const char* known[] = {"libpocl.so.2", "libpocl.so", "libRusticlOpenCL.so.1",
                           "libnvidia-opencl.so.1", "libamdocl64.so", "libigdrcl.so",
                           "libintelocl.so", "libGLES_mali.so"};
#endif
    for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); i++)
        append_opencl_candidate(candidates, known[i]);
    return candidates;
}

static OpenCLLibrary open_opencl_library(const char* path)
{
#if defined(_WIN32)
    return LoadLibraryA(path);
#else
    return dlopen(path, RTLD_LOCAL | RTLD_NOW);
#endif
}

static void close_opencl_library(OpenCLLibrary library)
{
    if (!library)
        return;
#if defined(_WIN32)
    FreeLibrary(library);
#else
    dlclose(library);
#endif
}

struct OpenCLLibraryReleaser
{
    ~OpenCLLibraryReleaser()
    {
        close_opencl_library(g_opencl_library);
        g_opencl_library = 0;
    }
};

static OpenCLLibraryReleaser g_opencl_library_releaser;

static void* opencl_library_symbol(OpenCLLibrary library, const char* name)
{
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(library, name));
#else
    return dlsym(library, name);
#endif
}

static bool enumerate_opencl_platforms(clGetPlatformIDs_fn get_platform_ids,
                                       std::vector<cl_platform_id>& platforms)
{
    cl_uint count = 0;
    if (!get_platform_ids || get_platform_ids(0, 0, &count) != CL_SUCCESS || count == 0)
        return false;
    platforms.resize(count);
    if (get_platform_ids(count, platforms.data(), 0) != CL_SUCCESS)
    {
        platforms.clear();
        return false;
    }
    return true;
}

static bool validate_opencl_functions(const OpenCLApi& functions)
{
#define MOMOTEN_VALIDATE_OPENCL_FUNCTION(name) \
    if (!functions.p_##name) return false;
    MOMOTEN_OPENCL_FUNCTIONS(MOMOTEN_VALIDATE_OPENCL_FUNCTION)
#undef MOMOTEN_VALIDATE_OPENCL_FUNCTION
    return true;
}

static bool has_compatible_opencl_device(const OpenCLApi& functions,
                                         const std::vector<cl_platform_id>& platforms)
{
    for (size_t p = 0; p < platforms.size(); p++)
    {
        cl_uint count = 0;
        cl_int ret = functions.p_clGetDeviceIDs(
            platforms[p], CL_DEVICE_TYPE_ALL, 0, 0, &count);
        if (ret == CL_DEVICE_NOT_FOUND || ret != CL_SUCCESS || count == 0)
            continue;
        std::vector<cl_device_id> devices(count);
        if (functions.p_clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                                       count, devices.data(), 0)
            != CL_SUCCESS)
            continue;
        for (size_t d = 0; d < devices.size(); d++)
        {
            cl_bool available = CL_FALSE;
            cl_bool compiler = CL_FALSE;
            cl_bool little_endian = CL_FALSE;
            cl_uint address_bits = 0;
            cl_uint dimensions = 0;
            if (functions.p_clGetDeviceInfo(devices[d], CL_DEVICE_AVAILABLE,
                                            sizeof(available), &available, 0)
                    == CL_SUCCESS
                && functions.p_clGetDeviceInfo(devices[d], CL_DEVICE_COMPILER_AVAILABLE,
                                               sizeof(compiler), &compiler, 0)
                       == CL_SUCCESS
                && functions.p_clGetDeviceInfo(devices[d], CL_DEVICE_ENDIAN_LITTLE,
                                               sizeof(little_endian), &little_endian, 0)
                       == CL_SUCCESS
                && functions.p_clGetDeviceInfo(devices[d], CL_DEVICE_ADDRESS_BITS,
                                               sizeof(address_bits), &address_bits, 0)
                       == CL_SUCCESS
                && functions.p_clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                               sizeof(dimensions), &dimensions, 0)
                       == CL_SUCCESS
                && available == CL_TRUE && compiler == CL_TRUE && little_endian == CL_TRUE && (address_bits == 32 || address_bits == 64) && dimensions >= 3)
                return true;
        }
    }
    return false;
}

static bool try_opencl_driver(const std::string& path)
{
    OpenCLLibrary library = open_opencl_library(path.c_str());
    if (!library)
        return false;

    OpenCLApi functions = {};
    std::vector<cl_platform_id> platforms;
    clIcdGetPlatformIDsKHR_fn get_icd_platform_ids = reinterpret_cast<clIcdGetPlatformIDsKHR_fn>(
        opencl_library_symbol(library, "clIcdGetPlatformIDsKHR"));
    // PoCL intentionally prefixes its public vendor entry point; the system
    // ICD loader knows this convention, so a loader which bypasses libOpenCL
    // must recognize it as well.
    if (!get_icd_platform_ids)
        get_icd_platform_ids = reinterpret_cast<clIcdGetPlatformIDsKHR_fn>(
            opencl_library_symbol(library, "POclIcdGetPlatformIDsKHR"));
    if (get_icd_platform_ids)
    {
        if (!enumerate_opencl_platforms(get_icd_platform_ids, platforms))
        {
            close_opencl_library(library);
            return false;
        }
        const cl_icd_dispatch* dispatch = 0;
        memcpy(&dispatch, platforms[0], sizeof(dispatch));
        if (!dispatch)
        {
            close_opencl_library(library);
            return false;
        }
#define MOMOTEN_ASSIGN_ICD_FUNCTION(name) functions.p_##name = dispatch->name;
        MOMOTEN_OPENCL_FUNCTIONS(MOMOTEN_ASSIGN_ICD_FUNCTION)
#undef MOMOTEN_ASSIGN_ICD_FUNCTION
        functions.p_clGetExtensionFunctionAddressForPlatform = reinterpret_cast<momo_clGetExtensionFunctionAddressForPlatform_fn>(
            dispatch->clGetExtensionFunctionAddressForPlatform);
        functions.p_clGetKernelSubGroupInfoKHR = dispatch->clGetKernelSubGroupInfoKHR;
    }
    else
    {
        clGetPlatformIDs_fn get_platform_ids = reinterpret_cast<clGetPlatformIDs_fn>(
            opencl_library_symbol(library, "clGetPlatformIDs"));
#define MOMOTEN_ASSIGN_DIRECT_FUNCTION(name) \
    functions.p_##name = reinterpret_cast<name##_fn>(opencl_library_symbol(library, #name));
        MOMOTEN_OPENCL_FUNCTIONS(MOMOTEN_ASSIGN_DIRECT_FUNCTION)
#undef MOMOTEN_ASSIGN_DIRECT_FUNCTION
        if (!enumerate_opencl_platforms(get_platform_ids, platforms))
        {
            close_opencl_library(library);
            return false;
        }
        functions.p_clGetExtensionFunctionAddressForPlatform = reinterpret_cast<momo_clGetExtensionFunctionAddressForPlatform_fn>(
            opencl_library_symbol(library, "clGetExtensionFunctionAddressForPlatform"));
        functions.p_clGetKernelSubGroupInfoKHR = reinterpret_cast<clGetKernelSubGroupInfoKHR_fn>(
            opencl_library_symbol(library, "clGetKernelSubGroupInfoKHR"));
        if (!functions.p_clGetKernelSubGroupInfoKHR && functions.p_clGetExtensionFunctionAddressForPlatform)
        {
            functions.p_clGetKernelSubGroupInfoKHR = reinterpret_cast<clGetKernelSubGroupInfoKHR_fn>(
                functions.p_clGetExtensionFunctionAddressForPlatform(
                    platforms[0], "clGetKernelSubGroupInfoKHR"));
        }
    }

    if (!validate_opencl_functions(functions) || !has_compatible_opencl_device(functions, platforms))
    {
        close_opencl_library(library);
        return false;
    }

    g_opencl = functions;
    g_opencl_platforms.swap(platforms);
    g_opencl_library = library;
    g_opencl_driver_path = path;
    return true;
}

static void load_opencl_driver_once()
{
    const std::vector<std::string> candidates = opencl_driver_candidates();
    for (size_t i = 0; i < candidates.size(); i++)
    {
        if (try_opencl_driver(candidates[i]))
        {
            g_opencl_loaded = true;
            if (debug_enabled())
                fprintf(stderr, "[momoten] loaded OpenCL vendor driver %s without the system ICD loader\n",
                        g_opencl_driver_path.c_str());
            return;
        }
        if (debug_enabled())
            fprintf(stderr, "[momoten] rejected OpenCL driver candidate %s\n", candidates[i].c_str());
    }
    fprintf(stderr, "[momoten] no usable OpenCL vendor driver; set MOMOTEN_OPENCL_DRIVER to a vendor library or .icd file\n");
}

cl_int momo_clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms,
                             cl_uint* num_platforms)
{
    std::call_once(g_opencl_once, load_opencl_driver_once);
    if (!g_opencl_loaded)
        return CL_PLATFORM_NOT_FOUND_KHR;
    if ((!platforms && !num_platforms) || (platforms && num_entries == 0))
        return CL_INVALID_VALUE;
    if (num_platforms)
        *num_platforms = static_cast<cl_uint>(g_opencl_platforms.size());
    if (platforms)
    {
        const size_t count = std::min<size_t>(num_entries, g_opencl_platforms.size());
        for (size_t i = 0; i < count; i++)
            platforms[i] = g_opencl_platforms[i];
    }
    return CL_SUCCESS;
}

} // namespace momoten_detail
