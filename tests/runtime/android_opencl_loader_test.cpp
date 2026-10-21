// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

// Exercise the real Android branch on a host, without Android or a GPU. Include
// every host header first so __ANDROID__ only selects the loader implementation.
#include "../../src/backend/opencl_loader.h"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "android_opencl_loader_test:%d: %s\n", __LINE__, #condition); \
    abort(); } } while (0)

namespace mock {

static const char* const driver_path = "/vendor/lib64/libOpenCL.so";
static int driver_token, helper_token, platform_token, device_token;
struct State
{
    bool ordinary = false;
    bool next_symbol = false;
    bool helper_available = true;
    bool helper_symbol = true;
    bool sphal_success = true;
    bool consumed_error = false;
    bool expose_api = false;
    bool compatible = true;
    bool helper_live = false;
    bool driver_live = false;
    bool initialized = false;
    int generation = 0;
    int driver_opens = 0, driver_closes = 0;
    int next_lookups = 0, helper_opens = 0, helper_lookups = 0, helper_closes = 0;
    int sphal_calls[2] = {0, 0};
    int public_calls = 0, icd_calls = 0;
    std::string error, returned_error;
};
static State state;

static char* dlerror()
{
    state.returned_error = state.error;
    state.error.clear();
    return state.returned_error.empty() ? 0 : &state.returned_error[0];
}

static void* dlopen(const char* path, int flags)
{
    CHECK(flags == (RTLD_LOCAL | RTLD_NOW));
    CHECK(state.error.empty());
    if (strcmp(path, "libvndksupport.so") == 0)
    {
        CHECK(state.next_lookups > state.helper_opens);
        state.helper_opens++;
        if (!state.helper_available)
        {
            state.error = "helper unavailable";
            return 0;
        }
        CHECK(!state.helper_live);
        state.helper_live = true;
        return &helper_token;
    }
    CHECK(strcmp(path, driver_path) == 0);
    state.driver_opens++;
    if (!state.ordinary)
    {
        state.error = "driver namespace denied";
        return 0;
    }
    state.driver_live = true;
    return &driver_token;
}

static void* load_sphal(const char* path, int flags, int generation)
{
    CHECK(strcmp(path, driver_path) == 0);
    CHECK(flags == (RTLD_LOCAL | RTLD_NOW));
    CHECK(state.error.empty());
    CHECK(state.next_symbol || state.helper_live);
    CHECK(generation == state.generation); // A pointer from a closed helper is stale.
    state.sphal_calls[generation]++;
    if (!state.sphal_success)
    {
        state.error = "SP-HAL dependency missing";
        if (state.consumed_error)
            dlerror(); // Some libvndksupport implementations consume linker errors.
        return 0;
    }
    state.driver_live = true;
    return &driver_token;
}
static void* load_sphal_first(const char* path, int flags) { return load_sphal(path, flags, 0); }
static void* load_sphal_second(const char* path, int flags) { return load_sphal(path, flags, 1); }

static cl_int CL_API_CALL public_platforms(cl_uint count, cl_platform_id* ids, cl_uint* size)
{
    state.public_calls++;
    state.initialized = true;
    if (size) *size = 1;
    if (ids)
    {
        CHECK(count == 1);
        ids[0] = reinterpret_cast<cl_platform_id>(&platform_token);
    }
    return CL_SUCCESS;
}
static cl_int CL_API_CALL icd_platforms(cl_uint, cl_platform_id*, cl_uint*)
{
    state.icd_calls++;
    return CL_INVALID_OPERATION; // The vendor ICD export cannot initialize itself.
}
static cl_int CL_API_CALL devices(cl_platform_id, cl_device_type, cl_uint count,
                                  cl_device_id* ids, cl_uint* size)
{
    CHECK(state.initialized);
    if (!state.compatible) return CL_DEVICE_NOT_FOUND;
    if (size) *size = 1;
    if (ids)
    {
        CHECK(count == 1);
        ids[0] = reinterpret_cast<cl_device_id>(&device_token);
    }
    return CL_SUCCESS;
}
static cl_int CL_API_CALL device_info(cl_device_id, cl_device_info name,
                                      size_t size, void* value, size_t*)
{
    CHECK(state.initialized && size == sizeof(cl_uint));
    const cl_uint result = name == CL_DEVICE_ADDRESS_BITS ? 64
        : name == CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS ? 3 : CL_TRUE;
    memcpy(value, &result, sizeof(result));
    return CL_SUCCESS;
}
static void unused_entry_point() { CHECK(false); }

static void* dlsym(void* library, const char* name)
{
    if (library == RTLD_NEXT || library == &helper_token)
    {
        CHECK(strcmp(name, "android_load_sphal_library") == 0);
        CHECK(state.error.empty());
        bool found;
        if (library == RTLD_NEXT)
        {
            CHECK(state.driver_opens > state.next_lookups);
            state.next_lookups++;
            found = state.next_symbol;
        }
        else
        {
            CHECK(state.helper_live);
            state.helper_lookups++;
            found = state.helper_symbol;
        }
        if (found)
            return reinterpret_cast<void*>(state.generation == 0
                ? load_sphal_first : load_sphal_second);
        state.error = "helper symbol missing";
        return 0;
    }
    CHECK(library == &driver_token && state.driver_live);
    if (!state.expose_api) return 0;
    if (strcmp(name, "clGetPlatformIDs") == 0) return reinterpret_cast<void*>(public_platforms);
    if (strcmp(name, "clIcdGetPlatformIDsKHR") == 0) return reinterpret_cast<void*>(icd_platforms);
    if (strcmp(name, "clGetDeviceIDs") == 0) return reinterpret_cast<void*>(devices);
    if (strcmp(name, "clGetDeviceInfo") == 0) return reinterpret_cast<void*>(device_info);
#define MOCK_REQUIRED(name) if (strcmp(#name, name_string) == 0) return reinterpret_cast<void*>(unused_entry_point);
    const char* name_string = name;
    MOMOTEN_OPENCL_FUNCTIONS(MOCK_REQUIRED)
#undef MOCK_REQUIRED
    return 0;
}

static int dlclose(void* library)
{
    if (library == &helper_token)
    {
        CHECK(state.helper_live && state.error.empty());
        state.helper_live = false;
        state.helper_closes++;
    }
    else
    {
        CHECK(library == &driver_token && state.driver_live);
        state.driver_live = false;
        state.driver_closes++;
    }
    return 0;
}

static void reset()
{
    CHECK(!state.helper_live && !state.driver_live);
    state = State();
}

} // namespace mock

#define __ANDROID__ 1
#define dlopen mock::dlopen
#define dlsym mock::dlsym
#define dlclose mock::dlclose
#define dlerror mock::dlerror
#include "../../src/backend/loader.cpp"
#undef dlerror
#undef dlclose
#undef dlsym
#undef dlopen
#undef __ANDROID__

// Capture diagnostics with portable POSIX interfaces, including on macOS.
static std::string failed_load_log(bool consumed_error)
{
    mock::reset();
    mock::state.sphal_success = false;
    mock::state.consumed_error = consumed_error;
    FILE* log = tmpfile();
    CHECK(log);
    fflush(stderr);
    const int saved = dup(fileno(stderr));
    CHECK(saved >= 0 && dup2(fileno(log), fileno(stderr)) >= 0);
    CHECK(setenv("MOMOTEN_DEBUG", "1", 1) == 0);
    CHECK(!momoten_detail::open_opencl_library(mock::driver_path));
    CHECK(setenv("MOMOTEN_DEBUG", "0", 1) == 0);
    fflush(stderr);
    CHECK(dup2(saved, fileno(stderr)) >= 0);
    close(saved);
    rewind(log);
    std::string text;
    char buffer[256];
    for (size_t n; (n = fread(buffer, 1, sizeof(buffer), log)) != 0;)
        text.append(buffer, n);
    fclose(log);
    CHECK(mock::state.helper_closes == 1 && mock::state.driver_closes == 0);
    CHECK(mock::state.error.empty());
    return text;
}

int main()
{
    using namespace momoten_detail;
    using mock::state;
    CHECK(setenv("MOMOTEN_DEBUG", "0", 1) == 0);

    mock::reset();
    state.ordinary = true;
    state.error = "stale error from an earlier unrelated operation";
    void* driver = open_opencl_library(mock::driver_path);
    CHECK(driver && state.driver_opens == 1);
    CHECK(state.next_lookups == 0 && state.helper_opens == 0);
    close_opencl_library(driver);
    CHECK(state.driver_closes == 1);

    mock::reset();
    state.next_symbol = true;
    driver = open_opencl_library(mock::driver_path);
    CHECK(driver && state.next_lookups == 1 && state.sphal_calls[0] == 1);
    CHECK(state.helper_opens == 0 && state.helper_closes == 0);
    close_opencl_library(driver);
    CHECK(state.driver_closes == 1);

    mock::reset();
    for (int attempt = 0; attempt < 2; attempt++)
    {
        state.generation = attempt;
        driver = open_opencl_library(mock::driver_path);
        CHECK(driver && !state.helper_live && state.driver_live);
        CHECK(state.helper_opens == attempt + 1 && state.helper_lookups == attempt + 1);
        CHECK(state.helper_closes == attempt + 1 && state.sphal_calls[attempt] == 1);
        close_opencl_library(driver);
    }
    CHECK(state.next_lookups == 2 && state.driver_closes == 2);

    mock::reset();
    state.helper_available = false;
    CHECK(!open_opencl_library(mock::driver_path));
    CHECK(state.helper_opens == 1 && state.helper_lookups == 0 && state.helper_closes == 0);
    CHECK(state.sphal_calls[0] == 0 && state.error.empty());

    mock::reset();
    state.helper_symbol = false;
    CHECK(!open_opencl_library(mock::driver_path));
    CHECK(state.helper_lookups == 1 && state.helper_closes == 1);
    CHECK(state.sphal_calls[0] == 0 && state.error.empty());

    mock::reset();
    state.next_symbol = true;
    state.sphal_success = false;
    CHECK(!open_opencl_library(mock::driver_path));
    CHECK(state.sphal_calls[0] == 1 && state.helper_opens == 0 && state.helper_closes == 0);

    const std::string dependency_log = failed_load_log(false);
    CHECK(dependency_log.find("driver namespace denied") != std::string::npos);
    CHECK(dependency_log.find("android_load_sphal_library") != std::string::npos);
    CHECK(dependency_log.find("SP-HAL dependency missing") != std::string::npos);
    const std::string consumed_log = failed_load_log(true);
    CHECK(consumed_log.find("no dlerror; check the Android linker log") != std::string::npos);
    CHECK(consumed_log.find("SP-HAL dependency missing") == std::string::npos);

    // SP-HAL only changes opening the library, not public-first enumeration or
    // normal ownership of accepted/rejected vendor drivers.
    mock::reset();
    state.expose_api = true;
    CHECK(try_opencl_driver(mock::driver_path));
    CHECK(state.public_calls == 2 && state.icd_calls == 0);
    CHECK(state.helper_closes == 1 && state.driver_closes == 0 && state.driver_live);
    CHECK(g_opencl_library == &mock::driver_token && g_opencl_platforms.size() == 1);
    close_opencl_library(g_opencl_library);
    g_opencl_library = 0;
    g_opencl_platforms.clear();
    CHECK(state.driver_closes == 1);

    mock::reset();
    state.expose_api = true;
    state.compatible = false;
    CHECK(!try_opencl_driver(mock::driver_path));
    CHECK(state.public_calls == 2 && state.icd_calls == 0);
    CHECK(state.helper_closes == 1 && state.driver_closes == 1);
    mock::reset();
    return 0;
}
