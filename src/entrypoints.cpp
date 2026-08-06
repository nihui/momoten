// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "abi_guard.h"
#include "backend/device.h"
#include "backend/objects.h"
#include "vulkan_internal.h"

#include <cstring>

namespace momoten_detail {

enum ProcScope
{
    PROC_SCOPE_NONE = -1,
    PROC_SCOPE_GLOBAL,
    PROC_SCOPE_INSTANCE,
    PROC_SCOPE_DEVICE
};

struct VulkanProc
{
    const char* name;
    PFN_vkVoidFunction function;
    ProcScope scope;
    const char* required_extension;
    uint32_t required_api_version;
};

#define MOMOTEN_RESULT_PROC(api, impl, parameters, arguments, scope, extension, version) \
    {#api, guarded_function<decltype(&impl), &impl>(), scope, extension, version},
#define MOMOTEN_VOID_PROC MOMOTEN_RESULT_PROC
#define MOMOTEN_PFN_PROC  MOMOTEN_RESULT_PROC
static const VulkanProc g_vulkan_procs[] = {
#include "vulkan_procs.inc"
};
#undef MOMOTEN_RESULT_PROC
#undef MOMOTEN_VOID_PROC
#undef MOMOTEN_PFN_PROC

static bool extension_enabled(const std::vector<std::string>& enabled_extensions,
                              const char* extension)
{
    if (!extension)
        return false;
    for (size_t i = 0; i < enabled_extensions.size(); i++)
    {
        if (enabled_extensions[i] == extension)
            return true;
    }
    return false;
}

static bool proc_available(uint32_t api_version,
                           const std::vector<std::string>& enabled_extensions,
                           const VulkanProc& proc)
{
    if (proc.required_api_version && api_version < proc.required_api_version)
        return false;
    if (proc.required_extension && !extension_enabled(enabled_extensions, proc.required_extension))
        return false;
    return true;
}

static const VulkanProc* find_vulkan_proc(const char* name, ProcScope minimum_scope,
                                          ProcScope maximum_scope)
{
    if (!name)
        return 0;
    for (size_t i = 0; i < sizeof(g_vulkan_procs) / sizeof(g_vulkan_procs[0]); i++)
    {
        if (g_vulkan_procs[i].scope >= minimum_scope && g_vulkan_procs[i].scope <= maximum_scope && strcmp(g_vulkan_procs[i].name, name) == 0)
            return &g_vulkan_procs[i];
    }
    return 0;
}

PFN_vkVoidFunction impl_get_device_proc_addr(VkDevice device, const char* name)
{
    if (!device)
        return 0;
    const VulkanProc* proc = find_vulkan_proc(name, PROC_SCOPE_DEVICE, PROC_SCOPE_DEVICE);
    if (!proc || !proc_available(device->api_version, device->enabled_extensions, *proc))
        return 0;
    return proc->function;
}

PFN_vkVoidFunction impl_get_instance_proc_addr(VkInstance instance, const char* name)
{
    const VulkanProc* proc = find_vulkan_proc(name, PROC_SCOPE_GLOBAL,
                                              instance ? PROC_SCOPE_DEVICE : PROC_SCOPE_GLOBAL);
    if (!proc)
        return 0;
    if (proc->scope == PROC_SCOPE_GLOBAL)
        return proc->function;
    if (!proc_available(instance->api_version, instance->enabled_extensions, *proc))
    {
        if (proc->scope != PROC_SCOPE_DEVICE || !proc->required_extension)
            return 0;
        bool supported = false;
        for (size_t i = 0; i < instance->physical_devices.size(); i++)
            supported |= supports_device_extension(instance->physical_devices[i], proc->required_extension);
        if (!supported)
            return 0;
    }
    return proc->function;
}

} // namespace momoten_detail

#define MOMOTEN_RESULT_PROC(api, impl, parameters, arguments, scope, extension, version) \
    extern "C" MOMOTEN_EXPORT VKAPI_ATTR VkResult VKAPI_CALL api parameters noexcept     \
    {                                                                                    \
        return momoten_detail::VulkanAbiGuard<                                           \
            decltype(&momoten_detail::impl), &momoten_detail::impl>::call arguments;     \
    }
#define MOMOTEN_VOID_PROC(api, impl, parameters, arguments, scope, extension, version) \
    extern "C" MOMOTEN_EXPORT VKAPI_ATTR void VKAPI_CALL api parameters noexcept       \
    {                                                                                  \
        momoten_detail::VulkanAbiGuard<                                                \
            decltype(&momoten_detail::impl), &momoten_detail::impl>::call arguments;   \
    }
#define MOMOTEN_PFN_PROC(api, impl, parameters, arguments, scope, extension, version)          \
    extern "C" MOMOTEN_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL api parameters noexcept \
    {                                                                                          \
        return momoten_detail::VulkanAbiGuard<                                                 \
            decltype(&momoten_detail::impl), &momoten_detail::impl>::call arguments;           \
    }
#include "vulkan_procs.inc"
#undef MOMOTEN_RESULT_PROC
#undef MOMOTEN_VOID_PROC
#undef MOMOTEN_PFN_PROC
