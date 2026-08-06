// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

enum ProcScope
{
    PROC_SCOPE_NONE = -1,
    PROC_SCOPE_GLOBAL,
    PROC_SCOPE_INSTANCE,
    PROC_SCOPE_DEVICE
};

struct ProcExpectation
{
    const char* name;
    const char* implementation;
    ProcScope scope;
    const char* extension;
    uint32_t version;
};

#define MOMOTEN_RESULT_PROC(api, impl, parameters, arguments, scope, extension, version) \
    {#api, #impl, scope, extension, version},
#define MOMOTEN_VOID_PROC MOMOTEN_RESULT_PROC
#define MOMOTEN_PFN_PROC  MOMOTEN_RESULT_PROC
static const ProcExpectation expectations[] = {
#include "../../src/vulkan_procs.inc"
};
#undef MOMOTEN_RESULT_PROC
#undef MOMOTEN_VOID_PROC
#undef MOMOTEN_PFN_PROC

static bool has_extension(const std::vector<std::string>& extensions, const char* name)
{
    if (!name)
        return false;
    for (size_t i = 0; i < extensions.size(); i++)
    {
        if (extensions[i] == name)
            return true;
    }
    return false;
}

static bool expected_instance_proc(const ProcExpectation& proc, uint32_t api_version,
                                   const std::vector<std::string>& instance_extensions,
                                   const std::vector<std::string>& device_extensions)
{
    if (proc.scope == PROC_SCOPE_NONE)
        return false;
    if (proc.scope == PROC_SCOPE_GLOBAL)
        return true;
    if (proc.version && api_version < proc.version)
        return false;
    if (!proc.extension)
        return true;
    if (has_extension(instance_extensions, proc.extension))
        return true;
    return proc.scope == PROC_SCOPE_DEVICE && has_extension(device_extensions, proc.extension);
}

static bool expected_device_proc(const ProcExpectation& proc, uint32_t api_version,
                                 const std::vector<std::string>& enabled_extensions)
{
    return proc.scope == PROC_SCOPE_DEVICE && (!proc.version || api_version >= proc.version) && (!proc.extension || has_extension(enabled_extensions, proc.extension));
}

static void* open_library(const char* path)
{
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA(path));
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void* find_symbol(void* library, const char* name)
{
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(
        reinterpret_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

static void close_library(void* library)
{
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(library));
#else
    dlclose(library);
#endif
}

static int verify_instance_registry(VkInstance instance, uint32_t api_version,
                                    const std::vector<std::string>& instance_extensions,
                                    const std::vector<std::string>& device_extensions)
{
    for (size_t i = 0; i < sizeof(expectations) / sizeof(expectations[0]); i++)
    {
        const bool expected = expected_instance_proc(
            expectations[i], api_version, instance_extensions, device_extensions);
        const bool found = vkGetInstanceProcAddr(instance, expectations[i].name) != 0;
        if (found != expected)
        {
            fprintf(stderr, "proc_registry_test: instance lookup mismatch for %s\n",
                    expectations[i].name);
            return 1;
        }
    }
    return 0;
}

static int verify_device_registry(VkDevice device, uint32_t api_version,
                                  const std::vector<std::string>& enabled_extensions)
{
    for (size_t i = 0; i < sizeof(expectations) / sizeof(expectations[0]); i++)
    {
        const bool expected = expected_device_proc(
            expectations[i], api_version, enabled_extensions);
        const bool found = vkGetDeviceProcAddr(device, expectations[i].name) != 0;
        if (found != expected)
        {
            fprintf(stderr, "proc_registry_test: device lookup mismatch for %s\n",
                    expectations[i].name);
            return 1;
        }
    }
    return 0;
}

static int verify_aliases(VkInstance instance, VkDevice device)
{
    for (size_t i = 0; i < sizeof(expectations) / sizeof(expectations[0]); i++)
    {
        for (size_t j = i + 1; j < sizeof(expectations) / sizeof(expectations[0]); j++)
        {
            if (strcmp(expectations[i].implementation,
                       expectations[j].implementation)
                != 0)
                continue;
            PFN_vkVoidFunction first = expectations[i].scope == PROC_SCOPE_DEVICE
                                           ? vkGetDeviceProcAddr(device, expectations[i].name)
                                           : vkGetInstanceProcAddr(instance, expectations[i].name);
            PFN_vkVoidFunction second = expectations[j].scope == PROC_SCOPE_DEVICE
                                            ? vkGetDeviceProcAddr(device, expectations[j].name)
                                            : vkGetInstanceProcAddr(instance, expectations[j].name);
            if (first && second && first != second)
            {
                fprintf(stderr, "proc_registry_test: alias mismatch for %s and %s\n",
                        expectations[i].name, expectations[j].name);
                return 1;
            }
        }
    }
    return 0;
}

static VkResult create_instance(uint32_t api_version,
                                const std::vector<const char*>& extensions,
                                VkInstance* instance)
{
    VkApplicationInfo application_info = {};
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.apiVersion = api_version;
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &application_info;
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.empty() ? 0 : extensions.data();
    return vkCreateInstance(&create_info, 0, instance);
}

int main(int argc, char** argv)
{
    if (argc != 2)
        return 2;

    void* library = open_library(argv[1]);
    if (!library)
    {
        fprintf(stderr, "proc_registry_test: cannot open driver %s\n", argv[1]);
        return 1;
    }
    for (size_t i = 0; i < sizeof(expectations) / sizeof(expectations[0]); i++)
    {
        if (!find_symbol(library, expectations[i].name))
        {
            fprintf(stderr, "proc_registry_test: missing direct symbol %s\n",
                    expectations[i].name);
            close_library(library);
            return 1;
        }
        const bool global_found = vkGetInstanceProcAddr(VK_NULL_HANDLE, expectations[i].name) != 0;
        if (global_found != (expectations[i].scope == PROC_SCOPE_GLOBAL))
        {
            fprintf(stderr, "proc_registry_test: global lookup mismatch for %s\n",
                    expectations[i].name);
            close_library(library);
            return 1;
        }
    }

    VkInstance instance10 = VK_NULL_HANDLE;
    const std::vector<const char*> no_extensions;
    VkResult result = create_instance(VK_API_VERSION_1_0, no_extensions, &instance10);
    if (result == VK_ERROR_INCOMPATIBLE_DRIVER)
    {
        close_library(library);
        return 77;
    }
    if (result != VK_SUCCESS)
    {
        close_library(library);
        return 1;
    }

    uint32_t physical_count = 0;
    if (vkEnumeratePhysicalDevices(instance10, &physical_count, 0) != VK_SUCCESS || physical_count == 0)
    {
        vkDestroyInstance(instance10, 0);
        close_library(library);
        return 77;
    }
    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    if (vkEnumeratePhysicalDevices(instance10, &physical_count,
                                   physical_devices.data())
        != VK_SUCCESS)
        return 1;

    uint32_t extension_count = 0;
    if (vkEnumerateDeviceExtensionProperties(
            physical_devices[0], 0, &extension_count, 0)
        != VK_SUCCESS)
        return 1;
    std::vector<VkExtensionProperties> properties(extension_count);
    if (vkEnumerateDeviceExtensionProperties(
            physical_devices[0], 0, &extension_count, properties.data())
        != VK_SUCCESS)
        return 1;
    std::vector<std::string> available_device_extensions;
    for (size_t i = 0; i < properties.size(); i++)
        available_device_extensions.push_back(properties[i].extensionName);

    const std::vector<std::string> no_extension_strings;
    if (verify_instance_registry(instance10, VK_API_VERSION_1_0,
                                 no_extension_strings, available_device_extensions))
        return 1;
    vkDestroyInstance(instance10, 0);

    const char* properties2_extension = VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;
    std::vector<const char*> instance_extension_names(1, properties2_extension);
    VkInstance instance11 = VK_NULL_HANDLE;
    result = create_instance(VK_API_VERSION_1_1, instance_extension_names, &instance11);
    if (result != VK_SUCCESS)
        return 1;

    physical_count = 0;
    if (vkEnumeratePhysicalDevices(instance11, &physical_count, 0) != VK_SUCCESS || physical_count == 0)
        return 1;
    physical_devices.resize(physical_count);
    if (vkEnumeratePhysicalDevices(instance11, &physical_count,
                                   physical_devices.data())
        != VK_SUCCESS)
        return 1;
    VkPhysicalDevice physical = physical_devices[0];
    const std::vector<std::string> enabled_instance_extensions(1, properties2_extension);
    if (verify_instance_registry(instance11, VK_API_VERSION_1_1,
                                 enabled_instance_extensions,
                                 available_device_extensions))
        return 1;

    const float priority = 1.f;
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = 0;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;

    VkDevice minimal_device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical, &device_info, 0, &minimal_device) != VK_SUCCESS)
        return 1;
    if (verify_device_registry(minimal_device, VK_API_VERSION_1_1,
                               no_extension_strings))
        return 1;
    vkDestroyDevice(minimal_device, 0);

    std::vector<const char*> enabled_device_extension_names;
    for (size_t i = 0; i < available_device_extensions.size(); i++)
        enabled_device_extension_names.push_back(available_device_extensions[i].c_str());
    device_info.enabledExtensionCount = static_cast<uint32_t>(enabled_device_extension_names.size());
    device_info.ppEnabledExtensionNames = enabled_device_extension_names.empty()
                                              ? 0
                                              : enabled_device_extension_names.data();
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical, &device_info, 0, &device) != VK_SUCCESS)
        return 1;
    if (verify_device_registry(device, VK_API_VERSION_1_1,
                               available_device_extensions)
        || verify_aliases(instance11, device))
        return 1;

    vkDestroyDevice(device, 0);
    vkDestroyInstance(instance11, 0);
    close_library(library);
    return 0;
}
