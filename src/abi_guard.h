// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_ABI_GUARD_H
#define MOMOTEN_ABI_GUARD_H

#include "vulkan_internal.h"

#include <cstdio>
#include <exception>
#include <new>

namespace momoten_detail {

inline void log_abi_exception(const char* message) noexcept
{
    if (message)
        fprintf(stderr, "[momoten] Vulkan entry point failed: %s\n", message);
    else
        fprintf(stderr, "[momoten] Vulkan entry point failed with an unknown exception\n");
}

template<typename Signature, Signature Function>
struct VulkanAbiGuard;

template<typename... Args, VkResult (*Function)(Args...)>
struct VulkanAbiGuard<VkResult (*)(Args...), Function>
{
    static VKAPI_ATTR VkResult VKAPI_CALL call(Args... args) noexcept
    {
        try
        {
            return Function(args...);
        }
        catch (const std::bad_alloc&)
        {
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        catch (const std::exception& exception)
        {
            log_abi_exception(exception.what());
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        catch (...)
        {
            log_abi_exception(0);
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }
};

template<typename... Args, void (*Function)(Args...)>
struct VulkanAbiGuard<void (*)(Args...), Function>
{
    static VKAPI_ATTR void VKAPI_CALL call(Args... args) noexcept
    {
        try
        {
            Function(args...);
        }
        catch (const std::exception& exception)
        {
            log_abi_exception(exception.what());
        }
        catch (...)
        {
            log_abi_exception(0);
        }
    }
};

template<typename... Args, PFN_vkVoidFunction (*Function)(Args...)>
struct VulkanAbiGuard<PFN_vkVoidFunction (*)(Args...), Function>
{
    static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL call(Args... args) noexcept
    {
        try
        {
            return Function(args...);
        }
        catch (const std::exception& exception)
        {
            log_abi_exception(exception.what());
            return 0;
        }
        catch (...)
        {
            log_abi_exception(0);
            return 0;
        }
    }
};

template<typename Signature, Signature Function>
PFN_vkVoidFunction guarded_function() noexcept
{
    return reinterpret_cast<PFN_vkVoidFunction>(
        &VulkanAbiGuard<Signature, Function>::call);
}

} // namespace momoten_detail

#endif // MOMOTEN_ABI_GUARD_H
