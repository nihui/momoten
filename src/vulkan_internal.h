// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_VULKAN_INTERNAL_H
#define MOMOTEN_VULKAN_INTERNAL_H

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES 1
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define MOMOTEN_EXPORT __declspec(dllexport)
#else
#define MOMOTEN_EXPORT __attribute__((visibility("default")))
#endif

namespace momoten_detail {

template<typename Handle, bool IsPointer>
struct HandleValue;

template<typename Handle>
struct HandleValue<Handle, true>
{
    static uint64_t to_id(Handle handle)
    {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
    }

    static Handle from_id(uint64_t id)
    {
        return reinterpret_cast<Handle>(static_cast<uintptr_t>(id));
    }
};

template<typename Handle>
struct HandleValue<Handle, false>
{
    static uint64_t to_id(Handle handle)
    {
        return static_cast<uint64_t>(handle);
    }

    static Handle from_id(uint64_t id)
    {
        return static_cast<Handle>(id);
    }
};

template<typename Handle>
uint64_t handle_to_id(Handle handle)
{
    return HandleValue<Handle, std::is_pointer<Handle>::value>::to_id(handle);
}

template<typename Handle>
Handle id_to_handle(uint64_t id)
{
    return HandleValue<Handle, std::is_pointer<Handle>::value>::from_id(id);
}

template<typename Handle, typename Object>
class HandleTable
{
public:
    HandleTable()
        : next_id(1)
    {
    }

    Handle make_handle(const std::shared_ptr<Object>& object)
    {
        if (!object)
            return VK_NULL_HANDLE;

        uint64_t id = next_id.fetch_add(1, std::memory_order_relaxed);
        if (id == 0)
            id = next_id.fetch_add(1, std::memory_order_relaxed);

        std::lock_guard<std::mutex> lock(mutex);
        objects[id] = object;
        return id_to_handle<Handle>(id);
    }

    std::shared_ptr<Object> get_handle(Handle handle) const
    {
        const uint64_t id = handle_to_id(handle);
        if (id == 0)
            return std::shared_ptr<Object>();

        std::lock_guard<std::mutex> lock(mutex);
        typename std::unordered_map<uint64_t, std::shared_ptr<Object> >::const_iterator it = objects.find(id);
        return it == objects.end() ? std::shared_ptr<Object>() : it->second;
    }

    std::shared_ptr<Object> erase_handle(Handle handle)
    {
        const uint64_t id = handle_to_id(handle);
        if (id == 0)
            return std::shared_ptr<Object>();

        std::lock_guard<std::mutex> lock(mutex);
        typename std::unordered_map<uint64_t, std::shared_ptr<Object> >::iterator it = objects.find(id);
        if (it == objects.end())
            return std::shared_ptr<Object>();

        const std::shared_ptr<Object> object = it->second;
        objects.erase(it);
        return object;
    }

private:
    HandleTable(const HandleTable&);
    HandleTable& operator=(const HandleTable&);

    mutable std::mutex mutex;
    std::unordered_map<uint64_t, std::shared_ptr<Object> > objects;
    std::atomic<uint64_t> next_id;
};

enum CommandBufferState
{
    COMMAND_BUFFER_STATE_INITIAL,
    COMMAND_BUFFER_STATE_RECORDING,
    COMMAND_BUFFER_STATE_EXECUTABLE,
    COMMAND_BUFFER_STATE_PENDING,
    COMMAND_BUFFER_STATE_INVALID
};

#define MOMOTEN_RESULT_PROC(api, impl, parameters, arguments, scope, extension, version) \
    VkResult impl parameters;
#define MOMOTEN_VOID_PROC(api, impl, parameters, arguments, scope, extension, version) \
    void impl parameters;
#define MOMOTEN_PFN_PROC(api, impl, parameters, arguments, scope, extension, version) \
    PFN_vkVoidFunction impl parameters;
#include "vulkan_procs.inc"
#undef MOMOTEN_RESULT_PROC
#undef MOMOTEN_VOID_PROC
#undef MOMOTEN_PFN_PROC

} // namespace momoten_detail

template<typename T>
VkResult enumerate_values(const std::vector<T>& values, uint32_t* count, T* output)
{
    if (!count)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!output)
    {
        *count = static_cast<uint32_t>(values.size());
        return VK_SUCCESS;
    }

    const uint32_t requested = *count;
    const uint32_t written = std::min(requested, static_cast<uint32_t>(values.size()));
    for (uint32_t i = 0; i < written; i++)
        output[i] = values[i];
    *count = written;
    return written < values.size() ? VK_INCOMPLETE : VK_SUCCESS;
}

#endif // MOMOTEN_VULKAN_INTERNAL_H
