// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_MEMORY_STATE_H
#define MOMOTEN_BACKEND_MEMORY_STATE_H

#include "../vulkan_internal.h"

#include <vector>

namespace momoten_detail {

enum MemoryAuthority
{
    MEMORY_AUTHORITY_SYNCHRONIZED,
    MEMORY_AUTHORITY_HOST,
    MEMORY_AUTHORITY_DEVICE
};

struct MemoryRangeState
{
    VkDeviceSize begin;
    VkDeviceSize end;
    MemoryAuthority authority;
};

class MemoryRangeMap
{
public:
    MemoryRangeMap();

    void reset(VkDeviceSize size);
    void set(VkDeviceSize offset, VkDeviceSize size, MemoryAuthority authority);
    void replace(VkDeviceSize offset, VkDeviceSize size,
                 MemoryAuthority from, MemoryAuthority to);
    std::vector<MemoryRangeState> collect(VkDeviceSize offset, VkDeviceSize size,
                                          MemoryAuthority authority) const;

    const std::vector<MemoryRangeState>& states() const;

private:
    void split(VkDeviceSize position);
    void merge();

    VkDeviceSize total_size;
    std::vector<MemoryRangeState> ranges;
};

} // namespace momoten_detail

#endif // MOMOTEN_BACKEND_MEMORY_STATE_H
