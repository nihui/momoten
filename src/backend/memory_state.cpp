// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "memory_state.h"

#include <algorithm>

namespace momoten_detail {

MemoryRangeMap::MemoryRangeMap()
    : total_size(0)
{
}

void MemoryRangeMap::reset(VkDeviceSize size)
{
    total_size = size;
    ranges.clear();
    if (size != 0)
    {
        MemoryRangeState state;
        state.begin = 0;
        state.end = size;
        state.authority = MEMORY_AUTHORITY_SYNCHRONIZED;
        ranges.push_back(state);
    }
}

void MemoryRangeMap::split(VkDeviceSize position)
{
    if (position == 0 || position >= total_size)
        return;

    for (size_t i = 0; i < ranges.size(); i++)
    {
        if (position <= ranges[i].begin || position >= ranges[i].end)
            continue;

        MemoryRangeState right = ranges[i];
        right.begin = position;
        ranges[i].end = position;
        ranges.insert(ranges.begin() + i + 1, right);
        return;
    }
}

void MemoryRangeMap::merge()
{
    for (size_t i = 1; i < ranges.size();)
    {
        if (ranges[i - 1].end == ranges[i].begin
            && ranges[i - 1].authority == ranges[i].authority)
        {
            ranges[i - 1].end = ranges[i].end;
            ranges.erase(ranges.begin() + i);
        }
        else
        {
            i++;
        }
    }
}

void MemoryRangeMap::set(VkDeviceSize offset, VkDeviceSize size,
                         MemoryAuthority authority)
{
    if (size == 0 || offset >= total_size)
        return;
    const VkDeviceSize end = size > total_size - offset
                                 ? total_size
                                 : offset + size;
    split(offset);
    split(end);
    for (size_t i = 0; i < ranges.size(); i++)
    {
        if (ranges[i].begin >= offset && ranges[i].end <= end)
            ranges[i].authority = authority;
    }
    merge();
}

void MemoryRangeMap::replace(VkDeviceSize offset, VkDeviceSize size,
                             MemoryAuthority from, MemoryAuthority to)
{
    if (size == 0 || offset >= total_size || from == to)
        return;
    const VkDeviceSize end = size > total_size - offset
                                 ? total_size
                                 : offset + size;
    split(offset);
    split(end);
    for (size_t i = 0; i < ranges.size(); i++)
    {
        if (ranges[i].begin >= offset && ranges[i].end <= end
            && ranges[i].authority == from)
            ranges[i].authority = to;
    }
    merge();
}

std::vector<MemoryRangeState> MemoryRangeMap::collect(
    VkDeviceSize offset, VkDeviceSize size, MemoryAuthority authority) const
{
    std::vector<MemoryRangeState> result;
    if (size == 0 || offset >= total_size)
        return result;
    const VkDeviceSize end = size > total_size - offset
                                 ? total_size
                                 : offset + size;
    for (size_t i = 0; i < ranges.size(); i++)
    {
        if (ranges[i].authority != authority
            || ranges[i].end <= offset || ranges[i].begin >= end)
            continue;
        MemoryRangeState state = ranges[i];
        state.begin = std::max(state.begin, offset);
        state.end = std::min(state.end, end);
        result.push_back(state);
    }
    return result;
}

const std::vector<MemoryRangeState>& MemoryRangeMap::states() const
{
    return ranges;
}

} // namespace momoten_detail
