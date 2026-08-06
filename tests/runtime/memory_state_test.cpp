// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "../../src/backend/memory_state.h"

#include <cstdio>

static int expect_range(const momoten_detail::MemoryRangeState& range,
                        VkDeviceSize begin, VkDeviceSize end,
                        momoten_detail::MemoryAuthority authority)
{
    if (range.begin == begin && range.end == end
        && range.authority == authority)
        return 0;
    fprintf(stderr,
            "memory_state_test: range [%llu,%llu) authority=%d, expected [%llu,%llu) authority=%d\n",
            static_cast<unsigned long long>(range.begin),
            static_cast<unsigned long long>(range.end),
            static_cast<int>(range.authority),
            static_cast<unsigned long long>(begin),
            static_cast<unsigned long long>(end),
            static_cast<int>(authority));
    return 1;
}

int main()
{
    using namespace momoten_detail;

    MemoryRangeMap map;
    map.reset(256);
    if (map.states().size() != 1
        || expect_range(map.states()[0], 0, 256,
                        MEMORY_AUTHORITY_SYNCHRONIZED))
        return 1;

    map.set(64, 64, MEMORY_AUTHORITY_HOST);
    map.set(80, 32, MEMORY_AUTHORITY_DEVICE);
    if (map.states().size() != 5
        || expect_range(map.states()[0], 0, 64,
                        MEMORY_AUTHORITY_SYNCHRONIZED)
        || expect_range(map.states()[1], 64, 80, MEMORY_AUTHORITY_HOST)
        || expect_range(map.states()[2], 80, 112, MEMORY_AUTHORITY_DEVICE)
        || expect_range(map.states()[3], 112, 128, MEMORY_AUTHORITY_HOST)
        || expect_range(map.states()[4], 128, 256,
                        MEMORY_AUTHORITY_SYNCHRONIZED))
        return 1;

    const std::vector<MemoryRangeState> host =
        map.collect(72, 64, MEMORY_AUTHORITY_HOST);
    if (host.size() != 2
        || expect_range(host[0], 72, 80, MEMORY_AUTHORITY_HOST)
        || expect_range(host[1], 112, 128, MEMORY_AUTHORITY_HOST))
        return 1;

    map.replace(32, 128, MEMORY_AUTHORITY_SYNCHRONIZED,
                MEMORY_AUTHORITY_HOST);
    if (map.states().size() != 5
        || expect_range(map.states()[0], 0, 32,
                        MEMORY_AUTHORITY_SYNCHRONIZED)
        || expect_range(map.states()[1], 32, 80, MEMORY_AUTHORITY_HOST)
        || expect_range(map.states()[2], 80, 112, MEMORY_AUTHORITY_DEVICE)
        || expect_range(map.states()[3], 112, 160, MEMORY_AUTHORITY_HOST)
        || expect_range(map.states()[4], 160, 256,
                        MEMORY_AUTHORITY_SYNCHRONIZED))
        return 1;

    map.set(32, 128, MEMORY_AUTHORITY_SYNCHRONIZED);
    if (map.states().size() != 1
        || expect_range(map.states()[0], 0, 256,
                        MEMORY_AUTHORITY_SYNCHRONIZED))
        return 1;

    const std::vector<MemoryRangeState> tail =
        map.collect(250, VK_WHOLE_SIZE, MEMORY_AUTHORITY_SYNCHRONIZED);
    if (tail.size() != 1
        || expect_range(tail[0], 250, 256,
                        MEMORY_AUTHORITY_SYNCHRONIZED))
        return 1;

    return 0;
}
