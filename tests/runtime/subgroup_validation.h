// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_TEST_SUBGROUP_VALIDATION_H
#define MOMOTEN_TEST_SUBGROUP_VALIDATION_H

#include <stdint.h>
#include <vector>

struct SubgroupInvocation
{
    uint32_t size;
    uint32_t lane;
    uint32_t group;
    uint32_t count;
    uint32_t elected;
};

static_assert(sizeof(SubgroupInvocation) == 5 * sizeof(uint32_t), "subgroup output layout");

// Native OpenCL defines the mapping, so validate the complete partition rather
// than assuming consecutive local IDs belong to the same subgroup.
inline const char* validate_subgroup_output(const SubgroupInvocation* output, uint32_t local_size, uint32_t subgroup_size, bool emulated)
{
    const uint32_t count = output[0].count;
    if (!count || count > local_size)
        return "subgroup count out of range";
    std::vector<uint32_t> populations(count, 0);
    std::vector<uint32_t> elections(count, 0);
    std::vector<bool> lanes(count * subgroup_size, false);
    for (uint32_t i = 0; i < local_size; i++)
    {
        const SubgroupInvocation& value = output[i];
        if (value.size != subgroup_size || value.count != count)
            return "inconsistent subgroup size or count";
        if (value.group >= count || value.lane >= subgroup_size || value.elected > 1)
            return "subgroup ID, lane or election out of range";
        const uint32_t index = value.group * subgroup_size + value.lane;
        if (lanes[index])
            return "duplicate lane within a subgroup";
        lanes[index] = true;
        populations[value.group]++;
        elections[value.group] += value.elected;
        // Every invocation participates in the same election. The lowest
        // active subgroup lane must be elected, regardless of its local ID.
        if (value.elected != (value.lane == 0 ? 1u : 0u))
            return "incorrect elected lane";
        if (emulated && (value.lane != i % subgroup_size || value.group != i / subgroup_size || count != (local_size + subgroup_size - 1) / subgroup_size))
            return "incorrect emulated subgroup mapping";
    }
    for (uint32_t group = 0; group < count; group++)
    {
        if (!populations[group] || elections[group] != 1)
            return "subgroup missing invocations or a unique election";
        for (uint32_t lane = 0; lane < populations[group]; lane++)
            if (!lanes[group * subgroup_size + lane])
                return "noncontiguous subgroup lanes";
    }
    return 0;
}

#endif // MOMOTEN_TEST_SUBGROUP_VALIDATION_H
