// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "subgroup_validation.h"

#include <algorithm>
#include <stdio.h>

int main()
{
    std::vector<SubgroupInvocation> values(12);
    for (uint32_t i = 0; i < values.size(); i++)
        values[i] = {8, i % 8, i / 8, 2, i % 8 == 0 ? 1u : 0u};
    if (validate_subgroup_output(values.data(), 12, 8, true))
        return 1;
    std::reverse(values.begin(), values.end());
    if (validate_subgroup_output(values.data(), 12, 8, false) || !validate_subgroup_output(values.data(), 12, 8, true))
    {
        fprintf(stderr, "subgroup_validation_test: native permutation must be accepted only in native mode\n");
        return 1;
    }
    for (uint32_t field = 0; field < 5; field++)
    {
        std::vector<SubgroupInvocation> invalid = values;
        switch (field)
        {
        case 0: invalid[0].size = 4; break;
        case 1: invalid[0].lane = invalid[1].lane; break;
        case 2: invalid[0].group = 2; break;
        case 3: invalid[0].count = 1; break;
        case 4: invalid[0].elected = 1; break;
        }
        if (!validate_subgroup_output(invalid.data(), 12, 8, false))
        {
            fprintf(stderr, "subgroup_validation_test: invalid field %u was accepted\n", field);
            return 1;
        }
    }
    return 0;
}
