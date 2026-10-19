// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "spirv-tools/optimizer.hpp"

#include <cstdio>
#include <string>

// Exercise the public caller of the legacy-libc++ decimal parser, and the
// original std::from_chars implementation on builds which need no patch.
int main()
{
    const char* accepted[] = {"0:0", "1:2", "0001:0002", "4294967295:0", "0:4294967295", "4294967295:4294967295"};
    const char* rejected[] = {"", ":", "0", ":0", "0:", "-1:0", "+1:0", " 1:0", "0:-1", "0:+1", "0: 1", "1 :0", "0:1 ", "1x:0", "0:1x", "0:1:2", "4294967296:0", "0:4294967296", "99999999999999999999999999999999:0", "0:99999999999999999999999999999999"};
    for (int valid = 0; valid <= 1; ++valid)
    {
        const char* const* cases = valid ? accepted : rejected;
        const size_t count = valid ? sizeof(accepted) / sizeof(accepted[0]) : sizeof(rejected) / sizeof(rejected[0]);
        for (size_t i = 0; i < count; ++i)
        {
            spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_1);
            const std::string flag = std::string("--switch-descriptorset=") + cases[i];
            if (optimizer.RegisterPassFromFlag(flag.c_str()) != static_cast<bool>(valid))
            {
                std::fprintf(stderr, "Unexpected parsing result for %s\n", flag.c_str());
                return 1;
            }
        }
    }
    return 0;
}
