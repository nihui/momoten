// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "driver_policy.h"

namespace momoten_detail {

OpenCLDriverPolicy opencl_driver_policy(const std::string& vendor, const std::string& driver)
{
    OpenCLDriverPolicy policy;
    if (vendor == "QUALCOMM")
    {
        // Compiler revisions are shared across different driver builds.
        const std::string compiler_prefix = " Compiler ";
        const size_t compiler_pos = driver.find(compiler_prefix);
        std::string compiler;
        if (compiler_pos != std::string::npos)
        {
            const size_t version_pos = compiler_pos + compiler_prefix.size();
            compiler = driver.substr(version_pos, driver.find_first_of(" \t\r\n", version_pos) - version_pos);
        }
        // Select lowering before compiling the capability probe: a compiler
        // crash in negDp2accFolding cannot be recovered as a build error.
        policy.split_packed_dot_acc_sat = compiler == "E031.47.18.50" || compiler == "E031.47.18.51";
        // This compiler accepts unsigned x signed native saturation, including
        // when the accumulator aliases the original signed input.
        policy.swap_mixed_dot_acc_sat = compiler == "E031.41.03.62";
    }
    else if (vendor == "Imagination Technologies")
    {
        // Copy events remain running after clFlush until a blocking wait.
        // A kernel copy allows submissions and fences to progress asynchronously.
        policy.use_buffer_copy_kernel = driver == "1.10@5130912" || driver == "1.11@5425693";
    }
    return policy;
}

} // namespace momoten_detail
