// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "driver_policy.h"

namespace momoten_detail {

OpenCLDriverPolicy opencl_driver_policy(const std::string& vendor, const std::string& driver)
{
    OpenCLDriverPolicy policy;
    if (vendor == "QUALCOMM")
    {
        // Select lowering before compiling the capability probe: a compiler
        // crash in negDp2accFolding cannot be recovered as a build error.
        policy.split_packed_dot_acc_sat = driver == "OpenCL 3.0 QUALCOMM build: 0800.73.1 Compiler E031.47.18.50";
        // This compiler accepts unsigned x signed native saturation, including
        // when the accumulator aliases the original signed input.
        policy.swap_mixed_dot_acc_sat = driver == "OpenCL 3.0 QUALCOMM build: 0676.73 Compiler E031.41.03.62";
    }
    return policy;
}

} // namespace momoten_detail
