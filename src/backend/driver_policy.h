// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_DRIVER_POLICY_H
#define MOMOTEN_BACKEND_DRIVER_POLICY_H

#include <string>

namespace momoten_detail {

struct OpenCLDriverPolicy
{
    bool split_packed_dot_acc_sat = false;
    bool swap_mixed_dot_acc_sat = false;
    bool use_buffer_copy_kernel = false;
};

OpenCLDriverPolicy opencl_driver_policy(const std::string& vendor, const std::string& driver);

} // namespace momoten_detail

#endif // MOMOTEN_BACKEND_DRIVER_POLICY_H
