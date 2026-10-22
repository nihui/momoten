// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "integer_dot_product.h"

#include <stdexcept>

namespace momoten {

const char* integer_dot_product_extension(IntegerDotProductBackend backend)
{
    switch (backend)
    {
    case IntegerDotProductKhr: return "cl_khr_integer_dot_product";
    case IntegerDotProductArm: return "cl_arm_integer_dot_product_int8";
    case IntegerDotProductArmAccumulate: return "cl_arm_integer_dot_product_accumulate_int8";
    case IntegerDotProductArmAccumulateSaturate: return "cl_arm_integer_dot_product_accumulate_saturate_int8";
    case IntegerDotProductQcom: return "cl_qcom_dot_product8";
    default: throw std::runtime_error("invalid integer dot-product backend");
    }
}

std::string integer_dot_product_helpers(IntegerDotProductBackend backend)
{
    // Validate the enum even when no helper is needed.
    integer_dot_product_extension(backend);
    if (backend == IntegerDotProductKhr)
        return std::string();

    std::string source;
    if (backend == IntegerDotProductQcom)
    {
        // Qualcomm's signed builtin is signed x unsigned, and BOTH builtins
        // use signed 32-bit saturation. Zero accumulation is exact for 4x8.
        source += "int momo_dot_su(uint a, uint b) { return qcom_dot8_acc(a, b, 0); }\n"
                  "uint momo_dot_uu(uint a, uint b) { return (uint)qcom_udot8_acc(a, b, 0); }\n"
                  "int momo_dot_ss(uint a, uint b) { return momo_dot_su(a, b ^ 0x80808080u) - momo_dot_su(a, 0x80808080u); }\n";
    }
    else
    {
        const char* function = backend == IntegerDotProductArm ? "arm_dot" : backend == IntegerDotProductArmAccumulate ? "arm_dot_acc"
                                                                                                                       : "arm_dot_acc_sat";
        const char* signed_accumulator = backend == IntegerDotProductArm ? "" : ", 0";
        const char* unsigned_accumulator = backend == IntegerDotProductArm ? "" : ", 0u";
        source += "int momo_dot_ss(uint a, uint b) { return " + std::string(function) + "(as_char4(a), as_char4(b)" + signed_accumulator + "); }\n";
        source += "uint momo_dot_uu(uint a, uint b) { return " + std::string(function) + "(as_uchar4(a), as_uchar4(b)" + unsigned_accumulator + "); }\n";
        source += "int momo_dot_su(uint a, uint b) { return momo_dot_ss(a, b ^ 0x80808080u) - momo_dot_ss(a, 0x80808080u); }\n";
    }

    // Corrections must precede saturation: saturating an intermediate dot
    // plus accumulator can destroy cancellation at INT_MIN and INT_MAX.
    if (backend == IntegerDotProductArmAccumulateSaturate)
        source += "int momo_dot_acc_sat_ss(uint a, uint b, int c) { return arm_dot_acc_sat(as_char4(a), as_char4(b), c); }\n"
                  "uint momo_dot_acc_sat_uu(uint a, uint b, uint c) { return arm_dot_acc_sat(as_uchar4(a), as_uchar4(b), c); }\n";
    else
        source += "int momo_dot_acc_sat_ss(uint a, uint b, int c) { return add_sat(momo_dot_ss(a, b), c); }\n"
                  "uint momo_dot_acc_sat_uu(uint a, uint b, uint c) { return add_sat(momo_dot_uu(a, b), c); }\n";
    if (backend == IntegerDotProductQcom)
        source += "int momo_dot_acc_sat_su(uint a, uint b, int c) { return qcom_dot8_acc(a, b, c); }\n";
    else
        source += "int momo_dot_acc_sat_su(uint a, uint b, int c) { return add_sat(momo_dot_su(a, b), c); }\n";
    return source;
}

} // namespace momoten
