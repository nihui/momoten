// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_SPV_TO_CLC_H
#define MOMOTEN_SPV_TO_CLC_H

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

namespace momoten {

enum BufferAccess
{
    BufferAccessReadWrite = 0,
    BufferAccessReadOnly = 1,
    BufferAccessWriteOnly = 2
};

enum SubgroupMode
{
    // A portable Vulkan subgroup containing exactly one invocation. This is
    // the OpenCL C 1.0 fallback and provides the Vulkan 1.1 BASIC contract.
    SubgroupModeSingleton = 0,
    // Map Vulkan subgroup built-ins and BASIC operations to cl_khr_subgroups.
    // The driver selects this only after probing and validating the device.
    SubgroupModeNative = 1,
    // Emulate the Vulkan BASIC contract from the linear local invocation ID
    // and a trusted, fixed hardware execution width. This path requires no
    // OpenCL subgroup extension and deliberately does not expose vote,
    // arithmetic, ballot or shuffle operations.
    SubgroupModeEmulatedBasic = 2
};

enum WorkgroupMode
{
    // Preserve the logical Vulkan local size as a fixed OpenCL workgroup.
    WorkgroupModeDirect = 0,
    // Flatten independent invocations and reconstruct their Vulkan coordinates.
    // The driver may split one logical workgroup across physical workgroups.
    WorkgroupModeVirtual = 1
};

struct SpecializationValue
{
    uint32_t constant_id;
    std::vector<unsigned char> data;
};

struct BufferArgument
{
    uint32_t variable_id;
    uint32_t descriptor_set;
    uint32_t binding;
    uint32_t buffer_arg_index;
    uint32_t offset_arg_index;
    uint32_t size_arg_index;
    BufferAccess access;
    std::string name;
};

enum FloatingPointWidth
{
    FloatingPointWidth16 = 1,
    FloatingPointWidth32 = 2,
    FloatingPointWidth64 = 4
};

struct FloatingPointControls
{
    // Intersection of SPIR-V FPFastMathMode permissions across the selected
    // entry point's reachable floating-point operations.
    uint32_t fast_math_flags;
    uint32_t widths;
    // Bitmasks of FloatingPointWidth values requiring native preservation
    // of denormals or round-to-nearest-even arithmetic.
    uint32_t denorm_preserve_widths;
    uint32_t round_to_nearest_widths;
    uint32_t signed_zero_inf_nan_preserve_widths;
    // Fast math permissions do not relax the accuracy of math builtins.
    bool requires_builtin_accuracy;

    FloatingPointControls();
};

struct KernelABI
{
    std::string entry_point;
    uint32_t local_size[3];
    uint32_t address_bits;
    std::vector<BufferArgument> buffers;
    std::vector<std::string> required_extensions;
    SubgroupMode subgroup_mode;
    uint32_t subgroup_size;
    bool int64;
    bool fp64;
    bool integer_dot_product;
    bool float_controls2;
    FloatingPointControls floating_point;
    // True only when the logical Vulkan workgroup has no workgroup storage
    // or synchronization barrier and may therefore be split across independent
    // physical OpenCL workgroups by the driver.
    bool workgroup_splittable;
    WorkgroupMode workgroup_mode;
    // One by-value struct argument containing tightly packed 32-bit scalars.
    // Its complete byte size counts against CL_DEVICE_MAX_PARAMETER_SIZE.
    int push_constant_arg_index;
    size_t push_constant_size;

    KernelABI();
};

struct TranslationOptions
{
    std::string entry_point;
    uint32_t address_bits;
    uint32_t local_size[3];
    bool override_local_size;
    // Direct workgroups are preferred. Virtual workgroups require a shader
    // without workgroup storage or synchronization barriers.
    WorkgroupMode workgroup_mode;
    SubgroupMode subgroup_mode;
    uint32_t subgroup_size;
    // The driver enables these only after querying cl_khr_integer_dot_product
    // capabilities and compiling the corresponding OpenCL C builtins.
    bool integer_dot_product;
    bool integer_dot_product_input_4x8bit;
    bool integer_dot_product_input_4x8bit_packed;
    std::vector<SpecializationValue> specializations;

    TranslationOptions();
};

struct TranslationResult
{
    std::string source;
    KernelABI abi;
    std::string diagnostics;
};

// Translate one Vulkan GLCompute entry point to conservative OpenCL C 1.0.
// Returns false for invalid or currently unsupported SPIR-V and places a
// human-readable reason in result.diagnostics.
bool translate_spirv_to_opencl_c(const uint32_t* words, size_t word_count,
                                 const TranslationOptions& options,
                                 TranslationResult& result);

} // namespace momoten

#endif // MOMOTEN_SPV_TO_CLC_H
