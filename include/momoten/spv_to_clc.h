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

struct KernelABI
{
    std::string entry_point;
    uint32_t local_size[3];
    uint32_t address_bits;
    std::vector<BufferArgument> buffers;
    std::vector<std::string> required_extensions;
    SubgroupMode subgroup_mode;
    uint32_t subgroup_size;
    bool fp64;
    bool integer_dot_product;
    // True only when the logical Vulkan workgroup has no workgroup storage
    // or synchronization barrier and may therefore be split across independent
    // physical OpenCL workgroups by the driver.
    bool workgroup_splittable;
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
