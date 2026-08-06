// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <cstring>
#include <sstream>
#include <stdexcept>

namespace momoten {

KernelABI::KernelABI()
    : address_bits(0), subgroup_mode(SubgroupModeSingleton), subgroup_size(1), integer_dot_product(false), push_constant_arg_index(-1), push_constant_size(0)
{
    local_size[0] = 1;
    local_size[1] = 1;
    local_size[2] = 1;
}

TranslationOptions::TranslationOptions()
    : entry_point("main"), address_bits(32), override_local_size(false), subgroup_mode(SubgroupModeSingleton), subgroup_size(1), integer_dot_product(false), integer_dot_product_input_4x8bit(false), integer_dot_product_input_4x8bit_packed(false)
{
    local_size[0] = 1;
    local_size[1] = 1;
    local_size[2] = 1;
}

CompilerOpenCL::CompilerOpenCL(const uint32_t* words, size_t word_count, const TranslationOptions& options_,
                               bool requires_global_int32_atomics_)
    : CompilerGLSL(words, word_count), translation_options(options_), requires_global_int32_atomics(requires_global_int32_atomics_), requires_fp16(false), requires_integer_dot_product(false), uses_workgroup_storage(false), push_constant_variable_id(0)
{
}

void CompilerOpenCL::prepare(KernelABI& abi)
{
    if (translation_options.address_bits != 32 && translation_options.address_bits != 64)
        throw std::runtime_error("OpenCL address bits must be 32 or 64");
    if (translation_options.subgroup_mode != SubgroupModeSingleton && translation_options.subgroup_mode != SubgroupModeNative)
        throw std::runtime_error("invalid Vulkan subgroup translation mode");
    if (translation_options.subgroup_size == 0 || (translation_options.subgroup_size & (translation_options.subgroup_size - 1)) != 0)
        throw std::runtime_error("Vulkan subgroup size must be a nonzero power of two");
    if (translation_options.subgroup_mode == SubgroupModeSingleton && translation_options.subgroup_size != 1)
        throw std::runtime_error("singleton Vulkan subgroup size must be one");

    set_entry_point(translation_options.entry_point, ExecutionModelGLCompute);
    validate_capabilities();
    apply_specializations();

    if (translation_options.override_local_size)
        override_workgroup_size();

    localize_workgroup_variables();

    reflect_resources(abi);
    if (requires_global_int32_atomics)
        abi.required_extensions.push_back("cl_khr_global_int32_base_atomics");
    if (requires_fp16)
        abi.required_extensions.push_back("cl_khr_fp16");
    if (requires_integer_dot_product)
        abi.required_extensions.push_back("cl_khr_integer_dot_product");
    if (translation_options.subgroup_mode == SubgroupModeNative)
    {
        abi.required_extensions.push_back("cl_khr_subgroups");
        abi.required_extensions.push_back("cl_khr_subgroup_non_uniform_vote");
    }

    rename_entry_point(translation_options.entry_point, "momo_main", ExecutionModelGLCompute);

    abi.entry_point = "momo_main";
    abi.address_bits = translation_options.address_bits;
    abi.subgroup_mode = translation_options.subgroup_mode;
    abi.subgroup_size = translation_options.subgroup_size;
    abi.integer_dot_product = requires_integer_dot_product;
    abi.local_size[0] = get_execution_mode_argument(ExecutionModeLocalSize, 0);
    abi.local_size[1] = get_execution_mode_argument(ExecutionModeLocalSize, 1);
    abi.local_size[2] = get_execution_mode_argument(ExecutionModeLocalSize, 2);

    if (abi.local_size[0] == 0 || abi.local_size[1] == 0 || abi.local_size[2] == 0)
        throw std::runtime_error("compute local size must be fully resolved before OpenCL source emission");

    local_size[0] = abi.local_size[0];
    local_size[1] = abi.local_size[1];
    local_size[2] = abi.local_size[2];
}

bool CompilerOpenCL::needs_converged_returns() const
{
    return uses_workgroup_storage;
}

void CompilerOpenCL::validate_capabilities()
{
    const SmallVector<Capability>& capabilities = get_declared_capabilities();
    for (size_t i = 0; i < capabilities.size(); i++)
    {
        const Capability capability = capabilities[i];
        if (capability == CapabilityShader || capability == CapabilityGroupNonUniform || capability == CapabilityInt16 || capability == CapabilityStorageBuffer16BitAccess || capability == CapabilityUniformAndStorageBuffer16BitAccess)
            continue;
        if (capability == CapabilityDotProduct)
        {
            requires_integer_dot_product = true;
            continue;
        }
        if (capability == CapabilityDotProductInputAll)
        {
            requires_integer_dot_product = true;
            continue;
        }
        if (capability == CapabilityDotProductInput4x8Bit)
        {
            if (!translation_options.integer_dot_product_input_4x8bit)
                throw std::runtime_error("SPIR-V 4x8-bit vector dot product is unavailable in the OpenCL device profile");
            requires_integer_dot_product = true;
            continue;
        }
        if (capability == CapabilityDotProductInput4x8BitPacked)
        {
            if (!translation_options.integer_dot_product_input_4x8bit_packed)
                throw std::runtime_error("SPIR-V packed 4x8-bit dot product is unavailable in the OpenCL device profile");
            requires_integer_dot_product = true;
            continue;
        }
        if (capability == CapabilityFloat16)
        {
            requires_fp16 = true;
            continue;
        }

        std::ostringstream message;
        message << "SPIR-V capability " << capability_name(capability)
                << " (" << static_cast<unsigned>(capability)
                << ") is not supported by the initial OpenCL C 1.0 profile";
        throw std::runtime_error(message.str());
    }

    if (requires_integer_dot_product && !translation_options.integer_dot_product)
        throw std::runtime_error("SPIR-V integer dot product requires a probed cl_khr_integer_dot_product device profile");
}

void CompilerOpenCL::apply_specializations()
{
    std::map<uint32_t, const SpecializationValue*> values;
    for (size_t i = 0; i < translation_options.specializations.size(); i++)
        values[translation_options.specializations[i].constant_id] = &translation_options.specializations[i];

    const SmallVector<SpecializationConstant> constants = get_specialization_constants();
    for (size_t i = 0; i < constants.size(); i++)
    {
        const SpecializationConstant& spec = constants[i];
        const std::map<uint32_t, const SpecializationValue*>::const_iterator it = values.find(spec.constant_id);
        if (it == values.end())
            continue;

        SPIRConstant& constant = get_constant(spec.id);
        const SPIRType& type = get_type(constant.constant_type);
        if (type.vecsize != 1 || type.columns != 1 || !type.array.empty())
            throw std::runtime_error("only scalar specialization constants are currently supported");

        const size_t byte_width = type.width / 8;
        if (byte_width != 1 && byte_width != 2 && byte_width != 4 && byte_width != 8)
            throw std::runtime_error("unsupported specialization-constant width");
        if (it->second->data.size() != byte_width)
            throw std::runtime_error("specialization-constant data size does not match its SPIR-V type");

        uint64_t bits = 0;
        std::memcpy(&bits, it->second->data.data(), byte_width);
        constant.m.c[0].r[0].u64 = bits;
        constant.specialization = false;
        unset_decoration(spec.id, DecorationSpecId);
    }
}

void CompilerOpenCL::localize_workgroup_variables()
{
    SPIRFunction& entry = get<SPIRFunction>(get_entry_point().self);
    auto it = global_variables.begin();
    while (it != global_variables.end())
    {
        SPIRVariable& variable = get<SPIRVariable>(*it);
        if (variable.storage != StorageClassWorkgroup)
        {
            ++it;
            continue;
        }

        const SPIRType& type = get_variable_data_type(variable);
        const bool supported_32 = (type.basetype == SPIRType::Int || type.basetype == SPIRType::UInt || type.basetype == SPIRType::Float) && type.width == 32;
        const bool supported_half = type.basetype == SPIRType::Half && type.width == 16;
        if ((!supported_32 && !supported_half) || type.columns != 1 || type.vecsize == 3 || type.vecsize > 4)
            throw std::runtime_error("OpenCL workgroup storage requires scalar, vec2 or vec4 int32/uint32/float32/float16 elements");
        for (size_t dimension = 0; dimension < type.array.size(); dimension++)
        {
            if (!type.array_size_literal[dimension] || type.array[dimension] == 0)
                throw std::runtime_error("OpenCL workgroup arrays require fixed nonzero literal dimensions");
        }

        entry.add_local_variable(variable.self);
        uses_workgroup_storage = true;
        it = global_variables.erase(it);
    }
}

void CompilerOpenCL::override_workgroup_size()
{
    SpecializationConstant x = {};
    SpecializationConstant y = {};
    SpecializationConstant z = {};
    get_work_group_size_specialization_constants(x, y, z);

    const SpecializationConstant constants[3] = {x, y, z};
    for (size_t i = 0; i < 3; i++)
    {
        if (constants[i].id)
        {
            SPIRConstant& constant = get_constant(constants[i].id);
            constant.m.c[0].r[0].u32 = translation_options.local_size[i];
            constant.specialization = false;
            unset_decoration(constants[i].id, DecorationSpecId);
        }
    }
    set_execution_mode(ExecutionModeLocalSize, translation_options.local_size[0],
                       translation_options.local_size[1], translation_options.local_size[2]);
}

bool translate_spirv_to_opencl_c(const uint32_t* words, size_t word_count,
                                 const TranslationOptions& options,
                                 TranslationResult& result)
{
    result = TranslationResult();

    if (!words || word_count < 5)
    {
        result.diagnostics = "SPIR-V module is null or shorter than its five-word header";
        return false;
    }
    if (words[0] != MagicNumber)
    {
        result.diagnostics = "input does not start with the SPIR-V magic number";
        return false;
    }

    try
    {
        const std::vector<uint32_t> optimized = specialize_and_optimize_spirv(words, word_count, options);
        const bool requires_global_int32_atomics = spirv_contains_opcode(optimized, OpAtomicCompareExchange);
        CompilerOpenCL compiler(optimized.data(), optimized.size(), options,
                                requires_global_int32_atomics);
        compiler.prepare(result.abi);

        CompilerGLSL::Options common_options;
        common_options.version = 450;
        common_options.es = false;
        common_options.vulkan_semantics = true;
        common_options.enable_420pack_extension = false;
        common_options.use_entry_point_name = true;
        compiler.set_common_options(common_options);

        result.source = compiler.compile();
        if (compiler.needs_converged_returns())
        {
            // Some OpenCL 1.x CPU compilers incorrectly execute code after a
            // divergent return in a workgroup kernel. Keep all work-items
            // converged through the barriers and predicate only storage-buffer
            // side effects; workgroup writes must still be performed by every
            // lane. Volatile prevents the redundant-store guard from being
            // folded under the original return condition.
            const std::string return_statement = "return;";
            const std::string deactivate_statement = "momo_active = 0;";
            size_t position = 0;
            while ((position = result.source.find(return_statement, position)) != std::string::npos)
            {
                result.source.replace(position, return_statement.size(), deactivate_statement);
                position += deactivate_statement.size();
            }
        }
        return true;
    }
    catch (const std::exception& e)
    {
        result.diagnostics = e.what();
        return false;
    }
}

} // namespace momoten
