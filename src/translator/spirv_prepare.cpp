// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "spirv-tools/optimizer.hpp"

namespace momoten {

std::vector<uint32_t> specialize_and_optimize_spirv(
    const uint32_t* words, size_t word_count, const TranslationOptions& options)
{
    bool declares_workgroup_storage = false;
    for (size_t offset = 5; offset < word_count;)
    {
        const uint32_t instruction = words[offset];
        const uint32_t instruction_words = instruction >> 16;
        if (instruction_words == 0 || instruction_words > word_count - offset)
            throw std::runtime_error("malformed SPIR-V instruction stream");
        if ((instruction & 0xffffu) == static_cast<uint32_t>(OpVariable) && instruction_words >= 4 && words[offset + 3] == StorageClassWorkgroup)
            declares_workgroup_storage = true;
        offset += instruction_words;
    }

    std::unordered_map<uint32_t, std::vector<uint32_t> > specialization_values;
    for (size_t i = 0; i < options.specializations.size(); i++)
    {
        const SpecializationValue& value = options.specializations[i];
        if (value.data.empty() || value.data.size() > 8)
            throw std::runtime_error("specialization constants must contain between one and eight bytes");
        std::vector<uint32_t> bits((value.data.size() + 3) / 4, 0);
        memcpy(bits.data(), value.data.data(), value.data.size());
        specialization_values[value.constant_id] = bits;
    }

    std::ostringstream messages;
    // Vulkan 1.1 compute shaders commonly use SPIR-V 1.3. The corresponding
    // validation environment still accepts SPIR-V 1.0 modules used by older
    // applications and the baseline unit tests.
    spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_1);
    optimizer.SetMessageConsumer(
        [&messages](spv_message_level_t, const char* source,
                    const spv_position_t& position, const char* message) {
            if (source && source[0])
                messages << source << ':';
            messages << position.line << ':' << position.column << ": " << message << '\n';
        });
    if (!specialization_values.empty())
        optimizer.RegisterPass(spvtools::CreateSetSpecConstantDefaultValuePass(specialization_values));
    optimizer.RegisterPass(spvtools::CreateFreezeSpecConstantValuePass());
    optimizer.RegisterPass(spvtools::CreateFoldSpecConstantOpAndCompositePass());
    optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
    // Vulkan GLSL helper functions may directly reference descriptor and push
    // constant globals. OpenCL C represents those resources as kernel
    // parameters, so inline helpers before source emission to keep every such
    // access in kernel-parameter scope.
    // PoCL handles ordinary early exits more reliably after MergeReturn, but
    // workgroup kernels need real returns so momoten can converge all lanes
    // through barriers and predicate their storage side effects below.
    if (!declares_workgroup_storage)
        optimizer.RegisterPass(spvtools::CreateMergeReturnPass());
    optimizer.RegisterPass(spvtools::CreateInlineExhaustivePass());
    optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
    optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
    optimizer.RegisterPass(spvtools::CreateEliminateDeadFunctionsPass());

    std::vector<uint32_t> optimized;
    if (!optimizer.Run(words, word_count, &optimized))
    {
        std::string diagnostic = messages.str();
        if (diagnostic.empty())
            diagnostic = "SPIR-V specialization/dead-code optimization failed";
        throw std::runtime_error(diagnostic);
    }
    return optimized;
}

bool spirv_contains_opcode(const std::vector<uint32_t>& words, Op opcode)
{
    for (size_t offset = 5; offset < words.size();)
    {
        const uint32_t instruction = words[offset];
        const uint32_t instruction_words = instruction >> 16;
        if (instruction_words == 0 || instruction_words > words.size() - offset)
            throw std::runtime_error("malformed SPIR-V instruction stream after optimization");
        if ((instruction & 0xffffu) == static_cast<uint32_t>(opcode))
            return true;
        offset += instruction_words;
    }
    return false;
}

const char* capability_name(Capability capability)
{
    switch (capability)
    {
    case CapabilityShader:
        return "Shader";
    case CapabilityMatrix:
        return "Matrix";
    case CapabilityInt64:
        return "Int64";
    case CapabilityFloat64:
        return "Float64";
    case CapabilityInt16:
        return "Int16";
    case CapabilityFloat16:
        return "Float16";
    case CapabilityStorageBuffer16BitAccess:
        return "StorageBuffer16BitAccess";
    case CapabilityUniformAndStorageBuffer16BitAccess:
        return "UniformAndStorageBuffer16BitAccess";
    case CapabilityStoragePushConstant16:
        return "StoragePushConstant16";
    case CapabilityGroupNonUniform:
        return "GroupNonUniform";
    case CapabilityGroupNonUniformVote:
        return "GroupNonUniformVote";
    case CapabilityGroupNonUniformArithmetic:
        return "GroupNonUniformArithmetic";
    case CapabilityGroupNonUniformBallot:
        return "GroupNonUniformBallot";
    case CapabilityGroupNonUniformShuffle:
        return "GroupNonUniformShuffle";
    case CapabilityGroupNonUniformShuffleRelative:
        return "GroupNonUniformShuffleRelative";
    case CapabilityGroupNonUniformClustered:
        return "GroupNonUniformClustered";
    case CapabilityGroupNonUniformQuad:
        return "GroupNonUniformQuad";
    case CapabilityDotProductInputAll:
        return "DotProductInputAll";
    case CapabilityDotProductInput4x8Bit:
        return "DotProductInput4x8Bit";
    case CapabilityDotProductInput4x8BitPacked:
        return "DotProductInput4x8BitPacked";
    case CapabilityDotProduct:
        return "DotProduct";
    default:
        return "unsupported capability";
    }
}

} // namespace momoten
