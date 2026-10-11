// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <stdexcept>

#include "spirv-tools/libspirv.hpp"

namespace momoten {

static const uint32_t vulkan_fast_math_flags = FPFastMathModeNotNaNMask | FPFastMathModeNotInfMask | FPFastMathModeNSZMask | FPFastMathModeAllowRecipMask | FPFastMathModeAllowContractMask | FPFastMathModeAllowReassocMask | FPFastMathModeAllowTransformMask;

static uint32_t floating_point_width_mask(uint32_t width)
{
    switch (width)
    {
    case 16:
        return FloatingPointWidth16;
    case 32:
        return FloatingPointWidth32;
    case 64:
        return FloatingPointWidth64;
    default:
        throw std::runtime_error("unsupported floating-point width in SPIR-V float controls");
    }
}

// These builtins retain their normal accuracy under OpenCL unsafe math.
// Transcendentals and unrecognized extended instructions keep the normal
// compiler options: FloatControls2 does not authorize approximate functions.
static bool builtin_retains_accuracy(uint32_t instruction)
{
    switch (instruction)
    {
    case GLSLstd450Round:
    case GLSLstd450RoundEven:
    case GLSLstd450Trunc:
    case GLSLstd450FAbs:
    case GLSLstd450FSign:
    case GLSLstd450Floor:
    case GLSLstd450Ceil:
    case GLSLstd450FMin:
    case GLSLstd450FMax:
    case GLSLstd450FClamp:
    case GLSLstd450FMix:
    case GLSLstd450Step:
    case GLSLstd450SmoothStep:
    case GLSLstd450Fma:
    case GLSLstd450NMin:
    case GLSLstd450NMax:
    case GLSLstd450NClamp:
        return true;
    default:
        return false;
    }
}

struct FloatingPointInstruction
{
    uint32_t result_id;
    uint32_t type_id;
    uint32_t callee;
    bool requires_builtin_accuracy;
    std::vector<uint32_t> operands;
};

FloatingPointControls::FloatingPointControls()
    : fast_math_flags(0), widths(0), denorm_preserve_widths(0), round_to_nearest_widths(0), signed_zero_inf_nan_preserve_widths(0), requires_builtin_accuracy(false)
{
}

uint32_t CompilerOpenCL::floating_point_scalar_type(uint32_t type_id) const
{
    if (type_id == 0)
        return 0;
    const SPIRType& type = get<SPIRType>(type_id);
    if (type.pointer || !type.array.empty() || (type.basetype != SPIRType::Half && type.basetype != SPIRType::Float && type.basetype != SPIRType::Double))
        return 0;
    if (type.vecsize == 1 && type.columns == 1)
        return type_id;
    return floating_point_scalar_type(type.parent_type);
}

void CompilerOpenCL::reflect_float_controls(KernelABI& abi)
{
    FloatingPointControls& controls = abi.floating_point;
    const uint32_t entry_id = get_entry_point().self;
    std::map<uint32_t, uint32_t> value_types;
    std::map<uint32_t, uint32_t> defaults;
    std::map<uint32_t, std::vector<FloatingPointInstruction> > functions;
    uint32_t current_function = 0;
    bool contraction_off = false;

    // Use the grammar's operand kinds so literal indices cannot be mistaken
    // for floating-point IDs. Loads, stores, comparisons and conversions all
    // matter when selecting an option that affects the entire OpenCL program.
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
    if (!tools.Parse(ir.spirv, [](spv_endianness_t, const spv_parsed_header_t&) { return SPV_SUCCESS; }, [&](const spv_parsed_instruction_t& instruction) {
            const uint32_t* words = instruction.words;
            const Op opcode = static_cast<Op>(instruction.opcode);
            if (instruction.type_id && instruction.result_id)
                value_types[instruction.result_id] = instruction.type_id;
            if (opcode == OpExecutionModeId && words[1] == entry_id && words[2] == ExecutionModeFPFastMathDefault)
                defaults[words[3]] = static_cast<const CompilerOpenCL&>(*this).get<SPIRConstant>(words[4]).scalar();
            if (opcode == OpExecutionMode && words[1] == entry_id)
            {
                const ExecutionMode mode = static_cast<ExecutionMode>(words[2]);
                if (mode == ExecutionModeContractionOff)
                    contraction_off = true;
                else if (mode == ExecutionModeDenormPreserve)
                    controls.denorm_preserve_widths |= floating_point_width_mask(words[3]);
                else if (mode == ExecutionModeSignedZeroInfNanPreserve)
                    controls.signed_zero_inf_nan_preserve_widths |= floating_point_width_mask(words[3]);
                else if (mode == ExecutionModeRoundingModeRTE)
                    controls.round_to_nearest_widths |= floating_point_width_mask(words[3]);
                else if (mode == ExecutionModeDenormFlushToZero)
                    throw std::runtime_error("SPIR-V DenormFlushToZero cannot be guaranteed by OpenCL compiler options");
                else if (mode == ExecutionModeRoundingModeRTZ)
                    throw std::runtime_error("SPIR-V RoundingModeRTZ is not supported by OpenCL C translation");
            }
            if (opcode == OpFunction)
            {
                current_function = instruction.result_id;
                return SPV_SUCCESS;
            }
            if (opcode == OpFunctionEnd)
            {
                current_function = 0;
                return SPV_SUCCESS;
            }
            if (!current_function)
                return SPV_SUCCESS;

            FloatingPointInstruction operation = {};
            operation.result_id = instruction.result_id;
            operation.type_id = instruction.type_id;
            if (opcode == OpFunctionCall)
                operation.callee = words[3];
            if (opcode == OpExtInst)
                operation.requires_builtin_accuracy = instruction.ext_inst_type != SPV_EXT_INST_TYPE_GLSL_STD_450 || !builtin_retains_accuracy(words[4]);
            for (uint32_t i = 0; i < instruction.num_operands; i++)
            {
                const spv_parsed_operand_t& operand = instruction.operands[i];
                if (operand.type == SPV_OPERAND_TYPE_ID)
                    operation.operands.push_back(words[operand.offset]);
            }
            functions[current_function].push_back(operation);
            return SPV_SUCCESS; }))
        throw std::runtime_error("failed to parse SPIR-V floating-point operands");

    uint32_t common_flags = vulkan_fast_math_flags;
    std::set<uint32_t> visited;
    std::vector<uint32_t> pending(1, entry_id);
    while (!pending.empty())
    {
        const uint32_t function = pending.back();
        pending.pop_back();
        if (!visited.insert(function).second)
            continue;
        const std::vector<FloatingPointInstruction>& operations = functions[function];
        for (size_t i = 0; i < operations.size(); i++)
        {
            const FloatingPointInstruction& operation = operations[i];
            if (operation.callee)
                pending.push_back(operation.callee);
            std::set<uint32_t> types;
            const uint32_t result_scalar = floating_point_scalar_type(operation.type_id);
            if (result_scalar)
                types.insert(result_scalar);
            for (size_t j = 0; j < operation.operands.size(); j++)
            {
                const uint32_t scalar = floating_point_scalar_type(value_types[operation.operands[j]]);
                if (scalar)
                    types.insert(scalar);
            }
            if (types.empty())
                continue;

            uint32_t flags = defaults.empty() ? vulkan_fast_math_flags : 0;
            uint32_t widths = 0;
            for (std::set<uint32_t>::const_iterator type = types.begin(); type != types.end(); ++type)
            {
                const uint32_t width = floating_point_width_mask(get<SPIRType>(*type).width);
                widths |= width;
                if (!defaults.empty())
                    flags |= defaults[*type];
            }
            controls.widths |= widths;
            // Legacy preservation applies when any operand or result width
            // requests it. Explicit FloatControls2 defaults instead combine
            // the permissions for the participating scalar types above.
            if (defaults.empty() && (controls.signed_zero_inf_nan_preserve_widths & widths))
                flags &= ~(FPFastMathModeNotNaNMask | FPFastMathModeNotInfMask | FPFastMathModeNSZMask);
            if (operation.result_id && has_decoration(operation.result_id, DecorationFPFastMathMode))
                flags = get_decoration(operation.result_id, DecorationFPFastMathMode);
            if (contraction_off || (operation.result_id && has_decoration(operation.result_id, DecorationNoContraction)))
                flags &= ~(FPFastMathModeAllowRecipMask | FPFastMathModeAllowContractMask | FPFastMathModeAllowReassocMask | FPFastMathModeAllowTransformMask);
            common_flags &= flags;
            controls.requires_builtin_accuracy = controls.requires_builtin_accuracy || operation.requires_builtin_accuracy;
        }
    }
    if (controls.widths)
        controls.fast_math_flags = common_flags;
    floating_point_controls = controls;
}

} // namespace momoten
