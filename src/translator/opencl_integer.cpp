// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

namespace momoten {

std::string CompilerOpenCL::convert_value(const SPIRType& type, const std::string& value)
{
    if (type.vecsize == 1)
        return "((" + type_to_glsl(type) + ")(" + value + "))";
    return "convert_" + type_to_glsl(type) + "(" + value + ")";
}

std::string CompilerOpenCL::narrow_int16(const SPIRType& type, const std::string& value)
{
    SPIRType bits = type;
    bits.basetype = SPIRType::UShort;
    // preserve the low 16 bits before interpreting the result's signedness
    std::string expression = convert_value(bits, "(" + value + ") & 65535");
    if (type.basetype == SPIRType::Short)
        expression = "as_" + type_to_glsl(type) + "(" + expression + ")";
    return expression;
}

bool CompilerOpenCL::emit_integer_conversion_instruction(const Instruction& instruction, const uint32_t* ops)
{
    const Op opcode = static_cast<Op>(instruction.op);
    const bool integer_width = opcode == OpSConvert || opcode == OpUConvert;
    const bool to_float = opcode == OpConvertSToF || opcode == OpConvertUToF;
    const bool from_float = opcode == OpConvertFToS || opcode == OpConvertFToU;
    if (!integer_width && !to_float && !from_float)
        return false;

    const SPIRType& result_type = get<SPIRType>(ops[0]);
    const SPIRType& source_type = expression_type(ops[2]);
    const bool signed_value = opcode == OpSConvert || opcode == OpConvertSToF || opcode == OpConvertFToS;
    std::string value = to_expression(ops[2]);
    // conversion opcodes determine signedness independently of the SPIR-V type
    if (!from_float)
        value = bitcast_expression(signed_value ? to_signed_basetype(source_type.width) : to_unsigned_basetype(source_type.width), ops[2]);

    std::string expression;
    if (integer_width && result_type.width == 16 && source_type.width > 16)
    {
        expression = narrow_int16(result_type, value);
    }
    else
    {
        SPIRType converted = result_type;
        if (!to_float)
            converted.basetype = signed_value ? to_signed_basetype(result_type.width) : to_unsigned_basetype(result_type.width);
        expression = convert_value(converted, value);
        if (converted.basetype != result_type.basetype)
            expression = "as_" + type_to_glsl(result_type) + "(" + expression + ")";
    }
    emit_op(ops[0], ops[1], expression, should_forward(ops[2]));
    inherit_expression_dependencies(ops[1], ops[2]);
    return true;
}

bool CompilerOpenCL::emit_int16_arithmetic_instruction(const Instruction& instruction, const uint32_t* ops)
{
    const Op opcode = static_cast<Op>(instruction.op);
    const bool shift = opcode == OpShiftLeftLogical || opcode == OpShiftRightLogical || opcode == OpShiftRightArithmetic;
    if (opcode != OpIAdd && opcode != OpISub && opcode != OpIMul && opcode != OpSNegate && opcode != OpSMod && !shift)
        return false;
    const SPIRType& type = get<SPIRType>(ops[0]);
    if (type.width != 16 || type.columns != 1 || (type.basetype != SPIRType::Short && type.basetype != SPIRType::UShort))
        return false;

    // preserve modulo-16-bit arithmetic without signed overflow or scalar integer promotion
    SPIRType wide = type;
    wide.width = 32;
    wide.basetype = opcode == OpShiftRightArithmetic || opcode == OpSMod ? SPIRType::Int : SPIRType::UInt;
    std::string left = to_unpacked_expression(ops[2]);
    if (shift || opcode == OpSMod)
    {
        SPIRType shifted = type;
        shifted.basetype = wide.basetype == SPIRType::Int ? SPIRType::Short : SPIRType::UShort;
        if (expression_type(ops[2]).basetype != shifted.basetype)
            left = "as_" + type_to_glsl(shifted) + "(" + left + ")";
    }
    std::string expression = convert_value(wide, left);
    bool forward = should_forward(ops[2]);
    if (opcode == OpSNegate)
        expression = "0u - " + expression;
    else
    {
        const char* operation;
        switch (opcode)
        {
        case OpIAdd: operation = " + "; break;
        case OpISub: operation = " - "; break;
        case OpIMul: operation = " * "; break;
        case OpSMod: operation = " % "; break;
        case OpShiftLeftLogical: operation = " << "; break;
        default: operation = " >> "; break;
        }
        SPIRType right = expression_type(ops[3]);
        right.basetype = SPIRType::UInt;
        right.width = 32;
        const SPIRType& right_type = shift ? right : wide;
        std::string right_expression = to_unpacked_expression(ops[3]);
        if (opcode == OpSMod && expression_type(ops[3]).basetype != SPIRType::Short)
        {
            SPIRType signed_type = type;
            signed_type.basetype = SPIRType::Short;
            right_expression = "as_" + type_to_glsl(signed_type) + "(" + right_expression + ")";
        }
        right_expression = convert_value(right_type, right_expression);
        expression += operation + right_expression;
        if (opcode == OpSMod)
        {
            const std::string remainder = "(" + expression + ")";
            const std::string condition = "(" + remainder + " != 0) & ((" + remainder + " < 0) != (" + right_expression + " < 0))";
            const std::string correction = type.vecsize == 1 ? "((" + condition + ") ? " + right_expression + " : 0)" : "select((" + type_to_glsl(wide) + ")(0), " + right_expression + ", " + condition + ")";
            expression = remainder + " + " + correction;
        }
        forward = forward && should_forward(ops[3]);
    }
    expression = narrow_int16(type, expression);
    emit_op(ops[0], ops[1], expression, forward);
    inherit_expression_dependencies(ops[1], ops[2]);
    if (opcode != OpSNegate)
        inherit_expression_dependencies(ops[1], ops[3]);
    return true;
}

} // namespace momoten
