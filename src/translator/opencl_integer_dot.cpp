// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <stdexcept>

namespace momoten {

void CompilerOpenCL::emit_integer_dot_product_helpers()
{
    // DotProductInputAll includes 32-bit vectors even though OpenCL's native
    // extension only accelerates 4x8-bit inputs. Accumulate exact magnitudes
    // in three uint limbs so saturating operations remain correct without
    // requiring optional OpenCL int64 support.
    // These helpers deliberately use ordinary definitions rather than inline;
    // see emit_header() for the OpenCL C 1.0 linkage constraint.
    statement("typedef struct { uint lo; uint mid; uint hi; } momo_u96;");
    statement("void momo_u96_add_u32(__private momo_u96* value, uint addend)");
    begin_scope();
    statement("uint old_lo = value->lo;");
    statement("value->lo += addend;");
    statement("if (value->lo < old_lo)");
    begin_scope();
    statement("uint old_mid = value->mid;");
    statement("value->mid++;");
    statement("if (value->mid < old_mid) value->hi++;");
    end_scope();
    end_scope();
    statement("void momo_u96_add_product(__private momo_u96* value, uint a, uint b)");
    begin_scope();
    statement("uint product_lo = a * b;");
    statement("uint product_hi = mul_hi(a, b);");
    statement("uint old_lo = value->lo;");
    statement("value->lo += product_lo;");
    statement("uint carry_lo = value->lo < old_lo;");
    statement("uint old_mid = value->mid;");
    statement("value->mid += product_hi;");
    statement("uint carry_hi = value->mid < old_mid;");
    statement("old_mid = value->mid;");
    statement("value->mid += carry_lo;");
    statement("value->hi += carry_hi + (value->mid < old_mid);");
    end_scope();
    statement("int momo_u96_compare(momo_u96 a, momo_u96 b)");
    begin_scope();
    statement("if (a.hi != b.hi) return a.hi > b.hi ? 1 : -1;");
    statement("if (a.mid != b.mid) return a.mid > b.mid ? 1 : -1;");
    statement("if (a.lo != b.lo) return a.lo > b.lo ? 1 : -1;");
    statement("return 0;");
    end_scope();
    statement("momo_u96 momo_u96_subtract(momo_u96 a, momo_u96 b)");
    begin_scope();
    statement("momo_u96 result;");
    statement("result.lo = a.lo - b.lo;");
    statement("uint borrow_lo = a.lo < b.lo;");
    statement("uint mid_without_borrow = a.mid - b.mid;");
    statement("uint borrow_mid = a.mid < b.mid;");
    statement("result.mid = mid_without_borrow - borrow_lo;");
    statement("uint borrow_carry = mid_without_borrow < borrow_lo;");
    statement("result.hi = a.hi - b.hi - borrow_mid - borrow_carry;");
    statement("return result;");
    end_scope();
    statement("uint momo_abs_int_bits(int value)");
    begin_scope();
    statement("return value < 0 ? 0u - as_uint(value) : as_uint(value);");
    end_scope();
    statement("void momo_u96_add_ss_product(__private momo_u96* positive, __private momo_u96* negative, int a, int b)");
    begin_scope();
    statement("__private momo_u96* destination = ((a < 0) != (b < 0)) ? negative : positive;");
    statement("momo_u96_add_product(destination, momo_abs_int_bits(a), momo_abs_int_bits(b));");
    end_scope();
    statement("void momo_u96_add_su_product(__private momo_u96* positive, __private momo_u96* negative, int a, uint b)");
    begin_scope();
    statement("momo_u96_add_product(a < 0 ? negative : positive, momo_abs_int_bits(a), b);");
    end_scope();
    statement("int momo_finalize_signed_dot(momo_u96 positive, momo_u96 negative, int accumulator)");
    begin_scope();
    statement("if (accumulator < 0) momo_u96_add_u32(&negative, 0u - as_uint(accumulator));");
    statement("else momo_u96_add_u32(&positive, as_uint(accumulator));");
    statement("int comparison = momo_u96_compare(positive, negative);");
    statement("momo_u96 magnitude = comparison >= 0 ? momo_u96_subtract(positive, negative) : momo_u96_subtract(negative, positive);");
    statement("if (magnitude.hi != 0u || magnitude.mid != 0u)");
    statement("    return comparison >= 0 ? as_int(0x7fffffffu) : as_int(0x80000000u);");
    statement("if (comparison >= 0)");
    statement("    return magnitude.lo > 0x7fffffffu ? as_int(0x7fffffffu) : as_int(magnitude.lo);");
    statement("return magnitude.lo > 0x80000000u ? as_int(0x80000000u) : as_int(0u - magnitude.lo);");
    end_scope();
    statement("uint momo_finalize_unsigned_dot(momo_u96 value, uint accumulator)");
    begin_scope();
    statement("momo_u96_add_u32(&value, accumulator);");
    statement("return value.hi != 0u || value.mid != 0u ? 0xffffffffu : value.lo;");
    end_scope();

    static const char components[] = "xyzw";
    for (uint32_t count = 2; count <= 4; count++)
    {
        const std::string suffix = std::to_string(count);
        std::string signed_sum = "return ";
        std::string unsigned_sum = "return ";
        std::string mixed_sum = "return ";
        for (uint32_t i = 0; i < count; i++)
        {
            if (i != 0)
            {
                signed_sum += " + ";
                unsigned_sum += " + ";
                mixed_sum += " + ";
            }
            signed_sum += "as_uint(a." + std::string(1, components[i]) + ") * as_uint(b." + std::string(1, components[i]) + ")";
            unsigned_sum += "a." + std::string(1, components[i]) + " * b." + std::string(1, components[i]);
            mixed_sum += "as_uint(a." + std::string(1, components[i]) + ") * b." + std::string(1, components[i]);
        }
        signed_sum += ";";
        unsigned_sum += ";";
        mixed_sum += ";";

        statement("uint momo_sdot32_" + suffix + "(int" + suffix + " a, int" + suffix + " b)");
        begin_scope();
        statement(signed_sum);
        end_scope();
        statement("uint momo_udot32_" + suffix + "(uint" + suffix + " a, uint" + suffix + " b)");
        begin_scope();
        statement(unsigned_sum);
        end_scope();
        statement("uint momo_sudot32_" + suffix + "(int" + suffix + " a, uint" + suffix + " b)");
        begin_scope();
        statement(mixed_sum);
        end_scope();

        statement("int momo_sdot_acc_sat32_" + suffix + "(int" + suffix + " a, int" + suffix + " b, int accumulator)");
        begin_scope();
        statement("momo_u96 positive = { 0u, 0u, 0u };");
        statement("momo_u96 negative = { 0u, 0u, 0u };");
        for (uint32_t i = 0; i < count; i++)
            statement("momo_u96_add_ss_product(&positive, &negative, a." + std::string(1, components[i]) + ", b." + std::string(1, components[i]) + ");");
        statement("return momo_finalize_signed_dot(positive, negative, accumulator);");
        end_scope();

        statement("uint momo_udot_acc_sat32_" + suffix + "(uint" + suffix + " a, uint" + suffix + " b, uint accumulator)");
        begin_scope();
        statement("momo_u96 value = { 0u, 0u, 0u };");
        for (uint32_t i = 0; i < count; i++)
            statement("momo_u96_add_product(&value, a." + std::string(1, components[i]) + ", b." + std::string(1, components[i]) + ");");
        statement("return momo_finalize_unsigned_dot(value, accumulator);");
        end_scope();

        statement("int momo_sudot_acc_sat32_" + suffix + "(int" + suffix + " a, uint" + suffix + " b, int accumulator)");
        begin_scope();
        statement("momo_u96 positive = { 0u, 0u, 0u };");
        statement("momo_u96 negative = { 0u, 0u, 0u };");
        for (uint32_t i = 0; i < count; i++)
            statement("momo_u96_add_su_product(&positive, &negative, a." + std::string(1, components[i]) + ", b." + std::string(1, components[i]) + ");");
        statement("return momo_finalize_signed_dot(positive, negative, accumulator);");
        end_scope();
    }
}

bool CompilerOpenCL::emit_integer_dot_product_instruction(
    const Instruction& instruction, const uint32_t* ops)
{
    const Op opcode = static_cast<Op>(instruction.op);
    const bool signed_dot = opcode == OpSDot || opcode == OpSDotAccSat;
    const bool unsigned_dot = opcode == OpUDot || opcode == OpUDotAccSat;
    const bool mixed_dot = opcode == OpSUDot || opcode == OpSUDotAccSat;
    if (!signed_dot && !unsigned_dot && !mixed_dot)
        return false;

    if (!translation_options.integer_dot_product)
        throw std::runtime_error("SPIR-V integer dot product is disabled in the OpenCL device profile");

    const bool accumulating = opcode == OpSDotAccSat || opcode == OpUDotAccSat || opcode == OpSUDotAccSat;
    const uint32_t unpacked_length = accumulating ? 5 : 4;
    const uint32_t packed_length = accumulating ? 6 : 5;
    if (instruction.length != unpacked_length && instruction.length != packed_length)
        throw std::runtime_error("SPIR-V integer dot product has an invalid operand count");
    const bool packed = instruction.length == packed_length;
    if (packed && ops[packed_length - 1] != PackedVectorFormatPackedVectorFormat4x8Bit)
        throw std::runtime_error("only PackedVectorFormat4x8Bit integer dot product is supported");

    const SPIRType& result_type = get<SPIRType>(ops[0]);
    const SPIRType& first_type = expression_type(ops[2]);
    const SPIRType& second_type = expression_type(ops[3]);
    const bool result_integer = result_type.basetype == SPIRType::Int || result_type.basetype == SPIRType::UInt;
    if (!result_integer || result_type.width != 32 || result_type.vecsize != 1 || result_type.columns != 1)
        throw std::runtime_error("the OpenCL Vulkan profile requires a scalar 32-bit integer dot-product result");
    if (unsigned_dot && result_type.basetype != SPIRType::UInt)
        throw std::runtime_error("OpUDot/OpUDotAccSat requires an unsigned result in the OpenCL Vulkan profile");
    if (first_type.columns != 1 || second_type.columns != 1 || first_type.width != second_type.width || first_type.vecsize != second_type.vecsize)
        throw std::runtime_error("integer dot-product operands must have matching scalar width and vector length");

    const auto integer_expression_as = [this](uint32_t id, SPIRType::BaseType expected) {
        const SPIRType& actual = expression_type(id);
        std::string expression = to_expression(id);
        if (actual.basetype == expected)
            return expression;

        SPIRType converted = actual;
        converted.basetype = expected;
        return std::string("as_") + type_to_glsl(converted) + "(" + expression + ")";
    };
    const auto convert_result = [this, &result_type](const std::string& expression,
                                                     SPIRType::BaseType expression_base) {
        if (result_type.basetype == expression_base)
            return expression;
        return std::string(result_type.basetype == SPIRType::Int ? "as_int(" : "as_uint(") + expression + ")";
    };

    std::string first;
    std::string second;
    std::string expression;
    SPIRType::BaseType native_result_base = unsigned_dot ? SPIRType::UInt : SPIRType::Int;
    if (packed)
    {
        if (!translation_options.integer_dot_product_input_4x8bit_packed || first_type.vecsize != 1 || second_type.vecsize != 1 || first_type.width != 32 || (first_type.basetype != SPIRType::Int && first_type.basetype != SPIRType::UInt) || (second_type.basetype != SPIRType::Int && second_type.basetype != SPIRType::UInt))
            throw std::runtime_error("packed integer dot product requires two scalar 32-bit integer operands and native packed 4x8 support");

        first = integer_expression_as(ops[2], SPIRType::UInt);
        second = integer_expression_as(ops[3], SPIRType::UInt);
        const char* function = 0;
        if (!accumulating)
            function = signed_dot ? "dot_4x8packed_ss_int" : unsigned_dot ? "dot_4x8packed_uu_uint"
                                                                          : "dot_4x8packed_su_int";
        else
            function = signed_dot ? "dot_acc_sat_4x8packed_ss_int" : unsigned_dot ? "dot_acc_sat_4x8packed_uu_uint"
                                                                                  : "dot_acc_sat_4x8packed_su_int";
        expression = std::string(function) + "(" + first + ", " + second;
        if (accumulating)
            expression += ", " + integer_expression_as(ops[4], unsigned_dot ? SPIRType::UInt : SPIRType::Int);
        expression += ")";
        expression = convert_result(expression, native_result_base);
    }
    else if (first_type.width == 8 && first_type.vecsize == 4)
    {
        if (!translation_options.integer_dot_product_input_4x8bit)
            throw std::runtime_error("4x8-bit vector integer dot product is unavailable in the OpenCL device profile");
        first = integer_expression_as(
            ops[2], unsigned_dot ? SPIRType::UByte : SPIRType::SByte);
        second = integer_expression_as(
            ops[3], signed_dot ? SPIRType::SByte : SPIRType::UByte);
        expression = std::string(accumulating ? "dot_acc_sat(" : "dot(") + first + ", " + second;
        if (accumulating)
            expression += ", " + integer_expression_as(ops[4], unsigned_dot ? SPIRType::UInt : SPIRType::Int);
        expression += ")";
        expression = convert_result(expression, native_result_base);
    }
    else if (first_type.width == 32 && first_type.vecsize >= 2 && first_type.vecsize <= 4)
    {
        first = integer_expression_as(
            ops[2], unsigned_dot ? SPIRType::UInt : SPIRType::Int);
        second = integer_expression_as(
            ops[3], signed_dot ? SPIRType::Int : SPIRType::UInt);
        const std::string suffix = std::to_string(first_type.vecsize);
        std::string function;
        if (!accumulating)
            function = signed_dot ? "momo_sdot32_" : unsigned_dot ? "momo_udot32_"
                                                                  : "momo_sudot32_";
        else
            function = signed_dot ? "momo_sdot_acc_sat32_" : unsigned_dot ? "momo_udot_acc_sat32_"
                                                                          : "momo_sudot_acc_sat32_";
        expression = function + suffix + "(" + first + ", " + second;
        if (accumulating)
            expression += ", " + integer_expression_as(ops[4], unsigned_dot ? SPIRType::UInt : SPIRType::Int);
        expression += ")";
        // Non-saturating software helpers return the modulo-2^32 bit pattern.
        expression = convert_result(expression,
                                    accumulating ? native_result_base : SPIRType::UInt);
    }
    else
    {
        throw std::runtime_error("integer dot-product operand type is outside the enabled Vulkan shader arithmetic profile");
    }

    bool forward = should_forward(ops[2]) && should_forward(ops[3]);
    if (accumulating)
        forward = forward && should_forward(ops[4]);
    emit_op(ops[0], ops[1], expression, forward);
    inherit_expression_dependencies(ops[1], ops[2]);
    inherit_expression_dependencies(ops[1], ops[3]);
    if (accumulating)
        inherit_expression_dependencies(ops[1], ops[4]);
    return true;
}

} // namespace momoten
