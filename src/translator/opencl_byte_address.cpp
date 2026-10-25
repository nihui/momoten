// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <stdexcept>

namespace momoten {

bool CompilerOpenCL::uses_byte_addressing(uint32_t variable) const
{
    return (raw_push_constants && variable == push_constant_variable_id) || raw_buffer_variables.count(variable)
           || (resource_indices.count(variable) && has_decoration(variable, DecorationAliased));
}

void CompilerOpenCL::track_byte_addressed_access(const Instruction& instruction, const uint32_t* ops)
{
    if (!get<SPIRType>(ops[0]).pointer)
        return;
    const SPIRVariable* backing = maybe_get_backing_variable(ops[2]);
    if (!backing || !uses_byte_addressing(backing->self))
        return;
    BufferAccessChain& access = buffer_access_chains[ops[1]];
    access.base = ops[2];
    access.indices.assign(ops + 3, ops + instruction.length);
}

std::string CompilerOpenCL::byte_address(uint32_t pointer, uint32_t* matrix_stride)
{
    const auto found = buffer_access_chains.find(pointer);
    if (found == buffer_access_chains.end())
    {
        if (matrix_stride)
            *matrix_stride = 0;
        if (raw_push_constants && pointer == push_constant_variable_id)
            return "((__private const uchar *)momo_push_constants)";
        const auto resource = resource_indices.find(pointer);
        if (resource == resource_indices.end())
            throw std::runtime_error("unsupported byte-addressed buffer pointer");
        const std::string index = std::to_string(resource->second);
        return "momo_buffer_" + index + " + momo_buffer_offset_" + index;
    }

    const BufferAccessChain& access = found->second;
    uint32_t matrix_bytes = 0;
    std::string address = byte_address(access.base, &matrix_bytes);
    uint32_t type_id = expression_type(access.base).parent_type;
    for (uint32_t index : access.indices)
    {
        const SPIRType* type = &get<SPIRType>(type_id);
        uint32_t stride;
        if (!type->array.empty())
        {
            stride = get_decoration(type_id, DecorationArrayStride);
        }
        else if (type->basetype == SPIRType::Struct)
        {
            const uint32_t member = evaluate_constant_u32(index);
            if (member >= type->member_types.size())
                throw std::runtime_error("buffer member index is out of bounds");
            address += " + " + std::to_string(get_member_decoration(type->self, member, DecorationOffset)) + "u";
            matrix_bytes = has_member_decoration(type->self, member, DecorationMatrixStride) ? get_member_decoration(type->self, member, DecorationMatrixStride) : 0;
            type_id = type->member_types[member];
            continue;
        }
        else
        {
            // preserve array and matrix strides without taking a vector lane address
            stride = type->columns > 1 && matrix_bytes ? matrix_bytes : type->width / 8 * (type->columns > 1 ? type->vecsize : 1);
            if (type->columns > 1)
                matrix_bytes = 0;
        }
        const char* offset_type = translation_options.address_bits == 64 ? "ulong" : "uint";
        address += " + ((" + std::string(offset_type) + ")(" + to_unpacked_expression(index) + ") * " + std::to_string(stride) + "u)";
        type_id = type->parent_type;
    }
    if (matrix_stride)
        *matrix_stride = matrix_bytes;
    return address;
}

std::string CompilerOpenCL::byte_addressed_load(const SPIRType& type, const std::string& address, uint32_t offset, const std::string& qualifiers)
{
    const uint32_t bytes = type.width / 8;
    if (bytes == 1)
    {
        const std::string value = "((" + qualifiers + " const uchar *)(" + address + "))[" + std::to_string(offset) + "]";
        return type.basetype == SPIRType::SByte ? "as_char(" + value + ")" : value;
    }
    std::string expression = "as_" + type_to_glsl(type) + "((uchar" + std::to_string(bytes) + ")(";
    for (uint32_t i = 0; i < bytes; i++)
    {
        if (i)
            expression += ", ";
        expression += "((" + qualifiers + " const uchar *)(" + address + "))[" + std::to_string(offset + i) + "]";
    }
    return expression + "))";
}

void CompilerOpenCL::emit_byte_addressed_store(const SPIRType& type, const std::string& address, uint32_t offset,
                                               const std::string& value, const std::string& temporary, const std::string& qualifiers)
{
    const uint32_t unit_bytes = type.width <= 16 ? 1 : 4;
    const uint32_t count = type.width / 8 / unit_bytes;
    const std::string unit_type = unit_bytes == 1 ? "uchar" : "uint";
    const std::string packed_type = unit_type + (count == 1 ? "" : std::to_string(count));
    const std::string expression = packed_type == type_to_glsl(type) ? value : "as_" + packed_type + "(" + value + ")";
    statement(packed_type, " ", temporary, " = ", expression, ";");
    for (uint32_t i = 0; i < count; i++)
        statement("((", qualifiers, " ", unit_type, " *)(", address, " + ", offset, "u))[", i, "] = ", temporary,
                  count == 1 ? "" : ".s" + std::to_string(i), ";");
}

bool CompilerOpenCL::emit_byte_addressed_access(const Instruction& instruction, const uint32_t* ops)
{
    const bool load = instruction.op == OpLoad;
    const uint32_t pointer = ops[load ? 2 : 0];
    const SPIRVariable* backing = maybe_get_backing_variable(pointer);
    const bool private_storage = backing && raw_push_constants && backing->self == push_constant_variable_id;
    if (!backing || !uses_byte_addressing(backing->self))
        return false;
    if (!load && (private_storage || (resource_indices.count(backing->self) && resources[resource_indices.at(backing->self)].uniform)))
        throw std::runtime_error("uniform and push-constant storage is read-only");
    const SPIRType& type = load ? get<SPIRType>(ops[0]) : expression_type(ops[1]);
    if (!type.array.empty())
        throw std::runtime_error("byte-addressed buffer array accesses are not supported");
    if (type.pointer || (type.width != 8 && type.width != 16 && type.width != 32 && type.width != 64) || type.vecsize > 4)
        throw std::runtime_error("byte-addressed buffer access has an unsupported value type");

    if (!load && type.width < 32 && !translation_options.byte_addressable_store)
        throw std::runtime_error("narrow byte-addressed stores require OpenCL C 1.1 or cl_khr_byte_addressable_store");

    require_statement_continue();
    flush_variable_declaration(pointer);
    // retain SPIRV-Cross pointer dependencies and write tracking
    to_expression(pointer, !load);
    uint32_t matrix_stride = 0;
    const std::string address = byte_address(pointer, &matrix_stride);

    const bool aliased = has_decoration(backing->self, DecorationAliased);
    const uint32_t memory_operand = load ? 3 : 2;
    const bool volatile_access = aliased || get_buffer_block_flags(backing->self).get(DecorationVolatile)
                                 || (instruction.length > memory_operand && (ops[memory_operand] & MemoryAccessVolatileMask));
    // aliasing and explicit volatile accesses retain ordered scalar memory operations
    const std::string qualifiers = std::string(volatile_access ? "volatile " : "") + (private_storage ? "__private" : "__global");
    const std::string temporary = std::string(aliased ? "momo_aliased_" : "momo_buffer_value_") + std::to_string(ops[1]);
    if (load)
        statement(type_to_glsl(type), " ", temporary, ";");
    else
    {
        flush_variable_declaration(ops[1]);
        const std::string value = to_expression(ops[1]);
        if (uses_workgroup_storage)
            statement("if (momo_active != 0)");
        begin_scope();
        statement(type_to_glsl(type), " ", temporary, " = ", value, ";");
    }
    SPIRType scalar_type = type;
    scalar_type.vecsize = scalar_type.columns = 1;
    const uint32_t bytes = type.width / 8;
    static const char components[] = "xyzw";
    for (uint32_t column = 0; column < type.columns; column++)
        for (uint32_t lane = 0; lane < type.vecsize; lane++)
        {
            std::string scalar = temporary;
            if (type.columns > 1)
                scalar += ".c[" + std::to_string(column) + "]";
            if (type.vecsize > 1)
                scalar += "." + std::string(1, components[lane]);
            const uint32_t offset = column * (matrix_stride ? matrix_stride : type.vecsize * bytes) + lane * bytes;
            if (load)
                statement(scalar, " = ", byte_addressed_load(scalar_type, address, offset, qualifiers), ";");
            else
            {
                const std::string packed = temporary + "_bits_" + std::to_string(column) + "_" + std::to_string(lane);
                emit_byte_addressed_store(scalar_type, address, offset, scalar, packed, qualifiers);
            }
        }
    if (load)
    {
        emit_buffer_load_result(ops[0], ops[1], pointer, temporary);
    }
    else
    {
        end_scope();
        register_write(pointer);
    }
    return true;
}

} // namespace momoten
